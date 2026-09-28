import type { BcEncodedSample } from "./pythonBcTrainerClient";
import type { RlFeatureSpec } from "./rlFeatureSpec";

export type PackedTensorDescriptor = {
  name: string;
  dtype: "float32" | "int32" | "uint8";
  shape: number[];
  byteOffset: number;
  byteLength: number;
};

type PendingTensor = Omit<PackedTensorDescriptor, "byteOffset" | "byteLength"> & { bytes: Buffer };
const strategicNames = [
  "siegeStates", "kingCampaignStates", "rewardPlacementRequests", "strategistCooldowns",
  "teleportCooldowns", "productionIntents", "movementIntents", "attackIntents",
  "strategistActionIntents", "teleportIntents",
] as const;

function floatMatrix(name: string, rows: number[][]): PendingTensor {
  const width = rows[0]?.length ?? 0;
  const values = new Float32Array(rows.length * width);
  rows.forEach((row, index) => values.set(row, index * width));
  return { name, dtype: "float32", shape: [rows.length, width], bytes: Buffer.from(values.buffer) };
}

function paddedFloatRows(name: string, values: number[][][], width: number): [PendingTensor, Uint8Array] {
  const maxRows = Math.max(0, ...values.map((rows) => rows.length));
  const packed = new Float32Array(values.length * maxRows * width);
  const presence = new Uint8Array(values.length * maxRows);
  values.forEach((rows, batch) => rows.forEach((row, rowIndex) => {
    packed.set(row, (batch * maxRows + rowIndex) * width);
    presence[batch * maxRows + rowIndex] = 1;
  }));
  return [{ name, dtype: "float32", shape: [values.length, maxRows, width], bytes: Buffer.from(packed.buffer) }, presence];
}

function maskTensor(name: string, masks: number[][], maxRows: number, presence?: Uint8Array): PendingTensor {
  const packed = new Uint8Array(masks.length * maxRows);
  masks.forEach((mask, batch) => mask.slice(0, maxRows).forEach((value, row) => {
    packed[batch * maxRows + row] = Number(Boolean(value));
  }));
  if (presence) for (let index = 0; index < packed.length; index += 1) packed[index] &= presence[index];
  return { name, dtype: "uint8", shape: [masks.length, maxRows], bytes: Buffer.from(packed.buffer) };
}

export type PackedRowCompaction = Partial<
  Record<"units" | "bases" | "constructions", number>
>;

export type PackedBcBatch = {
  payload: Buffer;
  tensors: PackedTensorDescriptor[];
  batchSize: number;
  rowCompaction?: PackedRowCompaction;
  actionSparseShape?: [number, number, number];
};

export type PackBcEncodedSamplesOptions = {
  compactMaskedPrefixes?: boolean;
  sparseActions?: boolean;
};

function validMaskPrefixLength(
  mask: number[],
  expectedRows: number,
  name: string,
) {
  if (mask.length !== expectedRows) {
    throw new Error(
      `${name} mask length does not match row count: ${mask.length} != ${expectedRows}`,
    );
  }
  let valid = 0;
  while (valid < mask.length && Boolean(mask[valid])) valid += 1;
  if (mask.slice(valid).some(Boolean)) {
    throw new Error(`${name} mask is not a valid 1-prefix`);
  }
  return valid;
}

export function packBcEncodedSamples(
  samples: BcEncodedSample[],
  featureSpec: RlFeatureSpec,
  options: PackBcEncodedSamplesOptions = {},
): PackedBcBatch {
  if (!samples.length) throw new Error("Cannot pack an empty BC batch");
  const observations = samples.map((sample) => sample.observation);
  const floats: PendingTensor[] = [];
  const masks: PendingTensor[] = [];
  const rowCompaction: PackedRowCompaction = {};
  const checkedMatrix = (name: string, rows: number[][], width: number) => {
    if (rows.some((row) => row.length !== width)) throw new Error(`${name} feature width does not match Feature Spec`);
    return floatMatrix(name, rows);
  };
  floats.push(checkedMatrix("global", observations.map((value) => value.global), featureSpec.globalWidth));
  floats.push(checkedMatrix("strategicGlobal", observations.map((value) => value.strategicState.global), featureSpec.strategicGlobalWidth));

  for (const [name, maskName, width] of [
    ["teams", "teamMask", featureSpec.teamWidth],
    ["units", "unitMask", featureSpec.unitWidth],
    ["bases", "baseMask", featureSpec.baseWidth],
    ["constructions", "constructionMask", featureSpec.constructionWidth],
  ] as const) {
    let rows = observations.map((value) => value[name]);
    let explicitMasks = observations.map((value) => value[maskName]);
    if (rows.some((batch) => batch.some((row) => row.length !== width))) {
      throw new Error(`${name} feature width does not match Feature Spec`);
    }

    if (
      options.compactMaskedPrefixes
      && (
        name === "units"
        || name === "bases"
        || name === "constructions"
      )
    ) {
      const originalRowCounts = rows.map((batch) => batch.length);
      if (
        originalRowCounts.some(
          (count) => count !== originalRowCounts[0],
        )
      ) {
        throw new Error(
          `${name} compact rows require a stable logical row count`,
        );
      }
      const originalRows = originalRowCounts[0] ?? 0;
      const validCounts = explicitMasks.map((mask) =>
        validMaskPrefixLength(mask, originalRows, name),
      );
      rows = rows.map((batch, index) => {
        const valid = validCounts[index];
        if (valid > 0) return batch.slice(0, valid);
        // Keep one masked zero input row so the encoder remains in the
        // autograd graph and Adam observes grad=0 instead of grad=None.
        if (batch.length) return [batch[0].map(() => 0)];
        return [Array(width).fill(0)];
      });
      explicitMasks = explicitMasks.map((mask, index) => {
        const valid = validCounts[index];
        return valid > 0
          ? mask.slice(0, valid)
          : [0];
      });
      rowCompaction[name] = originalRows;
    }

    const [tensor, presence] = paddedFloatRows(
      name,
      rows,
      width,
    );
    floats.push(tensor);
    masks.push(
      maskTensor(
        maskName,
        explicitMasks,
        tensor.shape[1],
        presence,
      ),
    );
  }
  const mapRows = observations.map((value) => value.map.flat());
  if (mapRows.some((batch) => batch.some((row) => row.length !== featureSpec.mapTileWidth))) throw new Error("map feature width does not match Feature Spec");
  const [map, mapPresence] = paddedFloatRows("map", mapRows, featureSpec.mapTileWidth);
  floats.push(map);
  masks.push({ name: "mapMask", dtype: "uint8", shape: [samples.length, map.shape[1]], bytes: Buffer.from(mapPresence.buffer) });

  for (const name of strategicNames) {
    const rows = observations.map((value) => value.strategicState[name]);
    const width = featureSpec.strategicTableRowWidths[name];
    if (rows.some((batch) => batch.some((row) => row.length !== width))) throw new Error(`${name} feature width does not match Feature Spec`);
    const [tensor, presence] = paddedFloatRows(`strategic.${name}`, rows, width);
    floats.push(tensor);
    masks.push({ name: `strategicMask.${name}`, dtype: "uint8", shape: [samples.length, tensor.shape[1]], bytes: Buffer.from(presence.buffer) });
  }
  const actionRows = samples.map((sample) => sample.actions);
  if (actionRows.some((batch) => batch.some((row) => row.length !== featureSpec.actionFeatureWidth))) {
    throw new Error("action feature width does not match Feature Spec");
  }
  const maxActionRows = Math.max(0, ...actionRows.map((rows) => rows.length));
  const actionPresence = new Uint8Array(samples.length * maxActionRows);
  actionRows.forEach((rows, batch) => rows.forEach((_row, rowIndex) => {
    actionPresence[batch * maxActionRows + rowIndex] = 1;
  }));
  const targets = Int32Array.from(samples.map((sample) => sample.targetIndex));
  const integers: PendingTensor[] = [{ name: "targets", dtype: "int32", shape: [samples.length], bytes: Buffer.from(targets.buffer) }];
  let actionSparseShape: [number, number, number] | undefined;
  if (options.sparseActions) {
    const sparseIndices: number[] = [];
    const sparseValues: number[] = [];
    actionRows.forEach((rows, batch) => rows.forEach((row, rowIndex) => row.forEach((item, featureIndex) => {
      if (item !== 0) {
        sparseIndices.push((batch * maxActionRows + rowIndex) * featureSpec.actionFeatureWidth + featureIndex);
        sparseValues.push(item);
      }
    })));
    const indices = Int32Array.from(sparseIndices);
    const values = Float32Array.from(sparseValues);
    integers.push({ name: "actionSparseIndices", dtype: "int32", shape: [indices.length], bytes: Buffer.from(indices.buffer) });
    floats.push({ name: "actionSparseValues", dtype: "float32", shape: [values.length], bytes: Buffer.from(values.buffer) });
    actionSparseShape = [samples.length, maxActionRows, featureSpec.actionFeatureWidth];
  } else {
    const [actions] = paddedFloatRows("actions", actionRows, featureSpec.actionFeatureWidth);
    floats.push(actions);
  }
  masks.push({ name: "actionMask", dtype: "uint8", shape: [samples.length, maxActionRows], bytes: Buffer.from(actionPresence.buffer) });

  let byteOffset = 0;
  const tensors: PackedTensorDescriptor[] = [];
  const buffers: Buffer[] = [];
  for (const tensor of [...floats, ...integers, ...masks]) {
    tensors.push({ name: tensor.name, dtype: tensor.dtype, shape: tensor.shape, byteOffset, byteLength: tensor.bytes.byteLength });
    buffers.push(tensor.bytes);
    byteOffset += tensor.bytes.byteLength;
  }
  return {
    payload: Buffer.concat(buffers, byteOffset),
    tensors,
    batchSize: samples.length,
    ...(Object.keys(rowCompaction).length
      ? { rowCompaction }
      : {}),
    ...(actionSparseShape ? { actionSparseShape } : {}),
  };
}
