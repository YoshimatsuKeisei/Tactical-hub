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

export type PackedBcBatch = {
  payload: Buffer;
  tensors: PackedTensorDescriptor[];
  batchSize: number;
};

export function packBcEncodedSamples(samples: BcEncodedSample[], featureSpec: RlFeatureSpec): PackedBcBatch {
  if (!samples.length) throw new Error("Cannot pack an empty BC batch");
  const observations = samples.map((sample) => sample.observation);
  const floats: PendingTensor[] = [];
  const masks: PendingTensor[] = [];
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
    const rows = observations.map((value) => value[name]);
    if (rows.some((batch) => batch.some((row) => row.length !== width))) throw new Error(`${name} feature width does not match Feature Spec`);
    const [tensor, presence] = paddedFloatRows(name, rows, width);
    floats.push(tensor);
    masks.push(maskTensor(maskName, observations.map((value) => value[maskName]), tensor.shape[1], presence));
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
  const [actions, actionPresence] = paddedFloatRows("actions", actionRows, featureSpec.actionFeatureWidth);
  floats.push(actions);
  masks.push({ name: "actionMask", dtype: "uint8", shape: [samples.length, actions.shape[1]], bytes: Buffer.from(actionPresence.buffer) });
  const targets = Int32Array.from(samples.map((sample) => sample.targetIndex));
  const integers: PendingTensor[] = [{ name: "targets", dtype: "int32", shape: [samples.length], bytes: Buffer.from(targets.buffer) }];

  let byteOffset = 0;
  const tensors: PackedTensorDescriptor[] = [];
  const buffers: Buffer[] = [];
  for (const tensor of [...floats, ...integers, ...masks]) {
    tensors.push({ name: tensor.name, dtype: tensor.dtype, shape: tensor.shape, byteOffset, byteLength: tensor.bytes.byteLength });
    buffers.push(tensor.bytes);
    byteOffset += tensor.bytes.byteLength;
  }
  return { payload: Buffer.concat(buffers, byteOffset), tensors, batchSize: samples.length };
}


const VARIABLE_BATCH_ROW_TENSORS = new Set([
  "teams",
  "units",
  "bases",
  "constructions",
  "map",
  "actions",
  ...strategicNames.map((name) => `strategic.${name}`),
  "teamMask",
  "unitMask",
  "baseMask",
  "constructionMask",
  "mapMask",
  "actionMask",
  ...strategicNames.map((name) => `strategicMask.${name}`),
]);

const PACKED_DTYPE_BYTES: Record<PackedTensorDescriptor["dtype"], number> = {
  float32: 4,
  int32: 4,
  uint8: 1,
};

/**
 * Combine batch-size-1 packed samples without reconstructing JS number arrays.
 * Tensor bytes remain bit-identical; only batch padding/layout is rebuilt.
 */
export function combinePackedSingleSampleBatches(
  samples: PackedBcBatch[],
): PackedBcBatch {
  if (!samples.length) {
    throw new Error("Cannot combine an empty packed sample list");
  }
  if (samples.some((sample) => sample.batchSize !== 1)) {
    throw new Error("Packed worker samples must all have batchSize=1");
  }

  const template = samples[0].tensors;
  for (let sampleIndex = 1; sampleIndex < samples.length; sampleIndex += 1) {
    const descriptors = samples[sampleIndex].tensors;
    if (descriptors.length !== template.length) {
      throw new Error("Packed worker tensor count mismatch");
    }
    for (let index = 0; index < template.length; index += 1) {
      const left = template[index];
      const right = descriptors[index];
      if (left.name !== right.name || left.dtype !== right.dtype) {
        throw new Error(
          `Packed worker tensor mismatch at ${index}: ${left.name}/${right.name}`,
        );
      }
    }
  }

  let byteOffset = 0;
  const tensors: PackedTensorDescriptor[] = [];
  const buffers: Buffer[] = [];

  for (let descriptorIndex = 0; descriptorIndex < template.length; descriptorIndex += 1) {
    const descriptors = samples.map((sample) => sample.tensors[descriptorIndex]);
    const first = descriptors[0];
    const itemBytes = PACKED_DTYPE_BYTES[first.dtype];

    if (VARIABLE_BATCH_ROW_TENSORS.has(first.name)) {
      if (first.shape.length !== 2 && first.shape.length !== 3) {
        throw new Error(
          `Unexpected variable packed tensor rank for ${first.name}: ${first.shape.length}`,
        );
      }

      const trailing = first.shape.slice(2);
      for (const descriptor of descriptors) {
        if (descriptor.shape[0] !== 1 || descriptor.shape.length !== first.shape.length) {
          throw new Error(`Packed worker variable shape mismatch: ${first.name}`);
        }
        if (
          descriptor.shape.slice(2).length !== trailing.length
          || descriptor.shape.slice(2).some((value, index) => value !== trailing[index])
        ) {
          throw new Error(`Packed worker variable width mismatch: ${first.name}`);
        }
      }

      const maxRows = Math.max(...descriptors.map((descriptor) => descriptor.shape[1]));
      const rowWidth = first.shape.length === 3 ? first.shape[2] : 1;
      const rowBytes = rowWidth * itemBytes;
      const output = Buffer.alloc(samples.length * maxRows * rowBytes);

      descriptors.forEach((descriptor, sampleIndex) => {
        const source = samples[sampleIndex].payload.subarray(
          descriptor.byteOffset,
          descriptor.byteOffset + descriptor.byteLength,
        );
        const expectedBytes = descriptor.shape[1] * rowBytes;
        if (source.byteLength !== expectedBytes) {
          throw new Error(`Packed worker byte length mismatch: ${first.name}`);
        }
        source.copy(output, sampleIndex * maxRows * rowBytes);
      });

      const shape = first.shape.length === 3
        ? [samples.length, maxRows, first.shape[2]]
        : [samples.length, maxRows];
      tensors.push({
        name: first.name,
        dtype: first.dtype,
        shape,
        byteOffset,
        byteLength: output.byteLength,
      });
      buffers.push(output);
      byteOffset += output.byteLength;
      continue;
    }

    const trailingShape = first.shape.slice(1);
    for (const descriptor of descriptors) {
      if (
        descriptor.shape[0] !== 1
        || descriptor.shape.length !== first.shape.length
        || descriptor.shape.slice(1).some(
          (value, index) => value !== trailingShape[index],
        )
      ) {
        throw new Error(`Packed worker fixed tensor shape mismatch: ${first.name}`);
      }
    }

    const sampleByteLength = first.byteLength;
    if (descriptors.some((descriptor) => descriptor.byteLength !== sampleByteLength)) {
      throw new Error(`Packed worker fixed byte length mismatch: ${first.name}`);
    }

    const output = Buffer.allocUnsafe(samples.length * sampleByteLength);
    descriptors.forEach((descriptor, sampleIndex) => {
      const source = samples[sampleIndex].payload.subarray(
        descriptor.byteOffset,
        descriptor.byteOffset + descriptor.byteLength,
      );
      source.copy(output, sampleIndex * sampleByteLength);
    });

    tensors.push({
      name: first.name,
      dtype: first.dtype,
      shape: [samples.length, ...trailingShape],
      byteOffset,
      byteLength: output.byteLength,
    });
    buffers.push(output);
    byteOffset += output.byteLength;
  }

  return {
    payload: Buffer.concat(buffers, byteOffset),
    tensors,
    batchSize: samples.length,
  };
}
