import type { BcEncodedSample } from "./pythonBcTrainerClient";
import type { SparseActionRow } from "./rlActionEncoder";
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
  directSparseActions?: SparseActionRow[][];
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
      const declaredLogicalRows = observations.map(
        (value) => value.rowCompaction?.[name],
      );
      const directCompact = declaredLogicalRows.some(
        (count) => count !== undefined,
      );

      if (directCompact) {
        if (declaredLogicalRows.some((count) => count === undefined)) {
          throw new Error(
            `${name} direct compact rows require metadata for every sample`,
          );
        }
        const originalRows = declaredLogicalRows[0] ?? 0;
        if (
          declaredLogicalRows.some(
            (count) => count !== originalRows,
          )
        ) {
          throw new Error(
            `${name} direct compact rows require a stable logical row count`,
          );
        }
        const validCounts = rows.map((batch, index) => {
          if (batch.length > originalRows) {
            throw new Error(
              `${name} direct compact row count exceeds logical row count`,
            );
          }
          const mask = explicitMasks[index];
          if (mask.length !== batch.length || mask.some((value) => !value)) {
            throw new Error(
              `${name} direct compact mask must be all-valid and match rows`,
            );
          }
          return batch.length;
        });
        rows = rows.map((batch, index) => {
          if (validCounts[index] > 0) return batch;
          // Preserve the existing autograd behavior for an empty branch.
          return [Array(width).fill(0)];
        });
        explicitMasks = validCounts.map((valid) =>
          valid > 0 ? Array(valid).fill(1) : [0]
        );
        rowCompaction[name] = originalRows;
      } else {
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
  const directSparseActions = options.directSparseActions;
  if (directSparseActions && !options.sparseActions) {
    throw new Error("direct sparse actions require sparseActions transport");
  }
  if (directSparseActions && directSparseActions.length !== samples.length) {
    throw new Error("direct sparse action batch size does not match samples");
  }

  const actionRows = directSparseActions
    ? undefined
    : samples.map((sample) => sample.actions);
  if (
    actionRows
    && actionRows.some(
      (batch) => batch.some(
        (row) => row.length !== featureSpec.actionFeatureWidth,
      ),
    )
  ) {
    throw new Error("action feature width does not match Feature Spec");
  }
  if (directSparseActions) {
    for (const rows of directSparseActions) {
      for (const row of rows) {
        if (row.width !== featureSpec.actionFeatureWidth) {
          throw new Error(
            "direct sparse action feature width does not match Feature Spec",
          );
        }
        if (row.indices.length !== row.values.length) {
          throw new Error("direct sparse action index/value count mismatch");
        }
        let previous = -1;
        for (const index of row.indices) {
          if (
            !Number.isInteger(index)
            || index < 0
            || index >= featureSpec.actionFeatureWidth
            || index <= previous
          ) {
            throw new Error(
              "direct sparse action indices must be strictly increasing and in range",
            );
          }
          previous = index;
        }
      }
    }
  }

  const actionRowCounts = directSparseActions
    ? directSparseActions.map((rows) => rows.length)
    : actionRows!.map((rows) => rows.length);
  const maxActionRows = Math.max(0, ...actionRowCounts);
  const actionPresence = new Uint8Array(samples.length * maxActionRows);
  actionRowCounts.forEach((count, batch) => {
    for (let rowIndex = 0; rowIndex < count; rowIndex += 1) {
      actionPresence[batch * maxActionRows + rowIndex] = 1;
    }
  });

  const targets = Int32Array.from(samples.map((sample) => sample.targetIndex));
  const integers: PendingTensor[] = [{ name: "targets", dtype: "int32", shape: [samples.length], bytes: Buffer.from(targets.buffer) }];
  let actionSparseShape: [number, number, number] | undefined;
  if (options.sparseActions) {
    const sparseIndices: number[] = [];
    const sparseValues: number[] = [];
    if (directSparseActions) {
      directSparseActions.forEach((rows, batch) => rows.forEach((row, rowIndex) => {
        const base = (
          (batch * maxActionRows + rowIndex)
          * featureSpec.actionFeatureWidth
        );
        row.indices.forEach((featureIndex, offset) => {
          sparseIndices.push(base + featureIndex);
          sparseValues.push(row.values[offset]);
        });
      }));
    } else {
      actionRows!.forEach((rows, batch) => rows.forEach((row, rowIndex) => row.forEach((item, featureIndex) => {
        if (item !== 0) {
          sparseIndices.push((batch * maxActionRows + rowIndex) * featureSpec.actionFeatureWidth + featureIndex);
          sparseValues.push(item);
        }
      })));
    }
    const indices = Int32Array.from(sparseIndices);
    const values = Float32Array.from(sparseValues);
    integers.push({ name: "actionSparseIndices", dtype: "int32", shape: [indices.length], bytes: Buffer.from(indices.buffer) });
    floats.push({ name: "actionSparseValues", dtype: "float32", shape: [values.length], bytes: Buffer.from(values.buffer) });
    actionSparseShape = [samples.length, maxActionRows, featureSpec.actionFeatureWidth];
  } else {
    const [actions] = paddedFloatRows(
      "actions",
      actionRows!,
      featureSpec.actionFeatureWidth,
    );
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

function sameRowCompaction(
  left: PackedRowCompaction | undefined,
  right: PackedRowCompaction | undefined,
) {
  return (
    left?.units === right?.units
    && left?.bases === right?.bases
    && left?.constructions === right?.constructions
  );
}

/**
 * Combines batch-size-1 packed samples without rebuilding JS number arrays.
 *
 * V7-D uses this to let rollout workers perform compact observation + sparse
 * action packing locally, transfer only packed bytes/metadata to the parent,
 * and still reconstruct exactly the same central batch layout as the existing
 * packPpoActBatchInput path.
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
  const sparse = samples[0].actionSparseShape !== undefined;
  const rowCompaction = samples[0].rowCompaction;

  for (let sampleIndex = 1; sampleIndex < samples.length; sampleIndex += 1) {
    const sample = samples[sampleIndex];
    const descriptors = sample.tensors;
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
    if ((sample.actionSparseShape !== undefined) !== sparse) {
      throw new Error("Packed worker sparse-action mode mismatch");
    }
    if (!sameRowCompaction(rowCompaction, sample.rowCompaction)) {
      throw new Error("Packed worker row-compaction metadata mismatch");
    }
  }

  let sparseMaxRows = 0;
  let sparseWidth = 0;
  if (sparse) {
    for (const sample of samples) {
      const shape = sample.actionSparseShape;
      if (!shape || shape[0] !== 1) {
        throw new Error("Packed worker sparse-action shape must be batch-size 1");
      }
      if (!sparseWidth) sparseWidth = shape[2];
      if (shape[2] !== sparseWidth) {
        throw new Error("Packed worker sparse-action width mismatch");
      }
      sparseMaxRows = Math.max(sparseMaxRows, shape[1]);
    }
  }

  let byteOffset = 0;
  const tensors: PackedTensorDescriptor[] = [];
  const buffers: Buffer[] = [];

  const push = (
    descriptor: Omit<PackedTensorDescriptor, "byteOffset" | "byteLength">,
    output: Buffer,
  ) => {
    tensors.push({
      ...descriptor,
      byteOffset,
      byteLength: output.byteLength,
    });
    buffers.push(output);
    byteOffset += output.byteLength;
  };

  for (
    let descriptorIndex = 0;
    descriptorIndex < template.length;
    descriptorIndex += 1
  ) {
    const descriptors = samples.map(
      (sample) => sample.tensors[descriptorIndex],
    );
    const first = descriptors[0];
    const itemBytes = PACKED_DTYPE_BYTES[first.dtype];

    if (first.name === "actionSparseIndices") {
      if (!sparse || first.dtype !== "int32") {
        throw new Error("Unexpected sparse-action index tensor");
      }
      const remapped: number[] = [];
      descriptors.forEach((descriptor, sampleIndex) => {
        if (descriptor.shape.length !== 1) {
          throw new Error("Sparse-action indices must be rank 1");
        }
        const source = samples[sampleIndex].payload.subarray(
          descriptor.byteOffset,
          descriptor.byteOffset + descriptor.byteLength,
        );
        const local = new Int32Array(
          source.buffer,
          source.byteOffset,
          source.byteLength / Int32Array.BYTES_PER_ELEMENT,
        );
        for (const flatIndex of local) {
          const row = Math.floor(flatIndex / sparseWidth);
          const featureIndex = flatIndex % sparseWidth;
          if (
            row < 0
            || row >= (samples[sampleIndex].actionSparseShape?.[1] ?? 0)
            || featureIndex < 0
            || featureIndex >= sparseWidth
          ) {
            throw new Error("Sparse-action index is outside the local shape");
          }
          remapped.push(
            (
              (sampleIndex * sparseMaxRows + row)
              * sparseWidth
            ) + featureIndex,
          );
        }
      });
      const outputArray = Int32Array.from(remapped);
      push(
        {
          name: first.name,
          dtype: first.dtype,
          shape: [outputArray.length],
        },
        Buffer.from(
          outputArray.buffer,
          outputArray.byteOffset,
          outputArray.byteLength,
        ),
      );
      continue;
    }

    if (first.name === "actionSparseValues") {
      if (!sparse || first.dtype !== "float32") {
        throw new Error("Unexpected sparse-action values tensor");
      }
      const totalBytes = descriptors.reduce(
        (sum, descriptor) => sum + descriptor.byteLength,
        0,
      );
      const output = Buffer.allocUnsafe(totalBytes);
      let cursor = 0;
      descriptors.forEach((descriptor, sampleIndex) => {
        if (descriptor.shape.length !== 1) {
          throw new Error("Sparse-action values must be rank 1");
        }
        const source = samples[sampleIndex].payload.subarray(
          descriptor.byteOffset,
          descriptor.byteOffset + descriptor.byteLength,
        );
        source.copy(output, cursor);
        cursor += source.byteLength;
      });
      push(
        {
          name: first.name,
          dtype: first.dtype,
          shape: [totalBytes / Float32Array.BYTES_PER_ELEMENT],
        },
        output,
      );
      continue;
    }

    if (VARIABLE_BATCH_ROW_TENSORS.has(first.name)) {
      if (first.shape.length !== 2 && first.shape.length !== 3) {
        throw new Error(
          `Unexpected variable packed tensor rank for ${first.name}: ${first.shape.length}`,
        );
      }

      const trailing = first.shape.slice(2);
      for (const descriptor of descriptors) {
        if (
          descriptor.shape[0] !== 1
          || descriptor.shape.length !== first.shape.length
        ) {
          throw new Error(
            `Packed worker variable shape mismatch: ${first.name}`,
          );
        }
        if (
          descriptor.shape.slice(2).length !== trailing.length
          || descriptor.shape.slice(2).some(
            (value, index) => value !== trailing[index],
          )
        ) {
          throw new Error(
            `Packed worker variable width mismatch: ${first.name}`,
          );
        }
      }

      const maxRows = Math.max(
        ...descriptors.map((descriptor) => descriptor.shape[1]),
      );
      const rowWidth = first.shape.length === 3
        ? first.shape[2]
        : 1;
      const rowBytes = rowWidth * itemBytes;
      const output = Buffer.alloc(
        samples.length * maxRows * rowBytes,
      );

      descriptors.forEach((descriptor, sampleIndex) => {
        const source = samples[sampleIndex].payload.subarray(
          descriptor.byteOffset,
          descriptor.byteOffset + descriptor.byteLength,
        );
        const expectedBytes = descriptor.shape[1] * rowBytes;
        if (source.byteLength !== expectedBytes) {
          throw new Error(
            `Packed worker byte length mismatch: ${first.name}`,
          );
        }
        source.copy(
          output,
          sampleIndex * maxRows * rowBytes,
        );
      });

      push(
        {
          name: first.name,
          dtype: first.dtype,
          shape: first.shape.length === 3
            ? [samples.length, maxRows, first.shape[2]]
            : [samples.length, maxRows],
        },
        output,
      );
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
        throw new Error(
          `Packed worker fixed tensor shape mismatch: ${first.name}`,
        );
      }
    }

    const sampleByteLength = first.byteLength;
    if (
      descriptors.some(
        (descriptor) => descriptor.byteLength !== sampleByteLength,
      )
    ) {
      throw new Error(
        `Packed worker fixed byte length mismatch: ${first.name}`,
      );
    }

    const output = Buffer.allocUnsafe(
      samples.length * sampleByteLength,
    );
    descriptors.forEach((descriptor, sampleIndex) => {
      const source = samples[sampleIndex].payload.subarray(
        descriptor.byteOffset,
        descriptor.byteOffset + descriptor.byteLength,
      );
      source.copy(output, sampleIndex * sampleByteLength);
    });

    push(
      {
        name: first.name,
        dtype: first.dtype,
        shape: [samples.length, ...trailingShape],
      },
      output,
    );
  }

  return {
    payload: Buffer.concat(buffers, byteOffset),
    tensors,
    batchSize: samples.length,
    ...(rowCompaction
      ? { rowCompaction: { ...rowCompaction } }
      : {}),
    ...(sparse
      ? {
          actionSparseShape: [
            samples.length,
            sparseMaxRows,
            sparseWidth,
          ] as [number, number, number],
        }
      : {}),
  };
}
