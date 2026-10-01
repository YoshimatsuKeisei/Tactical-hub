import {
  type PackedBcBatch,
  type PackedRowCompaction,
  type PackedTensorDescriptor,
} from "./rlBcPackedBatch";

export type TransferablePackedBcBatch = {
  payload: Uint8Array;
  tensors: PackedTensorDescriptor[];
  batchSize: number;
  rowCompaction?: PackedRowCompaction;
  actionSparseShape?: [number, number, number];
};

const VARIABLE_BATCH_ROW_TENSORS = new Set([
  "teams",
  "units",
  "bases",
  "constructions",
  "map",
  "actionMask",
  "teamMask",
  "unitMask",
  "baseMask",
  "constructionMask",
  "mapMask",
  "strategic.siegeStates",
  "strategic.kingCampaignStates",
  "strategic.rewardPlacementRequests",
  "strategic.strategistCooldowns",
  "strategic.teleportCooldowns",
  "strategic.productionIntents",
  "strategic.movementIntents",
  "strategic.attackIntents",
  "strategic.strategistActionIntents",
  "strategic.teleportIntents",
  "strategicMask.siegeStates",
  "strategicMask.kingCampaignStates",
  "strategicMask.rewardPlacementRequests",
  "strategicMask.strategistCooldowns",
  "strategicMask.teleportCooldowns",
  "strategicMask.productionIntents",
  "strategicMask.movementIntents",
  "strategicMask.attackIntents",
  "strategicMask.strategistActionIntents",
  "strategicMask.teleportIntents",
]);

const PACKED_DTYPE_BYTES: Record<
  PackedTensorDescriptor["dtype"],
  number
> = {
  float32: 4,
  int32: 4,
  uint8: 1,
};

export function toTransferablePackedBcBatch(
  packed: PackedBcBatch,
): TransferablePackedBcBatch {
  const payload = new Uint8Array(packed.payload.byteLength);
  payload.set(packed.payload);
  return {
    payload,
    tensors: packed.tensors,
    batchSize: packed.batchSize,
    ...(packed.rowCompaction
      ? { rowCompaction: packed.rowCompaction }
      : {}),
    ...(packed.actionSparseShape
      ? { actionSparseShape: packed.actionSparseShape }
      : {}),
  };
}

export function fromTransferablePackedBcBatch(
  packed: TransferablePackedBcBatch,
): PackedBcBatch {
  return {
    payload: Buffer.from(
      packed.payload.buffer,
      packed.payload.byteOffset,
      packed.payload.byteLength,
    ),
    tensors: packed.tensors,
    batchSize: packed.batchSize,
    ...(packed.rowCompaction
      ? { rowCompaction: packed.rowCompaction }
      : {}),
    ...(packed.actionSparseShape
      ? { actionSparseShape: packed.actionSparseShape }
      : {}),
  };
}

function tensorBytes(
  packed: PackedBcBatch,
  descriptor: PackedTensorDescriptor,
) {
  return packed.payload.subarray(
    descriptor.byteOffset,
    descriptor.byteOffset + descriptor.byteLength,
  );
}

function combineRowCompaction(
  samples: PackedBcBatch[],
): PackedRowCompaction | undefined {
  const names = [
    "units",
    "bases",
    "constructions",
  ] as const;
  const result: PackedRowCompaction = {};

  for (const name of names) {
    const values = samples.map(
      (sample) => sample.rowCompaction?.[name],
    );
    const declared = values.filter(
      (value): value is number => value !== undefined,
    );
    if (!declared.length) continue;
    if (
      declared.length !== samples.length
      || declared.some((value) => value !== declared[0])
    ) {
      throw new Error(
        `Packed worker rowCompaction mismatch for ${name}`,
      );
    }
    result[name] = declared[0];
  }

  return Object.keys(result).length ? result : undefined;
}

function combineSparseActions(
  batches: PackedBcBatch[],
  descriptorByBatch: Array<Map<string, PackedTensorDescriptor>>,
) {
  const shapes = batches.map((batch) => batch.actionSparseShape);
  if (shapes.some((shape) => !shape)) {
    throw new Error("Sparse worker batches require actionSparseShape");
  }
  const width = shapes[0]![2];
  if (shapes.some((shape, index) =>
    shape![0] !== batches[index].batchSize || shape![2] !== width)) {
    throw new Error("Sparse worker action shape mismatch");
  }
  const maxRows = Math.max(0, ...shapes.map((shape) => shape![1]));
  const indices: number[] = [];
  const values: number[] = [];
  let sampleOffset = 0;
  batches.forEach((batch, batchIndex) => {
    const shape = shapes[batchIndex]!;
    const localRows = shape[1];
    const indexDescriptor = descriptorByBatch[batchIndex].get("actionSparseIndices");
    const valueDescriptor = descriptorByBatch[batchIndex].get("actionSparseValues");
    if (!indexDescriptor || !valueDescriptor) throw new Error("Sparse worker tensors missing");
    const indexBytes = tensorBytes(batch, indexDescriptor);
    const valueBytes = tensorBytes(batch, valueDescriptor);
    const localIndices = new Int32Array(indexBytes.buffer, indexBytes.byteOffset, indexDescriptor.byteLength / 4);
    const localValues = new Float32Array(valueBytes.buffer, valueBytes.byteOffset, valueDescriptor.byteLength / 4);
    if (localIndices.length !== localValues.length) throw new Error("Sparse worker index/value mismatch");
    const localPlane = localRows * width;
    if (localPlane <= 0 && localIndices.length) throw new Error("Sparse worker rows must be positive");
    for (let offset = 0; offset < localIndices.length; offset += 1) {
      const localIndex = localIndices[offset];
      const localSample = Math.floor(localIndex / localPlane);
      const withinSample = localIndex % localPlane;
      const row = Math.floor(withinSample / width);
      const feature = withinSample % width;
      if (localSample < 0 || localSample >= batch.batchSize || row < 0 || row >= localRows) {
        throw new Error("Sparse worker index out of range");
      }
      indices.push(((sampleOffset + localSample) * maxRows + row) * width + feature);
      values.push(localValues[offset]);
    }
    sampleOffset += batch.batchSize;
  });
  const packedIndices = Int32Array.from(indices);
  const packedValues = Float32Array.from(values);
  return {
    shape: [batches.reduce((sum, batch) => sum + batch.batchSize, 0), maxRows, width] as [number, number, number],
    indices: Buffer.from(packedIndices.buffer, packedIndices.byteOffset, packedIndices.byteLength),
    values: Buffer.from(packedValues.buffer, packedValues.byteOffset, packedValues.byteLength),
  };
}

/**
 * Rebuild the exact central packed-v1 batch from independent batch-size-1
 * worker payloads. No JS observation/action arrays are reconstructed.
 */
export function combinePackedWorkerBatchesV7(
  batches: PackedBcBatch[],
): PackedBcBatch {
  if (!batches.length) throw new Error("Cannot combine empty worker batches");
  if (batches.some((batch) => batch.batchSize <= 0)) throw new Error("Worker batchSize must be positive");
  const totalBatchSize = batches.reduce((sum, batch) => sum + batch.batchSize, 0);
  const template = batches[0].tensors;
  const descriptorByBatch = batches.map((batch) =>
    new Map(batch.tensors.map((descriptor) => [descriptor.name, descriptor])));
  for (let batchIndex = 1; batchIndex < batches.length; batchIndex += 1) {
    const descriptors = batches[batchIndex].tensors;
    if (descriptors.length !== template.length) throw new Error("Worker tensor count mismatch");
    for (let index = 0; index < template.length; index += 1) {
      const left = template[index];
      const right = descriptors[index];
      if (left.name !== right.name || left.dtype !== right.dtype) {
        throw new Error("Worker tensor mismatch at " + index + ": " + left.name + "/" + right.name);
      }
    }
  }

  const sparse = batches[0].actionSparseShape
    ? combineSparseActions(batches, descriptorByBatch)
    : undefined;
  if (Boolean(sparse) !== batches.every((batch) => Boolean(batch.actionSparseShape))) {
    throw new Error("Worker batches cannot mix sparse and dense actions");
  }

  let byteOffset = 0;
  const tensors: PackedTensorDescriptor[] = [];
  const buffers: Buffer[] = [];
  for (const first of template) {
    const descriptors = descriptorByBatch.map((map) => {
      const descriptor = map.get(first.name);
      if (!descriptor) throw new Error("Worker tensor missing: " + first.name);
      return descriptor;
    });

    if (first.name === "actionSparseIndices") {
      if (!sparse) throw new Error("Sparse indices without sparse shape");
      tensors.push({ name: first.name, dtype: "int32", shape: [sparse.indices.byteLength / 4], byteOffset, byteLength: sparse.indices.byteLength });
      buffers.push(sparse.indices); byteOffset += sparse.indices.byteLength; continue;
    }
    if (first.name === "actionSparseValues") {
      if (!sparse) throw new Error("Sparse values without sparse shape");
      tensors.push({ name: first.name, dtype: "float32", shape: [sparse.values.byteLength / 4], byteOffset, byteLength: sparse.values.byteLength });
      buffers.push(sparse.values); byteOffset += sparse.values.byteLength; continue;
    }

    const itemBytes = PACKED_DTYPE_BYTES[first.dtype];
    if (VARIABLE_BATCH_ROW_TENSORS.has(first.name)) {
      if (first.shape.length !== 2 && first.shape.length !== 3) throw new Error("Unexpected variable tensor rank");
      const trailing = first.shape.slice(2);
      descriptors.forEach((descriptor, batchIndex) => {
        if (descriptor.shape[0] !== batches[batchIndex].batchSize
          || descriptor.shape.length !== first.shape.length
          || descriptor.shape.slice(2).some((value, index) => value !== trailing[index])) {
          throw new Error("Worker variable shape mismatch: " + first.name);
        }
      });
      const maxRows = Math.max(0, ...descriptors.map((descriptor) => descriptor.shape[1]));
      const rowWidth = first.shape.length === 3 ? first.shape[2] : 1;
      const rowBytes = rowWidth * itemBytes;
      const output = Buffer.alloc(totalBatchSize * maxRows * rowBytes);
      let sampleOffset = 0;
      descriptors.forEach((descriptor, batchIndex) => {
        const batch = batches[batchIndex];
        const localRows = descriptor.shape[1];
        const source = tensorBytes(batch, descriptor);
        if (source.byteLength !== batch.batchSize * localRows * rowBytes) {
          throw new Error("Worker byte length mismatch: " + first.name);
        }
        for (let localSample = 0; localSample < batch.batchSize; localSample += 1) {
          const sourceStart = localSample * localRows * rowBytes;
          source.copy(output, (sampleOffset + localSample) * maxRows * rowBytes,
            sourceStart, sourceStart + localRows * rowBytes);
        }
        sampleOffset += batch.batchSize;
      });
      tensors.push({
        name: first.name,
        dtype: first.dtype,
        shape: first.shape.length === 3
          ? [totalBatchSize, maxRows, first.shape[2]]
          : [totalBatchSize, maxRows],
        byteOffset,
        byteLength: output.byteLength,
      });
      buffers.push(output); byteOffset += output.byteLength; continue;
    }

    const trailingShape = first.shape.slice(1);
    const trailingItems = trailingShape.reduce((product, value) => product * value, 1);
    const parts: Buffer[] = [];
    let fixedBytes = 0;
    descriptors.forEach((descriptor, batchIndex) => {
      const batch = batches[batchIndex];
      if (descriptor.shape[0] !== batch.batchSize
        || descriptor.shape.length !== first.shape.length
        || descriptor.shape.slice(1).some((value, index) => value !== trailingShape[index])) {
        throw new Error("Worker fixed shape mismatch: " + first.name);
      }
      if (descriptor.byteLength !== batch.batchSize * trailingItems * itemBytes) {
        throw new Error("Worker fixed byte length mismatch: " + first.name);
      }
      const part = tensorBytes(batch, descriptor);
      parts.push(part); fixedBytes += part.byteLength;
    });
    const output = Buffer.concat(parts, fixedBytes);
    tensors.push({ name: first.name, dtype: first.dtype, shape: [totalBatchSize, ...trailingShape], byteOffset, byteLength: output.byteLength });
    buffers.push(output); byteOffset += output.byteLength;
  }

  const rowCompaction = combineRowCompaction(batches);
  return {
    payload: Buffer.concat(buffers, byteOffset),
    tensors,
    batchSize: totalBatchSize,
    ...(rowCompaction ? { rowCompaction } : {}),
    ...(sparse ? { actionSparseShape: sparse.shape } : {}),
  };
}
