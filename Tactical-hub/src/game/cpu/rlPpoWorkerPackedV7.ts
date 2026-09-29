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
  samples: PackedBcBatch[],
  descriptorBySample: Array<
    Map<string, PackedTensorDescriptor>
  >,
) {
  const shapes = samples.map(
    (sample) => sample.actionSparseShape,
  );
  if (shapes.some((shape) => !shape)) {
    throw new Error(
      "Packed worker sparse batches require actionSparseShape for every sample",
    );
  }

  const width = shapes[0]![2];
  if (
    shapes.some(
      (shape) =>
        shape![0] !== 1
        || shape![2] !== width,
    )
  ) {
    throw new Error("Packed worker sparse action shape mismatch");
  }

  const maxRows = Math.max(
    0,
    ...shapes.map((shape) => shape![1]),
  );
  const indices: number[] = [];
  const values: number[] = [];

  samples.forEach((sample, sampleIndex) => {
    const indexDescriptor = descriptorBySample[sampleIndex].get(
      "actionSparseIndices",
    );
    const valueDescriptor = descriptorBySample[sampleIndex].get(
      "actionSparseValues",
    );
    if (!indexDescriptor || !valueDescriptor) {
      throw new Error(
        "Packed worker sparse tensors are missing",
      );
    }
    if (
      indexDescriptor.dtype !== "int32"
      || valueDescriptor.dtype !== "float32"
    ) {
      throw new Error(
        "Packed worker sparse tensor dtype mismatch",
      );
    }

    const localIndices = new Int32Array(
      tensorBytes(sample, indexDescriptor).buffer,
      tensorBytes(sample, indexDescriptor).byteOffset,
      indexDescriptor.byteLength / 4,
    );
    const localValues = new Float32Array(
      tensorBytes(sample, valueDescriptor).buffer,
      tensorBytes(sample, valueDescriptor).byteOffset,
      valueDescriptor.byteLength / 4,
    );
    if (localIndices.length !== localValues.length) {
      throw new Error(
        "Packed worker sparse index/value length mismatch",
      );
    }

    for (let offset = 0; offset < localIndices.length; offset += 1) {
      const localIndex = localIndices[offset];
      const row = Math.floor(localIndex / width);
      const feature = localIndex % width;
      indices.push(
        (
          sampleIndex * maxRows
          + row
        ) * width + feature,
      );
      values.push(localValues[offset]);
    }
  });

  const packedIndices = Int32Array.from(indices);
  const packedValues = Float32Array.from(values);

  return {
    shape: [
      samples.length,
      maxRows,
      width,
    ] as [number, number, number],
    indices: Buffer.from(
      packedIndices.buffer,
      packedIndices.byteOffset,
      packedIndices.byteLength,
    ),
    values: Buffer.from(
      packedValues.buffer,
      packedValues.byteOffset,
      packedValues.byteLength,
    ),
  };
}

/**
 * Rebuild the exact central packed-v1 batch from independent batch-size-1
 * worker payloads. No JS observation/action arrays are reconstructed.
 */
export function combinePackedSingleSampleBatchesV7(
  samples: PackedBcBatch[],
): PackedBcBatch {
  if (!samples.length) {
    throw new Error(
      "Cannot combine an empty packed worker sample list",
    );
  }
  if (samples.some((sample) => sample.batchSize !== 1)) {
    throw new Error(
      "Packed V7 worker samples must all have batchSize=1",
    );
  }

  const template = samples[0].tensors;
  const descriptorBySample = samples.map(
    (sample) =>
      new Map(
        sample.tensors.map((descriptor) => [
          descriptor.name,
          descriptor,
        ]),
      ),
  );

  for (let sampleIndex = 1; sampleIndex < samples.length; sampleIndex += 1) {
    const descriptors = samples[sampleIndex].tensors;
    if (descriptors.length !== template.length) {
      throw new Error("Packed worker tensor count mismatch");
    }
    for (let index = 0; index < template.length; index += 1) {
      const left = template[index];
      const right = descriptors[index];
      if (
        left.name !== right.name
        || left.dtype !== right.dtype
      ) {
        throw new Error(
          `Packed worker tensor mismatch at ${index}: ${left.name}/${right.name}`,
        );
      }
    }
  }

  const sparse = samples[0].actionSparseShape
    ? combineSparseActions(samples, descriptorBySample)
    : undefined;
  if (
    Boolean(sparse)
    !== samples.every(
      (sample) => Boolean(sample.actionSparseShape),
    )
  ) {
    throw new Error(
      "Packed worker samples cannot mix sparse and dense actions",
    );
  }

  let byteOffset = 0;
  const tensors: PackedTensorDescriptor[] = [];
  const buffers: Buffer[] = [];

  for (const first of template) {
    const descriptors = descriptorBySample.map((map) => {
      const descriptor = map.get(first.name);
      if (!descriptor) {
        throw new Error(
          `Packed worker is missing tensor ${first.name}`,
        );
      }
      return descriptor;
    });

    if (first.name === "actionSparseIndices") {
      if (!sparse) {
        throw new Error(
          "Unexpected sparse index tensor without sparse shape",
        );
      }
      tensors.push({
        name: first.name,
        dtype: "int32",
        shape: [sparse.indices.byteLength / 4],
        byteOffset,
        byteLength: sparse.indices.byteLength,
      });
      buffers.push(sparse.indices);
      byteOffset += sparse.indices.byteLength;
      continue;
    }

    if (first.name === "actionSparseValues") {
      if (!sparse) {
        throw new Error(
          "Unexpected sparse value tensor without sparse shape",
        );
      }
      tensors.push({
        name: first.name,
        dtype: "float32",
        shape: [sparse.values.byteLength / 4],
        byteOffset,
        byteLength: sparse.values.byteLength,
      });
      buffers.push(sparse.values);
      byteOffset += sparse.values.byteLength;
      continue;
    }

    const itemBytes = PACKED_DTYPE_BYTES[first.dtype];

    if (VARIABLE_BATCH_ROW_TENSORS.has(first.name)) {
      if (
        first.shape.length !== 2
        && first.shape.length !== 3
      ) {
        throw new Error(
          `Unexpected variable packed tensor rank for ${first.name}: ${first.shape.length}`,
        );
      }

      const trailing = first.shape.slice(2);
      for (const descriptor of descriptors) {
        if (
          descriptor.shape[0] !== 1
          || descriptor.shape.length !== first.shape.length
          || descriptor.shape.slice(2).some(
            (value, index) => value !== trailing[index],
          )
        ) {
          throw new Error(
            `Packed worker variable shape mismatch: ${first.name}`,
          );
        }
      }

      const maxRows = Math.max(
        0,
        ...descriptors.map(
          (descriptor) => descriptor.shape[1],
        ),
      );
      const rowWidth = first.shape.length === 3
        ? first.shape[2]
        : 1;
      const rowBytes = rowWidth * itemBytes;
      const output = Buffer.alloc(
        samples.length * maxRows * rowBytes,
      );

      descriptors.forEach((descriptor, sampleIndex) => {
        const source = tensorBytes(
          samples[sampleIndex],
          descriptor,
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

      tensors.push({
        name: first.name,
        dtype: first.dtype,
        shape: first.shape.length === 3
          ? [
              samples.length,
              maxRows,
              first.shape[2],
            ]
          : [
              samples.length,
              maxRows,
            ],
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
          (value, index) =>
            value !== trailingShape[index],
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
        (descriptor) =>
          descriptor.byteLength !== sampleByteLength,
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
      tensorBytes(
        samples[sampleIndex],
        descriptor,
      ).copy(
        output,
        sampleIndex * sampleByteLength,
      );
    });

    tensors.push({
      name: first.name,
      dtype: first.dtype,
      shape: [
        samples.length,
        ...trailingShape,
      ],
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
    ...(combineRowCompaction(samples)
      ? { rowCompaction: combineRowCompaction(samples) }
      : {}),
    ...(sparse
      ? { actionSparseShape: sparse.shape }
      : {}),
  };
}
