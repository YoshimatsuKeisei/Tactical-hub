import type { RlFeatureSpecV2 } from "./rlFeatureSpec";
import {
  packBcEncodedSamples,
  type PackBcEncodedSamplesOptions,
  type PackedBcBatch,
  type PackedTensorDescriptor,
} from "./rlBcPackedBatch";
import type { EncodedObservation } from "./rlObservationEncoder";

export type PpoEncodedSample = {
  observation: EncodedObservation;
  actions: number[][];
  targetIndex: number;
  oldLogProbability: number;
  advantage: number;
  return: number;
};

export type PpoUpdateScalarSample = Pick<
  PpoEncodedSample,
  "oldLogProbability" | "advantage" | "return"
>;

function appendFloatTensor(
  packed: PackedBcBatch,
  name: string,
  values: readonly number[],
): { buffer: Buffer; descriptor: PackedTensorDescriptor } {
  if (!values.every(Number.isFinite)) throw new Error(`${name} contains NaN or Inf`);
  const floats = Float32Array.from(values);
  const buffer = Buffer.from(floats.buffer, floats.byteOffset, floats.byteLength);
  return {
    buffer,
    descriptor: {
      name,
      dtype: "float32",
      shape: [values.length],
      byteOffset: packed.payload.byteLength,
      byteLength: buffer.byteLength,
    },
  };
}

export function appendPpoUpdateScalars(
  base: PackedBcBatch,
  samples: PpoUpdateScalarSample[],
): PackedBcBatch {
  if (!samples.length || samples.length !== base.batchSize) {
    throw new Error("PPO scalar batch size must match packed feature batch");
  }
  const additions = [
    appendFloatTensor(base, "oldLogProbabilities", samples.map((sample) => sample.oldLogProbability)),
    appendFloatTensor(base, "advantages", samples.map((sample) => sample.advantage)),
    appendFloatTensor(base, "returns", samples.map((sample) => sample.return)),
  ];
  let offset = base.payload.byteLength;
  const descriptors = additions.map(({ buffer, descriptor }) => {
    const adjusted = { ...descriptor, byteOffset: offset };
    offset += buffer.byteLength;
    return adjusted;
  });
  return {
    payload: Buffer.concat([base.payload, ...additions.map((entry) => entry.buffer)], offset),
    tensors: [...base.tensors, ...descriptors],
    batchSize: base.batchSize,
    ...(base.rowCompaction
      ? { rowCompaction: base.rowCompaction }
      : {}),
  };
}

export function packPpoEncodedSamples(samples: PpoEncodedSample[], featureSpec: RlFeatureSpecV2): PackedBcBatch {
  if (!samples.length) throw new Error("Cannot pack an empty PPO batch");
  return appendPpoUpdateScalars(packBcEncodedSamples(samples, featureSpec), samples);
}

export function packPpoActInput(
  observation: EncodedObservation,
  actions: number[][],
  featureSpec: RlFeatureSpecV2,
  options: PackBcEncodedSamplesOptions = {},
) {
  return packBcEncodedSamples(
    [{ observation, actions, targetIndex: 0 }],
    featureSpec,
    options,
  );
}

export function packPpoActBatchInput(
  samples: Array<{ observation: EncodedObservation; actions: number[][] }>,
  featureSpec: RlFeatureSpecV2,
  options: PackBcEncodedSamplesOptions = {},
) {
  if (!samples.length) throw new Error("Cannot pack an empty PPO action batch");
  return packBcEncodedSamples(
    samples.map((sample) => ({ ...sample, targetIndex: 0 })),
    featureSpec,
    options,
  );
}
