import { describe, expect, it } from "vitest";
import { encodeRlLegalActions } from "../cpu/rlActionEncoder";
import { RlEnvironment } from "../cpu/rlEnvironment";
import { createRlFeatureSpec } from "../cpu/rlFeatureSpec";
import { encodeRlObservation } from "../cpu/rlObservationEncoder";
import { packBcEncodedSamples } from "../cpu/rlBcPackedBatch";
import { PythonBcTrainerClient } from "../cpu/pythonBcTrainerClient";

function tensorView(
  packed: ReturnType<typeof packBcEncodedSamples>,
  name: string,
) {
  const descriptor = packed.tensors.find((value) => value.name === name)!;
  const buffer = packed.payload.subarray(descriptor.byteOffset, descriptor.byteOffset + descriptor.byteLength);
  return { descriptor, buffer };
}

describe("packed BC batches", () => {
  it("packs variable legal actions, targets and masks without changing encoder values", () => {
    const environment = new RlEnvironment();
    environment.reset(17, 4);
    const observation = environment.getObservation(environment.getCurrentActorTeamId()!);
    const legal = environment.getLegalActions(observation.observingTeamId);
    const encoded = encodeRlLegalActions(observation, legal);
    const encodedObservation = encodeRlObservation(observation);
    const spec = createRlFeatureSpec(observation);
    const samples = [
      { observation: encodedObservation, actions: encoded.actions, targetIndex: encoded.actions.length - 1 },
      { observation: encodedObservation, actions: encoded.actions.slice(0, 1), targetIndex: 0 },
    ];
    const packed = packBcEncodedSamples(samples, spec);

    const packedMap = tensorView(packed, "map");
    const expectedMapRows = encodedObservation.map.flat();
    const expectedMapValues = Float32Array.from([
      ...expectedMapRows.flat(),
      ...expectedMapRows.flat(),
    ]);
    expect(packedMap.descriptor.shape).toEqual([
      2,
      expectedMapRows.length,
      spec.mapTileWidth,
    ]);
    expect(Buffer.compare(
      packedMap.buffer,
      Buffer.from(
        expectedMapValues.buffer,
        expectedMapValues.byteOffset,
        expectedMapValues.byteLength,
      ),
    )).toBe(0);

    const actions = tensorView(packed, "actions");
    const actionValues = new Float32Array(
      actions.buffer.buffer,
      actions.buffer.byteOffset,
      actions.buffer.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );
    expect(actions.descriptor.shape).toEqual([2, encoded.actions.length, spec.actionFeatureWidth]);
    expect([...actionValues.slice(0, spec.actionFeatureWidth)]).toEqual([...Float32Array.from(encoded.actions[0])]);

    const mask = tensorView(packed, "actionMask");
    expect([...mask.buffer]).toEqual([
      ...Array(encoded.actions.length).fill(1),
      1,
      ...Array(Math.max(0, encoded.actions.length - 1)).fill(0),
    ]);
    const targets = tensorView(packed, "targets");
    expect([...new Int32Array(targets.buffer.buffer, targets.buffer.byteOffset, 2)])
      .toEqual([encoded.actions.length - 1, 0]);
  });

  it("round-trips sparse Action transport bytes exactly", () => {
    const environment = new RlEnvironment();
    environment.reset(19, 4);
    const observation = environment.getObservation(
      environment.getCurrentActorTeamId()!,
    );
    const legal = environment.getLegalActions(observation.observingTeamId);
    const encoded = encodeRlLegalActions(observation, legal);
    const encodedObservation = encodeRlObservation(observation);
    const spec = createRlFeatureSpec(observation);
    const samples = [
      { observation: encodedObservation, actions: encoded.actions, targetIndex: 0 },
      { observation: encodedObservation, actions: encoded.actions.slice(0, 2), targetIndex: 1 },
    ];

    const dense = packBcEncodedSamples(samples, spec);
    const sparse = packBcEncodedSamples(samples, spec, {
      sparseActions: true,
    });
    const denseActions = tensorView(dense, "actions");
    const sparseIndices = tensorView(sparse, "actionSparseIndices");
    const sparseValues = tensorView(sparse, "actionSparseValues");
    const restored = new Float32Array(
      denseActions.buffer.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );
    const indices = new Int32Array(
      sparseIndices.buffer.buffer,
      sparseIndices.buffer.byteOffset,
      sparseIndices.buffer.byteLength / Int32Array.BYTES_PER_ELEMENT,
    );
    const values = new Float32Array(
      sparseValues.buffer.buffer,
      sparseValues.buffer.byteOffset,
      sparseValues.buffer.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );
    expect(indices.length).toBe(values.length);
    indices.forEach((index, offset) => {
      restored[index] = values[offset];
    });

    const restoredBytes = Buffer.from(
      restored.buffer,
      restored.byteOffset,
      restored.byteLength,
    );
    expect(Buffer.compare(restoredBytes, denseActions.buffer)).toBe(0);
    expect(sparse.actionSparseShape).toEqual(denseActions.descriptor.shape);
    expect(Buffer.compare(
      tensorView(sparse, "actionMask").buffer,
      tensorView(dense, "actionMask").buffer,
    )).toBe(0);
  });

  it("preserves Feature Spec widths for empty strategic tables", () => {
    const environment = new RlEnvironment();
    environment.reset(23, 4);
    const observation = environment.getObservation(environment.getCurrentActorTeamId()!);
    const encodedObservation = encodeRlObservation(observation);
    const encodedActions = encodeRlLegalActions(observation, environment.getLegalActions(observation.observingTeamId));
    const spec = createRlFeatureSpec(observation);
    const packed = packBcEncodedSamples([{
      observation: encodedObservation,
      actions: encodedActions.actions,
      targetIndex: 0,
    }], spec);

    for (const [name, width] of Object.entries(spec.strategicTableRowWidths)) {
      expect(tensorView(packed, `strategic.${name}`).descriptor.shape).toEqual([1, 0, width]);
    }
  });

  it("produces the same targets, loss and correct count as the JSON regression path", async () => {
    const environment = new RlEnvironment();
    environment.reset(31, 4);
    const observation = environment.getObservation(environment.getCurrentActorTeamId()!);
    const encodedObservation = encodeRlObservation(observation);
    const encodedActions = encodeRlLegalActions(observation, environment.getLegalActions(observation.observingTeamId));
    const sample = { observation: encodedObservation, actions: encodedActions.actions, targetIndex: encodedActions.actions.length - 1 };
    const input = {
      featureSpec: createRlFeatureSpec(observation),
      learningRate: 1e-4,
      seed: 47,
      device: "cpu" as const,
      torchThreads: 1,
      torchInteropThreads: 1,
    };
    const json = new PythonBcTrainerClient();
    const packed = new PythonBcTrainerClient();
    try {
      await json.start(input);
      await packed.start(input);
      const jsonResult = await json.batchJson([sample], false);
      const packedResult = await packed.batch([sample], false);
      expect(packedResult.count).toBe(jsonResult.count);
      expect(packedResult.correct).toBe(jsonResult.correct);
      expect(packedResult.lossSum).toBeCloseTo(jsonResult.lossSum, 6);
    } finally {
      await Promise.all([json.close(), packed.close()]);
    }
  }, 20_000);
});
