import { describe, expect, it } from "vitest";
import {
  encodeRlLegalActionsSparseV2,
  encodeRlLegalActionsV2,
} from "../cpu/rlActionEncoder";
import { RlEnvironmentV2 } from "../cpu/rlEnvironment";
import { createRlFeatureSpecV2 } from "../cpu/rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
} from "../cpu/rlObservationEncoder";
import { packPpoActBatchInput } from "../cpu/rlPpoPackedBatch";

function floatBytes(values: ArrayLike<number>) {
  const floats = Float32Array.from(values);
  return Buffer.from(
    floats.buffer,
    floats.byteOffset,
    floats.byteLength,
  );
}

function restoreSparse(
  row: ReturnType<typeof encodeRlLegalActionsSparseV2>["sparseActions"][number],
) {
  const restored = new Float32Array(row.width);
  row.indices.forEach((index, offset) => {
    restored[index] = row.values[offset];
  });
  return restored;
}

describe("RL V2 direct sparse Action encoder", () => {
  it("is Float32-byte exact with the dense V2 encoder across rollout states", () => {
    let comparedRows = 0;
    const actionTypes = new Set<string>();

    for (const seed of [9, 10, 11, 12, 13, 14, 15, 16]) {
      const environment = new RlEnvironmentV2();
      environment.reset(seed, 4);

      for (let decision = 0; decision < 120; decision += 1) {
        if (environment.isTerminal()) break;
        const teamId = environment.getCurrentActorTeamId();
        if (!teamId) break;

        const observation = environment.getObservationForEncoding(teamId);
        const legal = environment.getLegalActionsForEncoding(teamId);
        if (!legal.length) break;

        const dense = encodeRlLegalActionsV2(observation, legal);
        const sparse = encodeRlLegalActionsSparseV2(observation, legal);

        expect(sparse.actionKeys).toEqual(dense.actionKeys);
        expect(sparse.sparseActions).toHaveLength(dense.actions.length);

        dense.actions.forEach((row, index) => {
          const sparseRow = sparse.sparseActions[index];
          expect(sparseRow.width).toBe(row.length);
          expect(sparseRow.indices).toHaveLength(sparseRow.values.length);
          expect(
            Buffer.compare(
              floatBytes(row),
              floatBytes(restoreSparse(sparseRow)),
            ),
          ).toBe(0);
          comparedRows += 1;
        });

        for (const action of legal) actionTypes.add(action.actionType);
        const chosen = legal[decision % legal.length];
        environment.stepWithoutObservation(chosen.actionKey);
      }
    }

    expect(comparedRows).toBeGreaterThan(0);
    expect(actionTypes.size).toBeGreaterThan(1);
  });

  it("packs the same sparse payload as the dense-scan transport path", () => {
    const environment = new RlEnvironmentV2();
    const initial = environment.reset(9, 4);
    const teamId = environment.getCurrentActorTeamId()!;
    const observation = environment.getObservationForEncoding(teamId);
    const legal = environment.getLegalActionsForEncoding(teamId);
    const dense = encodeRlLegalActionsV2(observation, legal);
    const direct = encodeRlLegalActionsSparseV2(observation, legal);
    const encodedObservation = encodeRlObservationV2(
      observation,
      createRlObservationEncoderCache(),
    );
    const featureSpec = createRlFeatureSpecV2(initial);

    const scanned = packPpoActBatchInput(
      [{
        observation: encodedObservation,
        actions: dense.actions,
      }],
      featureSpec,
      {
        compactMaskedPrefixes: true,
        sparseActions: true,
      },
    );
    const directPacked = packPpoActBatchInput(
      [{
        observation: encodedObservation,
        sparseActions: direct.sparseActions,
      }],
      featureSpec,
      {
        compactMaskedPrefixes: true,
        sparseActions: true,
      },
    );

    expect(directPacked.tensors).toEqual(scanned.tensors);
    expect(directPacked.rowCompaction).toEqual(scanned.rowCompaction);
    expect(directPacked.actionSparseShape).toEqual(scanned.actionSparseShape);
    expect(Buffer.compare(directPacked.payload, scanned.payload)).toBe(0);
  });
});
