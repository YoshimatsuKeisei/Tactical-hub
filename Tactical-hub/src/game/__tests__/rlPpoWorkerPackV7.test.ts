import { describe, expect, it } from "vitest";
import {
  encodeRlLegalActionsSparseV2,
} from "../cpu/rlActionEncoder";
import {
  combinePackedSingleSampleBatches,
} from "../cpu/rlBcPackedBatch";
import { RlEnvironmentV2 } from "../cpu/rlEnvironment";
import { createRlFeatureSpecV2 } from "../cpu/rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationCompactV2,
} from "../cpu/rlObservationEncoder";
import {
  packPpoActBatchInput,
} from "../cpu/rlPpoPackedBatch";

function createEnvironment() {
  return new RlEnvironmentV2(
    undefined,
    true,
    {
      cpuStep: {
        rlInPlacePhaseTransitions: true,
        rlPrevalidatedMovement: true,
        rlInPlaceProduction: true,
      },
    },
  );
}

describe("PPO V7-D worker packed combiner", () => {
  it("matches the current compact+sparse central batch byte-for-byte", () => {
    const environments = Array.from(
      { length: 8 },
      (_, environmentIndex) => {
        const environment = createEnvironment();
        const initial = environment.reset(9 + environmentIndex, 4);
        const actor = environment.getCurrentActorTeamId();
        if (!actor) throw new Error("missing V7-D test actor");

        const observation = environment.getObservationForEncoding(actor);
        const legal = environment.getLegalActionsForEncoding(actor);
        if (!legal.length) throw new Error("missing V7-D legal actions");

        return {
          initial,
          observation: encodeRlObservationCompactV2(
            observation,
            createRlObservationEncoderCache(),
          ),
          legalActions: encodeRlLegalActionsSparseV2(
            observation,
            legal,
          ),
        };
      },
    );

    const featureSpec = createRlFeatureSpecV2(
      environments[0].initial,
    );
    const options = {
      compactMaskedPrefixes: true,
      sparseActions: true,
    } as const;

    const central = packPpoActBatchInput(
      environments.map((sample) => ({
        observation: sample.observation,
        sparseActions: sample.legalActions.sparseActions,
      })),
      featureSpec,
      options,
    );

    const singles = environments.map((sample) =>
      packPpoActBatchInput(
        [{
          observation: sample.observation,
          sparseActions: sample.legalActions.sparseActions,
        }],
        featureSpec,
        options,
      )
    );

    const combined = combinePackedSingleSampleBatches(singles);

    expect(combined.batchSize).toBe(central.batchSize);
    expect(combined.rowCompaction).toEqual(central.rowCompaction);
    expect(combined.actionSparseShape).toEqual(
      central.actionSparseShape,
    );
    expect(combined.tensors).toEqual(central.tensors);
    expect(combined.payload.equals(central.payload)).toBe(true);
  });
});
