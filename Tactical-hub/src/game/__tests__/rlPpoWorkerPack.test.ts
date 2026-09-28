import { describe, expect, it } from "vitest";
import { encodeRlLegalActionsV2 } from "../cpu/rlActionEncoder";
import {
  combinePackedSingleSampleBatches,
} from "../cpu/rlBcPackedBatch";
import { RlEnvironmentV2 } from "../cpu/rlEnvironment";
import { createRlFeatureSpecV2 } from "../cpu/rlFeatureSpec";
import {
  encodeRlObservationV2,
} from "../cpu/rlObservationEncoder";
import {
  packPpoActBatchInput,
  packPpoActInput,
} from "../cpu/rlPpoPackedBatch";

describe("PPO worker packed-batch combiner", () => {
  it("matches the existing central batch packer byte-for-byte", () => {
    const samples = [9, 10].map((seed) => {
      const environment = new RlEnvironmentV2(
        undefined,
        true,
        { cpuStep: { rlInPlacePhaseTransitions: true } },
      );
      const initial = environment.reset(seed, 4);
      const actor = environment.getCurrentActorTeamId();
      if (!actor) throw new Error("missing PPO worker-pack actor");
      const observation = environment.getObservationForEncoding(actor);
      const legal = environment.getLegalActionsForEncoding(actor);
      if (!legal.length) throw new Error("missing PPO worker-pack legal action");
      return {
        initial,
        observation: encodeRlObservationV2(observation),
        legalActions: encodeRlLegalActionsV2(observation, legal),
      };
    });

    const featureSpec = createRlFeatureSpecV2(samples[0].initial);
    const direct = packPpoActBatchInput(
      samples.map((sample) => ({
        observation: sample.observation,
        actions: sample.legalActions.actions,
      })),
      featureSpec,
    );
    const combined = combinePackedSingleSampleBatches(
      samples.map((sample) =>
        packPpoActInput(
          sample.observation,
          sample.legalActions.actions,
          featureSpec,
        ),
      ),
    );

    expect(combined.batchSize).toBe(direct.batchSize);
    expect(combined.tensors).toEqual(direct.tensors);
    expect(combined.payload.equals(direct.payload)).toBe(true);
  });
});
