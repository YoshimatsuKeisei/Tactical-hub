import { describe, expect, it } from "vitest";
import { encodeRlLegalActionsV2 } from "../cpu/rlActionEncoder";
import { RlEnvironmentV2 } from "../cpu/rlEnvironment";
import { createRlFeatureSpecV2 } from "../cpu/rlFeatureSpec";
import {
  createRlObservationEncoderCache,
  encodeRlObservationV2,
} from "../cpu/rlObservationEncoder";
import { packPpoActBatchInput } from "../cpu/rlPpoPackedBatch";

function descriptor(
  packed: ReturnType<typeof packPpoActBatchInput>,
  name: string,
) {
  const value = packed.tensors.find(
    (entry) => entry.name === name,
  );
  if (!value) {
    throw new Error(`Missing packed tensor: ${name}`);
  }
  return value;
}

describe("PPO compact masked-prefix rows", () => {
  it("removes fixed zero padding while retaining logical row counts", () => {
    const samples = Array.from(
      { length: 8 },
      (_, index) => {
        const environment = new RlEnvironmentV2(
          undefined,
          true,
          {
            cpuStep: {
              rlInPlacePhaseTransitions: true,
            },
          },
        );
        const initial = environment.reset(9 + index, 4);
        const actor = environment.getCurrentActorTeamId();
        if (!actor) {
          throw new Error("Missing PPO compact-row actor");
        }
        const observation = environment.getObservationForEncoding(actor);
        const legal = environment.getLegalActionsForEncoding(actor);
        if (!legal.length) {
          throw new Error("Missing PPO compact-row legal action");
        }
        return {
          initial,
          observation: encodeRlObservationV2(
            observation,
            createRlObservationEncoderCache(),
          ),
          actions:
            encodeRlLegalActionsV2(
              observation,
              legal,
            ).actions,
        };
      },
    );

    const featureSpec = createRlFeatureSpecV2(
      samples[0].initial,
    );

    const normal = packPpoActBatchInput(
      samples.map(({ observation, actions }) => ({
        observation,
        actions,
      })),
      featureSpec,
    );

    const compact = packPpoActBatchInput(
      samples.map(({ observation, actions }) => ({
        observation,
        actions,
      })),
      featureSpec,
      {
        compactMaskedPrefixes: true,
      },
    );

    expect(compact.rowCompaction).toEqual({
      units: descriptor(normal, "units").shape[1],
      bases: descriptor(normal, "bases").shape[1],
      constructions:
        descriptor(normal, "constructions").shape[1],
    });

    expect(
      descriptor(compact, "units").shape[1],
    ).toBeLessThan(
      descriptor(normal, "units").shape[1],
    );

    expect(
      descriptor(compact, "constructions").shape[1],
    ).toBeLessThan(
      descriptor(normal, "constructions").shape[1],
    );

    expect(
      descriptor(compact, "actions").shape,
    ).toEqual(
      descriptor(normal, "actions").shape,
    );

    expect(
      descriptor(compact, "map").shape,
    ).toEqual(
      descriptor(normal, "map").shape,
    );

    expect(
      compact.payload.byteLength,
    ).toBeLessThan(
      normal.payload.byteLength,
    );
  });
});
