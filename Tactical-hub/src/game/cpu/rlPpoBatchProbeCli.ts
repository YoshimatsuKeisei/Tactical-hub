/**
 * Standalone multi-environment PPO inference probe input generator.
 * Does not run PPO training or write checkpoints.
 */
import { createWriteStream } from "node:fs";
import { once } from "node:events";
import { resolve } from "node:path";
import { RlEnvironmentV2 } from "./rlEnvironment";
import { createRlFeatureSpecV2 } from "./rlFeatureSpec";
import { createRlObservationEncoderCache, encodeRlObservationV2 } from "./rlObservationEncoder";
import { encodeRlLegalActionsV2 } from "./rlActionEncoder";

const args = process.argv.slice(2);
const value = (name: string) => {
  const index = args.indexOf(name);
  return index < 0 ? undefined : args[index + 1];
};
const environmentCount = Number(value("--environments") ?? "16");
const decisions = Number(value("--decisions") ?? "50");
const seed = Number(value("--seed") ?? "8");
const destination = value("--out");
if (!destination) throw new Error("Specify --out FILE (new probe fixture only)");
if (!Number.isInteger(environmentCount) || environmentCount < 2 || environmentCount > 16) {
  throw new Error("--environments must be an integer in 2..16");
}
if (!Number.isInteger(decisions) || decisions < 1 || decisions > 100) {
  throw new Error("--decisions must be an integer in 1..100");
}
if (!Number.isInteger(seed) || seed < 1) throw new Error("--seed must be a positive integer");

const environments = Array.from({ length: environmentCount }, (_, index) => {
  const environment = new RlEnvironmentV2();
  const initialObservation = environment.reset(seed + index, 4);
  return { environment, cache: createRlObservationEncoderCache(), initialObservation };
});
const seeds = environments.map((_, index) => seed + index);
const featureSpec = createRlFeatureSpecV2(environments[0].initialObservation);
const stream = createWriteStream(resolve(destination), { flags: "wx", encoding: "utf8" });
const write = async (payload: unknown) => {
  if (!stream.write(JSON.stringify(payload) + "\n")) await once(stream, "drain");
};
try {
  await write({ type: "header", schemaVersion: 2, featureSpec, seeds, decisions, environmentCount });
  for (let step = 0; step < decisions; step += 1) {
    const samples = environments.map(({ environment, cache }, environmentIndex) => {
      if (environment.isTerminal()) throw new Error(`Environment ${environmentIndex} terminated at step ${step}`);
      const teamId = environment.getCurrentActorTeamId();
      if (!teamId) throw new Error(`Environment ${environmentIndex} has no actor at step ${step}`);
      const observation = environment.getObservationForEncoding(teamId);
      const legal = environment.getLegalActionsForEncoding(teamId);
      if (!legal.length) throw new Error(`Environment ${environmentIndex} has no legal actions at step ${step}`);
      return {
        observation: encodeRlObservationV2(observation, cache),
        actions: encodeRlLegalActionsV2(observation, legal).actions,
        legalCount: legal.length,
      };
    });
    await write({ type: "pair", step, samples });
    // A fixed legal action advances every environment independently; no PPO sampling or model update.
    for (const { environment } of environments) {
      const actor = environment.getCurrentActorTeamId()!;
      environment.stepWithoutObservation(environment.getLegalActionsForEncoding(actor)[0].actionKey);
    }
  }
  await write({
    type: "end",
    decisionCounts: Array.from({ length: environmentCount }, () => decisions),
    finalStateHashes: environments.map(({ environment }) => environment.getStateHash()),
  });
} finally {
  stream.end();
  await once(stream, "close");
}
console.log(JSON.stringify({
  probe: "multi_environment_inputs",
  destination: resolve(destination),
  seeds,
  environmentCount,
  decisionsPerEnvironment: decisions,
  featureSpecSchemaVersion: featureSpec.schemaVersion,
}));
