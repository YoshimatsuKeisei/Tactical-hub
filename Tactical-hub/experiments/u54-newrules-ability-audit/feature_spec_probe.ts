import { readFileSync } from "node:fs";
import { RlEnvironmentV2 } from "../../src/game/cpu/rlEnvironment";
import { createRlFeatureSpecV2 } from "../../src/game/cpu/rlFeatureSpec";

const expectedPath = process.argv[2];
if (!expectedPath) throw new Error("Usage: vite-node feature_spec_probe.ts <expected-feature-spec.json>");
const environment = new RlEnvironmentV2(undefined, true);
const actual = createRlFeatureSpecV2(environment.reset(313, 4));
const expected = JSON.parse(readFileSync(expectedPath, "utf8"));
if (JSON.stringify(actual) !== JSON.stringify(expected)) {
  throw new Error(`Feature Spec mismatch\nexpected=${JSON.stringify(expected)}\nactual=${JSON.stringify(actual)}`);
}
process.stdout.write(JSON.stringify({ exact: true, featureSpec: actual }, null, 2) + "\n");
