import {
  createPythonPpoEvaluationClientFactory,
  runPpoCheckpointEvaluation,
  UPDATE2_CHECKPOINT_SHA256,
  UPDATE3_CHECKPOINT_SHA256,
  verifyPpoEvaluationCheckpoint,
} from "./rlPpoCheckpointEvaluation";
import { formatPpoCliJson } from "./rlPpoCliJson";
import { parseRlTorchDevice } from "./rlTorchDevice";

const args = process.argv.slice(2);
const value = (name: string) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};
const required = (name: string) => {
  const result = value(name);
  if (!result) throw new Error(`${name} is required`);
  return result;
};
const integer = (name: string, fallback: number, minimum: number) => {
  const parsed = Number(value(name) ?? fallback);
  if (!Number.isInteger(parsed) || parsed < minimum) {
    throw new Error(`${name} must be an integer >= ${minimum}`);
  }
  return parsed;
};

const update2Checkpoint = required("--update2-checkpoint");
const update3Checkpoint = required("--update3-checkpoint");
const update2Sha256 = await verifyPpoEvaluationCheckpoint(
  update2Checkpoint,
  value("--update2-sha256") ?? UPDATE2_CHECKPOINT_SHA256,
);
const update3Sha256 = await verifyPpoEvaluationCheckpoint(
  update3Checkpoint,
  value("--update3-sha256") ?? UPDATE3_CHECKPOINT_SHA256,
);

const result = await runPpoCheckpointEvaluation({
  update2Checkpoint,
  update3Checkpoint,
  update2Sha256,
  update3Sha256,
  seedStart: integer("--seed-start", 1, 0),
  seedCount: integer("--seed-count", 1, 1),
  maxTurns: integer("--max-turns", 1_000, 1),
  maxDecisions: integer("--max-decisions", 100_000, 1),
  clientFactory: createPythonPpoEvaluationClientFactory({
    python: value("--python") ?? "python",
    device: parseRlTorchDevice(value("--device") ?? "auto"),
  }),
});

process.stdout.write(formatPpoCliJson(result));
