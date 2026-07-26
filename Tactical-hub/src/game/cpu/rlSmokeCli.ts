import { runRlSmokeMatch } from "./rlSmokeRunner";
import { parseRlTorchDevice } from "./rlTorchDevice";

const args = process.argv.slice(2);
const value = (name: string) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};
const seed = Number(value("--seed") ?? 1);
const maxDecisions = Number(value("--max-decisions") ?? 100_000);
const pythonCommand = value("--python") ?? "python";
const device = parseRlTorchDevice(value("--device") ?? "auto");

const result = await runRlSmokeMatch({ seed, maxDecisions, python: { command: pythonCommand, device } });
console.log(JSON.stringify(result, null, 2));
if (!result.terminal || result.endReason !== "victory") process.exitCode = 1;
