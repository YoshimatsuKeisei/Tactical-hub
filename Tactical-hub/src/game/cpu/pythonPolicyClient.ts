import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import type { EncodedLegalActions } from "./rlActionEncoder";
import type { RlFeatureSpec } from "./rlFeatureSpec";
import type { EncodedObservation } from "./rlObservationEncoder";
import type { RlSelectedTorchDevice, RlTorchDevice } from "./rlTorchDevice";

type PythonResponse =
  | { type: "ready"; selectedDevice: RlSelectedTorchDevice }
  | { type: "action"; requestId: number; actionIndex: number; value: number }
  | { type: "closed" }
  | { type: "error"; message: string };

export type PythonPolicyClientOptions = {
  command?: string;
  args?: string[];
  cwd?: string;
  device?: RlTorchDevice;
};

export class PythonPolicyClient {
  private process?: ChildProcessWithoutNullStreams;
  private lines?: Interface;
  private stderr = "";
  private nextRequestId = 1;
  private selectedDevice?: RlSelectedTorchDevice;
  private readonly waiting: Array<{ resolve: (response: PythonResponse) => void; reject: (error: Error) => void }> = [];

  constructor(private readonly options: PythonPolicyClientOptions = {}) {}

  private waitForResponse() {
    return new Promise<PythonResponse>((resolve, reject) => this.waiting.push({ resolve, reject }));
  }

  private send(payload: unknown) {
    if (!this.process?.stdin.writable) throw new Error("Python policy process is not running");
    this.process.stdin.write(`${JSON.stringify(payload)}\n`);
  }

  async start(seed: number, featureSpec: RlFeatureSpec) {
    if (this.process) throw new Error("Python policy process is already running");
    const command = this.options.command ?? "python";
    const args = this.options.args ?? ["-u", "-m", "rl.policy_server"];
    this.process = spawn(command, args, { cwd: this.options.cwd ?? process.cwd(), stdio: ["pipe", "pipe", "pipe"] });
    this.process.stderr.on("data", (chunk) => {
      this.stderr += String(chunk);
      process.stderr.write(chunk);
    });
    this.lines = createInterface({ input: this.process.stdout });
    this.lines.on("line", (line) => {
      const pending = this.waiting.shift();
      if (!pending) return;
      try { pending.resolve(JSON.parse(line) as PythonResponse); }
      catch { pending.reject(new Error(`Invalid JSON from Python policy: ${line}`)); }
    });
    this.process.on("error", (error) => {
      while (this.waiting.length) this.waiting.shift()!.reject(error);
    });
    this.process.on("exit", (code) => {
      const error = new Error(`Python policy exited with code ${code}${this.stderr ? `: ${this.stderr.trim()}` : ""}`);
      while (this.waiting.length) this.waiting.shift()!.reject(error);
    });
    const responsePromise = this.waitForResponse();
    this.send({ type: "init", seed, featureSpec, device: this.options.device ?? "auto" });
    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    if (response.type !== "ready") throw new Error(`Unexpected Python initialization response: ${response.type}`);
    if (!["cpu", "cuda"].includes(response.selectedDevice)) throw new Error("Python policy returned an invalid selectedDevice");
    this.selectedDevice = response.selectedDevice;
  }

  async act(observation: EncodedObservation, legalActions: EncodedLegalActions) {
    if (!legalActions.actions.length) throw new Error("Cannot request NN action without legal actions");
    const requestId = this.nextRequestId++;
    const responsePromise = this.waitForResponse();
    this.send({ type: "act", requestId, observation, actions: legalActions.actions });
    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    if (response.type !== "action" || response.requestId !== requestId) throw new Error(`Unexpected Python action response`);
    if (!Number.isInteger(response.actionIndex) || response.actionIndex < 0 || response.actionIndex >= legalActions.actionKeys.length) {
      throw new Error(`Python returned illegal actionIndex ${response.actionIndex} for ${legalActions.actionKeys.length} legal actions`);
    }
    if (!Number.isFinite(response.value)) throw new Error(`Python returned non-finite value ${response.value}`);
    return { actionIndex: response.actionIndex, actionKey: legalActions.actionKeys[response.actionIndex], value: response.value };
  }

  async close() {
    if (!this.process) return;
    if (this.process.exitCode === null && this.process.stdin.writable) {
      const responsePromise = this.waitForResponse();
      this.send({ type: "close" });
      await responsePromise.catch(() => undefined);
    }
    this.lines?.close();
    this.process.kill();
    this.process = undefined;
  }

  getStderr() { return this.stderr; }
  getSelectedDevice() {
    if (!this.selectedDevice) throw new Error("Python policy has not reported a selected device");
    return this.selectedDevice;
  }
}
