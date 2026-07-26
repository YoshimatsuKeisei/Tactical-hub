import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface, type Interface } from "node:readline";
import type { RlFeatureSpec } from "./rlFeatureSpec";
import type { RlReplayEncodedDecision } from "./rlImitationCollector";
import { RL_PROJECT_ROOT } from "./rlProjectPaths";
import type { RlSelectedTorchDevice, RlTorchDevice } from "./rlTorchDevice";

type Response =
  | { type: "ready"; torchThreads: number; torchInteropThreads: number; selectedDevice: RlSelectedTorchDevice }
  | { type: "closed"; requestId?: number }
  | { type: "batchResult"; requestId: number; lossSum: number; correct: number; count: number }
  | { type: "saved" | "loaded"; requestId: number }
  | { type: "parameterHash"; requestId: number; hash: string }
  | { type: "error"; message: string };

export type BcEncodedSample = {
  observation: RlReplayEncodedDecision["encodedObservation"];
  actions: number[][];
  targetIndex: number;
};

export class PythonBcTrainerClient {
  private child?: ChildProcessWithoutNullStreams;
  private lines?: Interface;
  private stderr = "";
  private requestId = 1;
  private appliedSettings?: { torchThreads: number; torchInteropThreads: number; selectedDevice: RlSelectedTorchDevice };
  private readonly waiting: Array<{ resolve: (value: Response) => void; reject: (error: Error) => void }> = [];

  private wait() {
    return new Promise<Response>((resolve, reject) => this.waiting.push({ resolve, reject }));
  }
  private send(value: unknown) {
    if (!this.child?.stdin.writable) throw new Error("Python BC trainer is not running");
    this.child.stdin.write(`${JSON.stringify(value)}\n`);
  }
  private async request(value: Record<string, unknown>) {
    const responsePromise = this.wait();
    this.send(value);
    const response = await responsePromise;
    if (response.type === "error") throw new Error(response.message);
    return response;
  }

  async start(input: {
    featureSpec: RlFeatureSpec;
    learningRate: number;
    seed: number;
    command?: string;
    torchThreads?: number;
    torchInteropThreads?: number;
    device?: RlTorchDevice;
  }) {
    this.child = spawn(input.command ?? "python", ["-u", "-m", "rl.bc_server"], {
      cwd: RL_PROJECT_ROOT,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child.stderr.on("data", (chunk) => {
      this.stderr += String(chunk);
      process.stderr.write(chunk);
    });
    this.lines = createInterface({ input: this.child.stdout });
    this.lines.on("line", (line) => {
      const pending = this.waiting.shift();
      if (!pending) return;
      try { pending.resolve(JSON.parse(line) as Response); }
      catch { pending.reject(new Error(`Invalid JSON from Python BC trainer: ${line}`)); }
    });
    this.child.on("error", (error) => {
      while (this.waiting.length) this.waiting.shift()!.reject(error);
    });
    this.child.on("exit", (code) => {
      const error = new Error(`Python BC trainer exited with code ${code}${this.stderr ? `: ${this.stderr.trim()}` : ""}`);
      while (this.waiting.length) this.waiting.shift()!.reject(error);
    });
    const response = await this.request({
      type: "init",
      featureSpec: input.featureSpec,
      learningRate: input.learningRate,
      seed: input.seed,
      torchThreads: input.torchThreads,
      torchInteropThreads: input.torchInteropThreads,
      device: input.device ?? "auto",
    });
    if (response.type !== "ready") throw new Error(`Unexpected BC init response: ${response.type}`);
    if (!Number.isInteger(response.torchThreads) || !Number.isInteger(response.torchInteropThreads)
      || !["cpu", "cuda"].includes(response.selectedDevice)) {
      throw new Error("Python BC trainer returned invalid thread settings");
    }
    this.appliedSettings = {
      torchThreads: response.torchThreads,
      torchInteropThreads: response.torchInteropThreads,
      selectedDevice: response.selectedDevice,
    };
  }

  async batch(samples: BcEncodedSample[], train: boolean) {
    const requestId = this.requestId++;
    const response = await this.request({ type: "batch", requestId, train, samples });
    if (response.type !== "batchResult" || response.requestId !== requestId) throw new Error("Unexpected BC batch response");
    if (![response.lossSum, response.correct, response.count].every(Number.isFinite)) throw new Error("Non-finite BC metrics");
    return response;
  }

  async save(path: string, metadata: unknown) {
    const requestId = this.requestId++;
    const response = await this.request({ type: "save", requestId, path, metadata });
    if (response.type !== "saved" || response.requestId !== requestId) throw new Error("Unexpected BC save response");
  }
  async load(path: string) {
    const requestId = this.requestId++;
    const response = await this.request({ type: "load", requestId, path });
    if (response.type !== "loaded" || response.requestId !== requestId) throw new Error("Unexpected BC load response");
  }
  async parameterHash() {
    const requestId = this.requestId++;
    const response = await this.request({ type: "parameterHash", requestId });
    if (response.type !== "parameterHash" || response.requestId !== requestId) throw new Error("Unexpected BC parameter hash response");
    return response.hash;
  }
  async close() {
    if (!this.child) return;
    if (this.child.exitCode === null && this.child.stdin.writable) await this.request({ type: "close" }).catch(() => undefined);
    this.lines?.close();
    this.child.kill();
    this.child = undefined;
  }

  getAppliedThreads() {
    if (!this.appliedSettings) throw new Error("Python BC trainer has not reported settings");
    return { ...this.appliedSettings };
  }
}
