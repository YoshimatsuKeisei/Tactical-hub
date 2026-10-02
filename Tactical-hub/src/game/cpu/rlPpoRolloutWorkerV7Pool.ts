import { performance } from "node:perf_hooks";
import { Worker } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { RL_VITE_NODE_ENTRY } from "./rlProjectPaths";
import type { RlFeatureSpecV2 } from "./rlFeatureSpec";
import type { PpoHyperparameters } from "./pythonPpoClient";
import type {
  PpoRolloutWorkerV7Finalized,
  PpoRolloutWorkerV7PreparedGroup,
  PpoRolloutWorkerV7Request,
  PpoRolloutWorkerV7Response,
  PpoRolloutWorkerV7Timing,
} from "./rlPpoRolloutWorkerV7Messages";
import type { PpoRolloutEnvironmentDiagnosticV8 } from "./rlPpoRolloutDiagnosticsV8";
import type {
  PpoDefeatDiagnosticEventV8,
  PpoDefeatEnvironmentSnapshotV8,
} from "./rlPpoDefeatDiagnosticsV8";

type RequestWithoutId =
  PpoRolloutWorkerV7Request extends infer Request
    ? Request extends { requestId: number }
      ? Omit<Request, "requestId">
      : never
    : never;

type WorkerHandle = {
  workerId: number;
  worker: Worker;
  environmentIndices: number[];
  nextRequestId: number;
  pending?: {
    requestId: number;
    resolve: (response: PpoRolloutWorkerV7Response) => void;
    reject: (error: Error) => void;
  };
};

function emptyTiming(): PpoRolloutWorkerV7Timing {
  return {
    observationMs: 0,
    legalActionsMs: 0,
    encodeObservationMs: 0,
    encodeActionsMs: 0,
    packMs: 0,
    gameStepMs: 0,
  };
}

function addTiming(
  target: PpoRolloutWorkerV7Timing,
  source: PpoRolloutWorkerV7Timing,
) {
  target.observationMs += source.observationMs;
  target.legalActionsMs += source.legalActionsMs;
  target.encodeObservationMs += source.encodeObservationMs;
  target.encodeActionsMs += source.encodeActionsMs;
  target.packMs += source.packMs;
  target.gameStepMs += source.gameStepMs;
}

export type PpoRolloutWorkerV7PrepareResult = {
  groups: PpoRolloutWorkerV7PreparedGroup[];
  finalized: PpoRolloutWorkerV7Finalized[];
  mergeLegalActionCount: number;
  workerCpuTiming: PpoRolloutWorkerV7Timing;
  barrierMs: number;
};

export type PpoRolloutWorkerV7ApplyResult = {
  finalized: PpoRolloutWorkerV7Finalized[];
  workerCpuTiming: PpoRolloutWorkerV7Timing;
  barrierMs: number;
};

export type PpoRolloutWorkerV7AdvanceResult = {
  groups: PpoRolloutWorkerV7PreparedGroup[];
  finalized: PpoRolloutWorkerV7Finalized[];
  mergeLegalActionCount: number;
  workerCpuTiming: PpoRolloutWorkerV7Timing;
  barrierMs: number;
};

export type PpoRolloutWorkerV7WorkerPrepareResult = {
  workerId: number;
  group?: PpoRolloutWorkerV7PreparedGroup;
  finalized: PpoRolloutWorkerV7Finalized[];
  mergeLegalActionCount: number;
  workerCpuTiming: PpoRolloutWorkerV7Timing;
  operationMs: number;
};

export type PpoRolloutWorkerV7WorkerAdvanceResult =
  PpoRolloutWorkerV7WorkerPrepareResult & {
    defeatDiagnosticEvents: PpoDefeatDiagnosticEventV8[];
  };

export class PpoRolloutWorkerV7Pool {
  private constructor(
    private readonly handles: WorkerHandle[],
    private readonly environmentToWorker: Map<number, number>,
  ) {}

  static async create(input: {
    workerCount: number;
    environmentCount: number;
    firstGameSeed: number;
    featureSpec: RlFeatureSpecV2;
    hyperparameters: PpoHyperparameters;
    safetyMaxTurns: number;
    safetyMaxActions: number;
    autoRecycle?: boolean;
    recycleSeedStride?: number;
    defeatDiagnostics?: boolean;
  }) {
    if (!Number.isInteger(input.workerCount) || input.workerCount <= 0) {
      throw new Error("PPO V7 rollout workerCount must be a positive integer");
    }
    if (
      !Number.isInteger(input.environmentCount)
      || input.environmentCount <= 0
    ) {
      throw new Error(
        "PPO V7 rollout environmentCount must be a positive integer",
      );
    }
    if (
      input.recycleSeedStride !== undefined
      && (
        !Number.isInteger(input.recycleSeedStride)
        || input.recycleSeedStride <= 0
      )
    ) {
      throw new Error(
        "PPO V7 rollout recycleSeedStride must be a positive integer",
      );
    }

    const effectiveWorkerCount = Math.min(
      input.workerCount,
      input.environmentCount,
    );
    const workerEntry = fileURLToPath(
      new URL("./rlPpoRolloutWorkerV7.ts", import.meta.url),
    );
    const assignments = Array.from(
      { length: effectiveWorkerCount },
      () => [] as Array<{
        environmentIndex: number;
        seed: number;
      }>,
    );
    for (
      let environmentIndex = 0;
      environmentIndex < input.environmentCount;
      environmentIndex += 1
    ) {
      const targetWorker = Math.min(
        effectiveWorkerCount - 1,
        Math.floor(environmentIndex * effectiveWorkerCount / input.environmentCount),
      );
      assignments[targetWorker].push({
        environmentIndex,
        seed: input.firstGameSeed + environmentIndex,
      });
    }

    const handles: WorkerHandle[] = [];
    const environmentToWorker = new Map<number, number>();

    const cleanup = async () => {
      await Promise.allSettled(
        handles.map((handle) => handle.worker.terminate()),
      );
    };

    try {
      for (
        let workerId = 0;
        workerId < effectiveWorkerCount;
        workerId += 1
      ) {
        const worker = new Worker(
          RL_VITE_NODE_ENTRY,
          {
            argv: [workerEntry],
          },
        );
        const environmentIndices = assignments[workerId].map(
          (entry) => entry.environmentIndex,
        );
        for (const environmentIndex of environmentIndices) {
          environmentToWorker.set(environmentIndex, workerId);
        }

        const handle: WorkerHandle = {
          workerId,
          worker,
          environmentIndices,
          nextRequestId: 1,
        };
        handles.push(handle);

        worker.on(
          "message",
          (raw: PpoRolloutWorkerV7Response) => {
            const pending = handle.pending;
            if (!pending) return;

            if (
              raw.type === "workerError"
              && (
                raw.requestId === undefined
                || raw.requestId === pending.requestId
              )
            ) {
              handle.pending = undefined;
              pending.reject(
                new Error(
                  `PPO V7 rollout worker ${workerId}: ${raw.error}`,
                ),
              );
              return;
            }

            if (
              "requestId" in raw
              && raw.requestId === pending.requestId
            ) {
              handle.pending = undefined;
              pending.resolve(raw);
            }
          },
        );

        worker.on("error", (error) => {
          const pending = handle.pending;
          handle.pending = undefined;
          const message = error instanceof Error
            ? error.message
            : String(error);
          pending?.reject(
            new Error(
              `PPO V7 rollout worker ${workerId} process error: ${message}`,
            ),
          );
        });

        worker.on("exit", (code) => {
          const pending = handle.pending;
          handle.pending = undefined;
          if (pending) {
            pending.reject(
              new Error(
                `PPO V7 rollout worker ${workerId} exited unexpectedly with code ${code}`,
              ),
            );
          }
        });
      }

      const pool = new PpoRolloutWorkerV7Pool(
        handles,
        environmentToWorker,
      );
      await Promise.all(
        handles.map(async (handle) => {
          const response = await pool.request(
            handle,
            {
              type: "init",
              workerId: handle.workerId,
              environments: assignments[handle.workerId],
              featureSpec: input.featureSpec,
              hyperparameters: input.hyperparameters,
              safetyMaxTurns: input.safetyMaxTurns,
              safetyMaxActions: input.safetyMaxActions,
              ...(input.autoRecycle === undefined
                ? {}
                : { autoRecycle: input.autoRecycle }),
              ...(input.recycleSeedStride === undefined
                ? {}
                : { recycleSeedStride: input.recycleSeedStride }),
              ...(input.defeatDiagnostics === undefined
                ? {}
                : { defeatDiagnostics: input.defeatDiagnostics }),
            },
          );
          if (
            response.type !== "ready"
            || response.workerId !== handle.workerId
          ) {
            throw new Error(
              `Unexpected PPO V7 rollout worker init response from ${handle.workerId}`,
            );
          }
        }),
      );
      return pool;
    } catch (error) {
      await cleanup();
      throw error;
    }
  }

  get workerCount() {
    return this.handles.length;
  }

  getWorkerEnvironmentIndices(workerId: number) {
    return [...this.handleFor(workerId).environmentIndices];
  }

  private handleFor(workerId: number) {
    if (!Number.isInteger(workerId)) {
      throw new Error("PPO V7 rollout workerId must be an integer");
    }
    const handle = this.handles[workerId];
    if (!handle || handle.workerId !== workerId) {
      throw new Error(`Unknown PPO V7 rollout worker ${workerId}`);
    }
    return handle;
  }

  private request(
    handle: WorkerHandle,
    message: RequestWithoutId,
  ): Promise<PpoRolloutWorkerV7Response> {
    if (handle.pending) {
      throw new Error(
        `PPO V7 rollout worker ${handle.workerId} already has an active request`,
      );
    }
    const requestId = handle.nextRequestId++;
    return new Promise((resolve, reject) => {
      handle.pending = {
        requestId,
        resolve,
        reject,
      };
      handle.worker.postMessage({
        ...message,
        requestId,
      } as PpoRolloutWorkerV7Request);
    });
  }

  async setWorkerAutoRecycle(
    workerId: number,
    enabled: boolean,
  ) {
    const handle = this.handleFor(workerId);
    const response = await this.request(
      handle,
      { type: "setAutoRecycle", enabled },
    );
    if (
      response.type !== "autoRecycleSet"
      || response.workerId !== workerId
      || response.enabled !== enabled
    ) {
      throw new Error(
        `Unexpected PPO rollout auto-recycle response from worker ${workerId}`,
      );
    }
  }

  async getWorkerDiagnostics(
    workerId: number,
  ): Promise<PpoRolloutEnvironmentDiagnosticV8[]> {
    const handle = this.handleFor(workerId);
    const response = await this.request(
      handle,
      { type: "getDiagnostics" },
    );
    if (
      response.type !== "diagnostics"
      || response.workerId !== workerId
    ) {
      throw new Error(
        `Unexpected PPO rollout diagnostics response from worker ${workerId}`,
      );
    }
    return [...response.environments].sort(
      (left, right) =>
        left.environmentIndex - right.environmentIndex,
    );
  }

  async getWorkerDefeatDiagnostics(
    workerId: number,
  ): Promise<PpoDefeatEnvironmentSnapshotV8[]> {
    const handle = this.handleFor(workerId);
    const response = await this.request(
      handle,
      { type: "getDefeatDiagnostics" },
    );
    if (
      response.type !== "defeatDiagnostics"
      || response.workerId !== workerId
    ) {
      throw new Error(
        `Unexpected PPO rollout defeat diagnostics response from worker ${workerId}`,
      );
    }
    return [...response.environments].sort(
      (left, right) =>
        left.environmentIndex - right.environmentIndex,
    );
  }

  async prepareWorker(
    workerId: number,
    round: number,
  ): Promise<PpoRolloutWorkerV7WorkerPrepareResult> {
    const handle = this.handleFor(workerId);
    const started = performance.now();
    const response = await this.request(
      handle,
      { type: "prepare", round },
    );
    if (
      response.type !== "prepared"
      || response.workerId !== workerId
      || response.round !== round
    ) {
      throw new Error(
        `Unexpected PPO rollout prepare response from worker ${workerId}`,
      );
    }
    const finalized = [...response.finalized].sort(
      (left, right) => left.environmentIndex - right.environmentIndex,
    );
    return {
      workerId,
      ...(response.group ? { group: response.group } : {}),
      finalized,
      mergeLegalActionCount: response.mergeLegalActionCount,
      workerCpuTiming: response.timing,
      operationMs: performance.now() - started,
    };
  }

  async advanceWorker(
    workerId: number,
    round: number,
    actions: Array<{
      environmentIndex: number;
      actionIndex: number;
      actionKey: string;
      logProbability: number;
      value: number;
    }>,
  ): Promise<PpoRolloutWorkerV7WorkerAdvanceResult> {
    const handle = this.handleFor(workerId);
    for (const action of actions) {
      const owner = this.environmentToWorker.get(action.environmentIndex);
      if (owner === undefined) {
        throw new Error(
          `Unknown PPO V7 rollout environment ${action.environmentIndex}`,
        );
      }
      if (owner !== workerId) {
        throw new Error(
          `PPO V7 rollout environment ${action.environmentIndex} belongs to worker ${owner}, not ${workerId}`,
        );
      }
    }

    const started = performance.now();
    const response = await this.request(
      handle,
      { type: "advance", round, actions },
    );
    if (
      response.type !== "advanced"
      || response.workerId !== workerId
      || response.round !== round
    ) {
      throw new Error(
        `Unexpected PPO rollout advance response from worker ${workerId}`,
      );
    }
    const finalized = [...response.finalized].sort(
      (left, right) => left.environmentIndex - right.environmentIndex,
    );
    return {
      workerId,
      ...(response.group ? { group: response.group } : {}),
      finalized,
      mergeLegalActionCount: response.mergeLegalActionCount,
      defeatDiagnosticEvents:
        response.defeatDiagnosticEvents ?? [],
      workerCpuTiming: response.timing,
      operationMs: performance.now() - started,
    };
  }

  async prepare(
    round: number,
  ): Promise<PpoRolloutWorkerV7PrepareResult> {
    const started = performance.now();
    const responses = await Promise.all(
      this.handles.map((handle) => this.request(handle, { type: "prepare", round })),
    );
    const groups: PpoRolloutWorkerV7PreparedGroup[] = [];
    const finalized: PpoRolloutWorkerV7Finalized[] = [];
    const workerCpuTiming = emptyTiming();
    let mergeLegalActionCount = 0;
    for (const response of responses) {
      if (response.type !== "prepared") {
        throw new Error("Unexpected PPO rollout prepare response: " + response.type);
      }
      if (response.group) groups.push(response.group);
      finalized.push(...response.finalized);
      mergeLegalActionCount += response.mergeLegalActionCount;
      addTiming(workerCpuTiming, response.timing);
    }
    groups.sort((left, right) =>
      (left.environmentIndices[0] ?? Number.MAX_SAFE_INTEGER)
      - (right.environmentIndices[0] ?? Number.MAX_SAFE_INTEGER)
    );
    finalized.sort((left, right) => left.environmentIndex - right.environmentIndex);
    return {
      groups,
      finalized,
      mergeLegalActionCount,
      workerCpuTiming,
      barrierMs: performance.now() - started,
    };
  }

  async apply(
    round: number,
    actions: Array<{
      environmentIndex: number;
      actionIndex: number;
      actionKey: string;
      logProbability: number;
      value: number;
    }>,
  ): Promise<PpoRolloutWorkerV7ApplyResult> {
    const actionsByWorker = new Map<number, typeof actions>();
    for (const action of actions) {
      const workerId = this.environmentToWorker.get(
        action.environmentIndex,
      );
      if (workerId === undefined) {
        throw new Error(
          `Unknown PPO V7 rollout environment ${action.environmentIndex}`,
        );
      }
      const list = actionsByWorker.get(workerId) ?? [];
      list.push(action);
      actionsByWorker.set(workerId, list);
    }

    const started = performance.now();
    const responses = await Promise.all(
      this.handles.map((handle) =>
        this.request(handle, {
          type: "apply",
          round,
          actions: actionsByWorker.get(handle.workerId) ?? [],
        }),
      ),
    );

    const finalized: PpoRolloutWorkerV7Finalized[] = [];
    const workerCpuTiming = emptyTiming();

    for (const response of responses) {
      if (response.type !== "applied") {
        throw new Error(
          `Unexpected PPO V7 rollout apply response: ${response.type}`,
        );
      }
      finalized.push(...response.finalized);
      addTiming(workerCpuTiming, response.timing);
    }
    finalized.sort(
      (left, right) =>
        left.environmentIndex - right.environmentIndex,
    );

    return {
      finalized,
      workerCpuTiming,
      barrierMs: performance.now() - started,
    };
  }

  async advance(
    round: number,
    actions: Array<{
      environmentIndex: number;
      actionIndex: number;
      actionKey: string;
      logProbability: number;
      value: number;
    }>,
  ): Promise<PpoRolloutWorkerV7AdvanceResult> {
    const actionsByWorker = new Map<number, typeof actions>();
    for (const action of actions) {
      const workerId = this.environmentToWorker.get(action.environmentIndex);
      if (workerId === undefined) {
        throw new Error(
          "Unknown PPO rollout environment " + action.environmentIndex,
        );
      }
      const list = actionsByWorker.get(workerId) ?? [];
      list.push(action);
      actionsByWorker.set(workerId, list);
    }

    const started = performance.now();
    const responses = await Promise.all(
      this.handles.map((handle) =>
        this.request(handle, {
          type: "advance",
          round,
          actions: actionsByWorker.get(handle.workerId) ?? [],
        }),
      ),
    );

    const groups: PpoRolloutWorkerV7PreparedGroup[] = [];
    const finalized: PpoRolloutWorkerV7Finalized[] = [];
    const workerCpuTiming = emptyTiming();
    let mergeLegalActionCount = 0;

    for (const response of responses) {
      if (response.type !== "advanced") {
        throw new Error(
          "Unexpected PPO rollout advance response: " + response.type,
        );
      }
      if (response.group) groups.push(response.group);
      finalized.push(...response.finalized);
      mergeLegalActionCount += response.mergeLegalActionCount;
      addTiming(workerCpuTiming, response.timing);
    }
    groups.sort(
      (left, right) =>
        (left.environmentIndices[0] ?? Number.MAX_SAFE_INTEGER)
        - (right.environmentIndices[0] ?? Number.MAX_SAFE_INTEGER),
    );
    finalized.sort(
      (left, right) => left.environmentIndex - right.environmentIndex,
    );

    return {
      groups,
      finalized,
      mergeLegalActionCount,
      workerCpuTiming,
      barrierMs: performance.now() - started,
    };
  }

  async close() {
    const requests = this.handles.map(async (handle) => {
      if (handle.pending) {
        throw new Error(
          `Cannot close busy PPO V7 rollout worker ${handle.workerId}`,
        );
      }
      try {
        const response = await this.request(
          handle,
          { type: "shutdown" },
        );
        if (response.type !== "closed") {
          throw new Error(
            `Unexpected PPO V7 rollout close response: ${response.type}`,
          );
        }
      } finally {
        await handle.worker.terminate();
      }
    });
    await Promise.all(requests);
  }
}
