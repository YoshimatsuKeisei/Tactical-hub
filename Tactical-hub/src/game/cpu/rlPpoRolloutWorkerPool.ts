import { performance } from "node:perf_hooks";
import { Worker } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { RL_VITE_NODE_ENTRY } from "./rlProjectPaths";
import type { RlFeatureSpecV2 } from "./rlFeatureSpec";
import type { PpoHyperparameters } from "./pythonPpoClient";
import type {
  PpoRolloutWorkerFinalized,
  PpoRolloutWorkerPackedSample,
  PpoRolloutWorkerRequest,
  PpoRolloutWorkerResponse,
  PpoRolloutWorkerTiming,
} from "./rlPpoRolloutWorkerMessages";

type PpoRolloutWorkerRequestWithoutId =
  PpoRolloutWorkerRequest extends infer Request
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
    resolve: (response: PpoRolloutWorkerResponse) => void;
    reject: (error: Error) => void;
  };
};

function emptyTiming(): PpoRolloutWorkerTiming {
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
  target: PpoRolloutWorkerTiming,
  source: PpoRolloutWorkerTiming,
) {
  target.observationMs += source.observationMs;
  target.legalActionsMs += source.legalActionsMs;
  target.encodeObservationMs += source.encodeObservationMs;
  target.encodeActionsMs += source.encodeActionsMs;
  target.packMs += source.packMs;
  target.gameStepMs += source.gameStepMs;
}

export type PpoRolloutWorkerPrepareResult = {
  samples: PpoRolloutWorkerPackedSample[];
  finalized: PpoRolloutWorkerFinalized[];
  mergeLegalActionCount: number;
  workerCpuTiming: PpoRolloutWorkerTiming;
  barrierMs: number;
};

export type PpoRolloutWorkerApplyResult = {
  finalized: PpoRolloutWorkerFinalized[];
  workerCpuTiming: PpoRolloutWorkerTiming;
  barrierMs: number;
};

export class PpoRolloutWorkerPool {
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
  }) {
    if (!Number.isInteger(input.workerCount) || input.workerCount <= 0) {
      throw new Error("PPO rollout workerCount must be a positive integer");
    }
    if (
      !Number.isInteger(input.environmentCount)
      || input.environmentCount <= 0
    ) {
      throw new Error("PPO rollout environmentCount must be a positive integer");
    }

    const effectiveWorkerCount = Math.min(
      input.workerCount,
      input.environmentCount,
    );
    const workerEntry = fileURLToPath(
      new URL("./rlPpoRolloutWorker.ts", import.meta.url),
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
      assignments[
        environmentIndex % effectiveWorkerCount
      ].push({
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
          (raw: PpoRolloutWorkerResponse) => {
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
                  `PPO rollout worker ${workerId}: ${raw.error}`,
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
              `PPO rollout worker ${workerId} process error: ${message}`,
            ),
          );
        });

        worker.on("exit", (code) => {
          const pending = handle.pending;
          handle.pending = undefined;
          if (pending) {
            pending.reject(
              new Error(
                `PPO rollout worker ${workerId} exited unexpectedly with code ${code}`,
              ),
            );
          }
        });
      }

      const pool = new PpoRolloutWorkerPool(
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
            },
          );
          if (
            response.type !== "ready"
            || response.workerId !== handle.workerId
          ) {
            throw new Error(
              `Unexpected PPO rollout worker init response from ${handle.workerId}`,
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

  private request(
    handle: WorkerHandle,
    message: PpoRolloutWorkerRequestWithoutId,
  ): Promise<PpoRolloutWorkerResponse> {
    if (handle.pending) {
      throw new Error(
        `PPO rollout worker ${handle.workerId} already has an active request`,
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
      } as PpoRolloutWorkerRequest);
    });
  }

  async prepare(
    round: number,
  ): Promise<PpoRolloutWorkerPrepareResult> {
    const started = performance.now();
    const responses = await Promise.all(
      this.handles.map((handle) =>
        this.request(handle, {
          type: "prepare",
          round,
        }),
      ),
    );

    const samples: PpoRolloutWorkerPackedSample[] = [];
    const finalized: PpoRolloutWorkerFinalized[] = [];
    const workerCpuTiming = emptyTiming();
    let mergeLegalActionCount = 0;

    for (const response of responses) {
      if (response.type !== "prepared") {
        throw new Error(
          `Unexpected PPO rollout prepare response: ${response.type}`,
        );
      }
      samples.push(...response.samples);
      finalized.push(...response.finalized);
      mergeLegalActionCount += response.mergeLegalActionCount;
      addTiming(workerCpuTiming, response.timing);
    }

    samples.sort(
      (left, right) =>
        left.environmentIndex - right.environmentIndex,
    );
    finalized.sort(
      (left, right) =>
        left.environmentIndex - right.environmentIndex,
    );

    return {
      samples,
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
  ): Promise<PpoRolloutWorkerApplyResult> {
    const actionsByWorker = new Map<number, typeof actions>();
    for (const action of actions) {
      const workerId = this.environmentToWorker.get(
        action.environmentIndex,
      );
      if (workerId === undefined) {
        throw new Error(
          `Unknown PPO rollout environment ${action.environmentIndex}`,
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

    const finalized: PpoRolloutWorkerFinalized[] = [];
    const workerCpuTiming = emptyTiming();

    for (const response of responses) {
      if (response.type !== "applied") {
        throw new Error(
          `Unexpected PPO rollout apply response: ${response.type}`,
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

  async close() {
    const requests = this.handles.map(async (handle) => {
      if (handle.pending) {
        throw new Error(
          `Cannot close busy PPO rollout worker ${handle.workerId}`,
        );
      }
      try {
        const response = await this.request(
          handle,
          { type: "shutdown" },
        );
        if (response.type !== "closed") {
          throw new Error(
            `Unexpected PPO rollout close response: ${response.type}`,
          );
        }
      } finally {
        await handle.worker.terminate();
      }
    });
    await Promise.all(requests);
  }
}
