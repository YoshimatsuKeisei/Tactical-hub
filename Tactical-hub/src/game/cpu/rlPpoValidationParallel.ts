import { fork, type ChildProcess } from "node:child_process";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import {
  RL_VITE_NODE_ENTRY,
} from "./rlProjectPaths";
import type {
  PpoReplayValidationRollout,
} from "./rlPpoSelfPlay";
import type {
  PpoValidationWorkerRequest,
  PpoValidationWorkerResponse,
} from "./rlPpoValidationWorkerMessages";

export type ParallelPpoValidationResult = {
  requestedWorkerCount: number;
  effectiveWorkerCount: number;
  sampleCount: number;
  elapsedMs: number;
  perTask: Array<{
    taskId: string;
    seed: number;
    sampleCount: number;
    elapsedMs: number;
  }>;
};

export async function validatePpoRolloutsParallel(input: {
  rollouts: PpoReplayValidationRollout[];
  workerCount: number;
  memoryLogInterval: number;
  fastRlMovement: boolean;
  fastRlPhaseTransitions: boolean;
  workerEntryPath?: string;
}): Promise<ParallelPpoValidationResult> {
  if (!input.rollouts.length) {
    throw new Error("PPO parallel validation requires rollouts");
  }
  if (!Number.isInteger(input.workerCount) || input.workerCount <= 0) {
    throw new Error("PPO validation workerCount must be a positive integer");
  }

  const effectiveWorkerCount = Math.min(
    input.workerCount,
    input.rollouts.length,
  );
  const workerEntry = input.workerEntryPath
    ?? fileURLToPath(
      new URL("./rlPpoValidationWorker.ts", import.meta.url),
    );

  const tasks = input.rollouts.map((rollout, index) => ({
    taskId: `validation-${index}-seed-${rollout.seed}`,
    rollout,
  }));

  const started = performance.now();
  const workers: ChildProcess[] = [];
  const pending = new Map<
    string,
    { workerId: number; taskIndex: number }
  >();
  const completed = new Map<
    string,
    {
      taskId: string;
      seed: number;
      sampleCount: number;
      elapsedMs: number;
    }
  >();
  let nextTaskIndex = 0;

  return await new Promise((resolveBatch, rejectBatch) => {
    let settled = false;

    const cleanup = () => {
      for (const worker of workers) {
        if (worker.connected) {
          worker.send({
            type: "shutdown",
          } satisfies PpoValidationWorkerRequest);
        }
        worker.kill();
      }
    };

    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      cleanup();
      rejectBatch(error);
    };

    const finishIfDone = () => {
      if (
        !settled
        && completed.size === tasks.length
      ) {
        settled = true;
        cleanup();

        const perTask = tasks.map((task) => {
          const result = completed.get(task.taskId);
          if (!result) {
            throw new Error(
              `Missing PPO validation result for ${task.taskId}`,
            );
          }
          return result;
        });

        resolveBatch({
          requestedWorkerCount: input.workerCount,
          effectiveWorkerCount,
          sampleCount: perTask.reduce(
            (sum, result) => sum + result.sampleCount,
            0,
          ),
          elapsedMs: performance.now() - started,
          perTask,
        });
        return true;
      }
      return false;
    };

    const assign = (
      worker: ChildProcess,
      workerId: number,
    ) => {
      if (finishIfDone()) return;
      if (nextTaskIndex >= tasks.length) return;

      const taskIndex = nextTaskIndex++;
      const task = tasks[taskIndex];
      pending.set(task.taskId, {
        workerId,
        taskIndex,
      });

      worker.send({
        type: "validate",
        taskId: task.taskId,
        rollout: task.rollout,
        memoryLogInterval: input.memoryLogInterval,
        fastRlMovement: input.fastRlMovement,
        fastRlPhaseTransitions: input.fastRlPhaseTransitions,
      } satisfies PpoValidationWorkerRequest);
    };

    for (
      let workerId = 0;
      workerId < effectiveWorkerCount;
      workerId += 1
    ) {
      const worker = fork(
        RL_VITE_NODE_ENTRY,
        [workerEntry],
        {
          stdio: [
            "ignore",
            "ignore",
            "ignore",
            "ipc",
          ],
        },
      );
      workers.push(worker);

      worker.on(
        "message",
        (raw: PpoValidationWorkerResponse) => {
          if (
            settled
            || !raw
            || ![
              "validated",
              "workerError",
            ].includes(raw.type)
          ) {
            if (!settled) {
              fail(
                new Error(
                  `PPO validation worker ${workerId} sent invalid response`,
                ),
              );
            }
            return;
          }

          if (!raw.taskId) {
            fail(
              new Error(
                `PPO validation worker ${workerId} failed outside a known task: `
                + (
                  raw.type === "workerError"
                    ? raw.error
                    : "invalid response"
                ),
              ),
            );
            return;
          }

          const assignment = pending.get(raw.taskId);
          if (
            !assignment
            || assignment.workerId !== workerId
          ) {
            fail(
              new Error(
                `PPO validation worker ${workerId} completed unknown task ${raw.taskId}`,
              ),
            );
            return;
          }

          pending.delete(raw.taskId);
          const task = tasks[assignment.taskIndex];

          if (raw.type === "workerError") {
            fail(
              new Error(
                `PPO validation failed for seed=${task.rollout.seed}: ${raw.error}`,
              ),
            );
            return;
          }

          if (
            raw.seed !== task.rollout.seed
            || raw.sampleCount
              !== task.rollout.trajectory.length
          ) {
            fail(
              new Error(
                `PPO validation worker ${workerId} returned mismatched task ${raw.taskId}`,
              ),
            );
            return;
          }

          completed.set(raw.taskId, {
            taskId: raw.taskId,
            seed: raw.seed,
            sampleCount: raw.sampleCount,
            elapsedMs: raw.elapsedMs,
          });

          assign(worker, workerId);
        },
      );

      worker.on("error", (error) => {
        fail(
          new Error(
            `PPO validation worker ${workerId} process error: ${error.message}`,
          ),
        );
      });

      worker.on("exit", (code, signal) => {
        if (!settled) {
          const active = [...pending.entries()]
            .find(([, entry]) => entry.workerId === workerId);
          const suffix = active
            ? ` task=${active[0]}`
            : "";
          fail(
            new Error(
              `PPO validation worker ${workerId} exited unexpectedly${suffix} `
              + `(code=${code}, signal=${signal})`,
            ),
          );
        }
      });

      assign(worker, workerId);
    }
  });
}
