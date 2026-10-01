import { performance } from "node:perf_hooks";
import type { PackedBcBatch } from "./rlBcPackedBatch";
import type { PpoRolloutWorkerV7PreparedGroup } from "./rlPpoRolloutWorkerV7Messages";
import {
  combinePackedWorkerBatchesV7,
  fromTransferablePackedBcBatch,
} from "./rlPpoWorkerPackedV7";

export type PpoDynamicInferenceActionV8 = {
  actionIndex: number;
  actionKey: string;
  logProbability: number;
  value: number;
};

export type PpoDynamicInferenceFlushV8 = {
  flushIndex: number;
  workerGroupCount: number;
  batchSize: number;
  environmentIndices: number[];
};

export type PpoDynamicInferenceBrokerDiagnosticsV8 = {
  flushCount: number;
  batchSizeHistogram: Record<string, number>;
  workerGroupCountHistogram: Record<string, number>;
  combineMs: number;
  inferenceMs: number;
  queueToFlushWaitMs: number;
  averageQueueToFlushWaitMs: number;
  maxQueueToFlushWaitMs: number;
  submittedGroupCount: number;
  submittedSampleCount: number;
};

type Submission = {
  sequence: number;
  queuedAt: number;
  group: PpoRolloutWorkerV7PreparedGroup;
  resolve: (actions: PpoDynamicInferenceActionV8[]) => void;
  reject: (error: Error) => void;
};

export class PpoDynamicInferenceBrokerV8 {
  private readonly queue: Submission[] = [];
  private scheduled = false;
  private flushing = false;
  private sequence = 0;
  private failure?: Error;
  private readonly diagnosticsValue: PpoDynamicInferenceBrokerDiagnosticsV8 = {
    flushCount: 0,
    batchSizeHistogram: {},
    workerGroupCountHistogram: {},
    combineMs: 0,
    inferenceMs: 0,
    queueToFlushWaitMs: 0,
    averageQueueToFlushWaitMs: 0,
    maxQueueToFlushWaitMs: 0,
    submittedGroupCount: 0,
    submittedSampleCount: 0,
  };

  constructor(
    private readonly infer: (
      packed: PackedBcBatch,
      actionKeys: string[][],
      flush: PpoDynamicInferenceFlushV8,
    ) => Promise<PpoDynamicInferenceActionV8[]>,
  ) {}

  submit(group: PpoRolloutWorkerV7PreparedGroup) {
    try {
      this.validateGroup(group);
    } catch (error) {
      return Promise.reject<PpoDynamicInferenceActionV8[]>(
        this.asError(error),
      );
    }
    if (this.failure) {
      return Promise.reject<PpoDynamicInferenceActionV8[]>(this.failure);
    }

    return new Promise<PpoDynamicInferenceActionV8[]>((resolve, reject) => {
      this.queue.push({
        sequence: this.sequence++,
        queuedAt: performance.now(),
        group,
        resolve,
        reject,
      });
      this.diagnosticsValue.submittedGroupCount += 1;
      this.diagnosticsValue.submittedSampleCount += group.packed.batchSize;
      this.scheduleFlush();
    });
  }

  close(error: unknown = new Error("PPO dynamic inference broker closed")) {
    if (!this.failure) this.failure = this.asError(error);
    this.rejectQueued(this.failure);
  }

  diagnostics(): PpoDynamicInferenceBrokerDiagnosticsV8 {
    return {
      ...this.diagnosticsValue,
      batchSizeHistogram: {
        ...this.diagnosticsValue.batchSizeHistogram,
      },
      workerGroupCountHistogram: {
        ...this.diagnosticsValue.workerGroupCountHistogram,
      },
      averageQueueToFlushWaitMs:
        this.diagnosticsValue.submittedGroupCount
          ? this.diagnosticsValue.queueToFlushWaitMs
            / this.diagnosticsValue.submittedGroupCount
          : 0,
    };
  }

  private validateGroup(group: PpoRolloutWorkerV7PreparedGroup) {
    const count = group.environmentIndices.length;
    if (
      count <= 0
      || group.decisionIndices.length !== count
      || group.actionKeys.length !== count
      || group.packed.batchSize !== count
    ) {
      throw new Error("PPO dynamic inference group metadata mismatch");
    }
    if (group.actionKeys.some((keys) => keys.length === 0)) {
      throw new Error("PPO dynamic inference group has no legal actions");
    }
    if (new Set(group.environmentIndices).size !== count) {
      throw new Error("PPO dynamic inference group has duplicate environments");
    }
    for (let index = 1; index < count; index += 1) {
      if (
        group.environmentIndices[index - 1]
        >= group.environmentIndices[index]
      ) {
        throw new Error(
          "PPO dynamic inference group environments must be ordered",
        );
      }
    }
  }

  private scheduleFlush() {
    if (
      this.scheduled
      || this.flushing
      || this.failure
      || this.queue.length === 0
    ) {
      return;
    }
    this.scheduled = true;
    setImmediate(() => {
      this.scheduled = false;
      void this.flush();
    });
  }

  private async flush() {
    if (this.flushing || this.failure || this.queue.length === 0) return;

    this.flushing = true;
    const submissions = this.queue.splice(0).sort((left, right) => {
      const leftEnvironment = left.group.environmentIndices[0];
      const rightEnvironment = right.group.environmentIndices[0];
      return leftEnvironment - rightEnvironment
        || left.sequence - right.sequence;
    });

    try {
      const flushStarted = performance.now();
      for (const submission of submissions) {
        const waitMs = Math.max(0, flushStarted - submission.queuedAt);
        this.diagnosticsValue.queueToFlushWaitMs += waitMs;
        this.diagnosticsValue.maxQueueToFlushWaitMs = Math.max(
          this.diagnosticsValue.maxQueueToFlushWaitMs,
          waitMs,
        );
      }

      const environmentIndices = submissions.flatMap(
        ({ group }) => group.environmentIndices,
      );
      if (
        new Set(environmentIndices).size !== environmentIndices.length
      ) {
        throw new Error(
          "PPO dynamic inference flush has duplicate environments",
        );
      }

      const combineStarted = performance.now();
      const packed = combinePackedWorkerBatchesV7(
        submissions.map(({ group }) =>
          fromTransferablePackedBcBatch(group.packed)),
      );
      const actionKeys = submissions.flatMap(
        ({ group }) => group.actionKeys,
      );
      this.diagnosticsValue.combineMs +=
        performance.now() - combineStarted;

      const flushIndex = this.diagnosticsValue.flushCount;
      const flush: PpoDynamicInferenceFlushV8 = {
        flushIndex,
        workerGroupCount: submissions.length,
        batchSize: packed.batchSize,
        environmentIndices,
      };
      this.diagnosticsValue.flushCount += 1;
      this.incrementHistogram(
        this.diagnosticsValue.batchSizeHistogram,
        packed.batchSize,
      );
      this.incrementHistogram(
        this.diagnosticsValue.workerGroupCountHistogram,
        submissions.length,
      );

      const inferenceStarted = performance.now();
      const selected = await this.infer(
        packed,
        actionKeys,
        flush,
      );
      this.diagnosticsValue.inferenceMs +=
        performance.now() - inferenceStarted;

      if (selected.length !== packed.batchSize) {
        throw new Error(
          "PPO dynamic inference returned wrong action count",
        );
      }

      selected.forEach((action, index) => {
        const keys = actionKeys[index];
        if (
          !Number.isInteger(action.actionIndex)
          || action.actionIndex < 0
          || action.actionIndex >= keys.length
          || action.actionKey !== keys[action.actionIndex]
          || !Number.isFinite(action.logProbability)
          || !Number.isFinite(action.value)
        ) {
          throw new Error(
            `PPO dynamic inference returned invalid action at row ${index}`,
          );
        }
      });

      let offset = 0;
      for (const submission of submissions) {
        const count = submission.group.packed.batchSize;
        submission.resolve(
          selected.slice(offset, offset + count),
        );
        offset += count;
      }
      if (offset !== selected.length) {
        throw new Error("PPO dynamic inference action split mismatch");
      }
    } catch (error) {
      const failure = this.asError(error);
      this.failure = failure;
      for (const submission of submissions) {
        submission.reject(failure);
      }
      this.rejectQueued(failure);
    } finally {
      this.flushing = false;
      if (!this.failure) this.scheduleFlush();
    }
  }

  private incrementHistogram(
    histogram: Record<string, number>,
    value: number,
  ) {
    const key = String(value);
    histogram[key] = (histogram[key] ?? 0) + 1;
  }

  private rejectQueued(error: Error) {
    while (this.queue.length) {
      this.queue.shift()!.reject(error);
    }
  }

  private asError(error: unknown) {
    return error instanceof Error
      ? error
      : new Error(String(error));
  }
}