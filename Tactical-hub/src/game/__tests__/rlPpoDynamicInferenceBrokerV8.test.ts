import { describe, expect, it } from "vitest";
import type { PpoRolloutWorkerV7PreparedGroup } from "../cpu/rlPpoRolloutWorkerV7Messages";
import {
  PpoDynamicInferenceBrokerV8,
  type PpoDynamicInferenceActionV8,
  type PpoDynamicInferenceFlushV8,
} from "../cpu/rlPpoDynamicInferenceBrokerV8";

function group(
  environmentIndices: number[],
): PpoRolloutWorkerV7PreparedGroup {
  const values = new Float32Array(
    environmentIndices.map((value) => value),
  );
  return {
    environmentIndices,
    decisionIndices: environmentIndices.map(() => 0),
    actionKeys: environmentIndices.map((environmentIndex) => [
      `action-${environmentIndex}-0`,
      `action-${environmentIndex}-1`,
    ]),
    packed: {
      payload: new Uint8Array(values.buffer),
      tensors: [{
        name: "test.fixed",
        dtype: "float32",
        shape: [environmentIndices.length, 1],
        byteOffset: 0,
        byteLength: values.byteLength,
      }],
      batchSize: environmentIndices.length,
    },
  };
}

function actions(
  actionKeys: string[][],
): PpoDynamicInferenceActionV8[] {
  return actionKeys.map((keys, index) => ({
    actionIndex: index % keys.length,
    actionKey: keys[index % keys.length],
    logProbability: -0.1 - index,
    value: 0.25 + index,
  }));
}

describe("PPO V8 dynamic inference broker", () => {
  it("combines ready independent submissions, sorts them, and maps results back", async () => {
    const flushes: PpoDynamicInferenceFlushV8[] = [];
    const broker = new PpoDynamicInferenceBrokerV8(
      async (_packed, actionKeys, flush) => {
        flushes.push(flush);
        return actions(actionKeys);
      },
    );

    const laterEnvironments = broker.submit(group([4, 5]));
    const earlierEnvironments = broker.submit(group([0, 1]));
    const [later, earlier] = await Promise.all([
      laterEnvironments,
      earlierEnvironments,
    ]);

    expect(flushes).toHaveLength(1);
    expect(flushes[0].environmentIndices).toEqual([
      0, 1, 4, 5,
    ]);
    expect(flushes[0].batchSize).toBe(4);
    expect(flushes[0].workerGroupCount).toBe(2);

    expect(earlier.map((action) => action.actionKey)).toEqual([
      "action-0-0",
      "action-1-1",
    ]);
    expect(later.map((action) => action.actionKey)).toEqual([
      "action-4-0",
      "action-5-1",
    ]);

    const diagnostics = broker.diagnostics();
    expect(diagnostics.flushCount).toBe(1);
    expect(diagnostics.batchSizeHistogram).toEqual({
      "4": 1,
    });
    expect(
      diagnostics.workerGroupCountHistogram,
    ).toEqual({ "2": 1 });
  });

  it("completes an earlier worker request without waiting for a later worker", async () => {
    const flushes: PpoDynamicInferenceFlushV8[] = [];
    const broker = new PpoDynamicInferenceBrokerV8(
      async (_packed, actionKeys, flush) => {
        flushes.push(flush);
        return actions(actionKeys);
      },
    );

    const first = await broker.submit(group([0, 1]));
    expect(first).toHaveLength(2);
    expect(flushes).toHaveLength(1);
    expect(flushes[0].environmentIndices).toEqual([0, 1]);

    const second = await broker.submit(group([6, 7]));
    expect(second).toHaveLength(2);
    expect(flushes).toHaveLength(2);
    expect(flushes[1].environmentIndices).toEqual([6, 7]);
  });

  it("rejects a failed flush and all later submissions instead of hanging", async () => {
    const failure = new Error("synthetic inference failure");
    const broker = new PpoDynamicInferenceBrokerV8(
      async () => {
        throw failure;
      },
    );

    const first = broker.submit(group([0, 1]));
    const second = broker.submit(group([2, 3]));

    await expect(first).rejects.toThrow(
      "synthetic inference failure",
    );
    await expect(second).rejects.toThrow(
      "synthetic inference failure",
    );
    await expect(
      broker.submit(group([4, 5])),
    ).rejects.toThrow("synthetic inference failure");
  });
});