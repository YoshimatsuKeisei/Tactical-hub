import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
import type { PackedBcBatch } from "../cpu/rlBcPackedBatch";
import { appendPpoUpdateScalars } from "../cpu/rlPpoPackedBatch";
import {
  deletePpoTrajectorySpool,
  PpoTrajectorySpoolWriter,
  readPpoTrajectorySpool,
} from "../cpu/rlPpoTrajectorySpool";

function packed(payload: number[], batchSize = 2): PackedBcBatch {
  return {
    payload: Buffer.from(payload),
    batchSize,
    tensors: [{
      name: "targets",
      dtype: "int32",
      shape: [batchSize],
      byteOffset: 0,
      byteLength: payload.length,
    }],
  };
}

describe("PPO trajectory spool", () => {
  it("round-trips packed feature chunks and reports file statistics", () => {
    const path = join(tmpdir(), `ppo-spool-test-${process.pid}-${Date.now()}.bin`);
    const writer = new PpoTrajectorySpoolWriter(path);
    try {
      writer.write(packed([1, 2, 3, 4], 2));
      writer.write(packed([5, 6], 1));
      writer.close();

      const chunks = [...readPpoTrajectorySpool(path)];
      expect(chunks).toHaveLength(2);
      expect(chunks.map((chunk) => chunk.batchSize)).toEqual([2, 1]);
      expect([...chunks[0].payload]).toEqual([1, 2, 3, 4]);
      expect([...chunks[1].payload]).toEqual([5, 6]);

      const stats = writer.stats();
      expect(stats.chunkCount).toBe(2);
      expect(stats.sampleCount).toBe(3);
      expect(stats.payloadBytes).toBe(6);
      expect(stats.fileBytes).toBeGreaterThan(stats.payloadBytes);
    } finally {
      writer.discard();
    }
  });

  it("appends only PPO scalar tensors to an existing packed feature batch", () => {
    const base = packed([9, 8, 7, 6], 2);
    const result = appendPpoUpdateScalars(base, [
      { oldLogProbability: -0.2, advantage: 0.5, return: 1.5 },
      { oldLogProbability: -0.3, advantage: -0.5, return: -1.5 },
    ]);
    expect(result.batchSize).toBe(2);
    expect([...result.payload.subarray(0, base.payload.length)]).toEqual([...base.payload]);
    expect(result.tensors.slice(-3).map((tensor) => tensor.name)).toEqual([
      "oldLogProbabilities",
      "advantages",
      "returns",
    ]);
    expect(result.tensors[result.tensors.length - 3].byteOffset).toBe(base.payload.length);
    expect(result.payload.length).toBe(base.payload.length + 3 * 2 * 4);
  });

  it("deletes an already closed spool safely", () => {
    const path = join(tmpdir(), `ppo-spool-delete-${process.pid}-${Date.now()}.bin`);
    const writer = new PpoTrajectorySpoolWriter(path);
    writer.write(packed([1, 2, 3, 4], 2));
    writer.close();
    deletePpoTrajectorySpool(path);
    expect(() => deletePpoTrajectorySpool(path)).not.toThrow();
  });
});
