import {
  closeSync,
  openSync,
  readSync,
  unlinkSync,
  writeSync,
} from "node:fs";
import type { PackedBcBatch } from "./rlBcPackedBatch";

type SpoolHeader = {
  batchSize: number;
  byteLength: number;
  tensors: PackedBcBatch["tensors"];
};

export type PpoTrajectorySpoolStats = {
  chunkCount: number;
  sampleCount: number;
  payloadBytes: number;
  fileBytes: number;
};

export class PpoTrajectorySpoolWriter {
  private readonly fd: number;
  private closed = false;
  private statsValue: PpoTrajectorySpoolStats = {
    chunkCount: 0,
    sampleCount: 0,
    payloadBytes: 0,
    fileBytes: 0,
  };
  constructor(readonly path: string) {
    this.fd = openSync(path, "wx");
  }

  write(packed: PackedBcBatch) {
    if (this.closed) throw new Error("PPO trajectory spool is closed");
    if (!packed.batchSize) throw new Error("Cannot spool an empty PPO batch");
    const header: SpoolHeader = {
      batchSize: packed.batchSize,
      byteLength: packed.payload.byteLength,
      tensors: packed.tensors,
    };
    const encodedHeader = Buffer.from(JSON.stringify(header), "utf8");
    const prefix = Buffer.allocUnsafe(4);
    prefix.writeUInt32LE(encodedHeader.byteLength, 0);
    writeSync(this.fd, prefix);
    writeSync(this.fd, encodedHeader);
    writeSync(this.fd, packed.payload);
    this.statsValue = {
      chunkCount: this.statsValue.chunkCount + 1,
      sampleCount: this.statsValue.sampleCount + packed.batchSize,
      payloadBytes: this.statsValue.payloadBytes + packed.payload.byteLength,
      fileBytes: this.statsValue.fileBytes + 4 + encodedHeader.byteLength + packed.payload.byteLength,
    };
  }

  close() {
    if (this.closed) return;
    closeSync(this.fd);
    this.closed = true;
  }
  stats() {
    return { ...this.statsValue };
  }

  discard() {
    this.close();
    try { unlinkSync(this.path); } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
}

function readExact(fd: number, buffer: Buffer) {
  let offset = 0;
  while (offset < buffer.byteLength) {
    const count = readSync(fd, buffer, offset, buffer.byteLength - offset, null);
    if (!count) throw new Error("PPO trajectory spool ended early");
    offset += count;
  }
}

export function* readPpoTrajectorySpool(path: string): Generator<PackedBcBatch> {
  const fd = openSync(path, "r");
  try {
    while (true) {
      const prefix = Buffer.allocUnsafe(4);
      const first = readSync(fd, prefix, 0, 4, null);
      if (first === 0) return;
      if (first !== 4) throw new Error("PPO trajectory spool header prefix is truncated");
      const headerLength = prefix.readUInt32LE(0);
      if (!headerLength || headerLength > 1_000_000) {
        throw new Error("PPO trajectory spool header length is invalid");
      }
      const headerBuffer = Buffer.allocUnsafe(headerLength);
      readExact(fd, headerBuffer);
      const header = JSON.parse(headerBuffer.toString("utf8")) as SpoolHeader;
      if (
        !Number.isInteger(header.batchSize) || header.batchSize <= 0
        || !Number.isInteger(header.byteLength) || header.byteLength < 0
        || !Array.isArray(header.tensors)
      ) throw new Error("PPO trajectory spool header is invalid");
      const payload = Buffer.allocUnsafe(header.byteLength);
      readExact(fd, payload);
      yield {
        batchSize: header.batchSize,
        tensors: header.tensors,
        payload,
      };
    }
  } finally {
    closeSync(fd);
  }
}

export function deletePpoTrajectorySpool(path: string) {
  try { unlinkSync(path); } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
}
