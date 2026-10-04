import { webcrypto } from "node:crypto";
import { IDBFactory, IDBObjectStore } from "fake-indexeddb";
import { afterEach, describe, expect, it } from "vitest";
import { createHeuristicCpuPolicy } from "../cpu/heuristicCpuPolicy";
import { createCpuRuntime, type CpuTeamSettings } from "../cpu/types";
import { createInitialGameState } from "../initialState";
import { calculateLocalGameSaveSha256 } from "../save/localGameSaveChecksum";
import { IndexedDbLocalGameSaveRepository } from "../save/indexedDbLocalGameSaveRepository";
import { createLocalMatchSaveSnapshot, serializeLocalMatchSave } from "../save/localGameSaveSerializer";
import {
  LOCAL_GAME_SAVE_METADATA_STORE_NAME,
  LOCAL_GAME_SAVE_STORAGE_RECORD_VERSION,
  LOCAL_GAME_SAVE_STORE_NAME,
  type LocalGameSaveRepositoryResult,
  type LocalGameSaveStorageRecord,
} from "../save/localGameSaveStorageTypes";
import type { LocalMatchSaveResult, LocalMatchSaveV1 } from "../save/localGameSaveTypes";

const cryptoProvider = webcrypto as unknown as Crypto;
const settings: CpuTeamSettings = {
  "team-1": "human",
  "team-2": "random_cpu",
  "team-3": "heuristic_cpu",
  "team-4": "random_cpu",
};

let databaseSequence = 0;
const databases: Array<{ factory: IDBFactory; name: string }> = [];

function unwrapSave<T>(result: LocalMatchSaveResult<T>): T {
  if (!result.ok) throw new Error(`${result.error.code}: ${result.error.message}`);
  return result.value;
}

function unwrapRepository<T>(result: LocalGameSaveRepositoryResult<T>): T {
  if (!result.ok) throw new Error(`${result.error.code}: ${result.error.message}`);
  return result.value;
}

function expectRepositoryError<T>(result: LocalGameSaveRepositoryResult<T>, code: string) {
  expect(result.ok).toBe(false);
  if (!result.ok) expect(result.error.code).toBe(code);
}

function makeRepository() {
  const factory = new IDBFactory();
  const name = `tactical-hub-save-test-${++databaseSequence}`;
  databases.push({ factory, name });
  return {
    factory,
    name,
    repository: new IndexedDbLocalGameSaveRepository({ indexedDB: factory, crypto: cryptoProvider, databaseName: name }),
  };
}

function makeSave(input: {
  saveId?: string;
  displayName?: string;
  createdAt?: string;
  updatedAt?: string;
  cpuSettings?: CpuTeamSettings;
  rngState?: number;
} = {}) {
  const runtime = createCpuRuntime(12345);
  if (input.rngState !== undefined) runtime.rngState = input.rngState;
  return unwrapSave(createLocalMatchSaveSnapshot({
    saveId: input.saveId ?? "save-1",
    createdAt: input.createdAt ?? "2026-10-04T00:00:00.000Z",
    updatedAt: input.updatedAt ?? "2026-10-04T01:00:00.000Z",
    displayName: input.displayName ?? "Storage fixture",
    gameState: createInitialGameState(),
    cpuSettings: input.cpuSettings ?? settings,
    cpuRuntime: runtime,
    heuristicPolicyState: createHeuristicCpuPolicy().snapshotState(),
    resumeUi: { viewerTeamId: "team-1" },
  }));
}

function openDatabase(factory: IDBFactory, name: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = factory.open(name);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error);
    transaction.onerror = () => undefined;
  });
}

function requestValue<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function getRawRecord(factory: IDBFactory, name: string, saveId: string) {
  const database = await openDatabase(factory, name);
  const transaction = database.transaction(LOCAL_GAME_SAVE_STORE_NAME, "readonly");
  const completion = transactionDone(transaction);
  const result = await requestValue(transaction.objectStore(LOCAL_GAME_SAVE_STORE_NAME).get(saveId));
  await completion;
  database.close();
  return result as LocalGameSaveStorageRecord | undefined;
}

async function getRawMetadata(factory: IDBFactory, name: string, saveId: string) {
  const database = await openDatabase(factory, name);
  const transaction = database.transaction(LOCAL_GAME_SAVE_METADATA_STORE_NAME, "readonly");
  const completion = transactionDone(transaction);
  const result = await requestValue(transaction.objectStore(LOCAL_GAME_SAVE_METADATA_STORE_NAME).get(saveId));
  await completion;
  database.close();
  return result;
}

async function mutateRawRecord(factory: IDBFactory, name: string, saveId: string, mutate: (record: LocalGameSaveStorageRecord) => void) {
  const database = await openDatabase(factory, name);
  const transaction = database.transaction(LOCAL_GAME_SAVE_STORE_NAME, "readwrite");
  const completion = transactionDone(transaction);
  const store = transaction.objectStore(LOCAL_GAME_SAVE_STORE_NAME);
  const record = await requestValue(store.get(saveId)) as LocalGameSaveStorageRecord;
  mutate(record);
  store.put(record);
  await completion;
  database.close();
}

async function replaceCurrentJson(factory: IDBFactory, name: string, saveId: string, transform: (value: Record<string, unknown>) => void) {
  await mutateRawRecord(factory, name, saveId, (record) => {
    const parsed = JSON.parse(record.current.serializedSave) as Record<string, unknown>;
    transform(parsed);
    record.current.serializedSave = JSON.stringify(parsed);
  });
  const record = await getRawRecord(factory, name, saveId);
  const checksum = await calculateLocalGameSaveSha256(record!.current.serializedSave, cryptoProvider);
  await mutateRawRecord(factory, name, saveId, (next) => { next.current.sha256 = checksum; });
}

function deleteDatabase(factory: IDBFactory, name: string) {
  return new Promise<void>((resolve) => {
    const request = factory.deleteDatabase(name);
    request.onsuccess = () => resolve();
    request.onerror = () => resolve();
    request.onblocked = () => resolve();
  });
}

afterEach(async () => {
  const pending = databases.splice(0);
  await Promise.all(pending.map(({ factory, name }) => deleteDatabase(factory, name)));
});

describe("IndexedDbLocalGameSaveRepository", () => {
  it("puts a new save and restores the exact Stage 1 save from current", async () => {
    const { repository } = makeRepository();
    const save = makeSave();
    expect(unwrapRepository(await repository.put(save))).toEqual({ saveId: save.saveId, revision: 1 });
    expect(unwrapRepository(await repository.get(save.saveId))).toEqual({ save, source: "current", revision: 1 });
  });

  it("rotates the old current revision to the single previous slot", async () => {
    const { repository, factory, name } = makeRepository();
    const first = makeSave({ displayName: "A" });
    const second = makeSave({ displayName: "B", updatedAt: "2026-10-04T02:00:00.000Z" });
    await repository.put(first);
    expect(unwrapRepository(await repository.put(second)).revision).toBe(2);
    const record = (await getRawRecord(factory, name, first.saveId))!;
    expect(record.current.revision).toBe(2);
    expect(record.previous?.revision).toBe(1);
    expect(record.previous?.serializedSave).toBe(unwrapSave(serializeLocalMatchSave(first)));
    expect(unwrapRepository(await repository.get(first.saveId)).save).toEqual(second);
  });

  it("falls back to previous when the current checksum is corrupted", async () => {
    const { repository, factory, name } = makeRepository();
    const first = makeSave({ displayName: "A" });
    const second = makeSave({ displayName: "B", updatedAt: "2026-10-04T02:00:00.000Z" });
    await repository.put(first); await repository.put(second);
    await mutateRawRecord(factory, name, first.saveId, (record) => { record.current.sha256 = "0".repeat(64); });
    const loaded = unwrapRepository(await repository.get(first.saveId));
    expect(loaded).toEqual({ save: first, source: "previous", revision: 1 });
  });

  it("falls back when current contains checksum-valid malformed JSON", async () => {
    const { repository, factory, name } = makeRepository();
    const first = makeSave({ displayName: "A" });
    await repository.put(first);
    await repository.put(makeSave({ displayName: "B", updatedAt: "2026-10-04T02:00:00.000Z" }));
    const malformed = "{not-json";
    const checksum = await calculateLocalGameSaveSha256(malformed, cryptoProvider);
    await mutateRawRecord(factory, name, first.saveId, (record) => { record.current.serializedSave = malformed; record.current.sha256 = checksum; });
    expect(unwrapRepository(await repository.get(first.saveId)).source).toBe("previous");
  });

  it("falls back when current has an invalid game state", async () => {
    const { repository, factory, name } = makeRepository();
    const first = makeSave({ displayName: "A" });
    await repository.put(first);
    await repository.put(makeSave({ displayName: "B", updatedAt: "2026-10-04T02:00:00.000Z" }));
    await replaceCurrentJson(factory, name, first.saveId, (value) => {
      const payload = value.payload as { gameState: { units: Array<{ id: string }> } };
      payload.gameState.units[1].id = payload.gameState.units[0].id;
    });
    expect(unwrapRepository(await repository.get(first.saveId))).toEqual({ save: first, source: "previous", revision: 1 });
  });

  it("does not fall back for an unsupported save schema", async () => {
    const { repository, factory, name } = makeRepository();
    const first = makeSave({ displayName: "A" });
    await repository.put(first);
    await repository.put(makeSave({ displayName: "B", updatedAt: "2026-10-04T02:00:00.000Z" }));
    await replaceCurrentJson(factory, name, first.saveId, (value) => { value.saveSchemaVersion = 99; });
    expectRepositoryError(await repository.get(first.saveId), "UNSUPPORTED_SCHEMA_VERSION");
  });

  it("does not fall back for a map mismatch", async () => {
    const { repository, factory, name } = makeRepository();
    const first = makeSave({ displayName: "A" });
    await repository.put(first);
    await repository.put(makeSave({ displayName: "B", updatedAt: "2026-10-04T02:00:00.000Z" }));
    await replaceCurrentJson(factory, name, first.saveId, (value) => {
      (value.compatibility as { mapId: string }).mapId = "other-map";
    });
    expectRepositoryError(await repository.get(first.saveId), "MAP_MISMATCH");
  });

  it("reports no recoverable revision when both revisions are corrupted", async () => {
    const { repository, factory, name } = makeRepository();
    const save = makeSave();
    await repository.put(save);
    await repository.put(makeSave({ updatedAt: "2026-10-04T02:00:00.000Z" }));
    await mutateRawRecord(factory, name, save.saveId, (record) => {
      record.current.sha256 = "0".repeat(64);
      record.previous!.sha256 = "f".repeat(64);
    });
    const result = await repository.get(save.saveId);
    expectRepositoryError(result, "NO_RECOVERABLE_REVISION");
    if (!result.ok) expect(result.error).toMatchObject({ currentError: { code: "CORRUPTED_SAVE" }, previousError: { code: "CORRUPTED_SAVE" } });
  });

  it("lists metadata without loading or parsing the corrupted payload", async () => {
    const { repository, factory, name } = makeRepository();
    const save = makeSave(); await repository.put(save);
    await mutateRawRecord(factory, name, save.saveId, (record) => { record.current.serializedSave = "not-json"; });
    const listed = unwrapRepository(await repository.list());
    expect(listed).toHaveLength(1);
    expect(listed[0]).toMatchObject({ saveId: save.saveId, metadata: save.metadata });
  });

  it("lists multiple saves by updatedAt descending", async () => {
    const { repository } = makeRepository();
    await repository.put(makeSave({ saveId: "old", updatedAt: "2026-10-04T01:00:00.000Z" }));
    await repository.put(makeSave({ saveId: "new", updatedAt: "2026-10-04T03:00:00.000Z" }));
    await repository.put(makeSave({ saveId: "middle", updatedAt: "2026-10-04T02:00:00.000Z" }));
    expect(unwrapRepository(await repository.list()).map((entry) => entry.saveId)).toEqual(["new", "middle", "old"]);
  });

  it("deletes storage and metadata records in one transaction", async () => {
    const { repository, factory, name } = makeRepository();
    const save = makeSave(); await repository.put(save);
    expect(unwrapRepository(await repository.delete(save.saveId))).toEqual({ saveId: save.saveId, deleted: true });
    expect(await getRawRecord(factory, name, save.saveId)).toBeUndefined();
    expect(await getRawMetadata(factory, name, save.saveId)).toBeUndefined();
    expectRepositoryError(await repository.get(save.saveId), "NOT_FOUND");
  });

  it("keeps the old current when a storage-record write fails", async () => {
    const { repository } = makeRepository();
    const first = makeSave({ displayName: "A" }); await repository.put(first);
    const originalPut = IDBObjectStore.prototype.put;
    IDBObjectStore.prototype.put = function(value: unknown, key?: IDBValidKey) {
      if (this.name === LOCAL_GAME_SAVE_STORE_NAME) throw new DOMException("injected save write failure", "UnknownError");
      return originalPut.call(this, value, key);
    };
    try {
      expectRepositoryError(await repository.put(makeSave({ displayName: "B", updatedAt: "2026-10-04T02:00:00.000Z" })), "TRANSACTION_FAILED");
    } finally {
      IDBObjectStore.prototype.put = originalPut;
    }
    expect(unwrapRepository(await repository.get(first.saveId))).toMatchObject({ save: first, source: "current", revision: 1 });
  });

  it("rolls back the payload update when the metadata write fails", async () => {
    const { repository } = makeRepository();
    const first = makeSave({ displayName: "A" }); await repository.put(first);
    const originalPut = IDBObjectStore.prototype.put;
    IDBObjectStore.prototype.put = function(value: unknown, key?: IDBValidKey) {
      if (this.name === LOCAL_GAME_SAVE_METADATA_STORE_NAME) throw new DOMException("injected metadata failure", "UnknownError");
      return originalPut.call(this, value, key);
    };
    try {
      expectRepositoryError(await repository.put(makeSave({ displayName: "B", updatedAt: "2026-10-04T02:00:00.000Z" })), "TRANSACTION_FAILED");
    } finally {
      IDBObjectStore.prototype.put = originalPut;
    }
    expect(unwrapRepository(await repository.get(first.saveId))).toMatchObject({ save: first, revision: 1 });
  });

  it("classifies quota errors and does not delete the old save", async () => {
    const { repository } = makeRepository();
    const first = makeSave({ displayName: "A" }); await repository.put(first);
    const originalPut = IDBObjectStore.prototype.put;
    IDBObjectStore.prototype.put = function(value: unknown, key?: IDBValidKey) {
      if (this.name === LOCAL_GAME_SAVE_METADATA_STORE_NAME) throw new DOMException("quota fixture", "QuotaExceededError");
      return originalPut.call(this, value, key);
    };
    try {
      expectRepositoryError(await repository.put(makeSave({ displayName: "B", updatedAt: "2026-10-04T02:00:00.000Z" })), "QUOTA_EXCEEDED");
    } finally {
      IDBObjectStore.prototype.put = originalPut;
    }
    expect(unwrapRepository(await repository.get(first.saveId)).save).toEqual(first);
  });

  it("reports IndexedDB unavailable without falling back to another storage", async () => {
    const repository = new IndexedDbLocalGameSaveRepository({ indexedDB: null, crypto: cryptoProvider });
    expectRepositoryError(await repository.put(makeSave()), "STORAGE_UNAVAILABLE");
    expectRepositoryError(await repository.get("save-1"), "STORAGE_UNAVAILABLE");
    expectRepositoryError(await repository.list(), "STORAGE_UNAVAILABLE");
    expectRepositoryError(await repository.delete("save-1"), "STORAGE_UNAVAILABLE");
  });

  it("does not consume game RNG while calculating a checksum", async () => {
    const save = makeSave({ rngState: 0xdeadbeef });
    const before = save.payload.cpuRuntime.rngState;
    await calculateLocalGameSaveSha256(unwrapSave(serializeLocalMatchSave(save)), cryptoProvider);
    expect(save.payload.cpuRuntime.rngState).toBe(before);
  });

  it("does not consume game RNG across put, get, list, or delete", async () => {
    const { repository } = makeRepository();
    const save = makeSave({ rngState: 0x12345678 });
    const before = save.payload.cpuRuntime.rngState;
    await repository.put(save); await repository.get(save.saveId); await repository.list(); await repository.delete(save.saveId);
    expect(save.payload.cpuRuntime.rngState).toBe(before);
  });

  it("hashes the exact serialized UTF-8 string rather than parsed JSON", async () => {
    const compact = "{\"a\":1}";
    const spaced = "{ \"a\": 1 }";
    expect(JSON.parse(compact)).toEqual(JSON.parse(spaced));
    expect(await calculateLocalGameSaveSha256(compact, cryptoProvider)).not.toBe(await calculateLocalGameSaveSha256(spaced, cryptoProvider));
    expect(await calculateLocalGameSaveSha256("abc", cryptoProvider)).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });

  it("rejects a storage record whose serialized payload has a different saveId", async () => {
    const { repository, factory, name } = makeRepository();
    const save = makeSave(); await repository.put(save);
    await replaceCurrentJson(factory, name, save.saveId, (value) => { value.saveId = "different-save"; });
    const result = await repository.get(save.saveId);
    expectRepositoryError(result, "NO_RECOVERABLE_REVISION");
    if (!result.ok) expect(result.error.currentError?.code).toBe("CORRUPTED_SAVE");
  });

  it("preserves the Stage 1 bc_cpu unsupported result at the storage boundary", async () => {
    const { repository } = makeRepository();
    const unsupported = makeSave() as LocalMatchSaveV1;
    unsupported.payload.cpuSettings["team-4"] = "bc_cpu";
    unsupported.metadata.controllersByTeamId["team-4"] = "bc_cpu";
    expectRepositoryError(await repository.put(unsupported), "UNSUPPORTED_CPU_CONTROLLER");
  });

  it("creates separate versioned stores and the updatedAt metadata index", async () => {
    const { repository, factory, name } = makeRepository();
    await repository.put(makeSave());
    const database = await openDatabase(factory, name);
    expect(database.version).toBe(1);
    expect(Array.from(database.objectStoreNames)).toEqual(expect.arrayContaining([LOCAL_GAME_SAVE_STORE_NAME, LOCAL_GAME_SAVE_METADATA_STORE_NAME]));
    const transaction = database.transaction(LOCAL_GAME_SAVE_METADATA_STORE_NAME, "readonly");
    expect(Array.from(transaction.objectStore(LOCAL_GAME_SAVE_METADATA_STORE_NAME).indexNames)).toContain("updatedAt");
    database.close();
  });

  it("keeps the storage record version independent from the game save schema", async () => {
    const { repository, factory, name } = makeRepository();
    const save = makeSave(); await repository.put(save);
    const record = (await getRawRecord(factory, name, save.saveId))!;
    expect(record.storageRecordVersion).toBe(LOCAL_GAME_SAVE_STORAGE_RECORD_VERSION);
    expect(save.saveSchemaVersion).toBe(1);
    expect(record.current.revision).toBe(1);
  });
});
