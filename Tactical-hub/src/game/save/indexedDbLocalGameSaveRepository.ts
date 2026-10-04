import { calculateLocalGameSaveSha256, LocalGameSaveChecksumUnavailableError } from "./localGameSaveChecksum";
import { parseLocalMatchSave, serializeLocalMatchSave } from "./localGameSaveSerializer";
import type { LocalMatchSaveError, LocalMatchSaveErrorCode, LocalMatchSaveV1 } from "./localGameSaveTypes";
import { validateLocalMatchSave } from "./localGameSaveValidator";
import {
  LOCAL_GAME_SAVE_DATABASE_NAME,
  LOCAL_GAME_SAVE_DATABASE_VERSION,
  LOCAL_GAME_SAVE_METADATA_STORE_NAME,
  LOCAL_GAME_SAVE_STORAGE_RECORD_VERSION,
  LOCAL_GAME_SAVE_STORE_NAME,
  LOCAL_GAME_SAVE_UPDATED_AT_INDEX,
  type DeleteLocalGameSaveResult,
  type GetLocalGameSaveResult,
  type LocalGameSaveMetadataRecord,
  type LocalGameSaveRepository,
  type LocalGameSaveRepositoryError,
  type LocalGameSaveRepositoryErrorCode,
  type LocalGameSaveRepositoryResult,
  type LocalGameSaveStorageRecord,
  type LocalGameSaveStorageRevision,
  type PutLocalGameSaveResult,
} from "./localGameSaveStorageTypes";

export type IndexedDbLocalGameSaveRepositoryOptions = {
  databaseName?: string;
  indexedDB?: IDBFactory | null;
  crypto?: Crypto | null;
};

const fallbackEligibleCodes = new Set<LocalMatchSaveErrorCode>([
  "MALFORMED_JSON",
  "STRUCTURAL_ERROR",
  "INVALID_GAME_STATE",
  "INVALID_CPU_RUNTIME",
  "INVALID_HEURISTIC_STATE",
]);

function success<T>(value: T): LocalGameSaveRepositoryResult<T> {
  return { ok: true, value };
}

function failure(code: LocalGameSaveRepositoryErrorCode, message: string, extra: Partial<LocalGameSaveRepositoryError> = {}): LocalGameSaveRepositoryResult<never> {
  return { ok: false, error: { code, message, ...extra } };
}

function saveFailure(error: LocalMatchSaveError): LocalGameSaveRepositoryResult<never> {
  return failure(error.code, error.message, { cause: error });
}

function errorName(error: unknown) {
  return error && typeof error === "object" && "name" in error ? String(error.name) : "";
}

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function storageFailure(error: unknown, fallbackCode: "OPEN_FAILED" | "UPGRADE_FAILED" | "TRANSACTION_FAILED"): LocalGameSaveRepositoryResult<never> {
  if (error instanceof LocalGameSaveChecksumUnavailableError || errorName(error) === "SecurityError") {
    return failure("STORAGE_UNAVAILABLE", errorMessage(error, "Browser storage is unavailable"));
  }
  if (errorName(error) === "QuotaExceededError") return failure("QUOTA_EXCEEDED", errorMessage(error, "IndexedDB quota was exceeded"));
  return failure(fallbackCode, errorMessage(error, `IndexedDB ${fallbackCode.toLowerCase().replaceAll("_", " ")}`));
}

function requestValue<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB transaction aborted"));
    transaction.onerror = () => undefined;
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isRevision(value: unknown): value is LocalGameSaveStorageRevision {
  return isRecord(value)
    && Number.isInteger(value.revision) && Number(value.revision) >= 1
    && typeof value.serializedSave === "string"
    && typeof value.sha256 === "string" && /^[0-9a-f]{64}$/.test(value.sha256);
}

function inspectStorageRecord(value: unknown, requestedSaveId: string): LocalGameSaveRepositoryResult<LocalGameSaveStorageRecord> {
  if (!isRecord(value)) return failure("CORRUPTED_SAVE", "IndexedDB save record is not an object");
  if (value.storageRecordVersion !== LOCAL_GAME_SAVE_STORAGE_RECORD_VERSION) {
    return failure("UNSUPPORTED_STORAGE_RECORD_VERSION", `Unsupported storage record version: ${String(value.storageRecordVersion)}`);
  }
  if (value.saveId !== requestedSaveId) return failure("CORRUPTED_SAVE", "IndexedDB key and storage record saveId disagree");
  if (!isRevision(value.current)) return failure("CORRUPTED_SAVE", "Current save revision is malformed");
  if (value.previous !== undefined && !isRevision(value.previous)) return failure("CORRUPTED_SAVE", "Previous save revision is malformed");
  return success(value as LocalGameSaveStorageRecord);
}

function isMetadataRecord(value: unknown): value is LocalGameSaveMetadataRecord {
  if (!isRecord(value) || value.storageRecordVersion !== LOCAL_GAME_SAVE_STORAGE_RECORD_VERSION) return false;
  if (typeof value.saveId !== "string" || typeof value.createdAt !== "string" || typeof value.updatedAt !== "string") return false;
  return isRecord(value.metadata);
}

export class IndexedDbLocalGameSaveRepository implements LocalGameSaveRepository {
  readonly databaseName: string;
  private readonly indexedDbFactory: IDBFactory | null | undefined;
  private readonly cryptoProvider: Crypto | null | undefined;

  constructor(options: IndexedDbLocalGameSaveRepositoryOptions = {}) {
    this.databaseName = options.databaseName ?? LOCAL_GAME_SAVE_DATABASE_NAME;
    this.indexedDbFactory = options.indexedDB === undefined ? globalThis.indexedDB : options.indexedDB;
    this.cryptoProvider = options.crypto === undefined ? globalThis.crypto : options.crypto;
  }

  private async openDatabase(): Promise<LocalGameSaveRepositoryResult<IDBDatabase>> {
    if (!this.indexedDbFactory) return failure("STORAGE_UNAVAILABLE", "IndexedDB is unavailable in this browser");
    return new Promise((resolve) => {
      let request: IDBOpenDBRequest;
      let upgradeError: unknown;
      try {
        request = this.indexedDbFactory!.open(this.databaseName, LOCAL_GAME_SAVE_DATABASE_VERSION);
      } catch (error) {
        resolve(storageFailure(error, "OPEN_FAILED"));
        return;
      }
      request.onupgradeneeded = () => {
        try {
          const database = request.result;
          if (!database.objectStoreNames.contains(LOCAL_GAME_SAVE_STORE_NAME)) {
            database.createObjectStore(LOCAL_GAME_SAVE_STORE_NAME, { keyPath: "saveId" });
          }
          if (!database.objectStoreNames.contains(LOCAL_GAME_SAVE_METADATA_STORE_NAME)) {
            const metadataStore = database.createObjectStore(LOCAL_GAME_SAVE_METADATA_STORE_NAME, { keyPath: "saveId" });
            metadataStore.createIndex(LOCAL_GAME_SAVE_UPDATED_AT_INDEX, "updatedAt", { unique: false });
          }
        } catch (error) {
          upgradeError = error;
          request.transaction?.abort();
        }
      };
      request.onsuccess = () => {
        const database = request.result;
        database.onversionchange = () => database.close();
        resolve(success(database));
      };
      request.onerror = () => resolve(storageFailure(upgradeError ?? request.error, upgradeError ? "UPGRADE_FAILED" : "OPEN_FAILED"));
      request.onblocked = () => resolve(failure("OPEN_FAILED", "IndexedDB open was blocked by another connection"));
    });
  }

  async put(save: LocalMatchSaveV1): Promise<LocalGameSaveRepositoryResult<PutLocalGameSaveResult>> {
    const validated = validateLocalMatchSave(save);
    if (!validated.ok) return saveFailure(validated.error);
    const serialized = serializeLocalMatchSave(validated.value);
    if (!serialized.ok) return saveFailure(serialized.error);
    let sha256: string;
    try {
      sha256 = await calculateLocalGameSaveSha256(serialized.value, this.cryptoProvider);
    } catch (error) {
      return storageFailure(error, "TRANSACTION_FAILED");
    }
    const opened = await this.openDatabase();
    if (!opened.ok) return opened;
    const database = opened.value;
    try {
      const transaction = database.transaction([LOCAL_GAME_SAVE_STORE_NAME, LOCAL_GAME_SAVE_METADATA_STORE_NAME], "readwrite");
      const completion = transactionDone(transaction);
      const saveStore = transaction.objectStore(LOCAL_GAME_SAVE_STORE_NAME);
      const metadataStore = transaction.objectStore(LOCAL_GAME_SAVE_METADATA_STORE_NAME);
      let revision = 1;
      try {
        const existingValue = await requestValue(saveStore.get(validated.value.saveId));
        let previous: LocalGameSaveStorageRevision | undefined;
        if (existingValue !== undefined) {
          const inspected = inspectStorageRecord(existingValue, validated.value.saveId);
          if (!inspected.ok) {
            transaction.abort();
            await completion.catch(() => undefined);
            return inspected;
          }
          previous = inspected.value.current;
          revision = previous.revision + 1;
        }
        const record: LocalGameSaveStorageRecord = {
          storageRecordVersion: LOCAL_GAME_SAVE_STORAGE_RECORD_VERSION,
          saveId: validated.value.saveId,
          current: { revision, serializedSave: serialized.value, sha256 },
          ...(previous ? { previous } : {}),
        };
        const metadataRecord: LocalGameSaveMetadataRecord = {
          storageRecordVersion: LOCAL_GAME_SAVE_STORAGE_RECORD_VERSION,
          saveId: validated.value.saveId,
          createdAt: validated.value.createdAt,
          updatedAt: validated.value.updatedAt,
          metadata: validated.value.metadata,
        };
        saveStore.put(record);
        metadataStore.put(metadataRecord);
        await completion;
        return success({ saveId: validated.value.saveId, revision });
      } catch (error) {
        try { transaction.abort(); } catch { /* already inactive */ }
        await completion.catch(() => undefined);
        return storageFailure(error, "TRANSACTION_FAILED");
      }
    } catch (error) {
      return storageFailure(error, "TRANSACTION_FAILED");
    } finally {
      database.close();
    }
  }

  private async validateRevision(recordSaveId: string, revision: LocalGameSaveStorageRevision): Promise<LocalGameSaveRepositoryResult<LocalMatchSaveV1>> {
    let actualChecksum: string;
    try {
      actualChecksum = await calculateLocalGameSaveSha256(revision.serializedSave, this.cryptoProvider);
    } catch (error) {
      return storageFailure(error, "TRANSACTION_FAILED");
    }
    if (actualChecksum !== revision.sha256) return failure("CORRUPTED_SAVE", "Saved JSON checksum does not match");
    const parsed = parseLocalMatchSave(revision.serializedSave);
    if (!parsed.ok) return saveFailure(parsed.error);
    if (parsed.value.saveId !== recordSaveId) return failure("CORRUPTED_SAVE", "Storage record saveId and serialized saveId disagree");
    return success(parsed.value);
  }

  async get(saveId: string): Promise<LocalGameSaveRepositoryResult<GetLocalGameSaveResult>> {
    const opened = await this.openDatabase();
    if (!opened.ok) return opened;
    const database = opened.value;
    let rawRecord: unknown;
    try {
      const transaction = database.transaction(LOCAL_GAME_SAVE_STORE_NAME, "readonly");
      const completion = transactionDone(transaction);
      rawRecord = await requestValue(transaction.objectStore(LOCAL_GAME_SAVE_STORE_NAME).get(saveId));
      await completion;
    } catch (error) {
      database.close();
      return storageFailure(error, "TRANSACTION_FAILED");
    }
    database.close();
    if (rawRecord === undefined) return failure("NOT_FOUND", `No LOCAL match save exists for ${saveId}`);
    const inspected = inspectStorageRecord(rawRecord, saveId);
    if (!inspected.ok) return inspected;
    const current = await this.validateRevision(saveId, inspected.value.current);
    if (current.ok) return success({ save: current.value, source: "current", revision: inspected.value.current.revision });
    if (!this.canFallback(current.error)) return current;
    if (!inspected.value.previous) {
      return failure("NO_RECOVERABLE_REVISION", "Current save is corrupted and no previous revision exists", { currentError: current.error });
    }
    const previous = await this.validateRevision(saveId, inspected.value.previous);
    if (previous.ok) return success({ save: previous.value, source: "previous", revision: inspected.value.previous.revision });
    return failure("NO_RECOVERABLE_REVISION", "Neither current nor previous save revision can be loaded", {
      currentError: current.error,
      previousError: previous.error,
    });
  }

  private canFallback(error: LocalGameSaveRepositoryError) {
    return error.code === "CORRUPTED_SAVE" || fallbackEligibleCodes.has(error.code as LocalMatchSaveErrorCode);
  }

  async list(): Promise<LocalGameSaveRepositoryResult<LocalGameSaveMetadataRecord[]>> {
    const opened = await this.openDatabase();
    if (!opened.ok) return opened;
    const database = opened.value;
    try {
      const transaction = database.transaction(LOCAL_GAME_SAVE_METADATA_STORE_NAME, "readonly");
      const completion = transactionDone(transaction);
      const index = transaction.objectStore(LOCAL_GAME_SAVE_METADATA_STORE_NAME).index(LOCAL_GAME_SAVE_UPDATED_AT_INDEX);
      const records: LocalGameSaveMetadataRecord[] = [];
      await new Promise<void>((resolve, reject) => {
        const request = index.openCursor(null, "prev");
        request.onerror = () => reject(request.error ?? new Error("IndexedDB metadata cursor failed"));
        request.onsuccess = () => {
          const cursor = request.result;
          if (!cursor) { resolve(); return; }
          if (!isMetadataRecord(cursor.value)) { reject(new Error("IndexedDB metadata record is malformed")); return; }
          records.push(cursor.value);
          cursor.continue();
        };
      });
      await completion;
      return success(records);
    } catch (error) {
      return storageFailure(error, "TRANSACTION_FAILED");
    } finally {
      database.close();
    }
  }

  async delete(saveId: string): Promise<LocalGameSaveRepositoryResult<DeleteLocalGameSaveResult>> {
    const opened = await this.openDatabase();
    if (!opened.ok) return opened;
    const database = opened.value;
    try {
      const transaction = database.transaction([LOCAL_GAME_SAVE_STORE_NAME, LOCAL_GAME_SAVE_METADATA_STORE_NAME], "readwrite");
      const completion = transactionDone(transaction);
      try {
        const saveStore = transaction.objectStore(LOCAL_GAME_SAVE_STORE_NAME);
        const existing = await requestValue(saveStore.getKey(saveId));
        saveStore.delete(saveId);
        transaction.objectStore(LOCAL_GAME_SAVE_METADATA_STORE_NAME).delete(saveId);
        await completion;
        return success({ saveId, deleted: existing !== undefined });
      } catch (error) {
        try { transaction.abort(); } catch { /* already inactive */ }
        await completion.catch(() => undefined);
        return storageFailure(error, "TRANSACTION_FAILED");
      }
    } catch (error) {
      return storageFailure(error, "TRANSACTION_FAILED");
    } finally {
      database.close();
    }
  }
}
