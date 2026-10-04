import type { LocalMatchMetadataV1, LocalMatchSaveError, LocalMatchSaveErrorCode, LocalMatchSaveV1 } from "./localGameSaveTypes";

export const LOCAL_GAME_SAVE_DATABASE_NAME = "tactical-hub-local-game-saves";
export const LOCAL_GAME_SAVE_DATABASE_VERSION = 1;
export const LOCAL_GAME_SAVE_STORAGE_RECORD_VERSION = 1 as const;
export const LOCAL_GAME_SAVE_STORE_NAME = "local-match-saves";
export const LOCAL_GAME_SAVE_METADATA_STORE_NAME = "local-match-save-metadata";
export const LOCAL_GAME_SAVE_UPDATED_AT_INDEX = "updatedAt";

export type LocalGameSaveStorageRevision = {
  revision: number;
  serializedSave: string;
  sha256: string;
};

export type LocalGameSaveStorageRecord = {
  storageRecordVersion: typeof LOCAL_GAME_SAVE_STORAGE_RECORD_VERSION;
  saveId: string;
  current: LocalGameSaveStorageRevision;
  previous?: LocalGameSaveStorageRevision;
};

export type LocalGameSaveMetadataRecord = {
  storageRecordVersion: typeof LOCAL_GAME_SAVE_STORAGE_RECORD_VERSION;
  saveId: string;
  createdAt: string;
  updatedAt: string;
  metadata: LocalMatchMetadataV1;
};

export type LocalGameSaveRepositoryErrorCode =
  | LocalMatchSaveErrorCode
  | "STORAGE_UNAVAILABLE"
  | "OPEN_FAILED"
  | "UPGRADE_FAILED"
  | "TRANSACTION_FAILED"
  | "QUOTA_EXCEEDED"
  | "NOT_FOUND"
  | "CORRUPTED_SAVE"
  | "NO_RECOVERABLE_REVISION"
  | "UNSUPPORTED_STORAGE_RECORD_VERSION";

export type LocalGameSaveRepositoryError = {
  code: LocalGameSaveRepositoryErrorCode;
  message: string;
  cause?: LocalMatchSaveError;
  currentError?: LocalGameSaveRepositoryError;
  previousError?: LocalGameSaveRepositoryError;
};

export type LocalGameSaveRepositoryResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: LocalGameSaveRepositoryError };

export type PutLocalGameSaveResult = {
  saveId: string;
  revision: number;
};

export type GetLocalGameSaveResult = {
  save: LocalMatchSaveV1;
  source: "current" | "previous";
  revision: number;
};

export type DeleteLocalGameSaveResult = {
  saveId: string;
  deleted: boolean;
};

export interface LocalGameSaveRepository {
  put(save: LocalMatchSaveV1): Promise<LocalGameSaveRepositoryResult<PutLocalGameSaveResult>>;
  get(saveId: string): Promise<LocalGameSaveRepositoryResult<GetLocalGameSaveResult>>;
  list(): Promise<LocalGameSaveRepositoryResult<LocalGameSaveMetadataRecord[]>>;
  delete(saveId: string): Promise<LocalGameSaveRepositoryResult<DeleteLocalGameSaveResult>>;
}
