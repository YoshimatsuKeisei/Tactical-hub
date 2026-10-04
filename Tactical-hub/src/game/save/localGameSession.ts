import { createVisualCpuPolicyRouter, type VisualCpuPolicyRouter } from "../cpu/cpuPolicyRouter";
import type { HeuristicCpuPolicyState } from "../cpu/heuristicCpuPolicy";
import type { CpuRuntime, CpuTeamSettings } from "../cpu/types";
import type { GameState } from "../types";
import { createLocalMatchRestoreBundle, createLocalMatchSaveSnapshot } from "./localGameSaveSerializer";
import type { LocalMatchResumeUiV1, LocalMatchSaveError, LocalMatchSaveV1 } from "./localGameSaveTypes";
import { isLocalMatchContinuable } from "./localGameSaveValidator";
import type {
  GetLocalGameSaveResult,
  LocalGameSaveRepository,
  LocalGameSaveRepositoryError,
  LocalGameSaveRepositoryResult,
} from "./localGameSaveStorageTypes";

export const LOCAL_GAME_AUTOSAVE_DEBOUNCE_MS = 1_000;

export type LocalAutosaveStatus = "disabled" | "saved" | "pending" | "saving" | "error";

export type LocalAutosaveState = {
  status: LocalAutosaveStatus;
  saveId?: string;
  lastSavedAt?: string;
  lastSavedRevision?: number;
  pendingRevision?: number;
  lastError?: LocalGameSaveRepositoryError;
};

export type RevisionedLocalSessionValue<T> = {
  revision: number;
  value: T;
};

export type LocalGameStableSnapshotParts = {
  gameState: RevisionedLocalSessionValue<GameState>;
  cpuRuntime: RevisionedLocalSessionValue<CpuRuntime>;
  cpuSettings: RevisionedLocalSessionValue<CpuTeamSettings>;
  heuristicPolicyState: RevisionedLocalSessionValue<HeuristicCpuPolicyState>;
  resumeUi?: LocalMatchResumeUiV1;
};

export type LocalGameStableSnapshot = {
  revision: number;
  gameState: GameState;
  cpuRuntime: CpuRuntime;
  cpuSettings: CpuTeamSettings;
  heuristicPolicyState: HeuristicCpuPolicyState;
  resumeUi?: LocalMatchResumeUiV1;
};

export type LocalGameSessionScheduler = {
  setTimeout(callback: () => void, delayMs: number): unknown;
  clearTimeout(handle: unknown): void;
};

export type LocalGameSessionOptions = {
  repository: LocalGameSaveRepository;
  cpuSettings: CpuTeamSettings;
  displayName?: string;
  saveId?: string;
  createdAt?: string;
  initialLastSavedAt?: string;
  uuid?: () => string;
  clock?: () => string;
  scheduler?: LocalGameSessionScheduler;
  debounceMs?: number;
  onStatusChange?: (status: LocalAutosaveState) => void;
};

type PendingOperation =
  | { kind: "put"; snapshot: LocalGameStableSnapshot }
  | { kind: "delete"; revision: number };

const defaultScheduler: LocalGameSessionScheduler = {
  setTimeout: (callback, delayMs) => globalThis.setTimeout(callback, delayMs),
  clearTimeout: (handle) => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
};

function defaultClock() {
  return new Date().toISOString();
}

function defaultUuid() {
  if (!globalThis.crypto?.randomUUID) throw new Error("Web Crypto randomUUID is unavailable");
  return globalThis.crypto.randomUUID();
}

function repositoryErrorFromSave(error: LocalMatchSaveError): LocalGameSaveRepositoryError {
  return { code: error.code, message: error.message, cause: error };
}

export function assembleLocalGameStableSnapshot(parts: LocalGameStableSnapshotParts): LocalGameStableSnapshot | undefined {
  const revisions = [
    parts.gameState.revision,
    parts.cpuRuntime.revision,
    parts.cpuSettings.revision,
    parts.heuristicPolicyState.revision,
  ];
  if (!revisions.every(Number.isSafeInteger) || revisions.some((revision) => revision < 1) || !revisions.every((revision) => revision === revisions[0])) return undefined;
  return {
    revision: revisions[0],
    gameState: parts.gameState.value,
    cpuRuntime: parts.cpuRuntime.value,
    cpuSettings: parts.cpuSettings.value,
    heuristicPolicyState: parts.heuristicPolicyState.value,
    ...(parts.resumeUi ? { resumeUi: parts.resumeUi } : {}),
  };
}

export function shouldFlushLocalGameSnapshot(previous: GameState | undefined, next: GameState) {
  if (!previous) return true;
  if (previous.turnNumber !== next.turnNumber || previous.phase !== next.phase) return true;
  const previousStatuses = previous.teams.map((team) => `${team.id}:${team.status}`).join("|");
  const nextStatuses = next.teams.map((team) => `${team.id}:${team.status}`).join("|");
  if (previousStatuses !== nextStatuses) return true;
  const previousRewards = previous.rewardPlacementRequests.map((request) => `${request.id}:${request.completed}:${request.expired}`).join("|");
  const nextRewards = next.rewardPlacementRequests.map((request) => `${request.id}:${request.completed}:${request.expired}`).join("|");
  if (previousRewards !== nextRewards) return true;
  if (previous.teleportIntents.length > next.teleportIntents.length) return true;
  return previous.strategistActionIntents.length > next.strategistActionIntents.length;
}

export class LocalGameSession {
  readonly saveId?: string;
  readonly createdAt?: string;
  readonly enabled: boolean;
  private readonly repository: LocalGameSaveRepository;
  private readonly displayName: string;
  private readonly clock: () => string;
  private readonly scheduler: LocalGameSessionScheduler;
  private readonly debounceMs: number;
  private readonly onStatusChange?: (status: LocalAutosaveState) => void;
  private autosaveState: LocalAutosaveState;
  private latestRevision = 0;
  private latestSnapshot?: LocalGameStableSnapshot;
  private pending?: PendingOperation;
  private timer?: unknown;
  private processing?: Promise<void>;
  private finished = false;

  constructor(options: LocalGameSessionOptions) {
    this.repository = options.repository;
    this.displayName = options.displayName ?? "LOCAL GAME";
    this.clock = options.clock ?? defaultClock;
    this.scheduler = options.scheduler ?? defaultScheduler;
    this.debounceMs = options.debounceMs ?? LOCAL_GAME_AUTOSAVE_DEBOUNCE_MS;
    this.onStatusChange = options.onStatusChange;
    if (Object.values(options.cpuSettings).includes("bc_cpu")) {
      this.enabled = false;
      this.autosaveState = { status: "disabled" };
      return;
    }
    try {
      this.saveId = options.saveId ?? (options.uuid ?? defaultUuid)();
      this.createdAt = options.createdAt ?? this.clock();
      this.enabled = true;
      this.autosaveState = {
        status: "saved",
        saveId: this.saveId,
        ...(options.initialLastSavedAt ? { lastSavedAt: options.initialLastSavedAt, lastSavedRevision: 0 } : {}),
      };
    } catch (error) {
      this.enabled = false;
      this.autosaveState = {
        status: "error",
        lastError: { code: "STORAGE_UNAVAILABLE", message: error instanceof Error ? error.message : "Unable to create LOCAL save session" },
      };
    }
  }

  getStatus(): LocalAutosaveState {
    return { ...this.autosaveState };
  }

  private updateStatus(update: LocalAutosaveState) {
    this.autosaveState = update;
    this.onStatusChange?.({ ...update });
  }

  commit(parts: LocalGameStableSnapshotParts, options: { immediate?: boolean } = {}) {
    if (!this.enabled || this.finished) return false;
    const snapshot = assembleLocalGameStableSnapshot(parts);
    if (!snapshot || snapshot.revision <= this.latestRevision) return false;
    const immediate = options.immediate ?? shouldFlushLocalGameSnapshot(this.latestSnapshot?.gameState, snapshot.gameState);
    this.latestRevision = snapshot.revision;
    this.latestSnapshot = snapshot;
    if (!isLocalMatchContinuable(snapshot.gameState)) {
      this.pending = { kind: "delete", revision: snapshot.revision };
      this.setPendingStatus(snapshot.revision);
      void this.flush();
      return true;
    }
    this.pending = { kind: "put", snapshot };
    this.setPendingStatus(snapshot.revision);
    if (immediate) void this.flush(); else this.schedule();
    return true;
  }

  private setPendingStatus(revision: number) {
    this.updateStatus({
      ...this.autosaveState,
      status: "pending",
      saveId: this.saveId,
      pendingRevision: revision,
      lastError: undefined,
    });
  }

  private schedule() {
    if (this.timer !== undefined) this.scheduler.clearTimeout(this.timer);
    this.timer = this.scheduler.setTimeout(() => {
      this.timer = undefined;
      void this.flush();
    }, this.debounceMs);
  }

  private pendingOperationRevision() {
    const pending: PendingOperation | undefined = this.pending;
    return pending ? (pending.kind === "put" ? pending.snapshot.revision : pending.revision) : undefined;
  }

  async flush(): Promise<LocalAutosaveState> {
    if (!this.enabled) return this.getStatus();
    if (this.timer !== undefined) {
      this.scheduler.clearTimeout(this.timer);
      this.timer = undefined;
    }
    if (!this.processing) {
      this.processing = this.drain().finally(() => { this.processing = undefined; });
    }
    await this.processing;
    return this.getStatus();
  }

  private async drain() {
    while (this.pending) {
      const operation = this.pending;
      this.pending = undefined;
      const revision = operation.kind === "put" ? operation.snapshot.revision : operation.revision;
      this.updateStatus({ ...this.autosaveState, status: "saving", saveId: this.saveId, pendingRevision: revision, lastError: undefined });
      let result: LocalGameSaveRepositoryResult<unknown>;
      let savedAt: string | undefined;
      if (operation.kind === "delete") {
        result = await this.repository.delete(this.saveId!);
      } else {
        savedAt = this.clock();
        const created = createLocalMatchSaveSnapshot({
          saveId: this.saveId!,
          createdAt: this.createdAt!,
          updatedAt: savedAt,
          displayName: this.displayName,
          gameState: operation.snapshot.gameState,
          cpuSettings: operation.snapshot.cpuSettings,
          cpuRuntime: operation.snapshot.cpuRuntime,
          heuristicPolicyState: operation.snapshot.heuristicPolicyState,
          ...(operation.snapshot.resumeUi ? { resumeUi: operation.snapshot.resumeUi } : {}),
        });
        result = created.ok ? await this.repository.put(created.value) : { ok: false, error: repositoryErrorFromSave(created.error) };
      }
      if (!result.ok) {
        if ((this.pendingOperationRevision() ?? -1) < revision) this.pending = operation;
        const pendingRevision = this.pendingOperationRevision()!;
        this.updateStatus({ ...this.autosaveState, status: "error", saveId: this.saveId, pendingRevision, lastError: result.error });
        return;
      }
      if (operation.kind === "delete") {
        this.finished = true;
        this.pending = undefined;
        this.updateStatus({ status: "saved", saveId: this.saveId, lastSavedAt: this.clock(), lastSavedRevision: revision });
        return;
      }
      this.updateStatus({
        status: this.pending ? "pending" : "saved",
        saveId: this.saveId,
        lastSavedAt: savedAt,
        lastSavedRevision: revision,
        ...(this.pendingOperationRevision() !== undefined ? { pendingRevision: this.pendingOperationRevision() } : {}),
      });
    }
  }

  dispose() {
    if (this.timer !== undefined) this.scheduler.clearTimeout(this.timer);
    this.timer = undefined;
  }
}

export function startLocalGameSession(options: LocalGameSessionOptions, initialSnapshot: LocalGameStableSnapshotParts) {
  const session = new LocalGameSession(options);
  session.commit(initialSnapshot, { immediate: true });
  return session;
}

export type ResumeLocalGameSessionOptions = Omit<LocalGameSessionOptions, "cpuSettings" | "saveId" | "createdAt" | "initialLastSavedAt"> & {
  saveId: string;
  policyRouter?: VisualCpuPolicyRouter;
};

export type ResumedLocalGameSession = {
  session: LocalGameSession;
  state: GameState;
  cpuRuntime: CpuRuntime;
  cpuSettings: CpuTeamSettings;
  visualCpuPolicy: VisualCpuPolicyRouter;
  cpuPaused: true;
  recoverySource: GetLocalGameSaveResult["source"];
  storageRevision: number;
};

export async function resumeLocalGameSession(options: ResumeLocalGameSessionOptions): Promise<LocalGameSaveRepositoryResult<ResumedLocalGameSession>> {
  const loaded = await options.repository.get(options.saveId);
  if (!loaded.ok) return loaded;
  const restored = createLocalMatchRestoreBundle(loaded.value.save);
  if (!restored.ok) return { ok: false, error: repositoryErrorFromSave(restored.error) };
  const policy = options.policyRouter ?? createVisualCpuPolicyRouter();
  policy.restoreHeuristicState(restored.value.heuristicPolicyState);
  const session = new LocalGameSession({
    ...options,
    cpuSettings: restored.value.cpuSettings,
    saveId: loaded.value.save.saveId,
    createdAt: loaded.value.save.createdAt,
    initialLastSavedAt: loaded.value.save.updatedAt,
    displayName: loaded.value.save.metadata.displayName,
  });
  return {
    ok: true,
    value: {
      session,
      state: restored.value.gameState,
      cpuRuntime: restored.value.cpuRuntime,
      cpuSettings: restored.value.cpuSettings,
      visualCpuPolicy: policy,
      cpuPaused: true,
      recoverySource: loaded.value.source,
      storageRevision: loaded.value.revision,
    },
  };
}

type LifecycleEventTarget = {
  addEventListener(type: string, listener: EventListener): void;
  removeEventListener(type: string, listener: EventListener): void;
};

type VisibilityEventTarget = LifecycleEventTarget & { visibilityState: string };

export function installLocalGameSessionLifecycle(
  session: LocalGameSession,
  pageTarget: LifecycleEventTarget = globalThis.window,
  visibilityTarget: VisibilityEventTarget = globalThis.document,
) {
  const flush = () => { void session.flush(); };
  const visibility = () => { if (visibilityTarget.visibilityState === "hidden") flush(); };
  pageTarget.addEventListener("pagehide", flush);
  visibilityTarget.addEventListener("visibilitychange", visibility);
  return () => {
    pageTarget.removeEventListener("pagehide", flush);
    visibilityTarget.removeEventListener("visibilitychange", visibility);
  };
}
