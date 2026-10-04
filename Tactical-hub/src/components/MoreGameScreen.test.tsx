import { act, create, type ReactTestInstance, type ReactTestRenderer } from "react-test-renderer";
import type { ComponentProps } from "react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { createVisualCpuPolicyRouter } from "../game/cpu/cpuPolicyRouter";
import { createCpuRuntime, type CpuTeamSettings } from "../game/cpu/types";
import { createInitialGameState } from "../game/initialState";
import { LocalGameSession, type ResumedLocalGameSession } from "../game/save/localGameSession";
import { createLocalMatchSaveSnapshot } from "../game/save/localGameSaveSerializer";
import type {
  DeleteLocalGameSaveResult,
  GetLocalGameSaveResult,
  LocalGameSaveMetadataRecord,
  LocalGameSaveRepository,
  LocalGameSaveRepositoryResult,
  PutLocalGameSaveResult,
} from "../game/save/localGameSaveStorageTypes";
import { MoreGameScreen } from "./MoreGameScreen";

beforeAll(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
});

function ok<T>(value: T): LocalGameSaveRepositoryResult<T> { return { ok: true, value }; }
function fail<T>(code: "TRANSACTION_FAILED" | "NOT_FOUND" = "TRANSACTION_FAILED"): LocalGameSaveRepositoryResult<T> {
  return { ok: false, error: { code, message: "fixture failure" } };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

const settings: CpuTeamSettings = {
  "team-1": "human",
  "team-2": "random_cpu",
  "team-3": "heuristic_cpu",
  "team-4": "random_cpu",
};

function metadata(saveId: string, updatedAt: string, turnNumber: number): LocalGameSaveMetadataRecord {
  return {
    storageRecordVersion: 1,
    saveId,
    createdAt: "2026-10-04T00:00:00.000Z",
    updatedAt,
    metadata: {
      displayName: `LOCAL GAME ${saveId}`,
      turnNumber,
      phase: "movement_input",
      humanTeamIds: ["team-1"],
      controllersByTeamId: settings,
      activeTeamIds: ["team-1", "team-2", "team-3", "team-4"],
      defeatedTeamIds: [],
      preview: { mapName: "Test Map", livingUnitCount: 18, ownedBaseCountByTeamId: { "team-1": 1 } },
    },
  };
}

class UiRepository implements LocalGameSaveRepository {
  listResult: LocalGameSaveRepositoryResult<LocalGameSaveMetadataRecord[]> = ok([]);
  deleteResult: LocalGameSaveRepositoryResult<DeleteLocalGameSaveResult> | undefined;
  list = vi.fn(async () => this.listResult);
  delete = vi.fn(async (saveId: string) => this.deleteResult ?? ok({ saveId, deleted: true }));
  put = vi.fn(async (save): Promise<LocalGameSaveRepositoryResult<PutLocalGameSaveResult>> => ok({ saveId: save.saveId, revision: 1 }));
  get = vi.fn(async (): Promise<LocalGameSaveRepositoryResult<GetLocalGameSaveResult>> => fail("NOT_FOUND"));
}

function resumed(source: "current" | "previous" = "current"): ResumedLocalGameSession {
  const repository = new UiRepository();
  const state = createInitialGameState();
  state.turnNumber = state.turnState.turnNumber = 9;
  const session = new LocalGameSession({ repository, cpuSettings: settings, saveId: "save-a", createdAt: "2026-10-04T00:00:00.000Z" });
  return {
    session,
    state,
    cpuRuntime: createCpuRuntime(9),
    cpuSettings: settings,
    visualCpuPolicy: createVisualCpuPolicyRouter(),
    cpuPaused: true,
    recoverySource: source,
    storageRevision: 2,
  };
}

async function renderMoreGame(
  repository: LocalGameSaveRepository,
  options: Partial<ComponentProps<typeof MoreGameScreen>> = {},
) {
  let renderer!: ReactTestRenderer;
  await act(async () => {
    renderer = create(<MoreGameScreen
      repository={repository}
      onResume={options.onResume ?? vi.fn()}
      resumeSession={options.resumeSession}
      formatDateTime={options.formatDateTime ?? ((value) => `formatted:${value}`)}
    />);
    await Promise.resolve();
  });
  return renderer;
}

function text(renderer: ReactTestRenderer) {
  return JSON.stringify(renderer.toJSON());
}

function buttons(root: ReactTestInstance, label: string) {
  return root.findAllByType("button").filter((button) => button.children.join("") === label);
}

describe("MoreGameScreen", () => {
  it("shows loading while repository metadata list is pending and calls list once", async () => {
    const pending = deferred<LocalGameSaveRepositoryResult<LocalGameSaveMetadataRecord[]>>();
    const repository = new UiRepository();
    repository.list.mockImplementationOnce(() => pending.promise);
    let renderer!: ReactTestRenderer;
    await act(async () => { renderer = create(<MoreGameScreen repository={repository} onResume={vi.fn()} />); });
    expect(text(renderer)).toContain("中断データを読み込んでいます");
    expect(repository.list).toHaveBeenCalledTimes(1);
    await act(async () => { pending.resolve(ok([])); await pending.promise; });
    await act(async () => { renderer.unmount(); });
  });

  it("distinguishes an empty list from loading", async () => {
    const renderer = await renderMoreGame(new UiRepository());
    expect(text(renderer)).toContain("中断中のLOCAL GAMEはありません");
    expect(text(renderer)).not.toContain("読み込んでいます");
  });

  it("shows a safe list error with the repository category available to tests", async () => {
    const repository = new UiRepository();
    repository.listResult = fail("TRANSACTION_FAILED");
    const renderer = await renderMoreGame(repository);
    expect(text(renderer)).toContain("保存データを読み込めませんでした");
    expect(renderer.root.findByProps({ "data-error-code": "TRANSACTION_FAILED" })).toBeDefined();
  });

  it("preserves repository updatedAt-desc order and renders card metadata", async () => {
    const repository = new UiRepository();
    repository.listResult = ok([
      metadata("new", "2026-10-04T02:00:00.000Z", 12),
      metadata("old", "2026-10-04T01:00:00.000Z", 4),
    ]);
    const renderer = await renderMoreGame(repository);
    const cards = renderer.root.findAllByProps({ className: "more-game-card" });
    expect(cards.map((card) => card.props["data-save-id"])).toEqual(["new", "old"]);
    expect(text(renderer)).toContain("Turn");
    expect(text(renderer)).toContain("movement_input");
    expect(text(renderer)).toContain("Random CPU");
    expect(text(renderer)).toContain("formatted:2026-10-04T02:00:00.000Z");
    expect(repository.get).not.toHaveBeenCalled();
  });

  it("uses the Stage 3 resume API by default before opening PlayScreen", async () => {
    const repository = new UiRepository();
    const state = createInitialGameState();
    state.turnNumber = state.turnState.turnNumber = 6;
    const policy = createVisualCpuPolicyRouter();
    const created = createLocalMatchSaveSnapshot({
      saveId: "save-real",
      createdAt: "2026-10-04T00:00:00.000Z",
      updatedAt: "2026-10-04T01:00:00.000Z",
      displayName: "LOCAL GAME save-real",
      gameState: state,
      cpuSettings: settings,
      cpuRuntime: createCpuRuntime(44),
      heuristicPolicyState: policy.snapshotHeuristicState(),
    });
    if (!created.ok) throw new Error(created.error.message);
    repository.listResult = ok([{
      storageRecordVersion: 1,
      saveId: created.value.saveId,
      createdAt: created.value.createdAt,
      updatedAt: created.value.updatedAt,
      metadata: created.value.metadata,
    }]);
    repository.get.mockResolvedValueOnce(ok({ save: created.value, source: "current", revision: 1 }));
    const onResume = vi.fn();
    const renderer = await renderMoreGame(repository, { onResume });
    await act(async () => { buttons(renderer.root, "CONTINUE")[0].props.onClick(); await Promise.resolve(); });
    expect(repository.get).toHaveBeenCalledWith("save-real");
    expect(onResume.mock.calls[0][0].state.turnNumber).toBe(6);
    expect(onResume.mock.calls[0][0].cpuPaused).toBe(true);
  });

  it("continues a save once and passes the restored saved state onward", async () => {
    const repository = new UiRepository();
    repository.listResult = ok([metadata("save-a", "2026-10-04T02:00:00.000Z", 9)]);
    const restored = resumed();
    const onResume = vi.fn();
    const resumeSession = vi.fn(async () => ok(restored));
    const renderer = await renderMoreGame(repository, { onResume, resumeSession });
    await act(async () => { buttons(renderer.root, "CONTINUE")[0].props.onClick(); await Promise.resolve(); });
    expect(resumeSession).toHaveBeenCalledWith({ repository, saveId: "save-a" });
    expect(onResume).toHaveBeenCalledWith(restored);
    expect(onResume.mock.calls[0][0].state.turnNumber).toBe(9);
  });

  it("blocks duplicate CONTINUE calls while the same save is resuming", async () => {
    const repository = new UiRepository();
    repository.listResult = ok([metadata("save-a", "2026-10-04T02:00:00.000Z", 9)]);
    const pending = deferred<LocalGameSaveRepositoryResult<ResumedLocalGameSession>>();
    const resumeSession = vi.fn(() => pending.promise);
    const renderer = await renderMoreGame(repository, { resumeSession });
    const button = buttons(renderer.root, "CONTINUE")[0];
    await act(async () => { button.props.onClick(); button.props.onClick(); });
    expect(resumeSession).toHaveBeenCalledTimes(1);
    expect(buttons(renderer.root, "読み込み中…")[0].props.disabled).toBe(true);
    await act(async () => { pending.resolve(ok(resumed())); await pending.promise; });
  });

  it("keeps the list and other saves operable after a resume failure", async () => {
    const repository = new UiRepository();
    repository.listResult = ok([
      metadata("bad", "2026-10-04T02:00:00.000Z", 9),
      metadata("good", "2026-10-04T01:00:00.000Z", 8),
    ]);
    const resumeSession = vi.fn(async () => fail<ResumedLocalGameSession>("TRANSACTION_FAILED"));
    const renderer = await renderMoreGame(repository, { resumeSession });
    await act(async () => { buttons(renderer.root, "CONTINUE")[0].props.onClick(); await Promise.resolve(); });
    expect(text(renderer)).toContain("この中断データを再開できませんでした");
    expect(renderer.root.findAllByProps({ className: "more-game-card" })).toHaveLength(2);
    expect(buttons(renderer.root, "CONTINUE")[1].props.disabled).toBe(false);
  });

  it("requires delete confirmation and CANCEL never calls the repository", async () => {
    const repository = new UiRepository();
    repository.listResult = ok([metadata("save-a", "2026-10-04T02:00:00.000Z", 9)]);
    const renderer = await renderMoreGame(repository);
    await act(async () => { buttons(renderer.root, "DELETE")[0].props.onClick(); });
    expect(text(renderer)).toContain("この中断データを削除しますか");
    await act(async () => { buttons(renderer.root, "CANCEL")[0].props.onClick(); });
    expect(repository.delete).not.toHaveBeenCalled();
    expect(text(renderer)).not.toContain("この中断データを削除しますか");
  });

  it("deletes after confirmation and removes the card", async () => {
    const repository = new UiRepository();
    repository.listResult = ok([metadata("save-a", "2026-10-04T02:00:00.000Z", 9)]);
    const renderer = await renderMoreGame(repository);
    await act(async () => { buttons(renderer.root, "DELETE")[0].props.onClick(); });
    await act(async () => { buttons(renderer.root, "DELETE")[1].props.onClick(); await Promise.resolve(); });
    expect(repository.delete).toHaveBeenCalledWith("save-a");
    expect(text(renderer)).toContain("中断中のLOCAL GAMEはありません");
  });

  it("retains the card and reports a delete failure", async () => {
    const repository = new UiRepository();
    repository.listResult = ok([metadata("save-a", "2026-10-04T02:00:00.000Z", 9)]);
    repository.deleteResult = fail("TRANSACTION_FAILED");
    const renderer = await renderMoreGame(repository);
    await act(async () => { buttons(renderer.root, "DELETE")[0].props.onClick(); });
    await act(async () => { buttons(renderer.root, "DELETE")[1].props.onClick(); await Promise.resolve(); });
    expect(text(renderer)).toContain("中断データを削除できませんでした");
    expect(renderer.root.findAllByProps({ className: "more-game-card" })).toHaveLength(1);
  });
});
