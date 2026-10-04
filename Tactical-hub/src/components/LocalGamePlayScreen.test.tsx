import { renderToStaticMarkup } from "react-dom/server";
import { act, create } from "react-test-renderer";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { PlayScreen } from "../App";
import { createVisualCpuPolicyRouter } from "../game/cpu/cpuPolicyRouter";
import { createCpuRuntime, type CpuTeamSettings } from "../game/cpu/types";
import { createInitialGameState } from "../game/initialState";
import { LocalGameSession, type ResumedLocalGameSession } from "../game/save/localGameSession";
import type { LocalGameSaveRepository } from "../game/save/localGameSaveStorageTypes";

beforeAll(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
});

const repository: LocalGameSaveRepository = {
  put: async (save) => ({ ok: true, value: { saveId: save.saveId, revision: 1 } }),
  get: async () => ({ ok: false, error: { code: "NOT_FOUND", message: "fixture" } }),
  list: async () => ({ ok: true, value: [] }),
  delete: async (saveId) => ({ ok: true, value: { saveId, deleted: true } }),
};

const cpuSettings: CpuTeamSettings = {
  "team-1": "human",
  "team-2": "random_cpu",
  "team-3": "heuristic_cpu",
  "team-4": "random_cpu",
};

function resumedFixture(
  source: "current" | "previous" = "current",
  session = new LocalGameSession({ repository, cpuSettings, saveId: "resumed", createdAt: "2026-10-04T00:00:00.000Z" }),
): ResumedLocalGameSession {
  const state = createInitialGameState();
  state.turnNumber = state.turnState.turnNumber = 7;
  const cpuRuntime = createCpuRuntime(777);
  cpuRuntime.appliedStepCount = 42;
  return {
    session,
    state,
    cpuRuntime,
    cpuSettings,
    visualCpuPolicy: createVisualCpuPolicyRouter(),
    cpuPaused: true,
    recoverySource: source,
    storageRevision: 3,
  };
}

function snapshot(session: ResumedLocalGameSession, revision = 1) {
  return {
    gameState: { revision, value: session.state },
    cpuRuntime: { revision, value: session.cpuRuntime },
    cpuSettings: { revision, value: session.cpuSettings },
    heuristicPolicyState: { revision, value: session.visualCpuPolicy.snapshotHeuristicState() },
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("LOCAL PlayScreen session initialization", () => {
  it("renders a resumed GameState as the initial state instead of booting a new game first", () => {
    const resumedSession = resumedFixture();
    const html = renderToStaticMarkup(<PlayScreen initialCpuSettings={cpuSettings} repository={repository} resumedSession={resumedSession} />);
    expect(html).toContain("ターン <strong>7</strong>");
    expect(html).toContain("Seed <input type=\"number\" value=\"777\"");
    expect(html).not.toContain("ターン <strong>1</strong>");
  });

  it("shows saved autosave status and no recovery warning for a current revision", () => {
    const html = renderToStaticMarkup(<PlayScreen initialCpuSettings={cpuSettings} repository={repository} resumedSession={resumedFixture("current")} />);
    expect(html).toContain("保存済み");
    expect(html).not.toContain("直前の正常な保存から復帰しました");
  });

  it("shows the previous-revision recovery warning without treating it as an error", () => {
    const html = renderToStaticMarkup(<PlayScreen initialCpuSettings={cpuSettings} repository={repository} resumedSession={resumedFixture("previous")} />);
    expect(html).toContain("直前の正常な保存から復帰しました");
    expect(html).not.toContain("自動保存に失敗しました");
  });

  it("shows pending and saving as 保存中 without changing the saved GameState", () => {
    const resumedSession = resumedFixture();
    resumedSession.session.commit(snapshot(resumedSession), { immediate: false });
    const html = renderToStaticMarkup(<PlayScreen initialCpuSettings={cpuSettings} repository={repository} resumedSession={resumedSession} />);
    expect(html).toContain("保存中…");
    expect(html).toContain("ターン <strong>7</strong>");
    resumedSession.session.dispose();
  });

  it("shows an autosave failure while keeping the game state playable", async () => {
    const failingRepository: LocalGameSaveRepository = {
      ...repository,
      put: async () => ({ ok: false, error: { code: "QUOTA_EXCEEDED", message: "full" } }),
    };
    const session = new LocalGameSession({ repository: failingRepository, cpuSettings, saveId: "failed", createdAt: "2026-10-04T00:00:00.000Z" });
    const resumedSession = resumedFixture("current", session);
    resumedSession.session.commit(snapshot(resumedSession), { immediate: true });
    await resumedSession.session.flush();
    const html = renderToStaticMarkup(<PlayScreen initialCpuSettings={cpuSettings} repository={failingRepository} resumedSession={resumedSession} />);
    expect(html).toContain("自動保存に失敗しました");
    expect(html).toContain("ターン <strong>7</strong>");
  });

  it("shows autosave disabled for a BC CPU game without blocking play", () => {
    const bcSettings = { ...cpuSettings, "team-4": "bc_cpu" as const };
    const html = renderToStaticMarkup(<PlayScreen initialCpuSettings={bcSettings} repository={repository} />);
    expect(html).toContain("このCPU構成では自動保存・再開は利用できません");
    expect(html).toContain("Tactical Hub Phase 1");
  });

  it("does not start CPU work before saved-state mount, then restores normal runner controls after mount", async () => {
    const setIntervalMock = vi.fn(() => 1);
    const clearIntervalMock = vi.fn();
    const eventTarget = { addEventListener: vi.fn(), removeEventListener: vi.fn() };
    vi.stubGlobal("window", { ...eventTarget, setInterval: setIntervalMock, clearInterval: clearIntervalMock });
    vi.stubGlobal("document", { ...eventTarget, visibilityState: "visible" });
    const resumedSession = resumedFixture();

    const serverHtml = renderToStaticMarkup(<PlayScreen initialCpuSettings={cpuSettings} repository={repository} resumedSession={resumedSession} />);
    expect(serverHtml).toContain("ターン <strong>7</strong>");
    expect(setIntervalMock).not.toHaveBeenCalled();

    let renderer!: ReturnType<typeof create>;
    await act(async () => { renderer = create(<PlayScreen initialCpuSettings={cpuSettings} repository={repository} resumedSession={resumedSession} />); });
    const settingsSection = renderer.root.findAllByType("section").find((section) => section.findAllByType("h2").some((heading) => heading.children.join("") === "プレイヤー構成"));
    expect(settingsSection).toBeDefined();
    const start = settingsSection!.findAllByType("button").find((button) => button.children.join("") === "自動進行を開始");
    const pause = settingsSection!.findAllByType("button").find((button) => button.children.join("") === "一時停止");
    expect(start?.props.disabled).toBe(true);
    expect(pause?.props.disabled).toBe(false);
    expect(setIntervalMock).toHaveBeenCalledTimes(1);
    await act(async () => { renderer.unmount(); });
  });
});
