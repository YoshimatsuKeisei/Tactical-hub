import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PlayScreen } from "../App";
import { createVisualCpuPolicyRouter } from "../game/cpu/cpuPolicyRouter";
import { createCpuRuntime, type CpuTeamSettings } from "../game/cpu/types";
import { createInitialGameState } from "../game/initialState";
import { LocalGameSession, type ResumedLocalGameSession } from "../game/save/localGameSession";
import type { LocalGameSaveRepository } from "../game/save/localGameSaveStorageTypes";

const repository: LocalGameSaveRepository = {
  put: async (save) => ({ ok: true, value: { saveId: save.saveId, revision: 1 } }),
  get: async () => ({ ok: false, error: { code: "NOT_FOUND", message: "fixture" } }),
  list: async () => ({ ok: true, value: [] }),
  delete: async (saveId) => ({ ok: true, value: { saveId, deleted: true } }),
};

describe("LOCAL PlayScreen session initialization", () => {
  it("renders a resumed GameState as the initial state instead of booting a new game first", () => {
    const state = createInitialGameState(); state.turnNumber = state.turnState.turnNumber = 7;
    const cpuRuntime = createCpuRuntime(777); cpuRuntime.appliedStepCount = 42;
    const cpuSettings: CpuTeamSettings = { "team-1": "human", "team-2": "random_cpu", "team-3": "heuristic_cpu", "team-4": "random_cpu" };
    const visualCpuPolicy = createVisualCpuPolicyRouter();
    const session = new LocalGameSession({ repository, cpuSettings, saveId: "resumed", createdAt: "2026-10-04T00:00:00.000Z" });
    const resumedSession: ResumedLocalGameSession = {
      session,
      state,
      cpuRuntime,
      cpuSettings,
      visualCpuPolicy,
      cpuPaused: true,
      recoverySource: "current",
      storageRevision: 3,
    };
    const html = renderToStaticMarkup(<PlayScreen initialCpuSettings={cpuSettings} repository={repository} resumedSession={resumedSession} />);
    expect(html).toContain("ターン <strong>7</strong>");
    expect(html).toContain("Seed <input type=\"number\" value=\"777\"");
    expect(html).not.toContain("ターン <strong>1</strong>");
  });
});
