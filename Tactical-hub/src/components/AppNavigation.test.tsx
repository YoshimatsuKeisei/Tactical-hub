import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import App from "../App";
import { createInitialGameState } from "../game/initialState";
import type { CpuTeamSettings } from "../game/cpu/types";
import { AppNavigation, getBackScreen, type MenuScreen } from "./AppNavigation";

const teams = createInitialGameState().teams;
const localCpuSettings: CpuTeamSettings = {
  "team-1": "human",
  "team-2": "random_cpu",
  "team-3": "heuristic_cpu",
  "team-4": "bc_cpu",
};

function renderScreen(screen: MenuScreen) {
  return renderToStaticMarkup(<AppNavigation
    screen={screen}
    teams={teams}
    localCpuSettings={localCpuSettings}
    onNavigate={vi.fn()}
    onLocalCpuChange={vi.fn()}
    onStartLocal={vi.fn()}
  />);
}

describe("AppNavigation", () => {
  it("opens on HOME", () => {
    const app = renderToStaticMarkup(<App />);
    expect(app).toContain(">HOME<");
    expect(app).toContain("NEW GAME");
  });

  it("renders every requested menu screen", () => {
    expect(renderScreen("home")).toContain("NEW GAME");
    expect(renderScreen("new-game")).toContain("ONLINE");
    expect(renderScreen("online")).toContain("部屋を作成する");
    expect(renderScreen("friend-match")).toContain("FRIEND MATCH");
    expect(renderScreen("local")).toContain("CPU設定領域");
    expect(renderScreen("more-game")).toContain("中断試合一覧");
    expect(renderScreen("settings")).toContain("設定項目は今後追加予定");
    expect(renderScreen("rules")).toContain("ゲームルールの説明");
    expect(renderScreen("friend")).toContain("フレンド一覧");
    expect(renderScreen("my-page")).toContain("プロフィール");
  });

  it("returns to the expected parent screen", () => {
    expect(getBackScreen("home")).toBeUndefined();
    expect(getBackScreen("new-game")).toBe("home");
    expect(getBackScreen("online")).toBe("new-game");
    expect(getBackScreen("friend-match")).toBe("new-game");
    expect(getBackScreen("local")).toBe("new-game");
    expect(getBackScreen("settings")).toBe("home");
  });

  it("keeps unavailable network actions disabled", () => {
    const online = renderScreen("online");
    expect(online.match(/disabled=""/g)).toHaveLength(3);
    expect(online).toContain("通信機能の実装後");
  });

  it("uses only existing local CPU controller values", () => {
    const local = renderScreen("local");
    expect(local).toContain("value=\"random_cpu\"");
    expect(local).toContain("value=\"heuristic_cpu\"");
    expect(local).toContain("value=\"bc_cpu\"");
    expect(local).toContain("人間（自分）");
  });
});
