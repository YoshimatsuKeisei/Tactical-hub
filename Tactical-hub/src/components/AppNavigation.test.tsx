import { useState } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { act, create } from "react-test-renderer";
import { describe, expect, it, vi } from "vitest";
import App from "../App";
import { createInitialGameState } from "../game/initialState";
import type { CpuTeamSettings } from "../game/cpu/types";
import type { LocalGameSaveRepository } from "../game/save/localGameSaveStorageTypes";
import { AppNavigation, getBackScreen, type MenuScreen } from "./AppNavigation";

const teams = createInitialGameState().teams;
const localCpuSettings: CpuTeamSettings = {
  "team-1": "human",
  "team-2": "random_cpu",
  "team-3": "heuristic_cpu",
  "team-4": "bc_cpu",
};

const repository: LocalGameSaveRepository = {
  put: async (save) => ({ ok: true, value: { saveId: save.saveId, revision: 1 } }),
  get: async () => ({ ok: false, error: { code: "NOT_FOUND", message: "fixture" } }),
  list: async () => ({ ok: true, value: [] }),
  delete: async (saveId) => ({ ok: true, value: { saveId, deleted: true } }),
};

function NavigationHarness({ onStartLocal = vi.fn() }: { onStartLocal?: () => void }) {
  const [screen, setScreen] = useState<MenuScreen>("home");
  return <AppNavigation
    screen={screen}
    teams={teams}
    localCpuSettings={localCpuSettings}
    localSaveRepository={repository}
    onNavigate={setScreen}
    onLocalCpuChange={vi.fn()}
    onResumeLocal={vi.fn()}
    onStartLocal={onStartLocal}
  />;
}

function renderScreen(screen: MenuScreen) {
  return renderToStaticMarkup(<AppNavigation
    screen={screen}
    teams={teams}
    localCpuSettings={localCpuSettings}
    localSaveRepository={repository}
    onNavigate={vi.fn()}
    onLocalCpuChange={vi.fn()}
    onResumeLocal={vi.fn()}
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
    expect(renderScreen("more-game")).toContain("中断データを読み込んでいます");
    expect(renderScreen("settings")).toContain("設定項目は今後追加予定");
    expect(renderScreen("rules")).toContain("基本ルール");
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
    expect(getBackScreen("rules")).toBe("home");
  });

  it("keeps unavailable network actions disabled", () => {
    const online = renderScreen("online");
    expect(online.match(/disabled=""/g)).toHaveLength(3);
    expect(online).toContain("通信機能の実装後");
  });

  it("navigates HOME to MORE GAME, loads repository metadata, and BACK returns HOME", async () => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    const list = vi.spyOn(repository, "list");
    let renderer!: ReturnType<typeof create>;
    await act(async () => { renderer = create(<NavigationHarness />); });
    const more = renderer.root.findAllByType("button").find((button) => button.children.join("") === "MORE GAME");
    await act(async () => { more!.props.onClick(); await Promise.resolve(); });
    expect(list).toHaveBeenCalledTimes(1);
    expect(renderer.root.findByType("h1").children.join("")).toBe("MORE GAME");
    const back = renderer.root.findAllByType("button").find((button) => button.children.join("") === "BACK");
    await act(async () => { back!.props.onClick(); });
    expect(renderer.root.findByType("h1").children.join("")).toBe("HOME");
    await act(async () => { renderer.unmount(); });
    list.mockRestore();
  });

  it("uses only existing local CPU controller values", () => {
    const local = renderScreen("local");
    expect(local).toContain("value=\"random_cpu\"");
    expect(local).toContain("value=\"heuristic_cpu\"");
    expect(local).toContain("value=\"bc_cpu\"");
    expect(local).toContain("人間（自分）");
    expect(local).toContain("BC CPUを含む試合では自動保存・再開は現在利用できません");
  });

  it("keeps BC CPU local game start available while showing autosave is unsupported", async () => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    const onStartLocal = vi.fn();
    let renderer!: ReturnType<typeof create>;
    await act(async () => {
      renderer = create(<AppNavigation
        screen="local"
        teams={teams}
        localCpuSettings={localCpuSettings}
        localSaveRepository={repository}
        onNavigate={vi.fn()}
        onLocalCpuChange={vi.fn()}
        onResumeLocal={vi.fn()}
        onStartLocal={onStartLocal}
      />);
    });
    expect(JSON.stringify(renderer.toJSON())).toContain("自動保存・再開は現在利用できません");
    const start = renderer.root.findAllByType("button").find((button) => button.children.join("") === "始める");
    await act(async () => { start!.props.onClick(); });
    expect(onStartLocal).toHaveBeenCalledTimes(1);
    await act(async () => { renderer.unmount(); });
  });

  it("renders the confirmed rules without pending or legacy specifications", () => {
    const rules = renderScreen("rules");

    expect(rules).toContain("ユニット・特殊能力");
    expect(rules).toContain("詳細ルール");
    expect(rules).toContain("移動は即時に確定します");
    expect(rules).toContain("重歩兵");
    expect(rules).toContain("湖から隣接する道路へ上陸できます");
    expect(rules).toContain("中立守備隊への王攻略褒賞は発生しません");
    expect(rules).toContain("攻撃成功率");
    expect(rules).toContain("拠点攻略の優先順位");
    expect(rules).toContain("<th scope=\"row\">工</th>");
    expect(rules).toContain(">BACK<");

    expect(rules).not.toContain("movement intent");
    expect(rules).not.toContain("一括移動");
    expect(rules).not.toContain("忍者見習い");
    expect(rules).not.toContain("兵種入れ替え");
    expect(rules).not.toContain("決戦モード");
    expect(rules).not.toContain("退却");
    expect(rules).not.toContain("工兵");
    expect(rules).not.toContain("工作隊");
  });
});
