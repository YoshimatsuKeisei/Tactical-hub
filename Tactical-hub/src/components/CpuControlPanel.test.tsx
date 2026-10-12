import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { createInitialGameState } from "../game/initialState";
import { CpuControlPanel } from "./CpuControlPanel";

describe("unified strategist setup UI", () => {
  it("shows one unified initial strategist without a role selector", () => {
    const state = createInitialGameState();
    const html = renderToStaticMarkup(<CpuControlPanel
      view="settings"
      teams={state.teams}
      settings={Object.fromEntries(state.teams.filter((team) => !team.isNeutral).map((team) => [team.id, "human" as const]))}
      onControllerChange={vi.fn()}
      running={false}
      paused={false}
      onStart={vi.fn()}
      onPause={vi.fn()}
      onResume={vi.fn()}
      onStep={vi.fn()}
      speed="normal"
      onSpeedChange={vi.fn()}
      seed={1}
      onSeedChange={vi.fn()}
      logs={[]}
    />);

    expect(html.match(/Initial strategist: Unified/g)).toHaveLength(4);
    expect(html).not.toContain("Builder</option>");
    expect(html).not.toContain("Teleporter</option>");
  });
});
