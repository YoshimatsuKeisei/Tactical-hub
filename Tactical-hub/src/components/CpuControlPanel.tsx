import type { CpuActionLog, CpuTeamSettings, TeamController } from "../game/cpu/types";
import type { Team } from "../game/types";

export type CpuRunnerSpeed = "normal" | "fast" | "instant";

type Props = {
  view: "settings" | "logs";
  teams: Team[];
  settings: CpuTeamSettings;
  onControllerChange: (teamId: string, controller: TeamController) => void;
  running: boolean;
  paused: boolean;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStep: () => void;
  speed: CpuRunnerSpeed;
  onSpeedChange: (speed: CpuRunnerSpeed) => void;
  currentCpuTeamId?: string;
  seed: number;
  onSeedChange: (seed: number) => void;
  logs: CpuActionLog[];
  stoppedReason?: string;
};

export function CpuControlPanel(props: Props) {
  if (props.view === "logs") return <section>
    <h2>CPU Action Log</h2>
    <p>Seed: {props.seed} / actions: {props.logs.length}</p>
    {props.stoppedReason ? <p>Stopped: {props.stoppedReason}</p> : null}
    <div className="intent-list">
      {props.logs.slice().reverse().map((entry) => <div className="intent-item" key={entry.id}>
        <strong>T{entry.turnNumber} {entry.phase} {entry.teamId ?? "system"}</strong>
        <span>{entry.action}{entry.detail ? `: ${entry.detail}` : ""}</span>
        {entry.error ? <span>{entry.error}</span> : null}
      </div>)}
    </div>
  </section>;

  return <section>
    <h2>プレイヤー構成</h2>
    <div className="intent-list">
      {props.teams.filter((team) => !team.isNeutral).map((team) => <div className="intent-item" key={team.id}>
        <strong>{team.name}</strong>
        <select value={props.settings[team.id] ?? "human"} onChange={(event) => props.onControllerChange(team.id, event.target.value as TeamController)}>
          <option value="human">人間</option>
          <option value="random_cpu">ランダムCPU</option>
          <option value="heuristic_cpu">Heuristic CPU</option>
          <option value="bc_cpu">BC CPU</option>
        </select>
        <span>Initial strategist: Unified (Encourage / Build / Teleport)</span>
      </div>)}
    </div>
    <div className="button-row">
      <button onClick={props.onStart} disabled={props.running && !props.paused}>自動進行を開始</button>
      <button onClick={props.onPause} disabled={!props.running || props.paused}>一時停止</button>
      <button onClick={props.onResume} disabled={!props.running || !props.paused}>再開</button>
      <button onClick={props.onStep}>CPU行動を1つ進める</button>
      <select value={props.speed} onChange={(event) => props.onSpeedChange(event.target.value as CpuRunnerSpeed)}>
        <option value="normal">通常</option>
        <option value="fast">高速</option>
        <option value="instant">待機なし</option>
      </select>
    </div>
    <p>現在行動中のCPU: {props.currentCpuTeamId ?? "待機中"}</p>
    <label>Seed <input type="number" value={props.seed} disabled={props.running && !props.paused} onChange={(event) => props.onSeedChange(Number(event.target.value) || 1)} /></label>
    {props.stoppedReason ? <p>Stopped: {props.stoppedReason}</p> : null}
  </section>;
}
