import type { CSSProperties, ReactNode } from "react";
import type { CpuTeamSettings, TeamController } from "../game/cpu/types";
import type { Team } from "../game/types";
import { NineSlice } from "../ui/NineSliceView";
import type { NineSliceLayout, NineSliceSkin } from "../ui/nineSlice";
import {
  HOME_MENU_BUTTON_LAYOUT,
  HOME_MENU_BUTTON_SKIN,
  HOME_MENU_BACKGROUND_IMAGE,
  HOME_MENU_PANEL_LAYOUT,
  HOME_MENU_PANEL_SKIN,
} from "../ui/skins/home";

export type MenuScreen =
  | "home"
  | "new-game"
  | "online"
  | "friend-match"
  | "local"
  | "more-game"
  | "settings"
  | "rules"
  | "friend"
  | "my-page";

export type AppScreen = MenuScreen | "play";

type Props = {
  screen: MenuScreen;
  teams: Team[];
  localCpuSettings: CpuTeamSettings;
  onNavigate: (screen: MenuScreen) => void;
  onLocalCpuChange: (teamId: string, controller: TeamController) => void;
  onStartLocal: () => void;
};

const cpuOptions: { value: Exclude<TeamController, "human">; label: string }[] = [
  { value: "random_cpu", label: "Random CPU" },
  { value: "heuristic_cpu", label: "Heuristic CPU" },
  { value: "bc_cpu", label: "BC CPU" },
];

export function getBackScreen(screen: MenuScreen): MenuScreen | undefined {
  if (screen === "home") return undefined;
  if (["online", "friend-match", "local"].includes(screen)) return "new-game";
  return "home";
}

type ScreenFrame = Readonly<{
  skin: NineSliceSkin;
  layout?: NineSliceLayout;
  className?: string;
}>;

const HOME_PANEL_FRAME: ScreenFrame = Object.freeze({
  skin: HOME_MENU_PANEL_SKIN,
  layout: HOME_MENU_PANEL_LAYOUT,
  className: "home-menu-panel",
});

const HOME_MENU_SHELL_STYLE: CSSProperties = Object.freeze({
  backgroundImage: `url("${HOME_MENU_BACKGROUND_IMAGE}")`,
});

function ScreenLayout({
  title,
  children,
  onBack,
  frame,
  shellClassName,
  shellStyle,
}: {
  title: string;
  children: ReactNode;
  onBack?: () => void;
  frame?: ScreenFrame;
  shellClassName?: string;
  shellStyle?: CSSProperties;
}) {
  const content = <>
      <h1 id="screen-title">{title}</h1>
      {children}
      {onBack ? <button className="menu-button menu-button-back" type="button" onClick={onBack}>BACK</button> : null}
    </>;
  const shellClasses = ["menu-shell", shellClassName].filter(Boolean).join(" ");

  return <main className={shellClasses} style={shellStyle}>
    {frame
      ? <NineSlice
          as="section"
          skin={frame.skin}
          {...frame.layout}
          className={["menu-card", frame.className].filter(Boolean).join(" ")}
          aria-labelledby="screen-title"
        >
          {content}
        </NineSlice>
      : <section className="menu-card" aria-labelledby="screen-title">{content}</section>}
  </main>;
}

type DecoratedMenuAction = Readonly<{
  label: string;
  onClick: () => void;
}>;

function DecoratedMenuScreen({ title, actions }: { title: string; actions: readonly DecoratedMenuAction[] }) {
  return <ScreenLayout
    title={title}
    frame={HOME_PANEL_FRAME}
    shellClassName="home-menu-shell"
    shellStyle={HOME_MENU_SHELL_STYLE}
  >
    <div className="menu-actions home-actions">
      {actions.map((action) => <NineSlice
        as="button"
        skin={HOME_MENU_BUTTON_SKIN}
        {...HOME_MENU_BUTTON_LAYOUT}
        className="menu-button home-menu-button"
        type="button"
        onClick={action.onClick}
        key={action.label}
      >
        {action.label}
      </NineSlice>)}
    </div>
  </ScreenLayout>;
}

function PlaceholderScreen({ title, description, onBack }: { title: string; description: string; onBack: () => void }) {
  return <ScreenLayout title={title} onBack={onBack}>
    <div className="placeholder-panel">
      <p>{description}</p>
    </div>
  </ScreenLayout>;
}

function NetworkMatchScreen({ title, onBack }: { title: "ONLINE" | "FRIEND MATCH"; onBack: () => void }) {
  return <ScreenLayout title={title} onBack={onBack}>
    <p className="screen-note">オンラインの部屋機能は現在準備中です。</p>
    <div className="menu-actions">
      <button className="menu-button" type="button" disabled>部屋を作成する</button>
      <button className="menu-button" type="button" disabled>部屋に入る</button>
      <button className="menu-button menu-button-primary" type="button" disabled>始める</button>
    </div>
    <p className="availability-note" role="status">部屋の作成・参加と開始は、通信機能の実装後に利用できます。</p>
  </ScreenLayout>;
}

export function AppNavigation({ screen, teams, localCpuSettings, onNavigate, onLocalCpuChange, onStartLocal }: Props) {
  const backScreen = getBackScreen(screen);
  const onBack = backScreen ? () => onNavigate(backScreen) : undefined;

  if (screen === "home") return <DecoratedMenuScreen title="HOME" actions={[
    { label: "NEW GAME", onClick: () => onNavigate("new-game") },
    { label: "MORE GAME", onClick: () => onNavigate("more-game") },
    { label: "SETTINGS", onClick: () => onNavigate("settings") },
    { label: "RULES", onClick: () => onNavigate("rules") },
    { label: "FRIEND", onClick: () => onNavigate("friend") },
    { label: "MY PAGE", onClick: () => onNavigate("my-page") },
  ]} />;

  if (screen === "new-game") return <DecoratedMenuScreen title="NEW GAME" actions={[
    { label: "ONLINE", onClick: () => onNavigate("online") },
    { label: "FRIEND", onClick: () => onNavigate("friend-match") },
    { label: "LOCAL", onClick: () => onNavigate("local") },
    { label: "BACK", onClick: onBack! },
  ]} />;

  if (screen === "online") return <NetworkMatchScreen title="ONLINE" onBack={onBack!} />;
  if (screen === "friend-match") return <NetworkMatchScreen title="FRIEND MATCH" onBack={onBack!} />;

  if (screen === "local") return <ScreenLayout title="LOCAL GAME" onBack={onBack}>
    <p className="screen-note">自分（Team 1）以外のCPUを設定してください。</p>
    <div className="cpu-settings" aria-label="CPU設定領域">
      {teams.filter((team) => !team.isNeutral).map((team, index) => <label className="cpu-setting-row" key={team.id}>
        <span>{team.name}</span>
        {index === 0
          ? <strong>人間（自分）</strong>
          : <select
              aria-label={`${team.name} CPU`}
              value={localCpuSettings[team.id] ?? "random_cpu"}
              onChange={(event) => onLocalCpuChange(team.id, event.target.value as Exclude<TeamController, "human">)}
            >
              {cpuOptions.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}
            </select>}
      </label>)}
    </div>
    <button className="menu-button menu-button-primary" type="button" onClick={onStartLocal}>始める</button>
  </ScreenLayout>;

  if (screen === "more-game") return <PlaceholderScreen
    title="MORE GAME"
    description="中断試合一覧を今後ここに表示します。"
    onBack={onBack!}
  />;
  if (screen === "settings") return <PlaceholderScreen
    title="SETTINGS"
    description="設定項目は今後追加予定です。"
    onBack={onBack!}
  />;
  if (screen === "rules") return <PlaceholderScreen
    title="RULES"
    description="ゲームルールの説明を今後ここに配置します。"
    onBack={onBack!}
  />;
  if (screen === "friend") return <PlaceholderScreen
    title="FRIEND"
    description="フレンド一覧と管理機能を今後ここに追加します。"
    onBack={onBack!}
  />;
  return <PlaceholderScreen
    title="MY PAGE"
    description="プロフィール、名前、アバターなどを今後ここに表示します。"
    onBack={onBack!}
  />;
}
