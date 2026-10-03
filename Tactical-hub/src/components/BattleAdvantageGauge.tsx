import { calculateBattleAdvantage } from "../game/engine/battleAdvantage";
import type { GameState } from "../game/types";

type GaugeEntry = {
  teamId: string;
  teamName: string;
  teamColor: string;
  share: number;
  percentage: number;
};

/** Converts normalized shares to display integers whose total is exactly 100. */
export function roundBattleAdvantagePercentages(shares: readonly number[]): number[] {
  if (shares.length === 0) return [];

  const percentages = shares.map((share) => Math.floor(share * 100));
  const remainderOrder = shares
    .map((share, index) => ({
      index,
      remainder: share * 100 - percentages[index],
    }))
    .sort((left, right) => right.remainder - left.remainder || left.index - right.index);

  let pointsToAllocate = 100 - percentages.reduce((sum, percentage) => sum + percentage, 0);
  for (let index = 0; pointsToAllocate > 0; index += 1, pointsToAllocate -= 1) {
    percentages[remainderOrder[index % remainderOrder.length].index] += 1;
  }

  return percentages;
}

export function BattleAdvantageGauge({ state }: { state: GameState }) {
  const teamsById = new Map(state.teams.map((team) => [team.id, team]));
  const activeAdvantages = calculateBattleAdvantage(state).filter((advantage) => advantage.active);
  const percentages = roundBattleAdvantagePercentages(
    activeAdvantages.map((advantage) => advantage.battleAdvantageShare),
  );
  const entries = activeAdvantages.flatMap((advantage, index): GaugeEntry[] => {
    const team = teamsById.get(advantage.teamId);
    return team ? [{
      teamId: team.id,
      teamName: team.name,
      teamColor: team.color,
      share: advantage.battleAdvantageShare,
      percentage: percentages[index],
    }] : [];
  });

  return (
    <section className="battle-advantage" aria-labelledby="battle-advantage-heading">
      <div className="battle-advantage-summary">
        <h2 id="battle-advantage-heading">戦況</h2>
        <div className="battle-advantage-legend" role="list">
          {entries.map((entry) => (
            <div className="battle-advantage-legend-item" role="listitem" key={entry.teamId}>
              <span className="battle-advantage-swatch" style={{ backgroundColor: entry.teamColor }} aria-hidden="true" />
              <span className="battle-advantage-team-name">{entry.teamName}</span>
              <strong>{entry.percentage}%</strong>
            </div>
          ))}
        </div>
      </div>
      {entries.length > 0 ? (
        <div className="battle-advantage-bar" aria-label="Battle Advantageによる現在の戦況">
          {entries.map((entry) => (
            <div
              className="battle-advantage-segment"
              key={entry.teamId}
              role="meter"
              aria-label={`${entry.teamName}の戦況`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={entry.percentage}
              style={{ backgroundColor: entry.teamColor, width: `${entry.share * 100}%` }}
              title={`${entry.teamName}: ${entry.percentage}%`}
            />
          ))}
        </div>
      ) : (
        <p className="battle-advantage-empty">表示できるactive teamがありません。</p>
      )}
    </section>
  );
}
