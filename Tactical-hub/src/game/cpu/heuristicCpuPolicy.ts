import { getEnemyControlledBases } from "../engine/retreat";
import type { Base, GameState, Unit, UnitPosition, UnitType } from "../types";
import { createRoadAttackDistanceLookup, createRoadAttackTopologyContext } from "../utils/roadTopology";
import { enumerateRlDecisions, getCpuDecisionActionKey, type EnumeratedDecision } from "./rlEnvironment";
import { isLegalProfilingEnabled, measureLegalSegment, recordLegalSegment } from "./legalEnumerationProfile";
import { nextCpuRandom } from "./randomCpuPolicy";
import type { CpuDecision, CpuPolicy, CpuRuntime, CpuTeamSettings } from "./types";

const PREFERRED_TYPES = new Set<UnitType>(["archer", "cavalry", "infantry"]);

type MatchTargets = { lastTurn: number; byTeamId: Map<string, string> };
export type HeuristicDecisionDiagnostics = { targetBaseId?: string; legalActionKeys: string[]; selectedActionKey?: string; distanceCache: HeuristicDistanceCacheStats };
export type HeuristicCpuPolicy = CpuPolicy & {
  getTargetBaseId(teamId: string, seed?: number): string | undefined;
  getLastDecisionDiagnostics(): HeuristicDecisionDiagnostics | undefined;
  setDecisionDiagnosticsEnabled(enabled: boolean): void;
};

function choose<T>(runtime: CpuRuntime, values: readonly T[]) {
  return values[Math.min(values.length - 1, Math.floor(nextCpuRandom(runtime) * values.length))];
}

function isFixedMover(unit: Unit | undefined) {
  return unit?.type === "king" || (unit?.type === "strategist" && unit.role === "builder");
}

export type HeuristicDistanceCacheStats = { requests: number; searches: number; hits: number; misses: number };
export function createHeuristicDistanceEvaluator(state: GameState) {
  const lookups = new Map<string, (position: UnitPosition) => number>();
  const topology = createRoadAttackTopologyContext(state);
  const stats: HeuristicDistanceCacheStats = { requests: 0, searches: 0, hits: 0, misses: 0 };
  const distance = (position: UnitPosition, baseId: string) => measureLegalSegment("heuristicDistanceEvaluation", () => {
    stats.requests += 1;
    let lookup = lookups.get(baseId);
    if (lookup) {
      stats.hits += 1;
      recordLegalSegment("heuristicDistanceCacheHit", 0);
    } else {
      stats.misses += 1;
      recordLegalSegment("heuristicDistanceCacheMiss", 0);
      const slotId = state.bases.find((base) => base.id === baseId)?.slots[0]?.id;
      if (!slotId) return Number.POSITIVE_INFINITY;
      lookup = measureLegalSegment("heuristicDistanceSearch", () => createRoadAttackDistanceLookup(state, { kind: "base", baseId, slotId }, topology));
      stats.searches += 1;
      lookups.set(baseId, lookup);
    }
    return lookup(position);
  });
  return { distance, stats };
}

function nearestTarget(state: GameState, teamId: string, runtime: CpuRuntime, candidates: Base[], distance: (position: UnitPosition, baseId: string) => number) {
  const movers = state.units.filter((unit) => unit.ownerTeamId === teamId && unit.hp > 0 && unit.position.kind !== "removed" && !isFixedMover(unit));
  const scored = candidates.map((base) => ({
    base,
    distance: Math.min(...movers.map((unit) => distance(unit.position, base.id))),
  }));
  const finite = scored.filter((entry) => Number.isFinite(entry.distance));
  const pool = finite.length ? finite : scored;
  if (!pool.length) return undefined;
  const minimum = Math.min(...pool.map((entry) => entry.distance));
  const tied = pool.filter((entry) => entry.distance === minimum).sort((left, right) => left.base.id.localeCompare(right.base.id));
  return choose(runtime, tied).base.id;
}

function choosePreferredProduction(runtime: CpuRuntime, decisions: EnumeratedDecision[]) {
  const preferred = decisions.filter((entry) => {
    const decision = entry.decision;
    return (decision.kind === "production" && decision.choice && PREFERRED_TYPES.has(decision.choice.unitType))
      || (decision.kind === "reward" && PREFERRED_TYPES.has(decision.unitType));
  });
  return preferred.length ? choose(runtime, preferred).decision : undefined;
}

export function createHeuristicCpuPolicy(): HeuristicCpuPolicy {
  const matches = new Map<number, MatchTargets>();
  let lastDiagnostics: HeuristicDecisionDiagnostics | undefined;
  let diagnosticsEnabled = false;

  const context = (state: GameState, runtime: CpuRuntime) => {
    let value = matches.get(runtime.seed);
    if (!value || state.turnNumber < value.lastTurn) value = { lastTurn: state.turnNumber, byTeamId: new Map() };
    value.lastTurn = state.turnNumber;
    matches.set(runtime.seed, value);
    return value;
  };

  const targetFor = (state: GameState, teamId: string, runtime: CpuRuntime, distance: (position: UnitPosition, baseId: string) => number) => {
    return measureLegalSegment("heuristicTargetSelection", () => {
      const match = context(state, runtime);
      const candidates = getEnemyControlledBases(state, teamId).filter((base) => state.bases.some((entry) => entry.id === base.id));
      const current = match.byTeamId.get(teamId);
      if (current && candidates.some((base) => base.id === current)) return current;
      const selected = nearestTarget(state, teamId, runtime, candidates, distance);
      if (selected) match.byTeamId.set(teamId, selected); else match.byTeamId.delete(teamId);
      return selected;
    });
  };

  const policy = ((state: GameState, runtime: CpuRuntime, settings: CpuTeamSettings): CpuDecision | undefined => {
    const profileLegal = isLegalProfilingEnabled();
    const legalStarted = profileLegal ? performance.now() : 0;
    const legal = enumerateRlDecisions(state, runtime);
    const legalMilliseconds = profileLegal ? performance.now() - legalStarted : 0;
    if (!legal.length) {
      if (diagnosticsEnabled) lastDiagnostics = { legalActionKeys: [], distanceCache: { requests: 0, searches: 0, hits: 0, misses: 0 } };
      return undefined;
    }
    const distanceEvaluator = createHeuristicDistanceEvaluator(state);
    const teamId = legal[0].decision.teamId;
    if (teamId !== "all" && settings[teamId] !== "random_cpu") {
      if (diagnosticsEnabled) lastDiagnostics = { legalActionKeys: legal.map((entry) => entry.action.actionKey), distanceCache: distanceEvaluator.stats };
      return undefined;
    }
    let diagnosticTargetBaseId: string | undefined;

    const kind = legal[0].decision.kind;
    const legalCategory = kind === "teleport" ? "teleport"
      : kind === "attack" ? "attack"
        : kind === "reward" ? "reward"
          : kind === "strategist" || kind === "submit_strategist" || kind === "resolve_strategists" ? "constructionStrategist"
            : kind === "movement" || kind === "submit_movement" ? "movement"
              : "production";
    recordLegalSegment(legalCategory, legalMilliseconds);
    const selected = measureLegalSegment<CpuDecision | undefined>("heuristicCandidateEvaluation", () => {
    if (kind === "production" || kind === "reward") {
      const selected = choosePreferredProduction(runtime, legal);
      if (selected) return selected;
      if (kind === "production") {
        const first = legal[0].decision;
        if (first.kind === "production") return { kind: "production", teamId: first.teamId, actorKey: first.actorKey };
      }
      return undefined;
    }

    if (kind === "movement") {
      const first = legal[0].decision;
      if (first.kind !== "movement") return undefined;
      const unit = state.units.find((entry) => entry.id === first.unitId);
      if (isFixedMover(unit)) return legal.find((entry) => entry.decision.kind === "movement" && !entry.decision.to)?.decision;
      const targetBaseId = targetFor(state, teamId, runtime, distanceEvaluator.distance);
      diagnosticTargetBaseId = targetBaseId;
      if (!unit || !targetBaseId) return legal.find((entry) => entry.decision.kind === "movement" && !entry.decision.to)?.decision;
      const currentDistance = distanceEvaluator.distance(unit.position, targetBaseId);
      const moving = legal.flatMap((entry) => entry.decision.kind === "movement" && entry.decision.to ? [{ decision: entry.decision, distance: distanceEvaluator.distance(entry.decision.to, targetBaseId) }] : []);
      const improving = moving.filter((entry) => entry.distance < currentDistance);
      if (!improving.length) return legal.find((entry) => entry.decision.kind === "movement" && !entry.decision.to)?.decision;
      const minimum = Math.min(...improving.map((entry) => entry.distance));
      return choose(runtime, improving.filter((entry) => entry.distance === minimum)).decision;
    }

    if (kind === "teleport") {
      const pass = legal.find((entry) => entry.decision.kind === "teleport" && !entry.decision.intent)?.decision;
      const targetBaseId = targetFor(state, teamId, runtime, distanceEvaluator.distance);
      diagnosticTargetBaseId = targetBaseId;
      if (!targetBaseId) return pass;
      const moving = legal.flatMap((entry) => entry.decision.kind === "teleport" && entry.decision.intent
        ? [{ decision: entry.decision, distance: distanceEvaluator.distance(entry.decision.intent.to, targetBaseId) }]
        : []);
      if (!moving.length) return pass;
      const minimum = Math.min(...moving.map((entry) => entry.distance));
      return choose(runtime, moving.filter((entry) => entry.distance === minimum)).decision;
    }

    if (kind === "attack") {
      const targetBaseId = targetFor(state, teamId, runtime, distanceEvaluator.distance);
      diagnosticTargetBaseId = targetBaseId;
      const preferred = legal.filter((entry) => {
        const decision = entry.decision;
        if (decision.kind !== "attack" || !decision.intent.target || !targetBaseId) return false;
        return state.units.find((unit) => unit.id === decision.intent.target!.unitId)?.position.kind === "base"
          && (state.units.find((unit) => unit.id === decision.intent.target!.unitId)?.position as Extract<UnitPosition, { kind: "base" }>).baseId === targetBaseId;
      });
      if (preferred.length) return choose(runtime, preferred).decision;
      return legal.find((entry) => entry.decision.kind === "attack" && entry.decision.intent.pass)?.decision;
    }

    if (kind === "strategist") return choose(runtime, legal).decision;
    return legal.length === 1 ? legal[0].decision : choose(runtime, legal).decision;
    });
    if (diagnosticsEnabled) {
      lastDiagnostics = { targetBaseId: diagnosticTargetBaseId, legalActionKeys: legal.map((entry) => entry.action.actionKey), selectedActionKey: selected ? getCpuDecisionActionKey(selected) : undefined, distanceCache: { ...distanceEvaluator.stats } };
    }
    return selected;
  }) as HeuristicCpuPolicy;

  policy.getTargetBaseId = (teamId, seed) => seed === undefined
    ? [...matches.values()].at(-1)?.byTeamId.get(teamId)
    : matches.get(seed)?.byTeamId.get(teamId);
  policy.getLastDecisionDiagnostics = () => lastDiagnostics ? { ...lastDiagnostics, legalActionKeys: [...lastDiagnostics.legalActionKeys], distanceCache: { ...lastDiagnostics.distanceCache } } : undefined;
  policy.setDecisionDiagnosticsEnabled = (enabled) => {
    diagnosticsEnabled = enabled;
    if (!enabled) lastDiagnostics = undefined;
  };
  return policy;
}
