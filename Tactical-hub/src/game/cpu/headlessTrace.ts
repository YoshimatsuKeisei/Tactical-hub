import type { GameState } from "../types";
import type { HeuristicDistanceCacheStats } from "./heuristicCpuPolicy";

export type HeadlessTraceStateSummary = {
  stateHash: string;
  turnPhaseHash: string;
  teamsHash: string;
  unitsHash: string;
  basesHash: string;
  constructionsHash: string;
  intentsHash: string;
  rewardsHash: string;
};

export type HeadlessTraceEntry = {
  actionIndex: number;
  seed: number;
  turn: number;
  phase: GameState["phase"];
  actorTeamId?: string;
  targetBaseId?: string;
  legalActionCount?: number;
  legalActionKeysHash?: string;
  legalActionKeys?: string[];
  selectedActionKey: string;
  rngStateBefore: number;
  rngStateAfter: number;
  stateBefore: HeadlessTraceStateSummary;
  stateAfter: HeadlessTraceStateSummary;
  distanceCache?: HeuristicDistanceCacheStats;
};

function canonical(value: unknown): string {
  if (value === undefined) return '"<undefined>"';
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? '"<undefined>"';
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0).map(([key, entry]) => `${JSON.stringify(key)}:${canonical(entry)}`).join(",")}}`;
}

export function stableDiagnosticHash(value: unknown) {
  let hash = 2166136261;
  for (const character of canonical(value)) hash = Math.imul(hash ^ character.charCodeAt(0), 16777619) >>> 0;
  return hash.toString(16).padStart(8, "0");
}

export function summarizeTraceState(state: GameState): HeadlessTraceStateSummary {
  return {
    stateHash: stableDiagnosticHash(state),
    turnPhaseHash: stableDiagnosticHash({ turnNumber: state.turnNumber, phase: state.phase, currentMovementTeamId: state.currentMovementTeamId, movementOrderTeamIds: state.movementOrderTeamIds, movementCompletedTeamIds: state.movementCompletedTeamIds }),
    teamsHash: stableDiagnosticHash(state.teams),
    unitsHash: stableDiagnosticHash(state.units),
    basesHash: stableDiagnosticHash(state.bases),
    constructionsHash: stableDiagnosticHash(state.constructions),
    intentsHash: stableDiagnosticHash({ actionIntents: state.turnState.actionIntents, teleportIntents: state.teleportIntents, strategistActionIntents: state.strategistActionIntents, strategistSubmittedTeamIds: state.strategistSubmittedTeamIds }),
    rewardsHash: stableDiagnosticHash(state.rewardPlacementRequests),
  };
}

export type HeadlessTraceDifferenceKind = "state_before" | "legal_actions" | "selected_action" | "state_after" | "rng" | "missing_action";
export type HeadlessTraceComparison = {
  equal: boolean;
  firstDifference?: {
    kind: HeadlessTraceDifferenceKind;
    actionIndex: number;
    turn?: number;
    phase?: string;
    actorTeamId?: string;
    message: string;
    differingStateFields?: string[];
    addedLegalActions?: string[];
    missingLegalActions?: string[];
    firstLegalOrderDifference?: { index: number; left?: string; right?: string };
    context: { left: Partial<HeadlessTraceEntry>[]; right: Partial<HeadlessTraceEntry>[] };
  };
};

const context = (entries: HeadlessTraceEntry[], index: number) => entries.slice(Math.max(0, index - 3), index + 4).map((entry) => ({
  actionIndex: entry.actionIndex,
  turn: entry.turn,
  phase: entry.phase,
  actorTeamId: entry.actorTeamId,
  targetBaseId: entry.targetBaseId,
  legalActionCount: entry.legalActionCount,
  legalActionKeysHash: entry.legalActionKeysHash,
  selectedActionKey: entry.selectedActionKey,
  rngStateBefore: entry.rngStateBefore,
  rngStateAfter: entry.rngStateAfter,
  stateBefore: { stateHash: entry.stateBefore.stateHash } as HeadlessTraceStateSummary,
  stateAfter: { stateHash: entry.stateAfter.stateHash } as HeadlessTraceStateSummary,
  distanceCache: entry.distanceCache,
}));
const stateFields = (left: HeadlessTraceStateSummary, right: HeadlessTraceStateSummary) => (Object.keys(left) as (keyof HeadlessTraceStateSummary)[]).filter((key) => left[key] !== right[key]);

export function compareHeadlessTraces(left: HeadlessTraceEntry[], right: HeadlessTraceEntry[]): HeadlessTraceComparison {
  const length = Math.max(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const a = left[index], b = right[index];
    const common = { actionIndex: a?.actionIndex ?? b?.actionIndex ?? index, turn: a?.turn ?? b?.turn, phase: a?.phase ?? b?.phase, actorTeamId: a?.actorTeamId ?? b?.actorTeamId, context: { left: context(left, index), right: context(right, index) } };
    if (!a || !b) return { equal: false, firstDifference: { ...common, kind: "missing_action", message: `Only ${a ? "left" : "right"} trace contains this action.` } };
    if (a.stateBefore.stateHash !== b.stateBefore.stateHash) return { equal: false, firstDifference: { ...common, kind: "state_before", message: "GameState already differs before the action.", differingStateFields: stateFields(a.stateBefore, b.stateBefore) } };
    const legalDifferent = a.legalActionKeysHash !== b.legalActionKeysHash || a.legalActionCount !== b.legalActionCount;
    if (legalDifferent) {
      const aKeys = a.legalActionKeys ?? [], bKeys = b.legalActionKeys ?? [];
      const firstOrderIndex = Array.from({ length: Math.max(aKeys.length, bKeys.length) }, (_, keyIndex) => keyIndex).find((keyIndex) => aKeys[keyIndex] !== bKeys[keyIndex]);
      return { equal: false, firstDifference: { ...common, kind: "legal_actions", message: "Legal Action contents or ordering differs.", addedLegalActions: bKeys.filter((key) => !new Set(aKeys).has(key)).slice(0, 10), missingLegalActions: aKeys.filter((key) => !new Set(bKeys).has(key)).slice(0, 10), firstLegalOrderDifference: firstOrderIndex === undefined ? undefined : { index: firstOrderIndex, left: aKeys[firstOrderIndex], right: bKeys[firstOrderIndex] } } };
    }
    if (a.rngStateBefore !== b.rngStateBefore) return { equal: false, firstDifference: { ...common, kind: "rng", message: `RNG state differs before selection: ${a.rngStateBefore} vs ${b.rngStateBefore}.` } };
    if (a.selectedActionKey !== b.selectedActionKey) return { equal: false, firstDifference: { ...common, kind: "selected_action", message: `Selected Action differs: ${a.selectedActionKey} vs ${b.selectedActionKey}.` } };
    if (a.stateAfter.stateHash !== b.stateAfter.stateHash) return { equal: false, firstDifference: { ...common, kind: "state_after", message: "The same selected Action produced a different GameState.", differingStateFields: stateFields(a.stateAfter, b.stateAfter) } };
    if (a.rngStateAfter !== b.rngStateAfter) return { equal: false, firstDifference: { ...common, kind: "rng", message: `RNG state differs after application: ${a.rngStateAfter} vs ${b.rngStateAfter}.` } };
  }
  return { equal: true };
}
