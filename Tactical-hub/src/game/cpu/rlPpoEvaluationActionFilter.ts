const LEGACY_UNIT_TYPES = new Set(["infantry", "cavalry", "archer"]);

export function filterPpoLegalActionsForEvaluation<
  T extends { actionType: string; unitType?: string }
>(
  actions: readonly T[],
  legacyProductionOnly: boolean,
): readonly T[] {
  if (!legacyProductionOnly) return actions;
  return actions.filter((action) => {
    if ((action.actionType === "production" || action.actionType === "reward") && action.unitType) {
      return LEGACY_UNIT_TYPES.has(action.unitType);
    }
    return true;
  });
}
