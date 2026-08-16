import type { RlObservation } from "./rlEnvironment";
import { getRlActionFeatureWidth, getRlActionFeatureWidthV2, RL_ACTION_ENCODER_VERSION, RL_ACTION_ENCODER_VERSION_V2 } from "./rlActionEncoder";
import { getRlObservationFeatureSpec, getRlObservationFeatureSpecV2 } from "./rlObservationEncoder";

export type RlFeatureSpec = ReturnType<typeof createRlFeatureSpec>;

export function createRlFeatureSpec(observation: RlObservation) {
  const observationSpec = getRlObservationFeatureSpec(observation);
  return {
    schemaVersion: 1,
    observationSchemaVersion: observationSpec.schemaVersion,
    actionSchemaVersion: RL_ACTION_ENCODER_VERSION,
    globalWidth: observationSpec.globalWidth,
    teamWidth: observationSpec.teamWidth,
    unitWidth: observationSpec.unitWidth,
    mapTileWidth: observationSpec.mapTileWidth,
    baseWidth: observationSpec.baseWidth,
    constructionWidth: observationSpec.constructionWidth,
    strategicGlobalWidth: observationSpec.strategicGlobalWidth,
    strategicTableRowWidths: observationSpec.strategicTableRowWidths,
    actionFeatureWidth: getRlActionFeatureWidth(observation),
  };
}
export const createRlFeatureSpecV1 = createRlFeatureSpec;

export type RlFeatureSpecV2 = ReturnType<typeof createRlFeatureSpecV2>;

export function createRlFeatureSpecV2(observation: RlObservation) {
  const observationSpec = getRlObservationFeatureSpecV2(observation);
  return {
    schemaVersion: 2 as const,
    observationSchemaVersion: observationSpec.schemaVersion,
    actionSchemaVersion: RL_ACTION_ENCODER_VERSION_V2,
    globalWidth: observationSpec.globalWidth,
    teamWidth: observationSpec.teamWidth,
    unitWidth: observationSpec.unitWidth,
    mapTileWidth: observationSpec.mapTileWidth,
    baseWidth: observationSpec.baseWidth,
    constructionWidth: observationSpec.constructionWidth,
    strategicGlobalWidth: observationSpec.strategicGlobalWidth,
    strategicTableRowWidths: observationSpec.strategicTableRowWidths,
    actionFeatureWidth: getRlActionFeatureWidthV2(observation),
  };
}
