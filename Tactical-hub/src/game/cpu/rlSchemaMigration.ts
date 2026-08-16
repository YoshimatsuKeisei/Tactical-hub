import type { RlObservation } from "./rlEnvironment";
import { RL_ACTION_TYPES, getRlActionFeatureWidth, getRlActionFeatureWidthV2 } from "./rlActionEncoder";
import { createRlFeatureSpecV1, createRlFeatureSpecV2 } from "./rlFeatureSpec";

export type FeatureColumnSegment = {
  sourceStart: number;
  targetStart: number;
  width: number;
};

export type RlV1ToV2MigrationManifest = {
  schemaVersion: 1;
  sourceFeatureSpec: ReturnType<typeof createRlFeatureSpecV1>;
  targetFeatureSpec: ReturnType<typeof createRlFeatureSpecV2>;
  unitInputColumns: FeatureColumnSegment[];
  actionInputColumns: FeatureColumnSegment[];
};

/**
 * Describes the exact v1 columns retained by schema v2. The omitted target
 * columns are the newly introduced features and must be zero initialized.
 */
export function createRlV1ToV2MigrationManifest(observation: RlObservation): RlV1ToV2MigrationManifest {
  const sourceFeatureSpec = createRlFeatureSpecV1(observation);
  const targetFeatureSpec = createRlFeatureSpecV2(observation);
  const actionTypeWidth = RL_ACTION_TYPES.length;
  const unitReferenceWidth = getRlActionFeatureWidthV2(observation) - getRlActionFeatureWidth(observation) - 1;
  if (unitReferenceWidth <= 0) throw new Error("Invalid v1/v2 action unit-reference width");
  const teamReferenceWidth = observation.teams.length + 1;
  const partnerInsertionSourceOffset = actionTypeWidth + 1 + teamReferenceWidth + unitReferenceWidth * 2;
  const sourceActionWidth = sourceFeatureSpec.actionFeatureWidth;
  if (partnerInsertionSourceOffset > sourceActionWidth) throw new Error("Invalid v1 action migration offset");
  return {
    schemaVersion: 1,
    sourceFeatureSpec,
    targetFeatureSpec,
    unitInputColumns: [
      { sourceStart: 0, targetStart: 0, width: 1 },
      { sourceStart: 1, targetStart: 2, width: sourceFeatureSpec.unitWidth - 1 },
    ],
    actionInputColumns: [
      { sourceStart: 0, targetStart: 0, width: actionTypeWidth },
      {
        sourceStart: actionTypeWidth,
        targetStart: actionTypeWidth + 1,
        width: partnerInsertionSourceOffset - actionTypeWidth,
      },
      {
        sourceStart: partnerInsertionSourceOffset,
        targetStart: partnerInsertionSourceOffset + 1 + unitReferenceWidth,
        width: sourceActionWidth - partnerInsertionSourceOffset,
      },
    ],
  };
}
