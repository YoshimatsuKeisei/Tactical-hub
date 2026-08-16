import { createInitialGameState } from "../initialState";
import { buildRlObservation } from "./rlEnvironment";
import { createRlV1ToV2MigrationManifest } from "./rlSchemaMigration";

const state = createInitialGameState();
const observation = buildRlObservation(state, "team-1", "team-1");
process.stdout.write(`${JSON.stringify(createRlV1ToV2MigrationManifest(observation), null, 2)}\n`);
