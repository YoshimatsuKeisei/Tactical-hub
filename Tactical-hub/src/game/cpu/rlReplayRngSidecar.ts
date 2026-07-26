import { createHash } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import type { RlImitationEpisode } from "./rlImitationCollector";

export const RL_REPLAY_RNG_SIDECAR_SCHEMA_VERSION = 1;

export type RlReplayRngSidecar = {
  schemaVersion: typeof RL_REPLAY_RNG_SIDECAR_SCHEMA_VERSION;
  replayIdentity: string;
  episodeId: string;
  seed: number;
  decisionCount: number;
  rngStatesAfterPolicy: number[];
};

export function getRlReplayIdentity(episode: RlImitationEpisode) {
  return createHash("sha256").update(JSON.stringify(episode)).digest("hex");
}

export function validateRlReplayRngSidecar(episode: RlImitationEpisode, sidecar: RlReplayRngSidecar) {
  const identity = getRlReplayIdentity(episode);
  if (sidecar.schemaVersion !== RL_REPLAY_RNG_SIDECAR_SCHEMA_VERSION) throw new Error(`Unsupported RNG sidecar schemaVersion ${sidecar.schemaVersion}`);
  if (sidecar.replayIdentity !== identity || sidecar.episodeId !== episode.header.episodeId || sidecar.seed !== episode.header.seed) {
    throw new Error(`RNG sidecar does not match replay ${episode.header.episodeId}`);
  }
  if (sidecar.decisionCount !== episode.decisions.length || sidecar.rngStatesAfterPolicy.length !== episode.decisions.length) {
    throw new Error(`RNG sidecar decision count mismatch for ${episode.header.episodeId}`);
  }
  if (sidecar.rngStatesAfterPolicy.some((value) => !Number.isInteger(value) || value < 0 || value > 0xffff_ffff)) {
    throw new Error(`RNG sidecar contains an invalid rngState for ${episode.header.episodeId}`);
  }
}

export function getDefaultRlReplaySidecarDirectory(dataPath: string) {
  return `${resolve(dataPath)}.rng-sidecars`;
}

export async function loadOrCreateRlReplayRngSidecar(
  episode: RlImitationEpisode,
  directory: string,
  generate: () => Promise<RlReplayRngSidecar>,
) {
  const identity = getRlReplayIdentity(episode);
  const resolvedDirectory = resolve(directory);
  const path = join(resolvedDirectory, `${identity}.json`);
  const load = async () => {
    const parsed = JSON.parse(await readFile(path, "utf8")) as RlReplayRngSidecar;
    validateRlReplayRngSidecar(episode, parsed);
    return parsed;
  };
  try {
    return { sidecar: await load(), generated: false, path };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  const sidecar = await generate();
  validateRlReplayRngSidecar(episode, sidecar);
  await mkdir(resolvedDirectory, { recursive: true });
  const temporaryPath = `${path}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(sidecar)}\n`, "utf8");
  try {
    await rename(temporaryPath, path);
  } catch (error) {
    await rm(temporaryPath, { force: true });
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    return { sidecar: await load(), generated: false, path };
  }
  return { sidecar, generated: true, path };
}
