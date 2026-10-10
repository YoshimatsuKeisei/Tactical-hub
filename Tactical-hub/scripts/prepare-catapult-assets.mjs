#!/usr/bin/env node

import { access, copyFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const CATAPULT_ATTRIBUTION = "Bleed - http://remusprites.carbonmade.com/";
export const CATAPULT_DIRECTION_LABELS = [
  "down",
  "down-right",
  "right",
  "up-right",
  "up",
  "up-left",
  "left",
  "down-left",
];
export const CATAPULT_ANIMATION_FRAME_COUNTS = {
  move: 31,
  load: 31,
  throw: 16,
  break: 31,
};

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(packageRoot, "..");
const defaultOutputRoot = path.join(packageRoot, "public", "local-assets", "catapult");

function frameName(animation, direction, frame) {
  return `catapult_${animation}_${direction}${String(frame).padStart(4, "0")}.png`;
}

function outputFrameName(frame) {
  return `${String(frame).padStart(4, "0")}.png`;
}

async function exists(candidate) {
  try {
    await access(candidate);
    return true;
  } catch {
    return false;
  }
}

async function resolveSourceRoot(explicitRoot) {
  const candidates = [
    explicitRoot,
    process.env.CATAPULT_ASSET_ROOT,
    path.join(repositoryRoot, "Catapult - Isometric"),
  ].filter(Boolean).map((candidate) => path.resolve(candidate));

  for (const candidate of candidates) {
    const isometric = path.basename(candidate) === "isometric"
      ? candidate
      : path.join(candidate, "isometric");
    if (await exists(path.join(isometric, "move"))) return isometric;
  }
  throw new Error(
    "Catapult asset unavailable. Place purchased 'Catapult - Isometric' at the repository root, "
      + "or set CATAPULT_ASSET_ROOT.",
  );
}

export function createCatapultManifest() {
  const animations = {};
  for (const [animation, frameCount] of Object.entries(CATAPULT_ANIMATION_FRAME_COUNTS)) {
    animations[animation] = Array.from({ length: 8 }, (_, direction) => ({
      direction,
      frames: Array.from(
        { length: frameCount },
        (_, frame) => `/local-assets/catapult/${animation}/dir${direction}/${outputFrameName(frame)}`,
      ),
    }));
  }
  return {
    version: 1,
    asset: "Catapult - Isometric",
    attribution: CATAPULT_ATTRIBUTION,
    sourceFps: "unverified",
    directions: CATAPULT_DIRECTION_LABELS.map((label, index) => ({ index, label })),
    idle: Array.from(
      { length: 8 },
      (_, direction) => `/local-assets/catapult/idle/dir${direction}.png`,
    ),
    animations,
  };
}

async function assertSourceLayout(sourceRoot) {
  for (const [animation, frameCount] of Object.entries(CATAPULT_ANIMATION_FRAME_COUNTS)) {
    for (let direction = 0; direction < 8; direction += 1) {
      for (let frame = 0; frame < frameCount; frame += 1) {
        const source = path.join(sourceRoot, animation, frameName(animation, direction, frame));
        if (!(await exists(source))) {
          throw new Error(
            `Catapult source layout mismatch: missing ${path.relative(sourceRoot, source)}`,
          );
        }
      }
    }
  }
}

export async function prepareCatapultAssets({
  sourceRoot: requestedSourceRoot,
  outputRoot = defaultOutputRoot,
  logger = console,
} = {}) {
  const sourceRoot = await resolveSourceRoot(requestedSourceRoot);
  const resolvedOutput = path.resolve(outputRoot);
  await assertSourceLayout(sourceRoot);

  for (const [animation, frameCount] of Object.entries(CATAPULT_ANIMATION_FRAME_COUNTS)) {
    for (let direction = 0; direction < 8; direction += 1) {
      const destinationDirectory = path.join(resolvedOutput, animation, `dir${direction}`);
      await mkdir(destinationDirectory, { recursive: true });
      for (let frame = 0; frame < frameCount; frame += 1) {
        await copyFile(
          path.join(sourceRoot, animation, frameName(animation, direction, frame)),
          path.join(destinationDirectory, outputFrameName(frame)),
        );
      }
    }
  }

  const idleDirectory = path.join(resolvedOutput, "idle");
  await mkdir(idleDirectory, { recursive: true });
  for (let direction = 0; direction < 8; direction += 1) {
    await copyFile(
      path.join(sourceRoot, "move", frameName("move", direction, 0)),
      path.join(idleDirectory, `dir${direction}.png`),
    );
  }

  const manifest = createCatapultManifest();
  await writeFile(
    path.join(resolvedOutput, "manifest.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf8",
  );
  logger.log(`Prepared Catapult assets from ${sourceRoot}`);
  logger.log(`Output: ${resolvedOutput}`);
  return { sourceRoot, outputRoot: resolvedOutput, manifest };
}

async function main() {
  const sourceArg = process.argv[2];
  const outputArg = process.argv[3];
  await prepareCatapultAssets({
    sourceRoot: sourceArg,
    outputRoot: outputArg ? path.resolve(outputArg) : defaultOutputRoot,
  });
}

const isDirectRun = process.argv[1]
  && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isDirectRun) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
