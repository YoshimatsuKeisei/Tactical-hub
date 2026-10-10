#!/usr/bin/env node

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";

export const HD_ENEMY_CHARACTERS = ["6Crusader", "10Caster"];
export const HD_ENEMY_REQUIRED_SHEETS = ["Idle.png", "Attack1.png", "TakeDamage.png", "Die.png"];
export const HD_ENEMY_FRAME_WIDTH = 128;
export const HD_ENEMY_FRAME_HEIGHT = 128;
export const HD_ENEMY_COLUMNS = 15;
export const HD_ENEMY_ROWS = 8;
export const HD_ENEMY_SHEET_WIDTH = HD_ENEMY_FRAME_WIDTH * HD_ENEMY_COLUMNS;
export const HD_ENEMY_SHEET_HEIGHT = HD_ENEMY_FRAME_HEIGHT * HD_ENEMY_ROWS;
export const HD_ENEMY_DIRECTION_ROWS = [6, 7, 0, 1, 2, 3, 4, 5];

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const DEFAULT_HD_ENEMY_ZIP = path.join(packageRoot, "2D HD Enemy pack 1.zip");
export const DEFAULT_HD_ENEMY_OUTPUT = path.join(packageRoot, "public", "local-assets", "hd-enemy");

function archivePath(character, filename) {
  return `2D HD Enemy pack 1/Spritesheets/With shadow/${character}/${filename}`;
}

export function getRequiredHdEnemyEntries() {
  return HD_ENEMY_CHARACTERS.flatMap((character) => (
    HD_ENEMY_REQUIRED_SHEETS.map((filename) => ({
      character,
      filename,
      archivePath: archivePath(character, filename),
    }))
  ));
}

export function readPngDimensions(buffer, label = "PNG") {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (buffer.length < 24 || !buffer.subarray(0, 8).equals(signature)
    || buffer.toString("ascii", 12, 16) !== "IHDR") {
    throw new Error(`${label} is not a valid PNG with an IHDR header.`);
  }
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

export function createHdEnemyManifest() {
  const stateMapping = {
    idle: "Idle.png",
    attack: "Attack1.png",
    hit: "TakeDamage.png",
    die: "Die.png",
  };
  return {
    version: 1,
    asset: "2D HD Enemy pack 1",
    sourceDirectory: "Spritesheets/With shadow",
    sourceFps: "unverified",
    frameWidth: HD_ENEMY_FRAME_WIDTH,
    frameHeight: HD_ENEMY_FRAME_HEIGHT,
    columns: HD_ENEMY_COLUMNS,
    rows: HD_ENEMY_ROWS,
    directionRows: HD_ENEMY_DIRECTION_ROWS,
    stateMapping,
    characters: Object.fromEntries(HD_ENEMY_CHARACTERS.map((character) => [
      character,
      { sheets: { ...stateMapping } },
    ])),
  };
}

export async function prepareHdEnemyAssets({
  zipPath = DEFAULT_HD_ENEMY_ZIP,
  outputRoot = DEFAULT_HD_ENEMY_OUTPUT,
  logger = console,
} = {}) {
  const resolvedZip = path.resolve(zipPath);
  const resolvedOutput = path.resolve(outputRoot);
  let zipBytes;
  try {
    zipBytes = await readFile(resolvedZip);
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
      throw new Error(
        `HD Enemy asset ZIP unavailable: ${resolvedZip}. Place '2D HD Enemy pack 1.zip' beside package.json.`,
      );
    }
    throw error;
  }

  const zip = new AdmZip(zipBytes);
  const entries = zip.getEntries();
  const prepared = [];
  for (const required of getRequiredHdEnemyEntries()) {
    const matches = entries.filter((entry) => !entry.isDirectory && entry.entryName === required.archivePath);
    if (matches.length !== 1) {
      throw new Error(
        `HD Enemy ZIP layout mismatch: expected exactly one '${required.archivePath}', found ${matches.length}.`,
      );
    }
    const bytes = matches[0].getData();
    const dimensions = readPngDimensions(bytes, required.archivePath);
    if (dimensions.width !== HD_ENEMY_SHEET_WIDTH || dimensions.height !== HD_ENEMY_SHEET_HEIGHT) {
      throw new Error(
        `HD Enemy sheet dimensions mismatch for ${required.archivePath}: `
          + `expected ${HD_ENEMY_SHEET_WIDTH}x${HD_ENEMY_SHEET_HEIGHT}, `
          + `found ${dimensions.width}x${dimensions.height}.`,
      );
    }
    prepared.push({ ...required, bytes });
  }

  // Output paths are built only from the fixed whitelist above. Archive paths are
  // never joined to the filesystem destination, so unrelated/traversal entries
  // cannot escape or be extracted.
  for (const entry of prepared) {
    const destinationDirectory = path.join(resolvedOutput, entry.character);
    await mkdir(destinationDirectory, { recursive: true });
    await writeFile(path.join(destinationDirectory, entry.filename), entry.bytes);
  }
  const manifest = createHdEnemyManifest();
  await mkdir(resolvedOutput, { recursive: true });
  await writeFile(path.join(resolvedOutput, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  logger.log(`Prepared 8 HD Enemy sheets from ${resolvedZip}`);
  logger.log(`Output: ${resolvedOutput}`);
  return { zipPath: resolvedZip, outputRoot: resolvedOutput, manifest };
}

async function main() {
  await prepareHdEnemyAssets({
    zipPath: process.argv[2] ? path.resolve(process.argv[2]) : DEFAULT_HD_ENEMY_ZIP,
    outputRoot: process.argv[3] ? path.resolve(process.argv[3]) : DEFAULT_HD_ENEMY_OUTPUT,
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
