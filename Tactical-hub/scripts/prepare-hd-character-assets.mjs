#!/usr/bin/env node

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";
import { readPngDimensions } from "./prepare-hd-enemy-assets.mjs";

export const HD_CHARACTER_SHEETS = {
  "1Knight": ["Idle.png", "Melee.png", "TakeDamage.png", "Die.png"],
  "2Archer": ["Idle.png", "Attack1.png", "TakeDamage.png", "Die.png"],
  "4Paladin": ["Idle.png", "Melee.png", "TakeDamage.png", "Die.png"],
  "7DeathKnight": ["Idle.png", "Melee.png", "TakeDamage.png", "Die.png"],
};
export const HD_CHARACTER_FRAME_WIDTH = 128;
export const HD_CHARACTER_FRAME_HEIGHT = 128;
export const HD_CHARACTER_COLUMNS = 15;
export const HD_CHARACTER_ROWS = 8;
export const HD_CHARACTER_SHEET_WIDTH = HD_CHARACTER_FRAME_WIDTH * HD_CHARACTER_COLUMNS;
export const HD_CHARACTER_SHEET_HEIGHT = HD_CHARACTER_FRAME_HEIGHT * HD_CHARACTER_ROWS;
export const HD_CHARACTER_DIRECTION_ROWS = [6, 7, 0, 1, 2, 3, 4, 5];
export const HD_CHARACTER_UNIT_MAPPING = {
  normalInfantry: "1Knight",
  archer: "2Archer",
  heavyInfantry: "4Paladin",
  ninja: "7DeathKnight",
};

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const DEFAULT_HD_CHARACTER_ZIP = path.join(packageRoot, "2D HD Character pack 1 V1.2.zip");
export const DEFAULT_HD_CHARACTER_OUTPUT = path.join(packageRoot, "public", "local-assets", "hd-character");

function archivePath(character, filename) {
  return `2D HD Character pack 1 V1.2/Spritesheets/With shadow/${character}/${filename}`;
}

export function getRequiredHdCharacterEntries() {
  return Object.entries(HD_CHARACTER_SHEETS).flatMap(([character, filenames]) => (
    filenames.map((filename) => ({ character, filename, archivePath: archivePath(character, filename) }))
  ));
}

export function createHdCharacterManifest() {
  return {
    version: 1,
    asset: "2D HD Character pack 1 V1.2",
    sourceDirectory: "Spritesheets/With shadow",
    sourceFps: "unverified",
    frameWidth: HD_CHARACTER_FRAME_WIDTH,
    frameHeight: HD_CHARACTER_FRAME_HEIGHT,
    columns: HD_CHARACTER_COLUMNS,
    rows: HD_CHARACTER_ROWS,
    directionRows: HD_CHARACTER_DIRECTION_ROWS,
    unitMapping: HD_CHARACTER_UNIT_MAPPING,
    characters: Object.fromEntries(Object.entries(HD_CHARACTER_SHEETS).map(([character, filenames]) => [
      character,
      {
        sheets: {
          idle: "Idle.png",
          attack: filenames.includes("Attack1.png") ? "Attack1.png" : "Melee.png",
          hit: "TakeDamage.png",
          die: "Die.png",
        },
      },
    ])),
  };
}

export async function prepareHdCharacterAssets({
  zipPath = DEFAULT_HD_CHARACTER_ZIP,
  outputRoot = DEFAULT_HD_CHARACTER_OUTPUT,
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
        `HD Character asset ZIP unavailable: ${resolvedZip}. `
          + "Place '2D HD Character pack 1 V1.2.zip' beside package.json.",
      );
    }
    throw error;
  }

  const zip = new AdmZip(zipBytes);
  const entries = zip.getEntries();
  const prepared = [];
  for (const required of getRequiredHdCharacterEntries()) {
    const matches = entries.filter((entry) => !entry.isDirectory && entry.entryName === required.archivePath);
    if (matches.length !== 1) {
      throw new Error(
        `HD Character ZIP layout mismatch: expected exactly one '${required.archivePath}', found ${matches.length}.`,
      );
    }
    const bytes = matches[0].getData();
    const dimensions = readPngDimensions(bytes, required.archivePath);
    if (dimensions.width !== HD_CHARACTER_SHEET_WIDTH || dimensions.height !== HD_CHARACTER_SHEET_HEIGHT) {
      throw new Error(
        `HD Character sheet dimensions mismatch for ${required.archivePath}: `
          + `expected ${HD_CHARACTER_SHEET_WIDTH}x${HD_CHARACTER_SHEET_HEIGHT}, `
          + `found ${dimensions.width}x${dimensions.height}.`,
      );
    }
    prepared.push({ ...required, bytes });
  }

  // Archive entry names are validated against the fixed whitelist above. Only
  // fixed character/filename values form destinations, preventing traversal.
  for (const entry of prepared) {
    const destinationDirectory = path.join(resolvedOutput, entry.character);
    await mkdir(destinationDirectory, { recursive: true });
    await writeFile(path.join(destinationDirectory, entry.filename), entry.bytes);
  }
  const manifest = createHdCharacterManifest();
  await mkdir(resolvedOutput, { recursive: true });
  await writeFile(path.join(resolvedOutput, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  logger.log(`Prepared 16 HD Character sheets from ${resolvedZip}`);
  logger.log(`Output: ${resolvedOutput}`);
  return { zipPath: resolvedZip, outputRoot: resolvedOutput, manifest };
}

async function main() {
  await prepareHdCharacterAssets({
    zipPath: process.argv[2] ? path.resolve(process.argv[2]) : DEFAULT_HD_CHARACTER_ZIP,
    outputRoot: process.argv[3] ? path.resolve(process.argv[3]) : DEFAULT_HD_CHARACTER_OUTPUT,
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
