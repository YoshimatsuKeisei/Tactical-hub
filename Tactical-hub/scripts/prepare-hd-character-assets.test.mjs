import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_HD_CHARACTER_ZIP,
  HD_CHARACTER_DIRECTION_ROWS,
  HD_CHARACTER_SHEETS,
  createHdCharacterManifest,
  getRequiredHdCharacterEntries,
  prepareHdCharacterAssets,
} from "./prepare-hd-character-assets.mjs";

function syntheticPng(width = 1920, height = 1024, marker = 0) {
  const bytes = Buffer.alloc(32, marker);
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(bytes, 0);
  bytes.writeUInt32BE(13, 8);
  bytes.write("IHDR", 12, "ascii");
  bytes.writeUInt32BE(width, 16);
  bytes.writeUInt32BE(height, 20);
  return bytes;
}

function createFixtureZip({ missing, dimensions } = {}) {
  const zip = new AdmZip();
  getRequiredHdCharacterEntries().forEach((entry, index) => {
    if (entry.archivePath === missing) return;
    const size = dimensions?.[entry.archivePath];
    zip.addFile(entry.archivePath, syntheticPng(size?.width, size?.height, index));
  });
  zip.addFile("2D HD Character pack 1 V1.2/unrelated.png", syntheticPng());
  zip.addFile("2D HD Character pack 1 V1.2/../../must-not-extract.txt", Buffer.from("blocked"));
  return zip.toBuffer();
}

async function listFiles(root) {
  const result = [];
  async function visit(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const full = path.join(directory, entry.name);
      if (entry.isDirectory()) await visit(full);
      else result.push(path.relative(root, full).replaceAll(path.sep, "/"));
    }
  }
  await visit(root);
  return result.sort();
}

describe("HD Character local asset preparation", () => {
  it("defaults to the purchased ZIP beside package.json", () => {
    const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
    expect(DEFAULT_HD_CHARACTER_ZIP).toBe(path.join(packageRoot, "2D HD Character pack 1 V1.2.zip"));
  });

  it("whitelists the exact sixteen shadow sheets", () => {
    expect(HD_CHARACTER_SHEETS).toEqual({
      "1Knight": ["Idle.png", "Melee.png", "TakeDamage.png", "Die.png"],
      "2Archer": ["Idle.png", "Attack1.png", "TakeDamage.png", "Die.png"],
      "4Paladin": ["Idle.png", "Melee.png", "TakeDamage.png", "Die.png"],
      "7DeathKnight": ["Idle.png", "Melee.png", "TakeDamage.png", "Die.png"],
    });
    expect(getRequiredHdCharacterEntries()).toHaveLength(16);
    expect(getRequiredHdCharacterEntries().find((entry) => entry.character === "1Knight" && entry.filename === "Idle.png")?.archivePath)
      .toBe("Spritesheets/With shadow/1Knight/Idle.png");
  });

  it("byte-copies exactly sixteen sheets plus the manifest", async () => {
    const temporary = await mkdtemp(path.join(os.tmpdir(), "hd-character-assets-test-"));
    try {
      const zipPath = path.join(temporary, "pack.zip");
      const outputRoot = path.join(temporary, "prepared");
      const zipBytes = createFixtureZip();
      await writeFile(zipPath, zipBytes);
      await prepareHdCharacterAssets({ zipPath, outputRoot, logger: { log: () => undefined } });

      const files = await listFiles(outputRoot);
      expect(files).toHaveLength(17);
      expect(files).toContain("1Knight/Melee.png");
      expect(files).toContain("2Archer/Attack1.png");
      expect(files).toContain("4Paladin/Die.png");
      expect(files).toContain("7DeathKnight/TakeDamage.png");
      expect(files).not.toContain("unrelated.png");
      expect(files).not.toContain("must-not-extract.txt");

      const sourceZip = new AdmZip(zipBytes);
      const required = getRequiredHdCharacterEntries()[0];
      expect(await readFile(path.join(outputRoot, required.character, required.filename)))
        .toEqual(sourceZip.getEntry(required.archivePath)?.getData());
      const manifest = JSON.parse(await readFile(path.join(outputRoot, "manifest.json"), "utf8"));
      expect(manifest).toEqual(createHdCharacterManifest());
      expect(manifest).toMatchObject({
        sourceFps: "unverified",
        frameWidth: 128,
        frameHeight: 128,
        columns: 15,
        rows: 8,
        directionRows: [6, 7, 0, 1, 2, 3, 4, 5],
        unitMapping: {
          normalInfantry: "1Knight",
          heavyInfantry: "4Paladin",
          archer: "2Archer",
          ninja: "7DeathKnight",
        },
      });
      expect(HD_CHARACTER_DIRECTION_ROWS).toEqual([6, 7, 0, 1, 2, 3, 4, 5]);
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });

  it("reports a missing purchased ZIP clearly", async () => {
    const temporary = await mkdtemp(path.join(os.tmpdir(), "hd-character-missing-test-"));
    try {
      await expect(prepareHdCharacterAssets({
        zipPath: path.join(temporary, "missing.zip"),
        outputRoot: path.join(temporary, "output"),
      })).rejects.toThrow("HD Character asset ZIP unavailable");
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });

  it("rejects a missing required entry before writing output", async () => {
    const temporary = await mkdtemp(path.join(os.tmpdir(), "hd-character-layout-test-"));
    try {
      const required = getRequiredHdCharacterEntries()[5];
      const zipPath = path.join(temporary, "pack.zip");
      await writeFile(zipPath, createFixtureZip({ missing: required.archivePath }));
      await expect(prepareHdCharacterAssets({
        zipPath,
        outputRoot: path.join(temporary, "output"),
      })).rejects.toThrow(required.archivePath);
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });

  it("rejects a sheet whose dimensions are not 1920x1024", async () => {
    const temporary = await mkdtemp(path.join(os.tmpdir(), "hd-character-size-test-"));
    try {
      const required = getRequiredHdCharacterEntries()[0];
      const zipPath = path.join(temporary, "pack.zip");
      await writeFile(zipPath, createFixtureZip({
        dimensions: { [required.archivePath]: { width: 1920, height: 1023 } },
      }));
      await expect(prepareHdCharacterAssets({
        zipPath,
        outputRoot: path.join(temporary, "output"),
      })).rejects.toThrow("expected 1920x1024, found 1920x1023");
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });
});
