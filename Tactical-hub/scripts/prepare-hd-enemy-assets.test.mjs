import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_HD_ENEMY_ZIP,
  HD_ENEMY_CHARACTERS,
  HD_ENEMY_DIRECTION_ROWS,
  HD_ENEMY_REQUIRED_SHEETS,
  createHdEnemyManifest,
  getRequiredHdEnemyEntries,
  prepareHdEnemyAssets,
} from "./prepare-hd-enemy-assets.mjs";

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
  getRequiredHdEnemyEntries().forEach((entry, index) => {
    if (entry.archivePath === missing) return;
    const size = dimensions?.[entry.archivePath];
    zip.addFile(
      entry.archivePath,
      syntheticPng(size?.width, size?.height, index),
    );
  });
  zip.addFile("2D HD Enemy pack 1/unrelated.png", syntheticPng());
  zip.addFile("2D HD Enemy pack 1/../../must-not-extract.txt", Buffer.from("blocked"));
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

describe("HD Enemy local asset preparation", () => {
  it("defaults to the purchased ZIP beside package.json", () => {
    const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
    expect(DEFAULT_HD_ENEMY_ZIP).toBe(path.join(packageRoot, "2D HD Enemy pack 1.zip"));
  });

  it("defines only the two characters and four required shadow sheets", () => {
    expect(HD_ENEMY_CHARACTERS).toEqual(["6Crusader", "10Caster"]);
    expect(HD_ENEMY_REQUIRED_SHEETS).toEqual(["Idle.png", "Attack1.png", "TakeDamage.png", "Die.png"]);
    expect(getRequiredHdEnemyEntries()).toHaveLength(8);
    expect(HD_ENEMY_DIRECTION_ROWS).toEqual([6, 7, 0, 1, 2, 3, 4, 5]);
  });

  it("byte-copies exactly eight whitelisted sheets and writes the manifest", async () => {
    const temporary = await mkdtemp(path.join(os.tmpdir(), "hd-enemy-assets-test-"));
    try {
      const zipPath = path.join(temporary, "pack.zip");
      const outputRoot = path.join(temporary, "prepared");
      const zipBytes = createFixtureZip();
      await import("node:fs/promises").then(({ writeFile }) => writeFile(zipPath, zipBytes));
      await prepareHdEnemyAssets({ zipPath, outputRoot, logger: { log: () => undefined } });

      const files = await listFiles(outputRoot);
      expect(files).toHaveLength(9);
      expect(files).toContain("6Crusader/Idle.png");
      expect(files).toContain("10Caster/Die.png");
      expect(files).not.toContain("unrelated.png");
      expect(files).not.toContain("must-not-extract.txt");

      const sourceZip = new AdmZip(zipBytes);
      const expected = sourceZip.getEntry(getRequiredHdEnemyEntries()[0].archivePath)?.getData();
      expect(await readFile(path.join(outputRoot, "6Crusader", "Idle.png"))).toEqual(expected);
      const manifest = JSON.parse(await readFile(path.join(outputRoot, "manifest.json"), "utf8"));
      expect(manifest).toEqual(createHdEnemyManifest());
      expect(manifest.sourceFps).toBe("unverified");
      expect(manifest).toMatchObject({
        frameWidth: 128,
        frameHeight: 128,
        columns: 15,
        rows: 8,
        directionRows: [6, 7, 0, 1, 2, 3, 4, 5],
      });
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });

  it("reports a missing purchased ZIP clearly", async () => {
    const temporary = await mkdtemp(path.join(os.tmpdir(), "hd-enemy-missing-test-"));
    try {
      await expect(prepareHdEnemyAssets({
        zipPath: path.join(temporary, "missing.zip"),
        outputRoot: path.join(temporary, "output"),
      })).rejects.toThrow("HD Enemy asset ZIP unavailable");
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });

  it("rejects a missing required entry before writing output", async () => {
    const temporary = await mkdtemp(path.join(os.tmpdir(), "hd-enemy-layout-test-"));
    try {
      const required = getRequiredHdEnemyEntries()[3];
      const zipPath = path.join(temporary, "pack.zip");
      await import("node:fs/promises").then(({ writeFile }) => writeFile(
        zipPath,
        createFixtureZip({ missing: required.archivePath }),
      ));
      await expect(prepareHdEnemyAssets({
        zipPath,
        outputRoot: path.join(temporary, "output"),
      })).rejects.toThrow(required.archivePath);
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });

  it("rejects a sheet whose dimensions are not 1920x1024", async () => {
    const temporary = await mkdtemp(path.join(os.tmpdir(), "hd-enemy-size-test-"));
    try {
      const required = getRequiredHdEnemyEntries()[0];
      const zipPath = path.join(temporary, "pack.zip");
      await import("node:fs/promises").then(({ writeFile }) => writeFile(
        zipPath,
        createFixtureZip({ dimensions: { [required.archivePath]: { width: 1919, height: 1024 } } }),
      ));
      await expect(prepareHdEnemyAssets({
        zipPath,
        outputRoot: path.join(temporary, "output"),
      })).rejects.toThrow("expected 1920x1024, found 1919x1024");
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });
});
