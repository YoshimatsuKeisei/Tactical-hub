import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  CATAPULT_ANIMATION_FRAME_COUNTS,
  createCatapultManifest,
  prepareCatapultAssets,
} from "./prepare-catapult-assets.mjs";

async function makeSourceFixture(root) {
  const isometric = path.join(root, "Catapult - Isometric", "isometric");
  for (const [animation, frameCount] of Object.entries(CATAPULT_ANIMATION_FRAME_COUNTS)) {
    const directory = path.join(isometric, animation);
    await mkdir(directory, { recursive: true });
    for (let direction = 0; direction < 8; direction += 1) {
      for (let frame = 0; frame < frameCount; frame += 1) {
        const filename = `catapult_${animation}_${direction}${String(frame).padStart(4, "0")}.png`;
        await writeFile(path.join(directory, filename), Buffer.from(`${animation}:${direction}:${frame}`));
      }
    }
  }
  return isometric;
}

describe("Catapult local asset preparation", () => {
  it("recognizes 8 directions and the exact source frame counts", () => {
    const manifest = createCatapultManifest();
    expect(manifest.directions).toHaveLength(8);
    for (const direction of manifest.directions) {
      expect(manifest.animations.move[direction.index].frames).toHaveLength(31);
      expect(manifest.animations.load[direction.index].frames).toHaveLength(31);
      expect(manifest.animations.break[direction.index].frames).toHaveLength(31);
      expect(manifest.animations.throw[direction.index].frames).toHaveLength(16);
      expect(manifest.idle[direction.index]).toBe(
        `/local-assets/catapult/idle/dir${direction.index}.png`,
      );
    }
  });

  it("byte-copies frames and maps idle to move frame 0000", async () => {
    const temporary = await mkdtemp(path.join(os.tmpdir(), "catapult-assets-test-"));
    try {
      const source = await makeSourceFixture(temporary);
      const output = path.join(temporary, "prepared");
      await prepareCatapultAssets({
        sourceRoot: source,
        outputRoot: output,
        logger: { log: () => undefined },
      });

      const moveSource = await readFile(path.join(source, "move", "catapult_move_30000.png"));
      const moveOutput = await readFile(path.join(output, "move", "dir3", "0000.png"));
      const idleOutput = await readFile(path.join(output, "idle", "dir3.png"));
      expect(moveOutput).toEqual(moveSource);
      expect(idleOutput).toEqual(moveSource);

      const manifest = JSON.parse(await readFile(path.join(output, "manifest.json"), "utf8"));
      expect(manifest.sourceFps).toBe("unverified");
      expect(manifest.attribution).toBe("Bleed - http://remusprites.carbonmade.com/");
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });

  it("rejects an incomplete purchased-pack directory before preparing output", async () => {
    const temporary = await mkdtemp(path.join(os.tmpdir(), "catapult-assets-invalid-test-"));
    try {
      const source = path.join(temporary, "Catapult - Isometric", "isometric");
      await mkdir(path.join(source, "move"), { recursive: true });
      await expect(prepareCatapultAssets({
        sourceRoot: source,
        outputRoot: path.join(temporary, "prepared"),
        logger: { log: () => undefined },
      })).rejects.toThrow("missing move/catapult_move_00000.png");
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });
});
