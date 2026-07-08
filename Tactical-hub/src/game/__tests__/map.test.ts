import { describe, expect, it } from "vitest";
import { TEST_MAP_4P_ROWS, testMap4p } from "../maps/testMap4p";

const expectedRows = [
  ". . . . . . . . . . . . . . . . . . . . . .",
  ". 1 1 G R R P R R G N N G R R P R R G 2 2 .",
  ". 1 1 G ~ ~ P ~ ~ G N N G ~ ~ P ~ ~ G 2 2 .",
  ". G G d ~ ~ ~ ~ ~ ~ R ~ ~ ~ ~ ~ ~ ~ d G G .",
  ". R ~ ~ d ~ ~ ~ ~ ~ R ~ ~ ~ ~ ~ ~ d ~ ~ R .",
  ". R ~ ~ ~ d ~ ~ ~ ~ P P ~ ~ ~ ~ d ~ ~ ~ R .",
  ". P P ~ ~ ~ d ~ ~ ~ R ~ ~ ~ ~ d ~ ~ ~ P P .",
  ". R ~ ~ ~ ~ ~ d ~ ~ R ~ ~ ~ d ~ ~ ~ ~ ~ R .",
  ". R ~ ~ ~ ~ ~ ~ d ~ R ~ ~ d ~ ~ ~ ~ ~ ~ R .",
  ". G G ~ ~ ~ ~ ~ ~ d G ~ d ~ ~ ~ ~ ~ ~ G G .",
  ". N N R R P R R R G C C G R R P R R R N N .",
  ". N N ~ ~ P ~ ~ ~ ~ C C ~ ~ ~ P ~ ~ ~ N N .",
  ". G G ~ ~ ~ ~ ~ ~ d G ~ d ~ ~ ~ ~ ~ ~ G G .",
  ". R ~ ~ ~ ~ ~ ~ d ~ R ~ ~ d ~ ~ ~ ~ ~ ~ R .",
  ". R ~ ~ ~ ~ ~ d ~ ~ R ~ ~ ~ d ~ ~ ~ ~ ~ R .",
  ". P P ~ ~ ~ d ~ ~ ~ R ~ ~ ~ ~ d ~ ~ ~ P P .",
  ". R ~ ~ ~ d ~ ~ ~ ~ P P ~ ~ ~ ~ d ~ ~ ~ R .",
  ". R ~ ~ d ~ ~ ~ ~ ~ R ~ ~ ~ ~ ~ ~ d ~ ~ R .",
  ". G G d ~ ~ ~ ~ ~ ~ R ~ ~ ~ ~ ~ ~ ~ d G G .",
  ". 4 4 G ~ ~ P ~ ~ G N N G ~ ~ P ~ ~ G 3 3 .",
  ". 4 4 G R R P R R G N N G R R P R R G 3 3 .",
  ". . . . . . . . . . . . . . . . . . . . . .",
];

describe("testMap4p", () => {
  it("matches logical map v0.2 exactly", () => {
    expect(TEST_MAP_4P_ROWS).toEqual(expectedRows);
    expect(testMap4p.width).toBe(22);
    expect(testMap4p.height).toBe(22);
  });

  it("treats N and C cells as bases", () => {
    const northTile = testMap4p.tiles.find((tile) => tile.x === 10 && tile.y === 1);
    const centerTile = testMap4p.tiles.find((tile) => tile.x === 10 && tile.y === 10);

    expect(northTile?.terrain).toBe("base");
    expect(northTile?.baseId).toBe("neutral-north");
    expect(centerTile?.terrain).toBe("base");
    expect(centerTile?.baseId).toBe("neutral-center");
  });

  it("keeps outside and lake terrain distinct", () => {
    expect(testMap4p.tiles.find((tile) => tile.x === 0 && tile.y === 0)?.terrain).toBe("outside");
    expect(testMap4p.tiles.find((tile) => tile.x === 4 && tile.y === 2)?.terrain).toBe("lake");
  });
});
