import type { Base, BoardMap, TerrainType, Tile } from "../types";

export const TEST_MAP_4P_ROWS = [
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

const homeProtectedSlots: Record<string, { localRow: 0 | 1; localCol: 0 | 1 }> = {
  "home-1": { localRow: 0, localCol: 0 },
  "home-2": { localRow: 0, localCol: 1 },
  "home-3": { localRow: 1, localCol: 1 },
  "home-4": { localRow: 1, localCol: 0 },
};

function makeSlots(baseId: string, home: boolean) {
  const protectedSlot = homeProtectedSlots[baseId];
  return ([0, 1] as const).flatMap((localRow) =>
    ([0, 1] as const).map((localCol) => {
      const isProtected =
        Boolean(protectedSlot) &&
        protectedSlot.localRow === localRow &&
        protectedSlot.localCol === localCol;
      return {
        id: `slot_${localRow}_${localCol}`,
        baseId,
        kind: isProtected ? ("protected" as const) : ("front" as const),
        localRow,
        localCol,
      };
    }),
  );
}

function terrainFor(symbol: string): TerrainType {
  if (symbol === ".") return "outside";
  if (symbol === "~") return "lake";
  if (symbol === "G") return "baseGate";
  if (symbol === "P") return "reorganize";
  if (["1", "2", "3", "4", "N", "C"].includes(symbol)) return "base";
  return "road";
}

type BaseSpec =
  | { id: string; name: string; symbol: string; ownerTeamId: string; type: "home" }
  | { id: string; name: string; coords: { x: number; y: number }[] };

const baseSpecs: BaseSpec[] = [
  { id: "home-1", name: "Team 1 Home", symbol: "1", ownerTeamId: "team-1", type: "home" as const },
  { id: "home-2", name: "Team 2 Home", symbol: "2", ownerTeamId: "team-2", type: "home" as const },
  { id: "home-3", name: "Team 3 Home", symbol: "3", ownerTeamId: "team-3", type: "home" as const },
  { id: "home-4", name: "Team 4 Home", symbol: "4", ownerTeamId: "team-4", type: "home" as const },
  { id: "neutral-north", name: "North Relay", coords: [{ x: 10, y: 1 }, { x: 11, y: 1 }, { x: 10, y: 2 }, { x: 11, y: 2 }] },
  { id: "neutral-east", name: "East Relay", coords: [{ x: 19, y: 10 }, { x: 20, y: 10 }, { x: 19, y: 11 }, { x: 20, y: 11 }] },
  { id: "neutral-south", name: "South Relay", coords: [{ x: 10, y: 19 }, { x: 11, y: 19 }, { x: 10, y: 20 }, { x: 11, y: 20 }] },
  { id: "neutral-west", name: "West Relay", coords: [{ x: 1, y: 10 }, { x: 2, y: 10 }, { x: 1, y: 11 }, { x: 2, y: 11 }] },
  { id: "neutral-center", name: "Center Relay", coords: [{ x: 10, y: 10 }, { x: 11, y: 10 }, { x: 10, y: 11 }, { x: 11, y: 11 }] },
];

const parsed = TEST_MAP_4P_ROWS.map((row) => row.split(" "));

const bases: Base[] = baseSpecs.map((spec) => {
  if ("symbol" in spec) {
    const coords = parsed.flatMap((row, y) =>
      row.flatMap((symbol, x) => (symbol === spec.symbol ? [{ x, y }] : [])),
    );
    return {
      id: spec.id,
      name: spec.name,
      type: spec.type,
      ownerTeamId: spec.ownerTeamId,
      coords,
      slots: makeSlots(spec.id, true),
      protectedSlotId: `slot_${homeProtectedSlots[spec.id].localRow}_${homeProtectedSlots[spec.id].localCol}`,
    };
  }

  return {
    id: spec.id,
    name: spec.name,
    type: "neutral",
    ownerTeamId: "neutral",
    coords: spec.coords,
    slots: makeSlots(spec.id, false),
  };
});

const tiles: Tile[] = parsed.flatMap((row, y) =>
  row.map((symbol, x) => {
    const base = bases.find((candidate) => candidate.coords.some((coord) => coord.x === x && coord.y === y));
    return { x, y, symbol, terrain: terrainFor(symbol), baseId: base?.id };
  }),
);

export const testMap4p: BoardMap = {
  id: "test-map-4p-v0.2",
  name: "Test Map 4P v0.2",
  width: parsed[0].length,
  height: parsed.length,
  tiles,
  bases,
};
