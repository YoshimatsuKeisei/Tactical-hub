import type { BcEncodedSample } from "./pythonBcTrainerClient";
import type { RlFeatureSpec } from "./rlFeatureSpec";

export type PackedTensorDescriptor = {
  name: string;
  dtype: "float32" | "int32" | "uint8";
  shape: number[];
  byteOffset: number;
  byteLength: number;
};

type PendingTensor = Omit<PackedTensorDescriptor, "byteOffset" | "byteLength"> & { bytes: Buffer };
const strategicNames = [
  "siegeStates", "kingCampaignStates", "rewardPlacementRequests", "strategistCooldowns",
  "teleportCooldowns", "productionIntents", "movementIntents", "attackIntents",
  "strategistActionIntents", "teleportIntents",
] as const;

function floatMatrix(name: string, rows: number[][]): PendingTensor {
  const width = rows[0]?.length ?? 0;
  const values = new Float32Array(rows.length * width);
  rows.forEach((row, index) => values.set(row, index * width));
  return { name, dtype: "float32", shape: [rows.length, width], bytes: Buffer.from(values.buffer) };
}

function paddedFloatRows(name: string, values: number[][][], width: number): [PendingTensor, Uint8Array] {
  const maxRows = Math.max(0, ...values.map((rows) => rows.length));
  const packed = new Float32Array(values.length * maxRows * width);
  const presence = new Uint8Array(values.length * maxRows);
  values.forEach((rows, batch) => rows.forEach((row, rowIndex) => {
    packed.set(row, (batch * maxRows + rowIndex) * width);
    presence[batch * maxRows + rowIndex] = 1;
  }));
  return [{ name, dtype: "float32", shape: [values.length, maxRows, width], bytes: Buffer.from(packed.buffer) }, presence];
}

function maskTensor(name: string, masks: number[][], maxRows: number, presence?: Uint8Array): PendingTensor {
  const packed = new Uint8Array(masks.length * maxRows);
  masks.forEach((mask, batch) => mask.slice(0, maxRows).forEach((value, row) => {
    packed[batch * maxRows + row] = Number(Boolean(value));
  }));
  if (presence) for (let index = 0; index < packed.length; index += 1) packed[index] &= presence[index];
  return { name, dtype: "uint8", shape: [masks.length, maxRows], bytes: Buffer.from(packed.buffer) };
}

export type PackedBcBatch = {
  payload: Buffer;
  tensors: PackedTensorDescriptor[];
  batchSize: number;
};

export function packBcEncodedSamples(samples: BcEncodedSample[], featureSpec: RlFeatureSpec): PackedBcBatch {
  if (!samples.length) throw new Error("Cannot pack an empty BC batch");
  const observations = samples.map((sample) => sample.observation);
  const floats: PendingTensor[] = [];
  const masks: PendingTensor[] = [];
  const checkedMatrix = (name: string, rows: number[][], width: number) => {
    if (rows.some((row) => row.length !== width)) throw new Error(`${name} feature width does not match Feature Spec`);
    return floatMatrix(name, rows);
  };
  floats.push(checkedMatrix("global", observations.map((value) => value.global), featureSpec.globalWidth));
  floats.push(checkedMatrix("strategicGlobal", observations.map((value) => value.strategicState.global), featureSpec.strategicGlobalWidth));

  for (const [name, maskName, width] of [
    ["teams", "teamMask", featureSpec.teamWidth],
    ["units", "unitMask", featureSpec.unitWidth],
    ["bases", "baseMask", featureSpec.baseWidth],
    ["constructions", "constructionMask", featureSpec.constructionWidth],
  ] as const) {
    const rows = observations.map((value) => value[name]);
    if (rows.some((batch) => batch.some((row) => row.length !== width))) throw new Error(`${name} feature width does not match Feature Spec`);
    const [tensor, presence] = paddedFloatRows(name, rows, width);
    floats.push(tensor);
    masks.push(maskTensor(maskName, observations.map((value) => value[maskName]), tensor.shape[1], presence));
  }
  const mapRows = observations.map((value) => value.map.flat());
  if (mapRows.some((batch) => batch.some((row) => row.length !== featureSpec.mapTileWidth))) throw new Error("map feature width does not match Feature Spec");
  const [map, mapPresence] = paddedFloatRows("map", mapRows, featureSpec.mapTileWidth);
  floats.push(map);
  masks.push({ name: "mapMask", dtype: "uint8", shape: [samples.length, map.shape[1]], bytes: Buffer.from(mapPresence.buffer) });

  for (const name of strategicNames) {
    const rows = observations.map((value) => value.strategicState[name]);
    const width = featureSpec.strategicTableRowWidths[name];
    if (rows.some((batch) => batch.some((row) => row.length !== width))) throw new Error(`${name} feature width does not match Feature Spec`);
    const [tensor, presence] = paddedFloatRows(`strategic.${name}`, rows, width);
    floats.push(tensor);
    masks.push({ name: `strategicMask.${name}`, dtype: "uint8", shape: [samples.length, tensor.shape[1]], bytes: Buffer.from(presence.buffer) });
  }
  const actionRows = samples.map((sample) => sample.actions);
  if (actionRows.some((batch) => batch.some((row) => row.length !== featureSpec.actionFeatureWidth))) {
    throw new Error("action feature width does not match Feature Spec");
  }
  const [actions, actionPresence] = paddedFloatRows("actions", actionRows, featureSpec.actionFeatureWidth);
  floats.push(actions);
  masks.push({ name: "actionMask", dtype: "uint8", shape: [samples.length, actions.shape[1]], bytes: Buffer.from(actionPresence.buffer) });
  const targets = Int32Array.from(samples.map((sample) => sample.targetIndex));
  const integers: PendingTensor[] = [{ name: "targets", dtype: "int32", shape: [samples.length], bytes: Buffer.from(targets.buffer) }];

  let byteOffset = 0;
  const tensors: PackedTensorDescriptor[] = [];
  const buffers: Buffer[] = [];
  for (const tensor of [...floats, ...integers, ...masks]) {
    tensors.push({ name: tensor.name, dtype: tensor.dtype, shape: tensor.shape, byteOffset, byteLength: tensor.bytes.byteLength });
    buffers.push(tensor.bytes);
    byteOffset += tensor.bytes.byteLength;
  }
  return { payload: Buffer.concat(buffers, byteOffset), tensors, batchSize: samples.length };
}
