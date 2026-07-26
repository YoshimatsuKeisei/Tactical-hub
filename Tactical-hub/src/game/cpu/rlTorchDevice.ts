export type RlTorchDevice = "auto" | "cpu" | "cuda";
export type RlSelectedTorchDevice = Exclude<RlTorchDevice, "auto">;

export function parseRlTorchDevice(value: string): RlTorchDevice {
  if (value !== "auto" && value !== "cpu" && value !== "cuda") {
    throw new Error("device must be auto, cpu, or cuda");
  }
  return value;
}
