import type { CSSProperties } from "react";

export type NineSliceInsets = Readonly<{
  top: number;
  right: number;
  bottom: number;
  left: number;
}>;

export type NineSliceSkin = Readonly<{
  image: string;
  slice: NineSliceInsets;
  borderWidth?: NineSliceInsets;
  padding?: CSSProperties["padding"];
  repeat?: "stretch" | "repeat" | "round" | "space";
}>;

export type NineSliceLayout = Readonly<{
  width?: CSSProperties["width"];
  height?: CSSProperties["height"];
  padding?: CSSProperties["padding"];
}>;

function toCssQuad({ top, right, bottom, left }: NineSliceInsets) {
  return `${top} ${right} ${bottom} ${left}`;
}

function toPixelQuad(insets: NineSliceInsets) {
  return `${insets.top}px ${insets.right}px ${insets.bottom}px ${insets.left}px`;
}

export function createNineSliceStyle(
  skin: NineSliceSkin,
  layout: NineSliceLayout = {},
): CSSProperties {
  const borderWidth = skin.borderWidth ?? skin.slice;

  return {
    width: layout.width,
    height: layout.height,
    padding: layout.padding ?? skin.padding,
    borderStyle: "solid",
    borderColor: "transparent",
    borderWidth: toPixelQuad(borderWidth),
    borderImageSource: `url("${skin.image}")`,
    borderImageSlice: `${toCssQuad(skin.slice)} fill`,
    borderImageWidth: toPixelQuad(borderWidth),
    borderImageRepeat: skin.repeat ?? "stretch",
  };
}
