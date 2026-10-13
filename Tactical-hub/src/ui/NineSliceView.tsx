import type {
  ComponentPropsWithoutRef,
  CSSProperties,
  ElementType,
  ReactNode,
} from "react";
import { createNineSliceStyle, type NineSliceSkin } from "./nineSlice";

type NineSliceOwnProps<T extends ElementType> = {
  as?: T;
  skin: NineSliceSkin;
  width?: CSSProperties["width"];
  height?: CSSProperties["height"];
  padding?: CSSProperties["padding"];
  className?: string;
  style?: CSSProperties;
  children?: ReactNode;
};

export type NineSliceProps<T extends ElementType = "div"> = NineSliceOwnProps<T>
  & Omit<ComponentPropsWithoutRef<T>, keyof NineSliceOwnProps<T>>;

export function NineSlice<T extends ElementType = "div">({
  as,
  skin,
  width,
  height,
  padding,
  className,
  style,
  children,
  ...elementProps
}: NineSliceProps<T>) {
  const Component = as ?? "div";
  const classes = ["nine-slice", className].filter(Boolean).join(" ");

  return <Component
    {...elementProps}
    className={classes}
    data-nine-slice=""
    style={{ ...createNineSliceStyle(skin, { width, height, padding }), ...style }}
  >
    {children}
  </Component>;
}
