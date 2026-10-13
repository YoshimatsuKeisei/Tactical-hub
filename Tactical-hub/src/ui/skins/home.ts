import type { NineSliceLayout, NineSliceSkin } from "../nineSlice";

export const HOME_MENU_BUTTON_SLICE = Object.freeze({
  top: 210,
  right: 230,
  bottom: 180,
  left: 230,
});

export const HOME_MENU_BUTTON_SKIN: NineSliceSkin = Object.freeze({
  image: "/ui/home/home-button-frame.png",
  slice: HOME_MENU_BUTTON_SLICE,
  borderWidth: { top: 10, right: 20, bottom: 10, left: 20 },
  padding: "8px 14px",
  repeat: "stretch",
});

export const HOME_MENU_BUTTON_LAYOUT: NineSliceLayout = Object.freeze({
  width: "100%",
  height: 62,
});

export const HOME_MENU_PANEL_SLICE = Object.freeze({
  top: 320,
  right: 215,
  bottom: 180,
  left: 215,
});

export const HOME_MENU_PANEL_SKIN: NineSliceSkin = Object.freeze({
  image: "/ui/home/home-panel-frame.png",
  slice: HOME_MENU_PANEL_SLICE,
  borderWidth: { top: 96, right: 64, bottom: 54, left: 64 },
  padding: "18px 28px 30px",
  repeat: "stretch",
});

export const HOME_MENU_PANEL_LAYOUT: NineSliceLayout = Object.freeze({
  width: "min(100%, 620px)",
});
