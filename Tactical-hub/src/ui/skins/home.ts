import type { NineSliceSkin } from "../nineSlice";

export const HOME_MENU_BUTTON_SLICE = Object.freeze({
  top: 14,
  right: 14,
  bottom: 14,
  left: 14,
});

export const HOME_MENU_BUTTON_SKIN: NineSliceSkin = Object.freeze({
  image: "/ui/home-button-frame.svg",
  slice: HOME_MENU_BUTTON_SLICE,
  borderWidth: { top: 8, right: 8, bottom: 8, left: 8 },
  padding: "8px 18px",
  repeat: "stretch",
});
