import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { NineSlice } from "./NineSliceView";
import { createNineSliceStyle, type NineSliceSkin } from "./nineSlice";

const testSkin: NineSliceSkin = {
  image: "/ui/test-frame.png",
  slice: { top: 3, right: 5, bottom: 7, left: 9 },
  borderWidth: { top: 6, right: 10, bottom: 14, left: 18 },
  padding: "4px 8px",
  repeat: "round",
};

describe("nine-slice UI primitive", () => {
  it("turns an independent skin and layout into border-image styles", () => {
    expect(createNineSliceStyle(testSkin, { width: 240, height: "3rem" })).toMatchObject({
      width: 240,
      height: "3rem",
      padding: "4px 8px",
      borderWidth: "6px 10px 14px 18px",
      borderImageSource: "url(\"/ui/test-frame.png\")",
      borderImageSlice: "3 5 7 9 fill",
      borderImageWidth: "6px 10px 14px 18px",
      borderImageRepeat: "round",
    });
  });

  it("renders children on a selectable host element and forwards interaction props", () => {
    const onClick = vi.fn();
    const markup = renderToStaticMarkup(
      <NineSlice as="button" skin={testSkin} padding={12} type="button" onClick={onClick}>
        <span>Button label</span>
      </NineSlice>,
    );

    expect(markup).toContain("<button");
    expect(markup).toContain("data-nine-slice");
    expect(markup).toContain("border-image-source:url(&quot;/ui/test-frame.png&quot;)");
    expect(markup).toContain("padding:12px");
    expect(markup).toContain("<span>Button label</span>");
  });

  it("falls back to slice dimensions for the rendered border", () => {
    expect(createNineSliceStyle({
      image: "/ui/other-frame.svg",
      slice: { top: 2, right: 4, bottom: 6, left: 8 },
    })).toMatchObject({
      borderWidth: "2px 4px 6px 8px",
      borderImageWidth: "2px 4px 6px 8px",
      borderImageRepeat: "stretch",
    });
  });
});
