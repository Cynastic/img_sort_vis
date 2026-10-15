#!/usr/bin/env python3
"""Generate N x M test images with continuous patterns.

Usage:
    python gen_test_image.py WIDTH HEIGHT [--pattern rainbow] [-o out.png]

Patterns:
    rainbow     horizontal hue sweep
    rainbow2d   hue sweeps across x, brightness fades down y
    diagonal    hue sweep along the diagonal
    radial      hue rings from the center
    angular     hue around the center (color wheel)
    gradient    smooth RGB gradient (R along x, G along y, B diagonal)

Requires: pip install numpy pillow
"""
import argparse

import numpy as np
from PIL import Image


def hsv_to_rgb(h, s, v):
    """Vectorized HSV -> RGB. h, s, v are arrays in [0, 1]. Returns uint8 HxWx3."""
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    choices = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ]
    r = np.choose(i, [c[0] for c in choices])
    g = np.choose(i, [c[1] for c in choices])
    b = np.choose(i, [c[2] for c in choices])
    return (np.stack([r, g, b], axis=-1) * 255).round().astype(np.uint8)


def make_image(width, height, pattern="rainbow", cycles=1.0):
    # Normalized coordinates in [0, 1]
    x = np.linspace(0, 1, width, endpoint=False)[None, :].repeat(height, 0)
    y = np.linspace(0, 1, height, endpoint=False)[:, None].repeat(width, 1)
    ones = np.ones((height, width))

    if pattern == "rainbow":
        h, s, v = (x * cycles) % 1, ones, ones
    elif pattern == "rainbow2d":
        h, s, v = (x * cycles) % 1, ones, 1 - 0.8 * y
    elif pattern == "diagonal":
        h, s, v = (((x + y) / 2) * cycles) % 1, ones, ones
    elif pattern == "radial":
        r = np.hypot(x - 0.5, y - 0.5) / np.hypot(0.5, 0.5)
        h, s, v = (r * cycles) % 1, ones, ones
    elif pattern == "angular":
        a = (np.arctan2(y - 0.5, x - 0.5) / (2 * np.pi)) % 1
        h, s, v = (a * cycles) % 1, ones, ones
    elif pattern == "gradient":
        rgb = np.stack([x, y, (x + y) / 2], axis=-1)
        return Image.fromarray((rgb * 255).round().astype(np.uint8))
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return Image.fromarray(hsv_to_rgb(h, s, v))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("-p", "--pattern", default="rainbow",
                    choices=["rainbow", "rainbow2d", "diagonal", "radial", "angular", "gradient"])
    ap.add_argument("-c", "--cycles", type=float, default=1.0,
                    help="how many times the hue repeats across the image (default 1)")
    ap.add_argument("-o", "--output", help="output file (default: <pattern>_<W>x<H>.png)")
    args = ap.parse_args()

    img = make_image(args.width, args.height, args.pattern, args.cycles)
    out = args.output or f"{args.pattern}_{args.width}x{args.height}.png"
    img.save(out)
    print(f"Saved {out} ({args.width}x{args.height})")


if __name__ == "__main__":
    main()
