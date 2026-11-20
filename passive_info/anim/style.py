"""Colours, drawing functions, bloom, tonemapping and text, shared by both styles.

The code composes frames in linear float RGB (HxWx3, about 0..1 for full brightness) and
converts them to 8-bit only at the end, so that glows add without clipping.
"""
from __future__ import annotations

import functools

import cv2
import numpy as np

# ---------------------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------------------

#: Background and neutral tones, linear RGB.
BG = np.array([0.006, 0.007, 0.011])
INK = np.array([0.55, 0.62, 0.75])           # faint structural lines
ASH = np.array([0.22, 0.20, 0.19])           # colour of dissipated energy
WHITE = np.array([1.0, 0.97, 0.92])
#: The one colour for information in the single-design loop, linear RGB.
INFO = np.array([0.30, 0.78, 1.0])

#: Force-mode ramp, low frequency -> high: amber, coral, violet, sky, aqua. sRGB hex.
_RAMP_HEX = ("#ffb347", "#ff6b6b", "#c56cf0", "#45aaf2", "#4cf2c4")


def _srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def _hex(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _lin_to_oklab(c):
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return (np.cbrt(c @ M1.T)) @ M2.T


def _oklab_to_lin(L):
    M2i = np.array([[1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]])
    M1i = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]])
    return np.clip((L @ M2i.T) ** 3 @ M1i.T, 0.0, None)


@functools.lru_cache(maxsize=8)
def mode_hues(k: int) -> np.ndarray:
    """`(k, 3)` linear-RGB colours for force modes 0..k-1, interpolated in OKLab."""
    stops = _lin_to_oklab(np.stack([_srgb_to_lin(_hex(h)) for h in _RAMP_HEX]))
    x = np.linspace(0.0, len(stops) - 1, k)
    i = np.clip(np.floor(x).astype(int), 0, len(stops) - 2)
    f = (x - i)[:, None]
    return _oklab_to_lin(stops[i] * (1 - f) + stops[i + 1] * f)


def mix_hue(weights: np.ndarray, hues: np.ndarray) -> np.ndarray:
    """Weighted colour of a mode distribution, renormalised to the ramp's brightness."""
    w = np.clip(np.asarray(weights, float), 0.0, None) ** 2     # favour the dominant modes
    s = w.sum()
    if s <= 0:
        return WHITE * 0.5
    c = (w / s) @ hues
    return c / max(c.max(), 1e-6) * 0.9


# ---------------------------------------------------------------------------------------
# Scales shared by every renderer
# ---------------------------------------------------------------------------------------

def level(e: float, e_ref: float) -> float:
    """Display level for an element with energy `e`. The scale is a square root with a soft
    knee, so that an element with a few percent of the reference energy stays dark."""
    a = float(np.sqrt(max(e, 0.0) / e_ref))
    return a * a / (a + 0.05)


#: Bits per force mode where the two scales of a spectrum bar meet. Every design has about
#: this many bits for every mode, so the first part of each bar is the same for all designs
#: and the second part shows the differences. The axis of the talk cut shows this break.
BAR_KNEE_BITS, BAR_MAX_BITS, BAR_KNEE_FRAC = 3.0, 5.5, 0.35


def bar_length(bits: float) -> float:
    """Fraction of a full bar for `bits`: linear to the knee, then linear to the max."""
    b = max(float(bits), 0.0)
    if b <= BAR_KNEE_BITS:
        return BAR_KNEE_FRAC * b / BAR_KNEE_BITS
    return BAR_KNEE_FRAC + (1 - BAR_KNEE_FRAC) * min(1.0, (b - BAR_KNEE_BITS)
                                                      / (BAR_MAX_BITS - BAR_KNEE_BITS))


# ---------------------------------------------------------------------------------------
# Drawing into float layers (cv2, anti-aliased, sub-pixel via shift)
# ---------------------------------------------------------------------------------------

_SHIFT = 4
_S = float(1 << _SHIFT)


def _pt(p):
    return (int(round(p[0] * _S)), int(round(p[1] * _S)))


def line(img, a, b, color, width=1.0):
    cv2.line(img, _pt(a), _pt(b), tuple(float(c) for c in color),
             max(1, int(round(width))), cv2.LINE_AA, _SHIFT)


def polyline(img, pts, color, width=1.0, closed=False):
    if len(pts) < 2:
        return
    P = np.round(np.asarray(pts) * _S).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [P], closed, tuple(float(c) for c in color),
                  max(1, int(round(width))), cv2.LINE_AA, _SHIFT)


def fill_poly(img, pts, color):
    P = np.round(np.asarray(pts) * _S).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img, [P], tuple(float(c) for c in color), cv2.LINE_AA, _SHIFT)


def disc(img, c, r, color):
    cv2.circle(img, _pt(c), max(1, int(round(r * _S))), tuple(float(x) for x in color),
               -1, cv2.LINE_AA, _SHIFT)


def splat(img, c, r, color):
    """Soft gaussian dot of radius ~r px, added in place."""
    h, w = img.shape[:2]
    R = int(np.ceil(3 * r)) + 1
    x0, y0 = int(c[0]) - R, int(c[1]) - R
    x1, y1 = x0 + 2 * R + 1, y0 + 2 * R + 1
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * r * r))
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    img[sy0:sy1, sx0:sx1] += (g[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0, None]
                              * np.asarray(color, np.float32))


# ---------------------------------------------------------------------------------------
# Post
# ---------------------------------------------------------------------------------------

def blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def bloom(img, scale: float, strength: float = 0.8) -> np.ndarray:
    """Glow at several scales around the bright parts. `scale` is in pixels at 1080p."""
    bright = np.clip(img - 0.35, 0.0, None)
    small = cv2.resize(bright, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    g = 0.0
    for k, w in ((1.0, 0.5), (3.0, 0.3), (8.0, 0.2)):
        g = g + w * blur(small, max(0.6, scale * k / 4.0))
    g = cv2.resize(g, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return img + strength * g


def glow(layer, scale: float, core: float = 1.0, halo: float = 1.6) -> np.ndarray:
    """A drawn light layer as a sharp core plus two soft halos."""
    return core * layer + halo * (0.6 * blur(layer, 1.5 * scale) + 0.4 * blur(layer, 6.0 * scale))


@functools.lru_cache(maxsize=4)
def _vignette(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    return (1.0 - 0.45 * np.clip(r - 0.35, 0.0, None) ** 1.5)[..., None].astype(np.float32)


def finish(img: np.ndarray, seed: int = 0, grain: float = 0.006) -> np.ndarray:
    """Tonemap linear RGB to 8-bit sRGB with a vignette and a little film grain."""
    h, w = img.shape[:2]
    x = img * _vignette(h, w)
    x = x * (1.0 + x / 6.0) / (1.0 + x)                       # soft shoulder
    out = _lin_to_srgb(x)
    if grain:
        out = out + np.random.default_rng(seed).normal(0.0, grain, out.shape[:2])[..., None]
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_linear(rgb8: np.ndarray) -> np.ndarray:
    return _srgb_to_lin(rgb8.astype(np.float32) / 255.0).astype(np.float32)


# ---------------------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------------------

_FONTS = {
    "light": "/usr/share/fonts/opentype/inter/Inter-Light.otf",
    "regular": "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
    "medium": "/usr/share/fonts/opentype/inter/Inter-Medium.otf",
    "semibold": "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
    "mono": "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
}


@functools.lru_cache(maxsize=32)
def font(weight: str, size: int):
    from PIL import ImageFont
    for path in (_FONTS.get(weight), _FONTS["regular"],
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, TypeError):
            continue
    return ImageFont.load_default()


def text(img, xy, s, size, color=(1, 1, 1), weight="regular", anchor="la", alpha=1.0):
    """Draw text into a float linear image (over, not additive)."""
    from PIL import Image, ImageDraw
    f = font(weight, int(size))
    x0, y0, x1, y1 = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(xy, s, font=f,
                                                                     anchor=anchor)
    x0, y0 = max(int(x0) - 2, 0), max(int(y0) - 2, 0)
    x1, y1 = min(int(x1) + 3, img.shape[1]), min(int(y1) + 3, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    m = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(m).text((xy[0] - x0, xy[1] - y0), s, font=f, fill=255, anchor=anchor)
    a = (np.asarray(m, np.float32) / 255.0 * alpha)[..., None]
    c = _srgb_to_lin(np.asarray(color, float)).astype(np.float32)
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + c * a
