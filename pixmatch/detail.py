"""
A check of fine detail between two copies of a picture, for Auto-select's safety net.

Two pictures can match even at the strictest strength while differing in a small patch: a different episode list on
    otherwise identical disc labels, or a caption or watermark added to a copy. The hashes are made from a picture
    shrunk to 32 pixels, where such a patch is a few grey pixels. This compares the pictures at up to 256 pixels
    instead, after lining them up (the right way round, mirrored or not, zoomed and shifted to match), block by block.
    A copy that's resized, recompressed, brightened, sharpened, mirrored or cropped by a sliver has no block much
    different.

It can't tell a watermark added from a different picture with different text (nothing can, see plans/auto_select.md),
    so Auto-select leaves such groups for the user rather than choosing.
"""
from pathlib import Path

import numpy as np

from PIL import Image, ImageOps

# A block this different (mean absolute difference of brightness, in standard deviations) is a real difference.
#   Measured: different Voyager disc labels (only their text differs) 0.45-0.49; their re-saved copies at most 0.35
#   (a 1% crop), and 0.0-0.2 for resized, recompressed, brightened, sharpened or mirrored ones
DETAIL_DIFFERENCE = 0.40

_BLOCKS = 16  # The comparison is split into _BLOCKS x _BLOCKS blocks
_MAX_SHIFT = 0.12  # Copies can be shifted against each other by up to this much of their size
_ORIENTATIONS = (
    None,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
)


def load(path: str | Path) -> Image.Image:
    """
    A picture as it's displayed, in grayscale, decoded at a reduced size where that's quick (JPEGs).

    Returns:
        Image.Image: The picture ("L" mode), at least 256 pixels on its short side if it's that big.
    """
    with Image.open(path) as im:
        im.draft("L", (512, 512))
        im = ImageOps.exif_transpose(im)
        if im.mode in {"RGBA", "LA", "PA", "P"} or "transparency" in im.info:
            im = im.convert("RGBA")
            im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im)
        elif im.mode.startswith("I;16"):
            im = Image.fromarray((np.asarray(im) >> 8).astype(np.uint8))
        return im.convert("L")


def _square(im: Image.Image, size: int) -> np.ndarray:
    return np.asarray(im.resize((size, size), Image.Resampling.BOX), dtype=np.float32)


def _normalized(a: np.ndarray) -> np.ndarray:
    """Brightness in standard deviations from its mean, so brightness and contrast changes don't count"""
    return (a - a.mean()) / (a.std() + 1e-6)


def _zoomed(b: np.ndarray, scale: float) -> np.ndarray:
    """Zoom b by scale about its middle, keeping it the same size (the edges of a smaller one are left at 0)"""
    n = b.shape[0]
    m = max(8, round(n * scale))
    arr = np.asarray(Image.fromarray(b).resize((m, m), Image.Resampling.BILINEAR), dtype=np.float32)
    if m >= n:
        offset = (m - n) // 2
        return arr[offset:offset + n, offset:offset + n]
    out = np.zeros((n, n), dtype=np.float32)
    offset = (n - m) // 2
    out[offset:offset + m, offset:offset + m] = arr
    return out


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """The shift that best lines b up with a (by phase correlation)"""
    product = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    product /= np.abs(product) + 1e-9
    correlation = np.fft.ifft2(product).real
    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    n = a.shape[0]
    return (y if y <= n // 2 else y - n), (x if x <= n // 2 else x - n)


def _difference(a: np.ndarray, b: np.ndarray, scale: float) -> tuple[float, float] | None:
    """
    Zoom b, shift it to line up with a, and compare where they overlap (not the zoom's empty edges).

    Returns:
        (mean difference, the most different block's difference), or None if b can't be lined up with a.
    """
    n = a.shape[0]
    zoomed = _normalized(_zoomed(b, scale)) if scale != 1 else b
    dy, dx = _phase_shift(a, zoomed)
    if max(abs(dy), abs(dx)) > n * _MAX_SHIFT:
        return None

    shifted = np.roll(zoomed, (dy, dx), axis=(0, 1))
    margin = max(abs(dy), abs(dx)) + int(np.ceil(abs(1 - scale) * n / 2)) + 1
    inner = slice(margin, n - margin)
    difference = np.abs(a[inner, inner] - shifted[inner, inner])
    block = n // _BLOCKS
    size = (difference.shape[0] // block) * block
    if size < block * 4:
        return None

    blocks = difference[:size, :size].reshape(size // block, block, size // block, block).mean(axis=(1, 3))
    return float(difference.mean()), float(blocks.max())


def detail_difference(a: Image.Image, b: Image.Image) -> float:
    """
    How different two copies of a picture are in their most different patch, once they're lined up.

    Returns:
        float: The most different block's mean difference, in standard deviations of brightness (see
            DETAIL_DIFFERENCE). Infinity if they can't be lined up at all.
    """
    # At the smaller picture's detail: a small copy blown up is blurrier, which isn't a difference
    short = min(*a.size, *b.size)
    size = 256 if short >= 256 else 128 if short >= 128 else 64

    # Which way round b is (a copy can be mirrored or turned), from a quick look at 32 pixels
    small_a = _normalized(_square(a, 32))
    b = min((b.transpose(o) if o is not None else b for o in _ORIENTATIONS),
            key=lambda turned: float(np.abs(small_a - _normalized(_square(turned, 32))).mean()))

    big_a, big_b = _normalized(_square(a, size)), _square(b, size)
    # Most copies are just resized or re-saved: lined up as they are, nothing is different
    first = _difference(big_a, _normalized(big_b), 1.0)
    if first is not None and first[1] < DETAIL_DIFFERENCE:
        return first[1]

    # Otherwise look for the zoom that lines them up best (cropped copies), roughly and then finely
    tried = [(s, _difference(big_a, big_b, s)) for s in np.arange(0.88, 1.121, 0.03)]
    tried = [(s, r) for s, r in tried if r is not None]
    if first is not None:
        tried.append((1.0, first))
    if not tried:
        return float("inf")

    best_scale, best = min(tried, key=lambda t: t[1][0])
    for s in np.arange(best_scale - 0.025, best_scale + 0.0251, 0.005):
        result = _difference(big_a, big_b, s)
        if result is not None and result[0] < best[0]:
            best = result
    return best[1]


def differ_in_detail(a: str | Path, b: str | Path) -> bool:
    """
    Do two copies of a picture differ in a patch of fine detail (see detail_difference)? Files that can't be read
        count as differing, so nothing is decided about them.

    Returns:
        bool: Whether they differ.
    """
    try:
        return detail_difference(load(a), load(b)) >= DETAIL_DIFFERENCE
    except (OSError, ValueError, Image.DecompressionBombError):
        return True
