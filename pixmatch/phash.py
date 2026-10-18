from functools import lru_cache

import numpy as np

from PIL import Image

HASH_SIZE = 8  # Hashes are HASH_SIZE x HASH_SIZE = 64 bits
HASH_IMG_SIZE = HASH_SIZE * 4  # Images are shrunk to this many pixels square to be hashed

# A frame whose pixels vary less than this (on a 0-255 scale, after resizing) is blank, like a GIF's empty first frame.
#   A blank frame has no detail for the hash to describe, so its hash bits would just be floating point noise.
FLAT_STD = 2.0

# Every rotation of a blank image is the same blank image, and its only meaningful bit is the first (DC) one
FLAT_HASH = 1 << (HASH_SIZE * HASH_SIZE - 1)

EXIF_ORIENTATION_TAG = 0x0112

# The numpy equivalent of the transpose that ImageOps.exif_transpose applies for each EXIF orientation
EXIF_ORIENTATIONS = {
    2: np.fliplr,  # FLIP_LEFT_RIGHT
    3: lambda px: np.rot90(px, 2),  # ROTATE_180
    4: np.flipud,  # FLIP_TOP_BOTTOM
    5: np.transpose,  # TRANSPOSE
    6: lambda px: np.rot90(px, 3),  # ROTATE_270
    7: lambda px: np.rot90(px, 2).T,  # TRANSVERSE
    8: lambda px: np.rot90(px, 1),  # ROTATE_90
}


@lru_cache
def _dct_basis(img_size: int, hash_size: int) -> np.ndarray:
    """The lowest hash_size frequency rows of an (unnormalized) DCT-II matrix for img_size samples"""
    n = np.arange(img_size)
    k = np.arange(hash_size)[:, None]
    return 2 * np.cos(np.pi * (2 * n + 1) * k / (2 * img_size))


def grayscale(im: Image.Image) -> Image.Image:
    """Convert an image (or the current frame of an animation) to grayscale ("L")"""
    if im.mode in {"RGBA", "RGBa", "LA", "La", "PA"} or "transparency" in im.info:
        # Converting straight to grayscale throws away alpha and hashes whatever colors hide under transparent pixels.
        #   So put the image on a white background first, the same as a copy of it that was flattened would be
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im)
    elif im.mode.startswith("I;16"):
        # Pillow clips 16-bit values to 0-255 when converting to grayscale, which turns nearly everything white
        im = Image.fromarray((np.asarray(im) >> 8).astype(np.uint8))

    return im.convert("L")


def resized_pixels(gray: Image.Image, img_size: int) -> np.ndarray:
    """Shrink a grayscale image to an img_size x img_size array"""
    return np.asarray(gray.resize((img_size, img_size), Image.Resampling.LANCZOS), dtype=np.float64)


def grayscale_pixels(im: Image.Image, img_size: int) -> np.ndarray:
    """Convert an image (or the current frame of an animation) to an img_size x img_size grayscale array"""
    return resized_pixels(grayscale(im), img_size)


def is_flat(px: np.ndarray) -> bool:
    """Is this grayscale array blank (nothing, or only a single color)?"""
    return px.std() < FLAT_STD


def orientations(px: np.ndarray):
    """Yield the 8 rotations and mirrors of px, starting with px itself"""
    for k in range(4):
        rotated = np.rot90(px, k)
        yield rotated
        yield np.fliplr(rotated)


def phash(px: np.ndarray) -> int:
    """
    Perceptual hash of an already resized grayscale array.

    This takes the 2D DCT of the array and keeps its lowest HASH_SIZE x HASH_SIZE frequencies,
        and each bit is whether that frequency's coefficient is above the median of them all.

    Returns:
        int: The 64 hash bits, in row-major order with the first bit being the most significant.
    """
    basis = _dct_basis(px.shape[0], HASH_SIZE)
    dct = basis @ px @ basis.T
    bits = (dct > np.median(dct)).ravel()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


# region Centre hash
#   Turning an image by a few degrees cuts off its corners, or fills them with a color on a bigger canvas, which moves
#   the whole image's hash about as far as an unrelated image's. The middle of the image survives that, so the centre
#   hash is the hash of a centered square, and of that square turned back by a few small angles.
#   An image's own centre hashes are its centered square, and its mirror image (to find flipped copies).
#   Its queries are its own centre hash, and its square turned by each of CENTRE_ANGLES at three sizes:
#     * The same size, for a copy turned within the same canvas.
#     * Smaller, as if this image is a copy turned onto a bigger canvas, which shows the middle smaller.
#     * Bigger, as if the other image is.
#   Comparing an image's queries with another image's centre hashes finds it if either is a slightly turned copy.

CENTRE_ANGLES = (3, -3, 6, -6, 9, -9, 12, -12)
CENTRE_WORK = 128  # Everything is sampled from a CENTRE_WORK x CENTRE_WORK array of the image's centered square
CENTRE_CROP = 0.5  # The hashed square is this fraction of the short side (at the same size)
CENTRE_SAMPLES = HASH_IMG_SIZE * 2  # Squares are sampled at twice the hash's image size, then averaged down to it

# The biggest size a square can be tried at, and still be inside the working array once it's turned (and one more
#   pixel for interpolating). Very wide or tall images can call for bigger, but then they wouldn't fit.
_CENTRE_SIDE = CENTRE_CROP * CENTRE_WORK
_CENTRE_MAX_SCALE = ((CENTRE_WORK - 1) / 2 - 1) / (_CENTRE_SIDE / 2 * np.sqrt(2))

# How many 64 bit hashes centre_hashes returns: the centre hash, its mirror, and 3 queries for each angle
CENTRE_HASHES = 2 + 3 * len(CENTRE_ANGLES)


@lru_cache
def _centre_offsets() -> tuple[np.ndarray, np.ndarray]:
    """
    Where to sample each square from, relative to the middle of the working array, at the same size.

    Returns:
        tuple[np.ndarray, np.ndarray]: The x and y offsets of the square as is, then turned by each of CENTRE_ANGLES,
            then mirrored. Each is shaped (squares, CENTRE_SAMPLES, CENTRE_SAMPLES).
    """
    unit = ((np.arange(CENTRE_SAMPLES) + 0.5) / CENTRE_SAMPLES - 0.5) * _CENTRE_SIDE
    ox, oy = np.meshgrid(unit, unit)
    xs, ys = [ox], [oy]
    for angle in CENTRE_ANGLES:
        cos, sin = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        xs.append(cos * ox - sin * oy)
        ys.append(sin * ox + cos * oy)

    xs.append(-ox)
    ys.append(oy)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


@lru_cache
def _dct_basis32(img_size: int, hash_size: int) -> np.ndarray:
    """_dct_basis as float32, which is twice as fast for centre hashes and gives the same bits"""
    return _dct_basis(img_size, hash_size).astype(np.float32)


def _turned_onto_bigger_canvas(angle: float, width: int, height: int) -> tuple[float, float]:
    """
    How much smaller and bigger the middle would be if this image was turned onto, or from, a bigger canvas.

    Turning a w x h image by angle onto a canvas big enough for all of it makes it W = w cos + h sin by
        H = w sin + h cos. The middle is a fraction of the short side, so it shows smaller by min(w, h) / min(W, H).

    Returns:
        (smaller, bigger): The scales to try if this image is that copy, and if the other image is.
    """
    cos, sin = np.cos(np.deg2rad(abs(angle))), np.sin(np.deg2rad(abs(angle)))

    # If this is the copy, then the original was...
    det = cos * cos - sin * sin
    w, h = (cos * width - sin * height) / det, (cos * height - sin * width) / det
    smaller = min(w, h) / min(width, height) if w > 0 and h > 0 else 1.0  # (no size it could have been)

    # If this is the original, then the copy is...
    bigger = min(width * cos + height * sin, width * sin + height * cos) / min(width, height)
    return smaller, min(bigger, _CENTRE_MAX_SCALE)


def centre_array(gray: Image.Image) -> np.ndarray:
    """The centered square of a grayscale image (the full short side), as a CENTRE_WORK x CENTRE_WORK array"""
    w, h = gray.size
    half = min(w, h) / 2
    box = (w / 2 - half, h / 2 - half, w / 2 + half, h / 2 + half)
    resized = gray.resize((CENTRE_WORK, CENTRE_WORK), Image.Resampling.BILINEAR, box=box, reducing_gap=2.0)
    return np.asarray(resized, dtype=np.float32)


def centre_hashes(work: np.ndarray, width: int, height: int) -> bytes | None:
    """
    Get the centre hashes of an image.

    Args:
        work: centre_array of the image, turned the way the image is displayed.
        width: The image's width, as displayed.
        height: The image's height, as displayed.

    Returns:
        bytes | None: CENTRE_HASHES 64-bit big-endian hashes: the centre hash, then every angle at the same size,
            smaller, then bigger, then the mirror's centre hash. None if the middle of the image is blank.
    """
    if work.std() < FLAT_STD:
        return None

    ox, oy = _centre_offsets()
    turned_x, turned_y = ox[1:-1], oy[1:-1]
    scales = np.array([_turned_onto_bigger_canvas(a, width, height) for a in CENTRE_ANGLES], dtype=np.float32)
    smaller, bigger = scales[:, 0, None, None], scales[:, 1, None, None]

    center = (CENTRE_WORK - 1) / 2
    x = np.concatenate((ox[:1], turned_x, turned_x * smaller, turned_x * bigger, ox[-1:])) + center
    y = np.concatenate((oy[:1], turned_y, turned_y * smaller, turned_y * bigger, oy[-1:])) + center

    # Bilinear interpolation of every sample at once
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    i = y0.astype(np.intp) * CENTRE_WORK + x0.astype(np.intp)
    flat = work.ravel()
    top_left, top_right = flat[i], flat[i + 1]
    bottom_left, bottom_right = flat[i + CENTRE_WORK], flat[i + CENTRE_WORK + 1]
    top = top_left + (top_right - top_left) * fx
    px = top + (bottom_left + (bottom_right - bottom_left) * fx - top) * fy

    # Average down to the hash's image size, then the same hash as phash, for every square at once
    n = px.shape[0]
    px = px.reshape(n, HASH_IMG_SIZE, 2, HASH_IMG_SIZE, 2).mean(axis=(2, 4))
    basis = _dct_basis32(HASH_IMG_SIZE, HASH_SIZE)
    dct = (basis @ px @ basis.T).reshape(n, -1)
    packed = np.packbits(dct > np.median(dct, axis=1, keepdims=True), axis=1)
    return packed.tobytes()


def unpack_centre_hashes(packed: bytes) -> tuple[tuple[int, int], tuple[int, ...]]:
    """
    Split centre_hashes into the hashes an image is found by, and the ones it looks for others with.

    Returns:
        ((centre hash, mirror's centre hash), queries): The queries are the centre hash and every turned square.
    """
    hashes = [int.from_bytes(packed[i:i + 8], "big") for i in range(0, len(packed), 8)]
    return (hashes[0], hashes[-1]), tuple(hashes[:-1])
# endregion
