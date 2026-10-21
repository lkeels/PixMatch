"""
Auto-select's decisions: which copy in a group of duplicates to keep, which to mark, and whether the kept copy should
    move into a higher folder in place of a copy there. See plans/auto_select.md.

This only decides, from what was measured when the images were hashed (see pixmatch.ImageQuality): it never touches a
    file, so every rule can be tried on made-up groups.
"""
import re

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePath
from typing import Callable, Sequence

from pixmatch import ImageQuality, ZipPath

# A copy is clearly better if it has at least this many times the pixels of another...
CLEARLY_MORE_PIXELS = 1.25
# ...or about the same pixels (less than CLEARLY_MORE_PIXELS times) and, in the same format, at least this many times
#   the file size (less compressed)...
CLEARLY_BIGGER_FILE = 1.25
# ...or as JPEGs, a quality setting at least this much higher...
CLEARLY_BETTER_JPEG_QUALITY = 10
# ...or at least this many times the sharpness for its contrast (a blurred copy has less)
CLEARLY_SHARPER = 1.15

# A picture is black and white if its colourfulness is under this...
GREY_COLOURFULNESS = 2.0
# ...or tinted black and white (like sepia) if its colour varies less than this among pixels of the same brightness
TINTED_COLOUR_VARIETY = 1.5

# Formats kept over others when everything else is equal: lower is preferred
FORMAT_PREFERENCE = {"PNG": 0, "JPEG": 0, "MPO": 0, "TIFF": 1, "BMP": 1, "GIF": 2, "WEBP": 3}


class Skip(Enum):
    """Why a group was left alone, for the user to decide"""
    TOO_FEW = "fewer than two copies that can be marked (the rest are ignored, or in zips)"
    ANIMATED = "it has an animation"
    COLOUR_CONFLICT = "the best copy is black and white, but there's a colour copy"
    TOO_CLOSE = "no copy is clearly better, and nothing else decides between them"
    DIFFERENT_DETAIL = "the copies differ in a patch of detail (text, a caption or watermark, or a different version)"


@dataclass(frozen=True)
class Candidate:
    """
    One image in a group, as Auto-select sees it.

    Attributes:
        path (ZipPath): The image.
        quality (ImageQuality): Its measurements.
        folder_rank (int | None): The position in the folder list of the listed folder it's under (0 is the top), or
            None if it isn't under one.
        ignored (bool): Whether it's IGNORED, so it's never marked or moved.
    """
    path: ZipPath
    quality: ImageQuality
    folder_rank: int | None
    ignored: bool = False


@dataclass(frozen=True)
class Move:
    """
    Move the kept copy into another copy's folder, in place of that copy (which is marked).

    Attributes:
        replaces (ZipPath): The copy it replaces.
        folder (Path): The folder it moves to (the replaced copy's).
        name (str): The file name it gets there (see moved_name).
    """
    replaces: ZipPath
    folder: Path
    name: str


@dataclass(frozen=True)
class Decision:
    """
    What Auto-select plans for a group.

    Attributes:
        keep (ZipPath | None): The copy kept, or None if the group is left alone (see skip).
        mark (list[ZipPath]): The copies marked to be deleted.
        move (Move | None): Where the kept copy moves, if it does.
        skip (Skip | None): Why the group was left alone, if it was.
        reason (str): Why the kept copy was kept, in a few words (for the summary).
    """
    keep: ZipPath | None = None
    mark: list[ZipPath] = field(default_factory=list)
    move: Move | None = None
    skip: Skip | None = None
    reason: str = ""


def folder_rank(path: ZipPath, folders: Sequence[str | Path]) -> int | None:
    """
    The position in the folder list of the listed folder an image is under (for an image in a zip, the zip's).

    Returns:
        int | None: 0 for the top folder of the list, or None if it isn't under any of them. An image under two listed
            folders (one inside the other) gets the higher one's position.
    """
    image = PurePath(path.path)
    for rank, folder in enumerate(folders):
        if image.is_relative_to(folder) or _same_case_insensitive_parent(image, PurePath(folder)):
            return rank

    return None


def _same_case_insensitive_parent(image: PurePath, folder: PurePath) -> bool:
    """Is the image under the folder, ignoring case and slashes (as Windows paths do)?"""
    image_parts = [part.lower() for part in image.parts]
    folder_parts = [part.lower() for part in folder.parts]
    return image_parts[:len(folder_parts)] == folder_parts


def is_black_and_white(quality: ImageQuality) -> bool:
    """Is it black and white, or tinted black and white like sepia? (see ImageQuality.colour_variety)"""
    return quality.colourfulness < GREY_COLOURFULNESS or quality.colour_variety < TINTED_COLOUR_VARIETY


def _relative_sharpness(quality: ImageQuality) -> float:
    """Sharpness for its contrast, so a faded copy isn't taken for a blurred one"""
    return quality.sharpness / quality.contrast if quality.contrast else 0.0


def clearly_better(a: ImageQuality, b: ImageQuality) -> bool:
    """
    Is a clearly better quality than b?

    By resolution first: at least CLEARLY_MORE_PIXELS times the pixels. At about the same resolution, by compression
        (a bigger file in the same format, or a higher JPEG quality setting) or by sharpness. A copy with clearly
        fewer pixels is never clearly better, however good it is otherwise.

    Returns:
        bool: Whether a is clearly better.
    """
    if a.pixels >= b.pixels * CLEARLY_MORE_PIXELS:
        return True

    if b.pixels >= a.pixels * CLEARLY_MORE_PIXELS:
        return False

    # About the same resolution
    if a.format == b.format and a.file_size >= b.file_size * CLEARLY_BIGGER_FILE > 0:
        return True

    if (a.jpeg_quality is not None and b.jpeg_quality is not None
            and a.jpeg_quality >= b.jpeg_quality + CLEARLY_BETTER_JPEG_QUALITY):
        return True

    return _relative_sharpness(a) >= _relative_sharpness(b) * CLEARLY_SHARPER > 0


# Names cameras, phones, apps and websites make up, which say nothing about the picture
_GENERATED_PREFIX = re.compile(
    r"""^(
        (img|dsc|dscn|dscf|dcim|pxl|mvimg|vid|sam|cimg|imag|gopr|dji|mov)[ _-]*\d   # cameras and phones: IMG_8842
        | (image|photo|picture|pic|file)[ _-]?\d                                    # image001, photo_12
        | screen[ _-]?shot | screenshot | unnamed | download | untitled            # apps
        | whatsapp[ _]image | fb[ _]img | received[ _] | snapchat | signal[ _-] | tumblr[ _] | instagram[ _]
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_GENERATED_WHOLE = re.compile(
    r"""^(
        [\d_ .x-]+                       # only digits and separators: 20210529_074856, 1234, 3456x5184
        | [0-9a-f]{8,}([_-][0-9a-f]{4,})*  # hex: hashes, UUIDs
        | [a-z0-9_-]*\d[a-z0-9_-]*        # letters and digits with no space: GA1vsExWsAAmo1W, E8RNZcjXoAISA2D
    )$""",
    re.IGNORECASE | re.VERBOSE,
)
# What's added to the name of a copy: " (2)", " - Copy", " copy 3", "_1" after a camera number is left alone
_COPY_SUFFIX = re.compile(r"(\s*\(\d+\)|\s*-\s*copy(\s*\(\d+\))?|\s+copy(\s+\d+)?)+$", re.IGNORECASE)


def _plain_stem(name: str) -> str:
    """A file name without its extension, and without what copying added to it (" (2)", " - Copy")"""
    return _COPY_SUFFIX.sub("", PurePath(name).stem).strip()


def is_copy_name(name: str) -> bool:
    """Does a file name end with what copying adds to it, like " (2)" or " - Copy"?"""
    return _plain_stem(name) != PurePath(name).stem.strip()


def looks_random(name: str) -> bool:
    """
    Does a file name look made up by a camera, phone, app or website, rather than written by someone?

    Random-looking: IMG_8842, DSC02794, DSC_0054_2, 587edaef10ead2cc..., GA1vsExWsAAmo1W, tumblr_m84falcI7K1qj6e7ro,
        Screenshot 2024-03-01, unnamed (3). Written: Tom Daley beach, Lilac Whimsy - 1300x900, Austin Sikora 2.

    Returns:
        bool: Whether it looks random. Names it doesn't recognize either way count as written.
    """
    stem = _plain_stem(name)
    if not stem or _GENERATED_PREFIX.match(stem):
        return True

    # A word of three or more letters (with a vowel) among others, separated by spaces, is something someone typed
    if " " in stem and re.search(r"(^|\s)[A-Za-z]*[aeiouyAEIOUY][A-Za-z]{2,}", stem):
        return False

    return bool(_GENERATED_WHOLE.match(stem))


def moved_name(kept: ZipPath, replaced: ZipPath) -> str:
    """
    The name a kept copy gets when it replaces another copy: the replaced copy's name (it's the name in the folder it
        belongs in), unless that name looks random and the kept copy's doesn't.

    Returns:
        str: The file name, with the kept copy's extension.
    """
    kept_name = PurePath(kept.subpath or kept.path).name
    replaced_name = PurePath(replaced.subpath or replaced.path).name
    stem = PurePath(replaced_name).stem
    if looks_random(replaced_name) and not looks_random(kept_name):
        stem = PurePath(kept_name).stem

    return stem + PurePath(kept_name).suffix


def _rank(candidate: Candidate) -> float:
    """A candidate's folder rank for comparing: a copy that isn't under a listed folder comes last"""
    return float("inf") if candidate.folder_rank is None else candidate.folder_rank


def _prefer(candidates: list[Candidate]) -> tuple[list[Candidate], str]:
    """
    Narrow down copies of about equal quality by preference: colour, then higher folder, then format, then bigger file.

    Returns:
        (the ones left, the preference that narrowed them down last, in a few words)
    """
    reason = ""
    for name, key in (
            ("colour", lambda c: is_black_and_white(c.quality)),
            ("higher folder", _rank),
            ("preferred format", lambda c: FORMAT_PREFERENCE.get(c.quality.format, 2)),
            ("bigger file", lambda c: -c.quality.file_size),
    ):
        best = min(key(c) for c in candidates)
        narrowed = [c for c in candidates if key(c) == best]
        if len(narrowed) < len(candidates):
            candidates, reason = narrowed, name
        if len(candidates) == 1:
            break

    return candidates, reason


def _exact_copies(candidates: list[Candidate]) -> bool:
    """Are these the same picture at the same quality, in the same folder?"""
    first = candidates[0]
    folder = PurePath(first.path.path).parent
    return all(PurePath(c.path.path).parent == folder and not c.path.subpath
               and (c.quality.width, c.quality.height, c.quality.file_size, c.quality.format)
               == (first.quality.width, first.quality.height, first.quality.file_size, first.quality.format)
               for c in candidates)


def decide(group: Sequence[Candidate]) -> Decision:
    """
    Decide what to do with a group of duplicates (see plans/auto_select.md, "How a group is decided").

    Only copies that aren't ignored and aren't in zips are kept, marked or moved. Ignored and zipped copies are left as
        they are, and don't count for anything.

    Returns:
        Decision: The copy to keep, the copies to mark, and where the kept copy moves to (if it does); or why the group
            was left alone.
    """
    if any(c.quality.animated for c in group):
        return Decision(skip=Skip.ANIMATED)

    candidates = [c for c in group if not c.ignored and not c.path.subpath]
    if len(candidates) < 2:
        return Decision(skip=Skip.TOO_FEW)

    # The best quality: the copies no other copy is clearly better than
    top = [c for c in candidates if not any(clearly_better(o.quality, c.quality) for o in candidates if o is not c)]
    if not top:
        # Every copy is clearly better than another in a circle (sharper, but a smaller file, but a higher JPEG
        #   quality...): no copy is the best, so they're all about equal
        top = list(candidates)
    if len(top) == 1:
        keeper = top[0]
        if is_black_and_white(keeper.quality) and not all(is_black_and_white(c.quality) for c in candidates):
            return Decision(skip=Skip.COLOUR_CONFLICT)
        reason = "clearly best quality"
    else:
        top, reason = _prefer(top)
        if len(top) > 1:
            if not _exact_copies(top):
                return Decision(skip=Skip.TOO_CLOSE)
            # The same picture saved twice in one folder: keep the one with a written name, then one that isn't named
            #   as a copy (" (2)"), then the first
            top.sort(key=lambda c: (looks_random(PurePath(c.path.path).name), is_copy_name(PurePath(c.path.path).name)))
            reason = "exact copy"
        keeper = top[0]

    mark = [c.path for c in candidates if c is not keeper]

    # A copy in a higher folder: the kept copy belongs there, in its place
    higher = [c for c in candidates if _rank(c) < _rank(keeper)]
    if not higher:
        return Decision(keep=keeper.path, mark=mark, reason=reason)

    best_rank = min(_rank(c) for c in higher)
    replaced = max((c for c in higher if _rank(c) == best_rank),
                   key=lambda c: (c.quality.pixels, c.quality.file_size))  # The best of them, if there are several
    return Decision(
        keep=keeper.path,
        mark=mark,
        move=Move(replaced.path, Path(replaced.path.path).parent, moved_name(keeper.path, replaced.path)),
        reason=reason,
    )


def check_detail(decision: Decision, differ: Callable[[ZipPath, ZipPath], bool]) -> Decision:
    """
    The safety net: leave a group alone if a copy that would be deleted differs from the one kept in a patch of fine
        detail (see pixmatch.detail), like disc labels that differ only by their text. Those match at any strength,
        and so does a copy with a watermark added: which is which can't be told, so the user decides.

    Args:
        decision (Decision): What decide planned for the group.
        differ (Callable[[ZipPath, ZipPath], bool]): Whether two copies differ in detail (pixmatch.detail).

    Returns:
        Decision: The same decision, or one leaving the group alone (Skip.DIFFERENT_DETAIL).
    """
    if decision.skip or decision.keep is None:
        return decision

    if any(differ(decision.keep, marked) for marked in decision.mark):
        return Decision(skip=Skip.DIFFERENT_DETAIL)

    return decision
