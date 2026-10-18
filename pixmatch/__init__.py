import hashlib
import logging
import os
import time

from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from itertools import combinations
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from queue import Queue
from threading import Event
from typing import ClassVar
from zipfile import BadZipFile, ZipFile

import numpy as np

from PIL import Image, ImageFile, UnidentifiedImageError

from pixmatch import phash

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow damaged images

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZipPath:
    """
    A general object describing a Path.

    All paths in pixmatch will be one of these. `subpath` will be empty for non-zip file paths.

    Attributes:
        path (str): The path to the file.
        subpath (str): The subpath in the zip if `path` is for a zip.
    """
    # TODO: At some point convert this to Path.
    #   When I tried that last it introduced problems with inter-process communication
    path: str
    subpath: str = ''

    @property
    def path_obj(self) -> Path:
        """The path as a Path object"""
        return Path(self.path)

    @property
    def is_gif(self) -> bool:
        """Is this a path to an animated image?"""
        movie_extensions = {'.gif', '.webp'}
        suffix = Path(self.subpath or self.path).suffix.lower()
        return suffix in movie_extensions

    @property
    def is_zip(self) -> bool:
        """Does this point to a file located in a zip?"""
        return bool(self.subpath)

    def absolute(self):
        """Get the absolute version of this ZipPath"""
        return ZipPath(str(self.path_obj.expanduser().resolve()), self.subpath)


def _is_under(folder_abs: str, target: str | Path) -> bool:
    """Return True if the ZipPath's real file (zp.path) is inside folder_abs."""
    try:
        Path(target).expanduser().resolve().relative_to(Path(folder_abs).expanduser().resolve())
    except ValueError:
        return False

    return True


# The most bits two images' hashes can differ by to match, for each strength from 0 (loose) to 10 (strict).
#   Every hash has exactly 32 of its 64 bits set, so two hashes always differ by an even number of bits.
#   From testing: re-saves, resizes, format changes, rotations and small brightness or color changes land within
#   2-4 bits, and trimming 1% off each side within 6. Unrelated images were more than 6 apart (and usually 20+).
#   Past 8 bits, HammingIndex gets much slower (~16ms per image with a million or more images loaded).
STRENGTH_RADIUS = (8, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0)

# Decoding JPEGs at a reduced size (see calculate_hashes' draft) can move a hash by up to 2 bits. That is harmless
#   when matches can differ by at least this many bits, but loses matches at stricter strengths.
DRAFT_RADIUS = 4


def calculate_hashes(f, *, is_gif=False, exact_match=False, draft=False) -> tuple[int | str, set[int], bytes | None]:
    """
    Calculate hashes for a given file.

    Args:
        f (IO or str or Path): Either a file path to process, or a in-memory BytesIO object ready for reading.
        is_gif (bool): Is this gif data? Needed if passing an in-memory BytesIO object.
        exact_match (bool): Use exact SHA256 hahes?
            If true, strength must be 10.
            If false, perceptual hashes will be used, even with high strength.
        draft (bool): Decode JPEGs at a reduced size? Much faster, but can change the hash by a couple bits.

    Returns:
        tuple[int | str, set, bytes | None]: The first element is the primary hash,
            the second element are any secondary hashes representing rotations, flips, etc...
            Perceptual hashes are 64-bit ints, and the SHA256 hash is a hex string (with no secondary hashes).
            The third element are the centre hashes (see phash.centre_hashes), for finding copies turned by a few
            degrees. None for SHA256 hashes, and images with a blank middle.
    """
    if exact_match:
        hasher = hashlib.sha256()
        block_size = 65536
        with (open(f, "rb") if isinstance(f, (str, Path)) else f) as file:  # ruff: ignore[builtin-open]
            for block in iter(lambda: file.read(block_size), b""):
                hasher.update(block)
        return hasher.hexdigest(), set(), None

    with Image.open(f) as im:
        if draft:
            # JPEGs can be decoded at 1/2 to 1/8 of their size (and straight to grayscale), which is several times
            #   faster than decoding all of a large photo just to shrink it. This does nothing for other formats.
            #   It leaves the short side at least CENTRE_WORK pixels, so the centre hashes lose nothing either.
            im.draft("L", (phash.HASH_IMG_SIZE * 4, phash.HASH_IMG_SIZE * 4))
        orientation = im.getexif().get(phash.EXIF_ORIENTATION_TAG, 1)
        gray = phash.grayscale(im)
        px = phash.resized_pixels(gray, phash.HASH_IMG_SIZE)
        if is_gif:
            # Some animations start with a blank frame (nothing, or only a single color). Its hash would be noise
            #   that happens to be shared by other blank frames, so use the first frame that has something in it.
            while phash.is_flat(px):
                try:
                    im.seek(im.tell() + 1)
                except EOFError:  # ruff: ignore[try-except-in-loop]
                    break
                else:
                    gray = phash.grayscale(im)
                    px = phash.resized_pixels(gray, phash.HASH_IMG_SIZE)

    centre = phash.centre_array(gray)
    width, height = gray.size

    # Hash the image the way it is displayed, not the way it is stored
    if orientation in phash.EXIF_ORIENTATIONS:
        px = phash.EXIF_ORIENTATIONS[orientation](px)
        centre = phash.EXIF_ORIENTATIONS[orientation](centre)
        if orientation >= 5:  # These turn it a quarter, so it's displayed with its width and height swapped
            width, height = height, width

    if phash.is_flat(px):
        return phash.FLAT_HASH, set(), None

    # All the rotations and mirrors are made from the one small resized array, which is far cheaper than
    #   transforming the full size image. For GIFs we'll look for mirrored versions but thats it
    variants = (px, np.fliplr(px)) if is_gif else tuple(phash.orientations(px))
    initial_hash, *extras = (phash.phash(variant) for variant in variants)
    return initial_hash, set(extras), phash.centre_hashes(np.ascontiguousarray(centre), width, height)


def thread_error_handler(func):
    """An error handler for the thread to return information about where the error occurred"""

    @wraps(func)
    def wrapper(path, *args, **kwargs):  # ruff: ignore[missing-return-type-private-function]
        try:
            return func(path, *args, **kwargs)
        except Exception as e:
            e.input_path = path
            raise

    return wrapper


@thread_error_handler
def _process_image(
        path: str | Path,
        supported_extensions: set | None = None,
        *,
        exact_match: bool = False,
        draft: bool = False,
) -> tuple[Path, tuple | dict[str, tuple]]:
    """Get the hashes for a given path. Is multiprocessing compatible"""
    path = Path(path)
    if path.suffix.lower() != '.zip':
        return path, calculate_hashes(path, is_gif=path.suffix.lower() in {".gif", ".webp"},
                                      exact_match=exact_match, draft=draft)

    if not supported_extensions:
        supported_extensions = ImageMatcher.SUPPORTED_EXTS

    results = {}
    with ZipFile(path) as zf:
        for f in zf.filelist:
            f_ext = Path(f.filename).suffix.lower()
            if f_ext not in supported_extensions:
                continue

            if f_ext == '.zip':
                logger.warning('Have not implemented nested zip support yet! Input file: %s (%s)', path, f)
                continue

            try:
                with zf.open(f) as zipped_file:
                    results[f.filename] = calculate_hashes(zipped_file, is_gif=f_ext in {".gif", ".webp"},
                                                           exact_match=exact_match, draft=draft)
            except BadZipFile as e:
                logger.warning("Could not read %s in %s due to %s", f.filename, path, e)
            except UnidentifiedImageError:
                logger.warning("Could not identify image %s in %s", f.filename, path)
            except Exception as e:  # ruff: ignore[blind-except]
                # One image that can't be hashed (too large, encrypted, ...) shouldn't lose every other image in the zip
                logger.warning("Could not process %s in %s due to %s: %s", f.filename, path, type(e).__name__, e)

    return path, results


@dataclass
class ImageMatch:
    """A match data structure containing the matches and where this match lies in the match list"""
    match_i: int | None = field(default=None)
    matches: list[ZipPath] = field(default_factory=list)

    def snapshot(self) -> "ImageMatch":
        """
        A copy of this group as it is now, for events.

        Events are read later on another thread, and the group keeps changing until then. Without a copy, the GUI
            would build a new group's row from images whose own NewMatch events are still queued, and show them twice.

        Returns:
            ImageMatch: The copy, which won't change as images are added to or removed from this group.
        """
        return ImageMatch(self.match_i, list(self.matches))


# region Events
@dataclass(frozen=True)
class NewGroup:
    """A new group event"""
    group: "ImageMatch"


@dataclass(frozen=True)
class NewMatch:
    """A new match event"""
    group: "ImageMatch"
    path: ZipPath


@dataclass(frozen=True)
class Finished:
    """A finished event"""


MatcherEvent = NewGroup | NewMatch | Finished
# endregion


class HammingIndex:
    """
    Finds the stored hashes within some number of differing bits (radius) of a hash, without checking all of them.

    This is multi-index hashing: every hash is split into chunks, and each chunk gets its own lookup table.
    If two hashes are within the radius, then (by the pigeonhole principle) at least one of their chunks differs by
    no more than radius // CHUNKS bits. So looking up every value that close to each of a hash's chunks finds every
    stored hash that could be in range, and only those candidates get compared in full.

    Args:
        radius (int): The most bits a stored hash can differ by to be found.
    """
    BITS = phash.HASH_SIZE * phash.HASH_SIZE
    CHUNKS = 3  # 21-22 bit chunks keep the tables' buckets small, even with a couple million hashes stored

    def __init__(self, radius: int):
        self.radius = radius
        widths = [self.BITS // self.CHUNKS + (i < self.BITS % self.CHUNKS) for i in range(self.CHUNKS)]

        # For each chunk: where it is in the hash, and every change to it of at most radius // CHUNKS bits
        self._chunks = []
        shift = 0
        for width in widths:
            flips = [sum(1 << bit for bit in bits)
                     for n in range(radius // self.CHUNKS + 1) for bits in combinations(range(width), n)]
            self._chunks.append((shift, (1 << width) - 1, flips))
            shift += width

        # Chunk value -> the stored hash with that chunk value, or a list of them if there are several.
        #   Most chunk values only ever belong to one hash, so skipping the list for those saves a lot of memory.
        self._tables = [{} for _ in widths]

    def add(self, hash_: int):
        """Store a hash"""
        for (shift, mask, _), table in zip(self._chunks, self._tables, strict=True):
            key = hash_ >> shift & mask
            stored = table.get(key)
            if stored is None:
                table[key] = hash_
            elif isinstance(stored, list):
                stored.append(hash_)
            else:
                table[key] = [stored, hash_]

    def remove(self, hash_: int):
        """Remove a stored hash"""
        for (shift, mask, _), table in zip(self._chunks, self._tables, strict=True):
            key = hash_ >> shift & mask
            stored = table[key]
            if stored == hash_:
                del table[key]
                continue

            stored.remove(hash_)
            if len(stored) == 1:
                table[key] = stored[0]

    def find(self, hash_: int) -> dict[int, int]:
        """Find the stored hashes within the radius of a hash, as a dict of stored hash -> bits different"""
        found = {}
        for (shift, mask, flips), table in zip(self._chunks, self._tables, strict=True):
            key = hash_ >> shift & mask
            for flip in flips:
                stored = table.get(key ^ flip)
                if stored is None:
                    continue

                for candidate in (stored if isinstance(stored, list) else (stored, )):
                    if candidate not in found:
                        distance = (candidate ^ hash_).bit_count()
                        if distance <= self.radius:
                            found[candidate] = distance

        return found


class CentreIndex(HammingIndex):
    """
    A HammingIndex for centre hashes, which are only matched within CENTRE_RADIUS bits.

    Each image looks itself up with every one of its centre hash queries, so lookups need to be cheap. With 5 chunks
    and a radius of 4, some chunk of a match is identical, so each lookup is only 5 table lookups.
    """
    CHUNKS = 5


# The most bits an image's centre hash queries can differ from another image's centre hash for it to be a turned copy.
#   From testing: copies turned 3-12 degrees, on the same or a bigger canvas, land within 0-4 bits.
#   Unrelated images were 12 or more apart.
CENTRE_RADIUS = 4


class ImageMatcher:
    """
    An image matching SDK

    Args:
        strength (int): The 0-10 strength to use for matching. Defaults to 5.
            STRENGTH_RADIUS has how many bits two images' hashes can differ by at each strength.
        exact_match (bool): Should use SHA-256 hashes? If False, the default, will use perceptual hashes.
            If True, strength must be 10.
        processes (int): The number of processes to use. Defaults to None.
        extensions (set): The extensions to process. Optional.
        rotation (bool): Also match copies turned by a few degrees (by their centre hashes)?
            Centre hashes are always calculated, so `regroup` can turn this on or off. Ignored for SHA-256 hashes.
    """
    SUPPORTED_EXTS: ClassVar = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".zip"}

    def __init__(self, strength: int = 5, processes: int | None = None, extensions: set | None = None,
                 *, exact_match: bool = False, rotation: bool = False):
        if not (0 <= strength <= 10):
            raise ValueError("Strength must be between 0 and 10!")

        self.extensions = extensions or self.SUPPORTED_EXTS

        self.strength = strength
        self.radius = STRENGTH_RADIUS[strength]
        self.exact_match = exact_match
        self.processes = processes

        self.found_images = 0
        self.processed_images = 0
        self.duplicate_images = 0
        self.matches = []

        # Both ends of these queues are in this process (the pool runs its callbacks on a thread here, not in the
        #   workers), so they don't need to be multiprocessing queues
        self.events = Queue()  # Events to go to higher level users
        self._new_paths = Queue()  # Inbound queue for new paths that are added while processing is running
        self._removed_paths = set()  # Paths that have been removed from processing after processing has been started
        self._ignored_files = set()  # Files which have been ignored and should be skipped from processing if re-ran
        self._processed_zips = {}  # Zips that have been successfully processed
        self._hashes = defaultdict(ImageMatch)  # Hash -> Paths
        self._reverse_hashes = {}  # Path -> Hash

        # Path -> (the order it was matched in, its own hashes with the initial orientation first, its centre hashes).
        #   _reverse_hashes only has the hash of the group each image joined, so these are what let `regroup`
        #   match everything again at another strength without hashing any of it again.
        self._image_hashes = {}
        self._next_order = 0

        # Matching turned copies: every loaded image's centre hashes are in _centre_index, and _centre_paths says
        #   which images have each of them. Not just each group's first image: that could be a quarter turn of
        #   the original, which no small angle matches, while another image in the group isn't.
        self.rotation = rotation and not exact_match
        self._centre_index = CentreIndex(CENTRE_RADIUS)
        self._centre_paths = {}  # Centre hash -> set of paths
        self._path_centres = {}  # Path -> its centre hashes (and its mirror's)

        # Finds the hashes of _hashes that are close to an image's hashes. Not needed if they must be identical
        self._index = HammingIndex(self.radius) if self.radius and not exact_match else None

        # Pausing and finished signaling...
        self._not_paused = Event()
        self._not_paused.set()
        self._finished = Event()
        self._finished.set()

    @property
    def left_to_process(self):
        """Files that are left to process"""
        return self.found_images - self.processed_images

    def add_path(self, path: str | Path):
        """Add a path for processing"""
        path = str(Path(path).expanduser().resolve())
        self._removed_paths.discard(path)
        self._new_paths.put(path)

    def remove_path(self, folder: str | Path):
        """
        Mark a folder to be skipped going forward, and remove already-indexed files
        that live under it. Pauses briefly if not already paused to keep state sane.
        """
        # TODO: This works but the biggest problem with it is that it will not remove any images which are still
        #   queue'd up for processing in the ThreadPool... I'm not sure how to fix that yet.
        folder = str(Path(folder).expanduser().resolve())
        paused = self.conditional_pause()
        self._removed_paths.add(folder)

        # Remove anything we've already seen under that folder
        # (iterate over a copy because remove() mutates structures)
        to_remove = [p for p in self._reverse_hashes if _is_under(folder, p.path)]
        for p in to_remove:
            self.remove(p)

        to_remove_zips = [p for p in self._processed_zips if _is_under(folder, p)]
        for p in to_remove_zips:
            self._processed_zips.pop(p)

        self.conditional_resume(paused)

    def conditional_pause(self) -> bool:
        """Pause if not paused and return if was paused"""
        _conditional_pause = self.is_paused()
        if not _conditional_pause:
            logger.debug('Performing conditional pause')
            self.pause()

        return _conditional_pause

    def conditional_resume(self, was_paused: bool):  # ruff: ignore[boolean-type-hint-positional-argument]
        """Resume if not paused previous (from call to `conditional_pause`)"""
        if not was_paused and not self.is_finished():
            logger.debug('Performing conditional resume')
            self.resume()

    def pause(self):
        """Pause processing"""
        logger.debug('Performing pause')
        self._not_paused.clear()

    def is_paused(self):
        """Is processing paused"""
        return not self._not_paused.is_set()

    def finish(self):
        """Finish processing"""
        logger.debug('Performing finished')
        self._finished.set()

    def is_finished(self):
        """Is processing finished"""
        return self._finished.is_set()

    def resume(self):
        """Resume processing"""
        logger.debug('Performing resume')
        self._not_paused.set()

    def running(self):
        """Currently running and loading hashes?"""
        return not self.is_paused() and (not self.is_finished() or self.left_to_process)

    def remove(self, path):
        """Remove a loaded path completely from the image matching system. Will not delete a file."""
        # Pause things while we remove things...
        logger.info('Removing %s from %s', path, self.__class__.__name__)
        paused = self.conditional_pause()

        hash_ = self._reverse_hashes.pop(path)
        self._image_hashes.pop(path, None)
        self._remove_path_centres(path)
        self._hashes[hash_].matches.remove(path)
        if len(self._hashes[hash_].matches) == 1:
            match_i = self._hashes[hash_].match_i
            logger.debug('Unmatching match group %s', match_i)
            self._hashes[hash_].match_i = None

            del self.matches[match_i]
            self.refresh_match_indexes(match_i)
            self.duplicate_images -= 2

        elif not self._hashes[hash_].matches:
            logger.debug('Removing empty match group')
            del self._hashes[hash_]
            if self._index is not None:
                self._index.remove(hash_)

        else:
            logger.debug('Simple removal performed')
            self.duplicate_images -= 1

        self.processed_images -= 1
        self.found_images -= 1
        self.conditional_resume(paused)

    def ignore(self, path):
        """Remove a path from the image matching service"""
        try:
            self.remove(path)
        except KeyError:
            logger.info("Failed to remove file %s, probably due to it being deleted already", path)

        if path.path_obj.suffix.lower() != '.zip':
            self._ignored_files.add(path.path)

    def rename(self, old_path: ZipPath, new_path: ZipPath):
        """Update a loaded path after its file was moved on disk. Will not move a file."""
        if old_path == new_path:
            return

        logger.info('Renaming %s to %s in %s', old_path, new_path, self.__class__.__name__)
        paused = self.conditional_pause()
        try:
            if new_path in self._reverse_hashes:
                # The file that was at new_path has been replaced by this one
                self.remove(new_path)

            hash_ = self._reverse_hashes.pop(old_path)
            self._reverse_hashes[new_path] = hash_
            self._image_hashes[new_path] = self._image_hashes.pop(old_path)  # Keeps its order for regrouping
            if old_path in self._path_centres:
                stored = self._path_centres[old_path]
                self._remove_path_centres(old_path)
                self._add_path_centres(new_path, stored)

            # Replace it in place, so the file keeps its position (column) in its group
            matches = self._hashes[hash_].matches
            matches[matches.index(old_path)] = new_path
        finally:
            self.conditional_resume(paused)

    def refresh_match_indexes(self, start=0):
        """Update the match_i value for all the matches passed a certain point"""
        for match_i, match in enumerate(self.matches[start:], start=start):
            match.match_i = match_i

    def _find_group(self, hashes: set, centre: bytes | None = None) -> int | str | None:
        """
        Find the group an image belongs to.

        Args:
            hashes (set): All of the image's hashes (every rotation and mirror of it).
            centre (bytes | None): The image's centre hashes. Only used if rotation is on, and no group matched
                the image as a whole.

        Returns:
            int | str | None: The key in _hashes of the closest group, or None if no group is close enough.
        """
        if self._index is None:
            # From testing at ~1.5m loaded images: it is ~10% faster to return a set and do this than it is to
            #   iterate over a list and do an `is in` check for each hash
            found_hashes = self._hashes.keys() & hashes
            if found_hashes:
                return min(found_hashes)
        else:
            closest = None
            for hash_ in hashes:
                for found_hash, distance in self._index.find(hash_).items():
                    if closest is None or (distance, found_hash) < closest:
                        closest = (distance, found_hash)

            if closest is not None:
                return closest[1]

        if not self.rotation or centre is None:
            return None

        # Nothing matched it as a whole, so is it a turned copy of a loaded image (or the other way around)?
        #   Only the turned queries are looked up, and an image only counts if turning explains the match better than
        #   not turning: an image whose middle matches as is (a recolored copy, say) is up to the strength to match.
        as_is, *turned = phash.unpack_centre_hashes(centre)[1]
        closest = None
        for query in turned:
            for found_centre, distance in self._centre_index.find(query).items():
                for found_path in self._centre_paths[found_centre]:
                    key = self._reverse_hashes[found_path]
                    if closest is not None and (distance, key) >= closest:
                        continue

                    if distance < min((as_is ^ stored).bit_count() for stored in self._path_centres[found_path]):
                        closest = (distance, key)

        return None if closest is None else closest[1]

    def _add_path_centres(self, path: ZipPath, stored: tuple[int, int]):
        """This image is loaded, so turned copies of it can find its group by its centre hashes"""
        self._path_centres[path] = stored
        for centre_hash in set(stored):
            paths = self._centre_paths.setdefault(centre_hash, set())
            if not paths:
                self._centre_index.add(centre_hash)
            paths.add(path)

    def _remove_path_centres(self, path: ZipPath):
        """This image is gone, so stop finding anything by its centre hashes"""
        for centre_hash in set(self._path_centres.pop(path, ())):
            paths = self._centre_paths[centre_hash]
            paths.discard(path)
            if not paths:
                del self._centre_paths[centre_hash]
                self._centre_index.remove(centre_hash)

    def _process_image_callback(self, result):
        """
        Handle the result of hashing an image.

        This needs to do quite a few things including sanitizing the results,
            actually checking if the hash matches an existing image,
            adding the image and any matches to the backend data structures, notify any listeners,
            update the found and processed image counts,
            and verify that this result wasn't added as a removed path since it was queued.

        Args:
            result: A tuple consisting of the path to the file, and the resultant hashes.
                If the hashes are a dict, then it is assumed that the path is for a zip. In that case,
                the individual zip files will sanitized and re-ran through this callback.
        """
        # TODO: This callback must return IMMEDIATELY and is currently too slow for large amounts of zips.
        #   Perhaps create a new queue/thread and queue up processing for zip results?
        #   I think the major slow point is adding to the data structures and I'm not sure if more threads will help
        # Check for paused or finished signals
        self._not_paused.wait()
        if self.is_finished():
            return

        # region Sanitize results
        path: Path | str | ZipPath
        path, hashes = result

        if any(_is_under(d, path.path if isinstance(path, ZipPath) else path) for d in self._removed_paths):
            # This image was removed AFTER it was queue'd! So decrement the found images count and just leave...
            self.found_images -= 1
            return

        if isinstance(hashes, dict):
            self.found_images -= 1
            subpaths = []
            for sub_path, sub_hashes in hashes.items():
                self.found_images += 1
                subpaths.append(ZipPath(str(path), sub_path))
                self._process_image_callback((subpaths[-1], sub_hashes))
            self._processed_zips[str(path)] = subpaths
            return

        initial_hash, extra_hashes, centre = hashes
        extra_hashes.add(initial_hash)
        if not isinstance(path, ZipPath):
            # From this point on, EVERYTHING should be a ZipPath
            path = ZipPath(str(path), "")
        # endregion

        if path in self._reverse_hashes:
            self.found_images -= 1
            return

        self.processed_images += 1
        self._image_hashes[path] = (self._next_order, (initial_hash, *(extra_hashes - {initial_hash})), centre)
        self._next_order += 1
        self._add_image(path, initial_hash, extra_hashes, centre)

    def _add_image(self, path: ZipPath, initial_hash: int | str, hashes: set, centre: bytes | None = None, *,
                   emit: bool = True):
        """
        Put an image in the group it matches, or in a group of its own if it matches none.

        Args:
            path (ZipPath): The image.
            initial_hash (int | str): The hash of the image as displayed.
            hashes (set): All of the image's hashes, including initial_hash.
            centre (bytes | None): The image's centre hashes, for matching turned copies.
            emit (bool): Send events for new groups and matches? Regrouping doesn't, the GUI redraws everything.
        """
        hash_ = self._find_group(hashes, centre)
        if hash_ is None:
            # This is a new image not matching any previous, so just add it to the hashmap and move on...
            #   Just use the initial orientation
            hash_ = initial_hash
            self._reverse_hashes[path] = hash_
            self._hashes[hash_].matches.append(path)
            if self._index is not None:
                self._index.add(hash_)
            if self.rotation and centre is not None:
                self._add_path_centres(path, phash.unpack_centre_hashes(centre)[0])
            return

        # We have found a match!
        self._reverse_hashes[path] = hash_
        self._hashes[hash_].matches.append(path)
        if self.rotation and centre is not None:
            self._add_path_centres(path, phash.unpack_centre_hashes(centre)[0])

        if self._hashes[hash_].match_i is None and len(self._hashes[hash_].matches) >= 2:
            # This is a brand new match group!
            self._hashes[hash_].match_i = len(self.matches)
            self.matches.append(self._hashes[hash_])
            self.duplicate_images += 2
            if emit:
                self.events.put(NewGroup(self._hashes[hash_].snapshot()))
                logger.debug('New match group found: %s', self._hashes[hash_].matches)
        else:
            # Just another match for an existing group...
            self.duplicate_images += 1
            if emit:
                self.events.put(NewMatch(self._hashes[hash_].snapshot(), path))
                logger.debug('New match found for group #%s: %s',
                             self._hashes[hash_].match_i,
                             self._hashes[hash_].matches)

    def regroup(self, strength: int, *, rotation: bool | None = None):
        """
        Group the loaded images again at another strength (or with rotation on or off), without hashing them again.

        Images are matched in the same order they were the first time, with the same logic, so the groups are what
            matching at this strength would have made from the same hashes.
        Processing must be finished (or stopped) first, since this rebuilds everything the callbacks write to.

        Args:
            strength (int): The 0-10 strength to group at.
            rotation (bool | None): Match copies turned by a few degrees? None leaves it as it is.

        Raises:
            ValueError: The strength is out of range, or the images have SHA-256 hashes, which only match identical
                files at any strength.
            RuntimeError: Processing is still running.
        """
        if not (0 <= strength <= 10):
            raise ValueError("Strength must be between 0 and 10!")

        if self.exact_match:
            raise ValueError("SHA-256 hashes only match identical files, so they can't be grouped at another strength")

        if not self.is_finished():
            raise RuntimeError("Can't regroup while processing is running")

        self.strength = strength
        self.radius = STRENGTH_RADIUS[strength]
        if rotation is not None:
            self.rotation = rotation
        self.matches = []
        self.duplicate_images = 0
        self._hashes = defaultdict(ImageMatch)
        self._reverse_hashes = {}
        self._index = HammingIndex(self.radius) if self.radius else None
        self._centre_index = CentreIndex(CENTRE_RADIUS)
        self._centre_paths = {}
        self._path_centres = {}

        for path, (_, hashes, centre) in sorted(self._image_hashes.items(), key=lambda item: item[1][0]):
            self._add_image(path, hashes[0], set(hashes), centre, emit=False)

    def _process_image_error_callback(self, e):
        """Temporary for testing"""
        self.processed_images += 1
        logger.error("%s: %s (input path %s)", type(e).__name__, e, e.input_path)

    def _root_stream(self):
        """This is to yield any paths for processing, then wait until processing is finished for any new paths"""
        while not self._new_paths.empty() or self.left_to_process:
            if self.is_finished():
                return

            if self._new_paths.empty():
                time.sleep(0.05)
                continue

            yield self._new_paths.get_nowait()

    def run(self, paths: list[str | Path]):
        """Do the work of matching!"""
        self._not_paused.set()
        self._finished.clear()

        for path in paths:
            self.add_path(path)

        with Pool(self.processes) as tp:
            for path in self._root_stream():
                path = Path(path)
                if not path.is_dir():
                    logger.warning('A path was entered that was not a directory : %s', path)
                    continue

                path = str(path.expanduser().resolve())
                if path in self._removed_paths:
                    continue

                for root, dirs, files in os.walk(path):
                    if self.is_finished():
                        break

                    dirs.sort()  # This actually works to ensure that os.walk goes in alphabetical order!
                    root = Path(root)

                    if any(_is_under(d, root) for d in self._removed_paths):
                        continue

                    for f in files:
                        self._not_paused.wait()
                        if self.is_finished():
                            break

                        f = root / f

                        if f.suffix.lower() not in self.extensions:
                            continue

                        if any(_is_under(d, f) for d in self._removed_paths):
                            continue

                        if str(f) in self._ignored_files:
                            continue

                        if f.suffix.lower() == '.zip':
                            if str(f.expanduser().resolve()) in self._processed_zips:
                                continue
                        elif ZipPath(str(f), "") in self._reverse_hashes:
                            continue

                        self.found_images += 1
                        tp.apply_async(
                            _process_image,
                            args=(f, ),
                            kwds={
                                'supported_extensions': self.extensions,
                                'exact_match': self.exact_match,
                                'draft': self.radius >= DRAFT_RADIUS,
                            },
                            callback=self._process_image_callback,
                            error_callback=self._process_image_error_callback,
                        )

            tp.close()

            if not self.is_finished():
                tp.join()

        # Leaving the pool terminates it, so if processing was stopped, whatever was still queued was never hashed.
        #   Forget about those files so the counts are right, and the next run finds them again.
        self.found_images = self.processed_images
        self._finished.set()
        self.events.put(Finished())

    def stop(self):
        """Stop processing. The images hashed so far are kept, and running again picks up the rest"""
        logger.debug('Performing stop')
        self._finished.set()
        self._not_paused.set()  # Anything waiting out a pause has to wake up to see that it should stop
