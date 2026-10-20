"""The images the user has ignored, saved between runs"""
import logging
import os
import sqlite3

from pathlib import Path
from typing import Iterable, Sequence

from pixmatch import ZipPath

logger = logging.getLogger(__name__)


def data_folder() -> Path:
    """
    Where PixMatch keeps what it saves between runs (other than settings, which are in QSettings).

    Returns:
        Path: The PixMatch folder in %LOCALAPPDATA% on Windows (~/.local/share/pixmatch elsewhere), or the
            PIXMATCH_DATA_DIR environment variable's folder if it's set (tests use it, to leave the real data alone).
    """
    if os.environ.get("PIXMATCH_DATA_DIR"):
        return Path(os.environ["PIXMATCH_DATA_DIR"])

    if os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "PixMatch"

    return Path.home() / ".local" / "share" / "pixmatch"


class IgnoredImages:
    """
    The images the user has ignored (marked IGNORED), saved to a file as soon as they change.

    Ignoring an image doesn't take it out of matching: it still finds its duplicates, and a group is only hidden when
        every image in it is ignored (see keeps_group). So when a new duplicate of an ignored image turns up, its group
        is shown again, with the ignored images still marked IGNORED.

    Images are kept by their path (and, for an image in a zip, its path in the zip). Checking an image is a set lookup,
        so it's cheap enough for group filters, which run whenever a group changes.
    """

    FILE_NAME = "ignored.db"

    def __init__(self, folder: Path | None = None):
        folder = data_folder() if folder is None else folder
        self._keys: set[tuple[str, str]] = set()
        self._db = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(folder / self.FILE_NAME)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS ignored (path TEXT NOT NULL, subpath TEXT NOT NULL, "
                "PRIMARY KEY (path, subpath)) WITHOUT ROWID",
            )
            self._keys = set(self._db.execute("SELECT path, subpath FROM ignored"))
        except sqlite3.Error as e:
            # Ignoring still works, it just isn't saved
            logger.warning("Couldn't open the ignored images file in %s: %s", folder, e)
            self._db = None

    @staticmethod
    def _key(path: ZipPath) -> tuple[str, str]:
        return path.path, path.subpath or ""

    def __contains__(self, path: ZipPath) -> bool:
        return self._key(path) in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, paths: Iterable[ZipPath]):
        """Ignore these images"""
        self._save("INSERT OR IGNORE INTO ignored VALUES (?, ?)", self._keys.add, paths)

    def discard(self, paths: Iterable[ZipPath]):
        """Stop ignoring these images (if they were)"""
        self._save("DELETE FROM ignored WHERE path = ? AND subpath = ?", self._keys.discard, paths)

    def _save(self, sql: str, change, paths: Iterable[ZipPath]):
        """Change the set, and the file in one go"""
        keys = [self._key(path) for path in paths]
        for key in keys:
            change(key)

        if self._db is not None and keys:
            try:
                with self._db:
                    self._db.executemany(sql, keys)
            except sqlite3.Error as e:
                logger.warning("Couldn't save the ignored images: %s", e)

    def keeps_group(self, images: Sequence[ZipPath]) -> bool:
        """A group filter (see ImageMatcher.set_group_filters) that hides the groups whose images are all ignored"""
        keys = self._keys
        return not all((image.path, image.subpath or "") in keys for image in images)

    def close(self):
        """Close the file"""
        if self._db is not None:
            self._db.close()
            self._db = None
