import io
import math
import subprocess as sp
import sys

from collections import OrderedDict, defaultdict
from contextlib import ExitStack, nullcontext, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import cache, lru_cache
from pathlib import Path
from typing import Callable, Sequence
from zipfile import BadZipFile, ZipFile

import numpy as np

from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from pixmatch import ZipPath
from pixmatch.gui.utils import MAX_SIZE_POLICY, NO_MARGIN
from pixmatch.utils import human_bytes

ZIP_ICON_PATH = Path(__file__).resolve().parent / 'zip.png'


class SelectionState(Enum):
    """
    Per-thumbnail action state.

    The Delete and Move actions act on every MARK file, and Save ignored on every IGNORE one.
    """
    KEEP = 'keep'
    MARK = 'mark'
    IGNORE = 'ignore'


@dataclass(frozen=True)
class SelectionValue:
    """A selection state"""
    state: SelectionState


class SelectionValues:
    """Similar to the SelectionState enum, but for SelectionValue instances"""
    KEEP = SelectionValue(SelectionState.KEEP)
    MARK = SelectionValue(SelectionState.MARK)
    IGNORE = SelectionValue(SelectionState.IGNORE)


class TileAction:
    """The names of the tile context menu actions that the main window handles"""
    IGNORE_GROUP = "ignore_group"
    UNMARK_GROUP = "unmark_group"
    IGNORE_ZIP = "ignore_zip"
    UNMARK_ZIP = "unmark_zip"
    KEEP_DELETE_OTHERS = "keep_delete_others"
    MOVE_LEAVE_OTHERS = "move_leave_others"
    MOVE_DELETE_OTHERS = "move_delete_others"


STATE_ORDER = [SelectionValues.KEEP, SelectionValues.MARK, SelectionValues.IGNORE]


@dataclass(frozen=True)
class StateBannerStyle:
    """How a tile shows a state: a tint over the thumbnail, and a labeled band across it"""
    text: str
    band: QtGui.QColor
    text_color: QtGui.QColor
    tint: QtGui.QColor


# Keeping a file has no banner, so the thumbnail shows as is
STATE_BANNERS = {
    SelectionState.MARK: StateBannerStyle(
        "MARKED", QtGui.QColor(170, 20, 20, 220), QtGui.QColor(255, 228, 228), QtGui.QColor(180, 30, 30, 80),
    ),
    SelectionState.IGNORE: StateBannerStyle(
        "IGNORED", QtGui.QColor(55, 55, 55, 215), QtGui.QColor(225, 225, 225), QtGui.QColor(0, 0, 0, 110),
    ),
}


# region Image view panel
def _load_thumbnail(path: ZipPath, thumb_size: int, zf: ZipFile | None = None) -> QtGui.QImage:
    """
    Load an image from disk as a square thumbnail.

    The image is decoded straight to the thumbnail's size, which lets JPEGs skip most of the work of decoding them
        in full (a 4K JPEG took ~78ms to load at full size, and ~19ms this way).
    If the image is in a zip that is already open, pass it as zf so it isn't opened again.

    Returns:
        QtGui.QImage: The thumbnail, which is null if the image couldn't be read.
    """
    if path.subpath:
        with nullcontext(zf) if zf is not None else ZipFile(path.path) as zip_file:
            return _decode_thumbnail(zip_file.read(path.subpath), thumb_size)

    reader = QtGui.QImageReader(path.path)
    reader.setScaledSize(QtCore.QSize(thumb_size, thumb_size))
    return reader.read()


def _decode_thumbnail(data: bytes, thumb_size: int) -> QtGui.QImage:
    """
    Decode an image's data (a file in a zip) as a square thumbnail, with Pillow.

    Qt could only read this data through a QBuffer made in Python, and that deadlocked the app: while Qt works out how
        to read an image it holds a lock and calls the buffer, and PySide needs the GIL for those calls. So a thread
        loading a thumbnail could hold the lock waiting for the GIL, while the GUI thread held the GIL waiting for the
        lock to load an image of its own (like the preview of an image being hovered over).

    Returns:
        QtGui.QImage: The thumbnail, which is null if the image couldn't be read (like QImageReader's).
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.draft("RGB", (thumb_size, thumb_size))  # JPEGs decode at a fraction of their size, as with QImageReader
            if im.mode.startswith("I;16"):
                # Pillow clips 16-bit values to 0-255 when converting, which turns nearly everything white
                im = Image.fromarray((np.asarray(im) >> 8).astype(np.uint8))
            thumbnail = im.convert("RGBA").resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        return QtGui.QImage()

    # Copied so the QImage has its own pixels, instead of pointing at the bytes (which are about to be freed)
    return QtGui.QImage(thumbnail.tobytes(), thumb_size, thumb_size, QtGui.QImage.Format.Format_RGBA8888).copy()


def movie_sizes(movie: QtGui.QMovie):
    """Get the dimensions of a QMovie, and uncompressed file size"""
    # Every frame is drawn onto the same canvas, so the first frame has the size of them all. Measuring each frame
    #   meant decoding the whole animation (twice per hover), which froze the GUI for ~0.4s per 100 800x500 frames
    movie.jumpToFrame(0)
    first_frame = movie.currentImage()
    return first_frame.size(), first_frame.sizeInBytes() * max(movie.frameCount(), 1)


class ImageViewPane(QtWidgets.QWidget):
    """Container with a stacked image viewer and a bottom overlay status label."""

    def __init__(self, parent=None):
        # This works by having 2 image labels depending on which scaling option was selected
        #   First is the simple scaled label handling scaling
        #   Second is a scroll area with a no-scaling label
        # Then a status label at the bottom
        super().__init__(parent)

        # --- viewers ---
        self.current_path = None
        self._buffer = self._qbytearray = None
        self.scaled = ScaledLabel(contentsMargins=NO_MARGIN, sizePolicy=MAX_SIZE_POLICY)
        self.scaled.setMinimumSize(10, 10)

        self.raw_label = QtWidgets.QLabel()
        self.raw_label.setContentsMargins(NO_MARGIN)
        self.raw_label.setMargin(0)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setContentsMargins(NO_MARGIN)
        self.scroll.setSizePolicy(MAX_SIZE_POLICY)
        self.scroll.setWidget(self.raw_label)

        # Only one visible at a time -> use a stack
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.scaled)
        self.stack.addWidget(self.scroll)

        self.status = ElidedLabel(text="Ready", alignment=QtCore.Qt.AlignmentFlag.AlignBottom)
        self.status.setContentsMargins(NO_MARGIN)
        self.status.setObjectName("imageStatus")
        self.status.setMaximumHeight(16)
        self.status.setStyleSheet("""
            QLabel#imageStatus {
                font-weight: bold;
            }
        """)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(NO_MARGIN)
        lay.addWidget(self.stack)
        lay.addWidget(self.status)

    def set_status(self, text: str):
        """Set text of the status bar"""
        self.status.setText(text)

    def set_index(self, index: int):
        """Set the index of which image view to use. 0 for scaled label, 1 for scroll area"""
        if index not in (0, 1):
            raise ValueError('Valid index must be 0 or 1 for the image pane to select!')

        self.stack.setCurrentIndex(index)

        if self.current_path:
            self.set_image(self.current_path)

        self.update()

    def clear(self):
        """Clear and reset the current object, and the two sub-objects"""
        self.raw_label.clear()
        self.scaled.clear()

        self.get_movie.cache_clear()
        self.get_pixmap.cache_clear()

    @lru_cache(maxsize=5)  # ruff: ignore[cached-instance-method]
    def get_movie(self, path: ZipPath) -> tuple[QtGui.QMovie, int, tuple]:
        """Load a QMovie and details from either a zip or just the file system"""
        file_size = modified = None
        # We're setting a movie...
        if path.subpath:
            # Need to load movie from a zipfile
            with ZipFile(path.path) as zf:
                st = zf.getinfo(path.subpath)
                modified = st.date_time
                file_size = st.file_size
                self._qbytearray = QtCore.QByteArray(zf.read(path.subpath))
                self._buffer = QtCore.QBuffer(self._qbytearray)
                self._buffer.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)

                movie = QtGui.QMovie()
                movie.setDevice(self._buffer)
        else:
            # Basic movie path
            movie = QtGui.QMovie(str(path.path))

        return movie, file_size, modified

    @lru_cache(maxsize=10)  # ruff: ignore[cached-instance-method]
    def get_pixmap(self, path: ZipPath) -> tuple[QtGui.QPixmap, int, tuple]:
        """Load a QPixmap and details from either a zip or just the file system"""
        file_size = modified = None
        # We're setting an image...
        if path.subpath:
            # Need to load image from a zipfile
            try:
                with ZipFile(path.path) as zf:
                    st = zf.getinfo(path.subpath)
                    modified = st.date_time
                    file_size = st.file_size
                    pixmap = QtGui.QPixmap()
                    pixmap.loadFromData(zf.read(path.subpath))
            except KeyError:
                return QtGui.QPixmap(), 0, ()
        else:
            # Basic image path
            pixmap = QtGui.QPixmap(str(path.path))

        return pixmap, file_size, modified

    def set_image(self, path: ZipPath):
        """Set the image to show in the image view area. Can be an animated image."""
        if path == self.current_path:
            return

        self.current_path = path
        file_size = modified = None
        extra = ''
        self.clear()
        if path.is_gif:
            movie, file_size, modified = self.get_movie(path)
            object_size, uncompressed_size = movie_sizes(movie)

            if self.stack.currentIndex() == 0:
                self.scaled.setMovie(movie)
            else:
                self.raw_label.setMovie(movie)

            # WEBP files which aren't animated will appear as movies with a single frame
            #   Thats boring. For the purposes of the statusbar, just treat them as images
            # TODO: Gee, that makes me wonder...
            #   could any image be treated as a movie and we could do away with this whole pixmap or movie thing?
            #   Worth investigating when I have more time...
            if movie.frameCount() > 1:
                extra = f'({human_bytes(uncompressed_size)}, {movie.frameCount()}) '
            else:
                extra = f'({human_bytes(uncompressed_size)}) '
            movie.start()
        else:
            # Not an animated image, just a normal boring pixmap
            pixmap, file_size, modified = self.get_pixmap(path)
            extra = f'({human_bytes(pixmap.toImage().sizeInBytes())}) '
            object_size = pixmap.size()

            if self.stack.currentIndex() == 0:
                self.scaled.setPixmap(pixmap)
            else:
                self.raw_label.setPixmap(pixmap)

        if self.stack.currentIndex() == 1:
            self.raw_label.resize(object_size)

        self.update()

        # region Update status text
        if not path.subpath:
            path = Path(path.path)
            st = path.stat()
            modified = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime('%m/%d/%Y')
            file_size = st.st_size
        elif modified:
            modified = f"{modified[1]}/{modified[2]}/{modified[0]}"

        # If it doesn't fit, shorten the path rather than the details that are for comparing duplicates
        self.status.set_elided_text(
            str(path.absolute()),
            f" ({human_bytes(file_size)} {extra}"
            f"- {object_size.width()},{object_size.height()}px "
            f"- {modified}"
            f")",
        )
        # endregion


class ScaledLabel(QtWidgets.QLabel):
    """
    A version of ScaledLabel but supporting gifs/movies as well as images

    https://stackoverflow.com/questions/72188903
    https://stackoverflow.com/questions/77602181
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._movieSize = QtCore.QSize()
        self._minSize = QtCore.QSize()
        self.object_size = QtCore.QSize()
        self.orig_pixmap = self.pixmap()
        self.orig_movie = self.movie()

    def clear(self):
        """Clear the label, make sure no files are left open"""
        super().clear()
        self.orig_pixmap = None
        if self.orig_movie:
            self.orig_movie.device().close()
            self.orig_movie.stop()
            self.orig_movie.deleteLater()
            self.orig_movie = None

    def minimumSizeHint(self):
        """Hijack minimumSizeHint"""
        # TODO: I don't know that this whole _minSize thing is required
        if self._minSize.isValid():
            return self._minSize
        return super().minimumSizeHint()

    def setPixmap(self, pixmap):
        """Hijack setPixmap"""
        # TODO: I'm not sure why I added this if statement but I think it can go after self.clear?
        if not pixmap:
            return

        self.clear()
        self.orig_pixmap = pixmap
        self._set_scaled_pixmap()

    def _set_scaled_pixmap(self):
        """Show orig_pixmap scaled to fit the label"""
        super().setPixmap(self.orig_pixmap.scaled(self.frameSize(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def resizeEvent(self, event):
        """Rescale the image to fit the new size (movies are scaled as they are painted)"""
        super().resizeEvent(event)
        if self.orig_pixmap and not self.movie():
            self._set_scaled_pixmap()

    def setMovie(self, movie):
        """Hijack setMovie"""
        if self.movie() == movie:
            return
        if self.orig_movie and movie:
            self.clear()
        super().setMovie(movie)
        self.orig_movie = movie

        if not isinstance(movie, QtGui.QMovie) or not movie.isValid():
            self._movieSize = QtCore.QSize()
            self._minSize = QtCore.QSize()
            self.updateGeometry()
            return

        self._movieSize, _ = movie_sizes(movie)
        width = self._movieSize.width()
        height = self._movieSize.height()

        minimum = min(width, height)
        maximum = max(width, height)
        ratio = maximum / minimum
        base = min(4, minimum)
        self._minSize = QtCore.QSize(base, round(base * ratio))
        if minimum == width:
            self._minSize.transpose()

        movie.jumpToFrame(0)
        self.updateGeometry()

    def paintEvent(self, event):
        """Paint the movie scaled to fit the label (images are scaled when they are set, and on resize)"""
        movie = self.movie()
        if not isinstance(movie, QtGui.QMovie) or not movie.isValid():
            # Setting the image again here would schedule another paint, repainting (and rescaling) nonstop
            super().paintEvent(event)
            return

        qp = QtGui.QPainter(self)
        self.drawFrame(qp)

        cr = self.contentsRect()
        margin = self.margin()
        cr.adjust(margin, margin, -margin, -margin)

        style = self.style()
        alignment = style.visualAlignment(self.layoutDirection(), self.alignment())
        maybeSize = self._movieSize.scaled(cr.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        if maybeSize != movie.scaledSize():
            movie.setScaledSize(maybeSize)
            style.drawItemPixmap(
                qp, cr, alignment,
                movie.currentPixmap().scaled(cr.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio),
            )

        else:
            style.drawItemPixmap(
                qp, cr, alignment,
                movie.currentPixmap(),
            )


class ElidedLabel(QtWidgets.QLabel):
    """
    A one line label that shortens its text with "…" to fit its width, instead of growing to fit its text.

    A long file path in the image preview's status used to force the preview pane (and the window) wider.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Don't let the width of the text set a minimum width for the label, and the layouts it's in
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, self.sizePolicy().verticalPolicy())
        self._text = self.text()
        self._suffix = ''

    def setText(self, text: str):
        """Set the text, which is shortened in the middle if it doesn't fit"""
        self.set_elided_text(text)

    def set_elided_text(self, text: str, suffix: str = ''):
        """
        Set the text to shorten in the middle if it doesn't fit, followed by a suffix that's kept whole.

        The suffix is only shortened (at its end) if it doesn't fit even after all of the text has been shortened away.
        """
        self._text = text
        self._suffix = suffix
        self._show_elided_text()

    def resizeEvent(self, event):
        """Shorten the text for the new width"""
        super().resizeEvent(event)
        self._show_elided_text()

    def _show_elided_text(self):
        """Show the text, shortened to fit"""
        self.ensurePolished()  # So the font is the style sheet's (e.g. bold), which is wider
        metrics = self.fontMetrics()
        width = self.contentsRect().width()
        text_width = max(0, width - metrics.horizontalAdvance(self._suffix))
        text = metrics.elidedText(self._text, QtCore.Qt.TextElideMode.ElideMiddle, text_width)
        super().setText(metrics.elidedText(text + self._suffix, QtCore.Qt.TextElideMode.ElideRight, width))
# endregion


# region Thumbnail tile panel
@cache
def get_overlay_icon(height, width):
    """Get the zip overlay icon"""
    return QtGui.QPixmap(ZIP_ICON_PATH).scaled(
        int(height), int(width),
        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        QtCore.Qt.TransformationMode.FastTransformation,
    )


class _ThumbnailJob(QtCore.QRunnable):
    """Loads thumbnails on a ThumbnailLoader's thread pool. A job's paths are all in the same zip, or it has one path"""

    def __init__(self, paths: list[ZipPath], thumb_size: int, loaded: QtCore.SignalInstance):
        super().__init__()
        self.paths = paths
        self.thumb_size = thumb_size
        self.loaded = loaded

    def run(self):
        """Load the thumbnails, handing each one to the GUI thread as it loads"""
        remaining = list(self.paths)
        try:
            with ExitStack() as open_zip:
                zf = None
                while remaining:
                    path = remaining[0]
                    image = QtGui.QImage()
                    # A file (or a file in a zip) that was changed or deleted outside of the app, like a zip whose
                    #   duplicates were cleaned out after it was scanned, gets an empty tile (like images Qt can't read)
                    with suppress(KeyError, OSError, BadZipFile):
                        if path.subpath and zf is None:
                            zf = open_zip.enter_context(ZipFile(path.path))
                        image = _load_thumbnail(path, self.thumb_size, zf)
                    self._report(remaining.pop(0), image)
        finally:
            # If loading failed unexpectedly, still report the rest (as empty) so they aren't left pending
            for path in remaining:
                self._report(path, QtGui.QImage())

    def _report(self, path: ZipPath, image: QtGui.QImage):
        """Hand a thumbnail to the GUI thread"""
        with suppress(RuntimeError):  # The loader was deleted, i.e. the app is closing
            self.loaded.emit(path, self.thumb_size, image)


class ThumbnailLoader(QtCore.QObject):
    """
    Loads thumbnails on a thread pool, and keeps the most recently used ones.

    Loading a page's thumbnails on the GUI thread froze it until every image was decoded (seconds, for large photos).
        This way the page shows straight away, each thumbnail appears as soon as it loads,
        and thumbnails that were shown recently appear instantly.

    Signals:
        thumbnailLoaded(path: ZipPath, thumb_size: int, pixmap: QPixmap): A thumbnail from `get` has loaded.
    """
    thumbnailLoaded = QtCore.Signal(ZipPath, int, QtGui.QPixmap)
    _imageLoaded = QtCore.Signal(ZipPath, int, QtGui.QImage)  # Emitted on the pool's threads, handled on the GUI's

    def __init__(self, parent=None, *, threads: int = 4, cache_size: int = 2000):
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(threads)
        self._cache_size = cache_size
        self._cache = OrderedDict()  # (path, thumb_size) -> QPixmap, least recently used first
        self._pending = set()  # (path, thumb_size) that are loading
        self._imageLoaded.connect(self._on_image_loaded)

        # Images in zips that were asked for, but not started yet: (zip path, thumb_size) -> paths.
        #   They're started together once the GUI is done asking (e.g. once a page is built), see _start_zip_jobs
        self._zip_requests = defaultdict(list)
        self._zip_timer = QtCore.QTimer(self, singleShot=True, interval=0)
        self._zip_timer.timeout.connect(self._start_zip_jobs)

        # The pool waits for every queued thumbnail when it's destroyed, so don't make closing wait for them too
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.cancel)

    def get(self, path: ZipPath, thumb_size: int) -> QtGui.QPixmap | None:
        """
        Get a thumbnail if it was loaded recently, otherwise start loading it.

        Returns:
            QtGui.QPixmap | None: The thumbnail, or None if it is loading. thumbnailLoaded is emitted when it's ready.
        """
        key = (path, thumb_size)
        pixmap = self._cache.get(key)
        if pixmap is not None:
            self._cache.move_to_end(key)
            return pixmap

        if key not in self._pending:
            self._pending.add(key)
            if path.subpath:
                self._zip_requests[path.path, thumb_size].append(path)
                self._zip_timer.start()
            else:
                self._pool.start(_ThumbnailJob([path], thumb_size, self._imageLoaded))

        return None

    def cancel(self):
        """Stop loading thumbnails that haven't started yet, e.g. because the page they were for has been left"""
        self._zip_timer.stop()
        self._zip_requests.clear()
        self._pool.clear()
        self._pending.clear()

    def _start_zip_jobs(self):
        """
        Start loading the images in zips that were asked for.

        Opening a zip reads its whole list of files, in Python: ~9ms for a zip of 3000 files, holding the GIL, which
            slowed down the GUI too. So each zip is opened once per thread that loads images from it,
            rather than once per image.
        """
        for (_, thumb_size), paths in self._zip_requests.items():
            # Every thread gets some of the zip's images, so they still load in parallel (and the top rows first)
            jobs = min(self._pool.maxThreadCount(), len(paths))
            for i in range(jobs):
                self._pool.start(_ThumbnailJob(paths[i::jobs], thumb_size, self._imageLoaded))

        self._zip_requests.clear()

    def _on_image_loaded(self, path: ZipPath, thumb_size: int, image: QtGui.QImage):
        """Keep a thumbnail that finished loading, and pass it on as a QPixmap (which only the GUI thread can make)"""
        key = (path, thumb_size)
        self._pending.discard(key)
        pixmap = QtGui.QPixmap.fromImage(image)
        if not pixmap.isNull():
            # Images that couldn't be read aren't kept, so they're tried again the next time they're shown
            self._cache[key] = pixmap
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        self.thumbnailLoaded.emit(path, thumb_size, pixmap)


class StateBanner(QtWidgets.QWidget):
    """
    Shows a tile's state over its thumbnail: a tint, and a band running diagonally from the bottom left corner to
        the top right one, with the state's name on it. Colored borders were too hard to see.

    Clicks go through it to the tile underneath.
    """

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, on=True)
        self.setGeometry(parent.rect())
        self._style: StateBannerStyle | None = None

    def set_state(self, state: SelectionState):
        """Show this state's banner, or none for a state that doesn't have one"""
        style = STATE_BANNERS.get(state)
        if style != self._style:
            self._style = style
            self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent):
        """Paint the tint and the band"""
        if self._style is None:
            return

        width, height = self.width(), self.height()
        diagonal = (width * width + height * height) ** 0.5

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        painter.fillRect(self.rect(), self._style.tint)

        # The biggest bold text that fits along most of the diagonal, and not taller than a fifth of the tile
        font = QtGui.QFont(self.font())
        font.setBold(True)
        font.setPixelSize(max(6, int(min(width, height) / 5)))
        while font.pixelSize() > 6 and QtGui.QFontMetrics(font).horizontalAdvance(self._style.text) > diagonal * 0.7:
            font.setPixelSize(font.pixelSize() - 1)
        metrics = QtGui.QFontMetrics(font)
        band_height = metrics.height() + 4

        # Turn the painter so its x axis runs up the diagonal, from the bottom left corner to the top right one,
        #   then the band is just a rectangle along it (the widget clips off the parts past the corners)
        painter.translate(width / 2, height / 2)
        painter.rotate(-math.degrees(math.atan2(height, width)))
        band = QtCore.QRectF(-diagonal / 2, -band_height / 2, diagonal, band_height)
        painter.fillRect(band, self._style.band)
        painter.setFont(font)
        painter.setPen(self._style.text_color)
        painter.drawText(band, QtCore.Qt.AlignmentFlag.AlignCenter, self._style.text)


class ThumbnailTile(QtWidgets.QFrame):
    """
    Clickable thumbnail tile that cycles between KEEP → MARK → IGNORE.

    Attributes:
        path: Image path (opaque identifier for the caller).
        stateChanged(path: str, state: SelectionValue): Emitted on state updates.
        hovered(path: str): Emitted on cursor hovering over thumbnail tile.
    """
    stateChanged = QtCore.Signal(ZipPath, SelectionValue)
    hovered = QtCore.Signal(ZipPath)

    # A context menu action that's handled by higher ups (one of the TileAction names), and the tile's path
    menuAction = QtCore.Signal(str, ZipPath)

    def __init__(self, path: ZipPath, pixmap: QtGui.QPixmap | None = None, thumb_size: int = 32, parent=None):
        super().__init__(parent, frameShape=QtWidgets.QFrame.Shape.Box, lineWidth=2)
        self.setObjectName("ThumbnailTile")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self._path = path
        self._state = SelectionValues.KEEP
        self._thumb_size = thumb_size

        self._image = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        if pixmap:
            self._image.setPixmap(pixmap)
        self._image.setFixedSize(thumb_size, thumb_size)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(NO_MARGIN)
        lay.setSpacing(0)
        lay.addWidget(self._image)

        if path.subpath:
            _overlay_icon = QtWidgets.QLabel(self._image)  # child of the tile so it floats over the image
            _overlay_icon.setObjectName("LockOverlay")
            _overlay_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            _overlay_icon.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, on=True)
            _overlay_icon.setFixedSize(thumb_size, thumb_size)  # small badge; adjust later if you want
            _overlay_icon.setPixmap(get_overlay_icon(thumb_size / 1.5, thumb_size / 1.5))

        # Made after the zip icon, so it's drawn over it
        self._banner = StateBanner(self._image)

        self.context_menu = self._build_context_menu()

        self.setStyleSheet(
            """
            QFrame#ThumbnailTile {
                border: 2px solid rgba(90, 90, 90, 220);
                border-radius: 6px;
                background: #202020;
            }
            """,
        )
        self._apply_state_style()

    def _build_context_menu(self) -> QtWidgets.QMenu:
        """
        Build the right click menu.

        The first sections mark files, which nothing happens to until an action (Delete, Move, Save ignored) is
            pressed. The "this, others" section moves or deletes files right away.

        Returns:
            QtWidgets.QMenu: The menu.
        """
        menu = QtWidgets.QMenu(self)
        in_zip = self._path.is_zip  # Files in zips are read-only, so they can't be deleted or moved

        def add(text: str, handler, *, enabled: bool = True):
            action = menu.addAction(text)
            action.setEnabled(enabled)
            action.triggered.connect(lambda _=False: handler())

        def emit(action: str) -> Callable[[], None]:
            return lambda: self.menuAction.emit(action, self._path)

        add("Mark", lambda: self.change_state_clicked(SelectionValues.MARK), enabled=not in_zip)
        add("Ignore", lambda: self.change_state_clicked(SelectionValues.IGNORE))
        add("Un-select", lambda: self.change_state_clicked(SelectionValues.KEEP))
        menu.addSeparator()
        add("Ignore group", emit(TileAction.IGNORE_GROUP))
        add("Un-select group", emit(TileAction.UNMARK_GROUP))
        menu.addSeparator()
        add("Keep this, delete others", emit(TileAction.KEEP_DELETE_OTHERS))
        add("Move this, leave others...", emit(TileAction.MOVE_LEAVE_OTHERS), enabled=not in_zip)
        add("Move this, delete others...", emit(TileAction.MOVE_DELETE_OTHERS), enabled=not in_zip)
        if in_zip:
            menu.addSeparator()
            add("Ignore zip", emit(TileAction.IGNORE_ZIP))
            add("Un-select zip", emit(TileAction.UNMARK_ZIP))
        menu.addSeparator()
        add("Open image", self.open_image, enabled=not in_zip)
        add("Open containing folder", self.open_folder)
        add("Copy path", self.copy_path)
        return menu

    def open_image(self):
        """Open the image in the default program for it"""
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self._path.path))

    def open_folder(self):
        """Open the folder the image (or its zip) is in, with it selected if possible"""
        path = Path(self._path.path)
        if sys.platform == "win32":
            sp.Popen(f'explorer /select,"{path}"')
        else:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path.parent)))

    def copy_path(self):
        """Copy the image's path, which for an image in a zip is the zip's path followed by its path in the zip"""
        text = str(Path(self._path.path, self._path.subpath)) if self._path.is_zip else self._path.path
        QtWidgets.QApplication.clipboard().setText(text)

    def set_pixmap(self, pixmap: QtGui.QPixmap):
        """Show the thumbnail image (a tile can be made before its thumbnail has loaded)"""
        self._image.setPixmap(pixmap)

    @property
    def path(self) -> ZipPath:
        """The internal file path"""
        return self._path

    @property
    def state(self) -> SelectionValue:
        """The internal state"""
        return self._state

    @state.setter
    def state(self, state: SelectionValue):
        """Set the tile selection state without cycling."""
        self.silent_set_state(state)
        self.stateChanged.emit(self._path, self._state)

    def silent_set_state(self, state: SelectionValue):
        """Set the state without invoking the stateChanged event"""
        if self._state == state:
            return

        locked = bool(self._path.subpath)
        if locked and state.state == SelectionState.MARK:
            raise ValueError(f"Cannot set a locked file to the {state.state.value} state!")

        self._state = state
        self._apply_state_style()

    def change_state_clicked(self, state: SelectionValue):
        """A change state button has been clicked in the context menu"""
        self.state = state

    def cycle_state(self):
        """Advance KEEP → MARK → IGNORE → KEEP."""
        idx = STATE_ORDER.index(self._state)
        locked = bool(self._path.subpath)
        next_state = STATE_ORDER[(idx + 1) % len(STATE_ORDER)]
        # Files in zips can't be marked, since they can't be deleted or moved:
        if locked and next_state == SelectionValues.MARK:
            next_state = STATE_ORDER[(idx + 2) % len(STATE_ORDER)]
        self.state = next_state

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        """Cycle state on button press"""
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self.cycle_state()
            e.accept()
        else:
            super().mousePressEvent(e)

    def enterEvent(self, e: QtGui.QEnterEvent):
        """Fire hover events"""
        self.hovered.emit(self._path)
        super().enterEvent(e)

    def _apply_state_style(self):
        """Show the state's banner over the thumbnail"""
        self._banner.set_state(self._state.state)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        """Handle opening the context menu"""
        self.context_menu.exec(event.globalPos())
        event.accept()


class DuplicateGroupRow(QtWidgets.QWidget):
    """
    A single row of thumbnails representing one duplicate group.

    Signals:
        tileStateChanged(path: ZipPath, state: SelectionValue)
        tileHovered(path: ZipPath)
    """
    tileStateChanged = QtCore.Signal(ZipPath, SelectionValue)
    tileHovered = QtCore.Signal(ZipPath)
    tileMenuAction = QtCore.Signal(str, ZipPath)

    def __init__(self, images: Sequence[ZipPath], thumb_size: int = 32, parent=None, *, thumbnails: ThumbnailLoader):
        super().__init__(parent)
        self._tiles: list[ThumbnailTile] = []
        self._thumb_size = thumb_size
        self._thumbnails = thumbnails
        self._thumbnails.thumbnailLoaded.connect(self._on_thumbnail_loaded)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(NO_MARGIN)
        self.layout.setSpacing(0)

        for path in images:
            self.add_tile(path)

        self.layout.addStretch(1)

    def __len__(self) -> int:
        """Define len operator for conveince"""
        return len(self._tiles)

    def tiles(self) -> list[ThumbnailTile]:
        """Fetch all the tiles (in a copy)"""
        return list(self._tiles)

    def add_tile(self, path: ZipPath):
        """Add a new tile to this duplicate group"""
        # If the thumbnail wasn't loaded recently, the tile starts empty and it's shown once it loads
        pixmap = self._thumbnails.get(path, self._thumb_size)
        tile = ThumbnailTile(path=path, pixmap=pixmap, thumb_size=self._thumb_size)
        tile.stateChanged.connect(self.tileStateChanged)
        tile.hovered.connect(self.tileHovered)
        tile.menuAction.connect(self.tileMenuAction)
        self._tiles.append(tile)
        self.layout.insertWidget(len(self._tiles) - 1, tile)

    def _on_thumbnail_loaded(self, path: ZipPath, thumb_size: int, pixmap: QtGui.QPixmap):
        """A thumbnail finished loading, so show it on this group's tiles for that image"""
        if thumb_size != self._thumb_size:
            return

        for tile in self._tiles:
            if tile.path == path:
                tile.set_pixmap(pixmap)


class ClickableLabel(QtWidgets.QLabel):
    """QLabel that emits a `clicked` signal on left mouse release."""
    clicked = QtCore.Signal()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Override the release event to emit to the clicked signal"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class DuplicateGroupList(QtWidgets.QWidget):
    """
    Scrollable list of duplicate groups. Each group renders as a row of thumbnails.

    Notes:
        - Clicking a thumbnail cycles KEEP → MARK → IGNORE.
        - Marked tiles have a banner across them saying how (see StateBanner).
    """

    groupTileStateChanged = QtCore.Signal(ZipPath, SelectionValue)  # path, state
    groupTileHovered = QtCore.Signal(ZipPath)
    pageIndicatorClicked = QtCore.Signal()

    groupTileMenuAction = QtCore.Signal(str, ZipPath)

    def __init__(self, parent=None, *, max_rows: int = 25, thumb_size: int = 64, **kwargs):
        super().__init__(parent, **kwargs)
        self._max_rows = max_rows
        self._thumb_size = thumb_size
        self._thumbnails = ThumbnailLoader(self)

        self._scroll = QtWidgets.QScrollArea(widgetResizable=True)
        self._container = QtWidgets.QWidget()
        self._vbox = QtWidgets.QVBoxLayout(self._container)
        self._vbox.setContentsMargins(NO_MARGIN)
        self._vbox.setSpacing(0)
        _tail_spacer = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._vbox.addItem(_tail_spacer)
        self._scroll.setWidget(self._container)

        # Main layout
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(NO_MARGIN)
        outer.setSpacing(0)
        outer.addWidget(self._scroll)

        self._rows: list[DuplicateGroupRow] = []

        # Status bar
        _status = QtWidgets.QHBoxLayout()
        self.first_page = QtWidgets.QPushButton("<<")
        self.page_down = QtWidgets.QPushButton("<")
        self.page_indicator = ClickableLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.page_indicator.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.page_indicator.clicked.connect(self.pageIndicatorClicked)
        self.page_up = QtWidgets.QPushButton(">")
        self.last_page = QtWidgets.QPushButton(">>")
        _status.addWidget(self.first_page)
        _status.addWidget(self.page_down)
        _status.addWidget(self.page_indicator)
        _status.addWidget(self.page_up)
        _status.addWidget(self.last_page)
        outer.addLayout(_status)

        self.update_page_indicator(1, 1)

    def set_max_rows(self, n: int) -> None:
        """Set maximum visible rows (groups)."""
        self._max_rows = max(1, int(n))

    def set_thumb_size(self, px: int) -> None:
        """Set square thumbnail size for subsequent loads."""
        self._thumb_size = max(32, int(px))

    def set_groups(self, groups: Sequence[Sequence[ZipPath]]) -> None:
        """
        Load duplicate groups.

        Args:
            groups: An iterable of groups; each group is an iterable of image file paths.
                    Only the first `max_rows` groups are shown.
        """
        self.clear()
        for group in groups[: self._max_rows]:
            self.add_group(group)

    def update_page_indicator(self, current_page: int, total_pages: int):
        """Update page indicator label"""
        self.page_indicator.setText(f"Page {current_page} of {total_pages}")

    def add_group(self, group: Sequence[ZipPath]) -> None:
        """Add a new match group, requiring a new row"""
        if len(self._rows) == self._max_rows:
            raise ValueError("Cannot add a new group to a fully filled group list!")

        row = DuplicateGroupRow(group, thumb_size=self._thumb_size, thumbnails=self._thumbnails)
        row.tileStateChanged.connect(self.groupTileStateChanged)
        row.tileHovered.connect(self.groupTileHovered)
        row.tileMenuAction.connect(self.groupTileMenuAction)
        tail_index = self._vbox.count() - 1
        self._vbox.insertWidget(tail_index, row)
        self._rows.append(row)

    def reset_states(self):
        """Set all tiles to KEEP."""
        for row in self._rows:
            for tile in row.tiles():
                tile.state = SelectionValues.KEEP

    def clear(self):
        """Reset this widget"""
        self._thumbnails.cancel()  # Any thumbnails still waiting to load were for these rows
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()
# endregion


class DirFileSystemModel(QtWidgets.QFileSystemModel):
    """
    Custom FileSystemModel.

    This is required so that if a directory is opened and it has no children, it still shows as opened...
    """

    def hasChildren(self, /, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = ...):
        """Return true only if there are files as children"""
        file_info = self.fileInfo(parent)
        dir_ = QtCore.QDir(file_info.absoluteFilePath())
        return bool(dir_.entryList(self.filter()))
