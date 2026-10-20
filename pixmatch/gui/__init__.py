# TODO: Validate that users don't select overlapping paths...


import bisect
import logging
import os
import shutil
import time

from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile
from PySide6 import QtCore, QtGui, QtWidgets
from send2trash import send2trash

from pixmatch import Finished, ImageMatcher, NewGroup, NewMatch, ZipPath, _is_under, has_image_outside_zips
from pixmatch.gui.utils import MAX_SIZE_POLICY, NO_MARGIN
from pixmatch.gui.widgets import (
    LISTED_MIME,
    DirFileSystemModel,
    DuplicateGroupList,
    FolderButton,
    FolderListWidget,
    ImageViewPane,
    SelectionState,
    SelectionValue,
    SelectionValues,
    TileAction,
    folder_button_icon,
    same_folder,
)
from pixmatch.ignored import IgnoredImages
from pixmatch.utils import human_bytes

# Allow damaged images, and very large ones. These Pillow settings apply to the whole process, so the app sets them
#   rather than pixmatch setting them for anything that imports it. ImageMatcher passes them on to the processes that
#   hash images.
#   Pillow warns about images over ~89 megapixels as a possible "decompression bomb" (a small file made to fill memory
#   when it's opened), and refuses ones over twice that. The images here are the user's own, so that's no reason to
#   leave a large photo or scan out.
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

ICON_PATH = Path(__file__).resolve().parent / 'pixmatch.ico'

# The options that hide groups, as they're named in the Options menu, and the group filter each one turns on
#   (see ImageMatcher.set_group_filters)
GROUP_FILTER_OPTIONS = {
    "Hide groups only in zips": has_image_outside_zips,
}

logger = logging.getLogger(__name__)


class FileAction(Enum):
    """What the Delete and Move actions do"""
    DELETE = 'delete'
    MOVE = 'move'


def project_version() -> str:
    """Get the version string to display in the title bar"""
    try:
        return version("pixmatch")
    except PackageNotFoundError:
        return "0.0.0+unknown"


class WorkerSignals(QtCore.QObject):
    """
    Signals from a running worker thread.

    new_group
        Signaled with the NewGroup event when a new match group is found.

    new_match
        Signaled with the NewMatch event when a new match is found.

    finish
        Signaled when execution is finished, and was not killed.
    """

    new_group = QtCore.Signal(object)
    new_match = QtCore.Signal(object)
    finish = QtCore.Signal()


class ProcessorThread(QtCore.QRunnable):
    """Handles executing the processor and transferring data from the processor to the GUI"""

    def __init__(self, processor, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.processor = processor
        self.signals = WorkerSignals()

        # timer lives on the GUI thread; it polls the library queue
        self._poller = QtCore.QTimer()
        self._poller.setInterval(250)
        self._poller.timeout.connect(self._drain_events)
        self._poller.start()

    def _drain_events(self):
        """Take events from the queue and forward them to the signals"""
        while not self.processor.events.empty():
            evt = self.processor.events.get_nowait()
            if isinstance(evt, NewGroup):
                self.signals.new_group.emit(evt)
            elif isinstance(evt, NewMatch):
                self.signals.new_match.emit(evt)
            elif isinstance(evt, Finished):
                self._poller.stop()
                self.signals.finish.emit()

    def run(self):
        """Execute the processor"""
        self.processor.run(*self.args, **self.kwargs)


class RegroupSignals(QtCore.QObject):
    """
    Signals from a running regroup.

    progress(done, total): How many images are regrouped so far, of how many.
    done(seconds): It finished, in this many seconds.
    cancelled(): It was cancelled, and the groups are back to what they were.
    failed(error): It failed.
    """

    progress = QtCore.Signal(int, int)
    done = QtCore.Signal(float)
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str)


class RegroupThread(QtCore.QRunnable):
    """Regroups the processor's images at another strength (or with rotation on or off) off of the GUI thread"""

    def __init__(self, processor: ImageMatcher, strength: int, *, rotation: bool):
        super().__init__()
        self.processor = processor
        self.strength = strength
        self.rotation = rotation
        self.signals = RegroupSignals()
        self.cancel_requested = False  # Set from the GUI thread to cancel, see cancel

    def cancel(self):
        """Stop the regroup at its next progress report, and put the groups back to what they were"""
        self.cancel_requested = True

    def _on_progress(self, done: int, total: int) -> bool:
        """Report how far it is, and say whether to carry on"""
        self.signals.progress.emit(done, total)
        return not self.cancel_requested

    def run(self):
        """Do the regroup"""
        start = time.perf_counter()
        try:
            finished = self.processor.regroup(self.strength, rotation=self.rotation, progress=self._on_progress)
        except Exception as e:
            logger.exception("Regrouping at strength %s failed", self.strength)
            self.signals.failed.emit(f"{type(e).__name__}: {e}")
        else:
            if finished:
                self.signals.done.emit(time.perf_counter() - start)
            else:
                self.signals.cancelled.emit()


class MainWindow(QtWidgets.QMainWindow):
    """
    The main application window emulating VisiPics' primary workflow.

    Key elements:
      - Toolbar with folder selection (stub), Load Test Image, Similarity slider
      - Central area that shows one group/page at a time
      - Prev/Next page navigation
    """

    def __init__(self, start_paths=None):
        super().__init__()
        self.setWindowTitle(f"PixMatch v{project_version()}")
        self.resize(1200, 800)

        # State
        self.deleted_files_count = 0
        self.deleted_files_size = 0
        self.current_page: int = 1
        # The index of the first group on each page (see DuplicateGroupList.page_starts), and of the first group on the
        #   page that's shown. The page that's shown always starts where it did, until it's rebuilt
        self._page_starts = [0]
        self._shown_from = 0
        # The seq of the last event the page already has (see update_group_list). Reset it if the processor is replaced
        self._page_seq = 0
        self.processor = None
        # MARK (and KEEP) states of files. Ignored files are in self.ignored instead, which is saved between runs
        self.file_states = {}
        self.ignored = IgnoredImages()
        self._threadpool = QtCore.QThreadPool()

        self.recycle = True

        # Remembers things between sessions, like the last folder files were moved to
        self._settings = QtCore.QSettings("PixMatch", "PixMatch")

        # UI build
        self.build_menubar()
        self.build_central()
        self.build_statusbar()
        self.build_extra()

        for start_path in start_paths or []:
            self.selected_file_path_display.addItem(str(start_path))

        if start_paths:
            self.selected_file_path_display.setCurrentRow(0)

        self.setWindowIcon(QtGui.QIcon(QtGui.QPixmap(ICON_PATH)))
        self.restore_position()

    def restore_position(self):
        """
        Put the window where it was when it was last closed, or in the middle of the screen the first time.

        If that would leave its title bar off every screen (say a monitor was unplugged, or Windows placed it off
            screen), it goes in the middle of the screen instead, so it can always be seen and moved.
        """
        geometry = self._settings.value("window/geometry")
        if geometry is not None and self.restoreGeometry(geometry) and self._title_bar_on_screen():
            return

        self.center_on_screen()

    def _title_bar_on_screen(self) -> bool:
        """Is enough of the window's top edge on a screen to grab it and move the window?"""
        frame = self.frameGeometry()
        top_edge = QtCore.QRect(frame.left(), frame.top(), frame.width(), 30)
        for screen in QtGui.QGuiApplication.screens():
            overlap = screen.availableGeometry().intersected(top_edge)
            if overlap.width() >= 100 and overlap.height() >= 20:
                return True

        return False

    def center_on_screen(self):
        """
        Move the window to the middle of its screen, shrinking it first if it's bigger than the screen.

        It can't shrink past the size its contents need, so on a screen smaller than that its top left corner is kept
            on the screen, where its title bar can be grabbed.
        """
        available = (self.screen() or QtGui.QGuiApplication.primaryScreen()).availableGeometry()
        self.resize(self.size().boundedTo(available.size() * 0.95))
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(max(frame.left(), available.left()), max(frame.top(), available.top()))

    def build_extra(self):
        """Extra items not involved with the main display"""
        self.exit_warning = QtWidgets.QMessageBox(self)
        self.exit_warning.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        self.exit_warning.setWindowTitle("Close?")
        self.exit_warning.setText("Are you sure you want to quit?")
        self.exit_warning.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        self.exit_warning.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
        self.exit_warning.setEscapeButton(QtWidgets.QMessageBox.StandardButton.No)
        self.exit_warning.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

    def build_menubar(self) -> None:
        """Creates the top menu bar"""
        menu = self.menuBar()

        # region File menu
        load_project = QtGui.QAction("Load Project...", self, enabled=False)
        save_project = QtGui.QAction("Save Project...", self, enabled=False)
        exit_project = QtGui.QAction("Exit", self)
        exit_project.triggered.connect(lambda *_: self.close())

        file_menu = menu.addMenu("&File")
        file_menu.addAction(load_project)
        file_menu.addAction(save_project)
        file_menu.addSeparator()
        file_menu.addAction(exit_project)
        # endregion

        # region Edit menu
        self.mark_menu = QtGui.QAction("Mark", self)
        mark_ignore = QtGui.QAction("Ignore", self)
        mark_ignore_group = QtGui.QAction("Ignore Group", self)
        self.mark_ignore_zip_menu = QtGui.QAction("Ignore Zip", self)
        unmark = QtGui.QAction("Un-select", self)
        unmark_group = QtGui.QAction("Un-select Group", self)
        self.unmark_zip_menu = QtGui.QAction("Un-select Zip", self)

        self.mark_menu.triggered.connect(self.mark_file)
        mark_ignore.triggered.connect(self.mark_ignore)
        mark_ignore_group.triggered.connect(self.mark_ignore_group)
        self.mark_ignore_zip_menu.triggered.connect(self.mark_ignore_zip)
        unmark.triggered.connect(self.mark_unmark)
        unmark_group.triggered.connect(self.mark_unmark_group)
        self.unmark_zip_menu.triggered.connect(self.mark_unmark_zip)

        edit_menu = menu.addMenu("&Edit")
        edit_menu.addAction(self.mark_menu)
        edit_menu.addSeparator()
        edit_menu.addAction(mark_ignore)
        edit_menu.addAction(mark_ignore_group)
        edit_menu.addAction(self.mark_ignore_zip_menu)
        edit_menu.addSeparator()
        edit_menu.addAction(unmark)
        edit_menu.addAction(unmark_group)
        edit_menu.addAction(self.unmark_zip_menu)
        # endregion

        # region View menu
        page_next = QtGui.QAction("Next page", self)
        page_next.triggered.connect(self.on_page_up)
        page_back = QtGui.QAction("Previous page", self)
        page_back.triggered.connect(self.on_page_down)
        self.preview_resized = QtGui.QAction("Preview resized", self, checked=True, checkable=True)
        preview_full_size = QtGui.QAction("Preview full size", self, checkable=True)

        preview_options_grp = QtGui.QActionGroup(self)
        preview_options_grp.setExclusive(True)
        preview_options_grp.addAction(preview_full_size)
        preview_options_grp.addAction(self.preview_resized)

        # TODO: I'm not sure why we would want to slow display?
        #   and show differences has never worked for me
        # QtGui.QAction("Slow preview display", self)
        # QtGui.QAction("Show differences", self)

        view_menu = menu.addMenu("&View")
        view_menu.addAction(page_next)
        view_menu.addAction(page_back)
        view_menu.addSeparator()
        view_menu.addAction(self.preview_resized)
        view_menu.addAction(preview_full_size)
        # endregion

        # region Tools menu
        autoselect = QtGui.QAction("Auto-select", self)
        autoselect.setEnabled(False)  # TODO:

        tool_menu = menu.addMenu("&Tools")
        tool_menu.addAction(autoselect)
        # endregion

        # region Actions menu
        run_move = QtGui.QAction("Move", self)
        run_move.triggered.connect(self.on_move)
        run_delete = QtGui.QAction("Delete", self)
        run_delete.triggered.connect(self.on_delete)
        actions_menu = menu.addMenu("&Actions")
        actions_menu.addAction(run_move)
        actions_menu.addAction(run_delete)
        # endregion

        # region Options menu
        option_hidden_folders = QtGui.QAction("Show hidden folders", self, checkable=True)
        option_subfolders = QtGui.QAction("Include subfolders", self, checkable=True, checked=True, enabled=False)
        option_rotations = QtGui.QAction("Scan for rotations", self, checkable=True, checked=True, enabled=False)
        option_recycle = QtGui.QAction("Use recycle bin", self, checkable=True, checked=True)

        # TODO: I'm not sure what these two do... I'm willing to add them if someone needs them but I don't.
        # ... = QtGui.QAction("Between folders only", self)
        # ... = QtGui.QAction("Loosen filter automatically", self)

        # Groups whose images are all ignored are hidden, unless this is checked (to un-ignore them)
        self.show_ignored_option = QtGui.QAction("Show fully ignored groups", self, checkable=True)
        self.show_ignored_option.triggered.connect(self.change_group_filters)

        # Each option that hides groups turns on a group filter, see group_filters
        self._group_filter_options = []  # (option, its group filter)
        for text, group_filter in GROUP_FILTER_OPTIONS.items():
            option = QtGui.QAction(text, self, checkable=True)
            option.triggered.connect(self.change_group_filters)
            self._group_filter_options.append((option, group_filter))

        option_hidden_folders.triggered.connect(self.change_hidden_folders_state)
        option_recycle.triggered.connect(self.change_recycle_state)

        options_menu = menu.addMenu("&Options")
        options_menu.addAction(option_hidden_folders)
        options_menu.addAction(option_subfolders)
        options_menu.addSeparator()
        options_menu.addAction(option_rotations)
        options_menu.addSeparator()
        for option, _ in self._group_filter_options:
            options_menu.addAction(option)
        options_menu.addAction(self.show_ignored_option)
        options_menu.addSeparator()
        options_menu.addAction(option_recycle)
        # endregion

    def build_central(self) -> None:
        """Creates the central stacked widget area for group pages."""
        style = QtWidgets.QApplication.instance().style()

        # region General controls area (top-right)
        # region Control buttons
        autoselect_btn = QtWidgets.QPushButton("Auto-select")
        autoselect_btn.setEnabled(False)  # TODO:

        tools = QtWidgets.QVBoxLayout()
        tools.addWidget(autoselect_btn)

        tool_box = QtWidgets.QGroupBox("Tools")
        tool_box.setLayout(tools)
        tool_box.setMaximumHeight(60)

        move_btn = QtWidgets.QPushButton("Move")
        move_btn.pressed.connect(self.on_move)
        delete_btn = QtWidgets.QPushButton("Delete")
        delete_btn.pressed.connect(self.on_delete)

        actions = QtWidgets.QVBoxLayout()
        actions.addWidget(move_btn)
        actions.addWidget(delete_btn)

        actions_box = QtWidgets.QGroupBox("Actions")
        actions_box.setLayout(actions)
        actions_box.setMaximumHeight(20 + 40 * 2)

        general_controls_btns = QtWidgets.QVBoxLayout()
        general_controls_btns.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        general_controls_btns.setContentsMargins(NO_MARGIN)
        general_controls_btns.addWidget(tool_box)
        general_controls_btns.addWidget(actions_box)
        # endregion

        # region Run Controls
        self.stop_btn = QtWidgets.QPushButton("\u25A0")
        self.stop_btn.setSizePolicy(MAX_SIZE_POLICY)
        self.stop_btn.setCheckable(True)
        self.stop_btn.setChecked(True)
        self.stop_btn.setStyleSheet('QPushButton {font-size: 26pt; color: maroon;}')
        self.stop_btn.clicked.connect(self.on_stop)

        self.start_btn = QtWidgets.QPushButton("\u25B6")
        self.start_btn.setSizePolicy(MAX_SIZE_POLICY)
        self.start_btn.setCheckable(True)
        self.start_btn.setStyleSheet('QPushButton {font-size: 32pt; color: green;}')
        self.start_btn.clicked.connect(self.on_start)

        self.pause_btn = QtWidgets.QPushButton()
        self.pause_btn.setSizePolicy(MAX_SIZE_POLICY)
        self.pause_btn.setCheckable(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.on_pause)
        self.pause_btn.setIcon(style.standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_MediaPause,
        ))
        self.pause_btn.setIconSize(QtCore.QSize(28, 28))

        run_controls_options_grp = QtWidgets.QButtonGroup(self, exclusive=True)
        run_controls_options_grp.addButton(self.stop_btn)
        run_controls_options_grp.addButton(self.start_btn)
        run_controls_options_grp.addButton(self.pause_btn)

        run_controls = QtWidgets.QHBoxLayout()
        run_controls.setContentsMargins(NO_MARGIN)
        run_controls.addWidget(self.stop_btn)
        run_controls.addWidget(self.start_btn)
        run_controls.addWidget(self.pause_btn)
        # endregion

        labels = QtWidgets.QVBoxLayout()
        labels.setContentsMargins(NO_MARGIN)

        # The counts: names on the left, and their numbers lined up on the right
        stats = QtWidgets.QGridLayout()
        stats.setContentsMargins(NO_MARGIN)
        stats.setHorizontalSpacing(6)
        stats.setVerticalSpacing(0)
        stats.setColumnStretch(1, 1)
        value_labels = []
        for row, name in enumerate(("Remaining files", "Loaded pictures", "Duplicate pictures", "Duplicate groups",
                                    "Deleted files", "Deleted size")):
            stats.addWidget(QtWidgets.QLabel(name), row, 0)
            value = QtWidgets.QLabel()
            value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            stats.addWidget(value, row, 1)
            value_labels.append(value)
        (self._remaining_files_label, self._loaded_pictures_label, self._dup_pictures_label, self._dup_groups_label,
         self._deleted_files_count_label, self._deleted_files_size_label) = value_labels
        self.set_remaining_files_label(0)
        self.set_loaded_pictures_label(0)
        self.set_duplicate_images_label(0)
        self.set_duplicate_groups_label(0)
        self.set_deleted_files_label(0)
        self.set_deleted_size_label(0)

        self._timer_label = QtWidgets.QLabel("00:00:00", alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        self._elapsed_secs = 0
        self._run_timer = QtCore.QTimer(self)
        self._run_timer.setInterval(1000)  # 1s ticks
        self._run_timer.timeout.connect(self._on_run_timer_tick)

        self._label_timer = QtCore.QTimer(self)
        self._label_timer.setInterval(50)
        self._label_timer.timeout.connect(self._on_labels_tick)

        self._progress_bar = QtWidgets.QProgressBar(value=50, textVisible=False)
        labels.addLayout(stats)
        labels.addWidget(self._timer_label)
        labels.addWidget(self._progress_bar)
        labels.addLayout(run_controls)

        # region Settings tabs
        # TODO: Add other setting tabs

        # region Filter tab
        slider_labels = QtWidgets.QVBoxLayout()
        slider_labels.setContentsMargins(NO_MARGIN)
        slider_labels.addWidget(QtWidgets.QLabel(text="Strict", alignment=QtCore.Qt.AlignmentFlag.AlignTop))
        slider_labels.addWidget(QtWidgets.QLabel(text="Basic", alignment=QtCore.Qt.AlignmentFlag.AlignVCenter))
        slider_labels.addWidget(QtWidgets.QLabel(text="Loose", alignment=QtCore.Qt.AlignmentFlag.AlignBottom))

        self.precision_slider = QtWidgets.QSlider(tickPosition=QtWidgets.QSlider.TickPosition.TicksLeft)
        self.precision_slider.setMaximum(10)
        self.precision_slider.setValue(5)
        self.precision_slider.valueChanged.connect(self.on_precision_adjust)
        self.precision_slider.sliderReleased.connect(self.schedule_regroup)

        # Wait for the slider to settle before regrouping, so keying or scrolling through it only regroups once
        self._regroup_timer = QtCore.QTimer(self, singleShot=True, interval=300)
        self._regroup_timer.timeout.connect(self.start_regroup)
        self._regrouping = False
        self._regroup_failed = False

        filter_tab_main = QtWidgets.QHBoxLayout()
        filter_tab_main.setContentsMargins(NO_MARGIN)
        filter_tab_main.addLayout(slider_labels)
        filter_tab_main.addWidget(self.precision_slider)
        filter_tab_main.addWidget(QtWidgets.QLabel(
            text="The slider determines\n"
                 "how strictly the program\n"
                 "checks for similarities\n"
                 "between the images.\n"
                 "Strict means it checks if\n"
                 "an image is the same or\n"
                 "slightly different, loose\n"
                 "allows for a greater\n"
                 "amount of differences.",
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        ))

        filter_tab = QtWidgets.QVBoxLayout()
        self.hash_match_chkbx = QtWidgets.QCheckBox(" Hash match")
        self.hash_match_chkbx.setEnabled(False)
        self.hash_match_chkbx.toggled.connect(self.on_hash_match_toggled)
        # Copies turned a quarter or half turn, and mirrored copies, are always matched: this only adds small angles
        self.rotation_chkbx = QtWidgets.QCheckBox(" Tilted copies")
        self.rotation_chkbx.setToolTip(
            "Also match copies turned by a few degrees (up to about 12), like a crooked scan\n"
            "or a re-photographed print. Copies turned 90 or 180 degrees, and mirror images,\n"
            "are always matched.\n"
            "It can also group some images that aren't copies, when only their middles look alike.",
        )
        self.rotation_chkbx.toggled.connect(self.schedule_regroup)
        self.rotation_chkbx.clicked.connect(self.on_rotation_clicked)
        filter_checkboxes = QtWidgets.QHBoxLayout()
        filter_checkboxes.setContentsMargins(NO_MARGIN)
        filter_checkboxes.addWidget(self.hash_match_chkbx)
        filter_checkboxes.addWidget(self.rotation_chkbx)
        filter_checkboxes.addStretch()
        filter_tab.addLayout(filter_checkboxes)
        filter_tab.addLayout(filter_tab_main)
        # endregion

        # region Settings tab
        cpus = os.cpu_count() or 1
        self._cpu_label = QtWidgets.QLabel()
        self.cpu_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal,
                                            tickPosition=QtWidgets.QSlider.TickPosition.TicksBelow)
        self.cpu_slider.setRange(1, cpus)
        self.cpu_slider.setPageStep(1)
        self.cpu_slider.setTickInterval(1)
        self.cpu_slider.setValue(max(1, cpus))
        self.cpu_slider.valueChanged.connect(self.on_cpu_adjust)
        self.on_cpu_adjust(self.cpu_slider.value())

        settings_tab = QtWidgets.QVBoxLayout()
        settings_tab.addWidget(self._cpu_label)
        settings_tab.addWidget(self.cpu_slider)
        settings_tab.addWidget(QtWidgets.QLabel(
            text="How many of the computer's CPUs a scan uses. Using fewer leaves more of the computer free "
                 "for other things, but the scan takes longer. This can't be changed while a scan is running.",
            wordWrap=True,
        ))
        settings_tab.addStretch(1)
        # endregion

        such = QtWidgets.QTabWidget()
        such.addTab(QtWidgets.QWidget(layout=filter_tab), "Filter")
        such.addTab(QtWidgets.QWidget(layout=settings_tab), "Settings")
        # endregion

        labels_and_such = QtWidgets.QHBoxLayout()
        labels_and_such.setContentsMargins(NO_MARGIN)
        stats_width = self.fontMetrics().horizontalAdvance("Duplicate pictures") + 6 + \
            self.fontMetrics().horizontalAdvance("9,999,999") + 4
        labels_and_such.addWidget(QtWidgets.QWidget(layout=labels, maximumWidth=max(140, stats_width)))
        labels_and_such.addWidget(such)

        primary_controls = QtWidgets.QVBoxLayout()
        primary_controls.setContentsMargins(NO_MARGIN)
        primary_controls.addLayout(self.build_file_path_selection_display())
        primary_controls.addWidget(QtWidgets.QWidget(layout=labels_and_such, fixedHeight=240))

        general_controls = QtWidgets.QHBoxLayout()
        general_controls.setContentsMargins(NO_MARGIN)
        general_controls.addLayout(primary_controls)
        general_controls.addWidget(QtWidgets.QWidget(layout=general_controls_btns, maximumWidth=130))

        # region File system explorer
        self.file_system_model = DirFileSystemModel()
        self.change_hidden_folders_state(checked=False)
        self.file_system_model.setRootPath("")
        # Every row is the same height, and saying so lets the tree skip measuring each one
        #   (updating one folder's arrow took ~2ms in a folder of 20,000 folders, and 15us like this)
        self.file_system_view = QtWidgets.QTreeView(headerHidden=True, uniformRowHeights=True)
        self.file_system_view.setContentsMargins(NO_MARGIN)
        self.file_system_view.setModel(self.file_system_model)
        self.file_system_view.expanded.connect(self.file_system_model.recheck_children)
        self.file_system_view.setRootIndex(self.file_system_model.index(""))
        self.file_system_view.hideColumn(1)  # Size
        self.file_system_view.hideColumn(2)  # Type
        self.file_system_view.hideColumn(3)  # Date Modified
        # Folders can be dragged to the folder list (several at once, with Ctrl or Shift), as well as added with >+
        self.file_system_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_system_view.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragOnly)
        file_view_splitter = QtWidgets.QHBoxLayout()
        file_view_splitter.setContentsMargins(NO_MARGIN)
        file_view_splitter.addWidget(self.file_system_view)
        file_view_splitter.addWidget(QtWidgets.QWidget(layout=general_controls, maximumWidth=600))
        # endregion
        # endregion

        outer_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, childrenCollapsible=False)
        inner_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, childrenCollapsible=False)

        inner_splitter.addWidget(QtWidgets.QWidget(layout=file_view_splitter))
        inner_splitter.addWidget(self.build_image_view_area())
        inner_splitter.setSizes([250, 250])

        self.duplicate_group_list = DuplicateGroupList(sizePolicy=MAX_SIZE_POLICY)
        self.duplicate_group_list.groupTileStateChanged.connect(self.on_match_state_changed)
        self.duplicate_group_list.groupTileHovered.connect(self.on_tile_hover)
        self.duplicate_group_list.groupTileMenuAction.connect(self.on_tile_menu_action)
        self.duplicate_group_list.page_down.pressed.connect(self.on_page_down)
        self.duplicate_group_list.page_up.pressed.connect(self.on_page_up)
        self.duplicate_group_list.first_page.pressed.connect(self.on_page_first)
        self.duplicate_group_list.last_page.pressed.connect(self.on_page_last)
        self.duplicate_group_list.pageIndicatorClicked.connect(self.on_page_jump_request)
        self.duplicate_group_list.pageSizeChanged.connect(self.refresh_pages)
        # While scanning, groups are added and grow, which changes what fits on each page: the pages are worked out
        #   again a little after, rather than on every new image
        self._pages_timer = QtCore.QTimer(self, singleShot=True, interval=300)
        self._pages_timer.timeout.connect(self.refresh_pages)
        outer_splitter.addWidget(self.duplicate_group_list)
        outer_splitter.addWidget(inner_splitter)

        # Create a central widget and layout to hold the splitters
        central_widget = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(central_widget)
        hbox.addWidget(outer_splitter)
        self.setCentralWidget(central_widget)

    def _on_run_timer_tick(self):
        """Timer tick for runtime label, increment the elapsed seconds and draw..."""
        if not self.processor:
            return

        # Only count time while actively running
        if not self.processor.is_paused() and not self.processor.is_finished():
            self._elapsed_secs += 1
            h, rem = divmod(self._elapsed_secs, 3600)
            m, s = divmod(rem, 60)
            self._timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def _on_labels_tick(self):
        """Timer tick for general labels, runs quickly"""
        if not self.processor:
            return

        self.update_labels()

    def on_precision_adjust(self, e):
        """The precision slider has been adjusted, so update the hash checkbox and regroup"""
        if e != 10:
            self.hash_match_chkbx.setEnabled(False)
            self.hash_match_chkbx.setChecked(False)
        else:
            self.hash_match_chkbx.setEnabled(True)

        self.schedule_regroup()

    def on_hash_match_toggled(self, checked: bool):  # ruff: ignore[boolean-type-hint-positional-argument]
        """SHA-256 hashes only match identical files, so they can't find turned copies"""
        self.rotation_chkbx.setEnabled(not checked and self.precision_slider.isEnabled())
        self.schedule_regroup()

    def on_rotation_clicked(self, checked: bool):  # ruff: ignore[boolean-type-hint-positional-argument]
        """Tilted copies was ticked: say that it can also group images that aren't copies"""
        if not checked or self._settings.value("warnings/hide_tilted_extra", defaultValue=False, type=bool):
            return

        # The regroup waits until the message is closed (toggled has already scheduled it)
        self._regroup_timer.stop()
        dlg = QtWidgets.QMessageBox(self)
        dlg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        dlg.setWindowTitle("Tilted copies")
        dlg.setText(
            "Tilted copies finds copies turned by a few degrees, but it can also group some images that "
            "aren't copies, when only their middles look alike.\n\n"
            "Check its groups before deleting, or untick it to see only the usual matches.",
        )
        # Kept here: PySide loses track of it, so dlg.checkBox() can't be read back
        dont_show = QtWidgets.QCheckBox("Don't show this again")
        dlg.setCheckBox(dont_show)
        dlg.exec()
        if dont_show.isChecked():
            self._settings.setValue("warnings/hide_tilted_extra", True)  # ruff: ignore[boolean-positional-value-in-call]

        self.schedule_regroup()

    def rotation_wanted(self) -> bool:
        """Should turned copies be matched, with the settings that are set now?"""
        return self.rotation_chkbx.isChecked() and not self.hash_match_chkbx.isChecked()

    def _scan_matches_settings(self) -> bool:
        """Was the loaded scan made with the strength, Hash match and Tilted copies settings that are set now?"""
        return (self.processor.strength == self.precision_slider.value()
                and self.processor.exact_match == self.hash_match_chkbx.isChecked()
                and self.processor.rotation == self.rotation_wanted())

    def schedule_regroup(self, *_):
        """The filter settings changed, so regroup what was loaded once they settle"""
        if not self.processor or not self.processor.is_finished() or self.precision_slider.isSliderDown():
            return

        self._regroup_timer.start()

    def start_regroup(self):
        """Regroup the loaded images at the strength now set, if they can be"""
        if not self.processor or not self.processor.is_finished() or self._regrouping:
            return

        if self._scan_matches_settings():
            self.statusBar().clearMessage()
            return

        if self.processor.exact_match or self.hash_match_chkbx.isChecked():
            # SHA-256 hashes and perceptual hashes can't be turned into each other, only a new scan can switch them
            self.statusBar().showMessage("Press start to scan again with the new Hash match setting.")
            return

        strength = self.precision_slider.value()
        rotation = self.rotation_wanted()
        logger.info("Regrouping %s images at strength %s, rotation %s",
                    self.processor.processed_images, strength, rotation)
        self.statusBar().showMessage(f"Regrouping at strength {strength}{' with tilted copies' if rotation else ''}...")

        # Everything that reads or changes the processor's groups waits until the regroup is done
        self._regrouping = True
        self.centralWidget().setEnabled(False)
        self.menuBar().setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        self._regroup_message = f"Regrouping at strength {strength}{' with tilted copies' if rotation else ''}"
        self._progress_bar.setRange(0, 0)  # Busy, until the first progress report says how many images there are
        self._cancel_regroup_btn.setText("Cancel")
        self._cancel_regroup_btn.setEnabled(True)
        self._cancel_regroup_btn.show()

        regroup = RegroupThread(self.processor, strength, rotation=rotation)
        regroup.signals.progress.connect(self.on_regroup_progress)
        regroup.signals.done.connect(self.on_regroup_done)
        regroup.signals.cancelled.connect(self.on_regroup_cancelled)
        regroup.signals.failed.connect(self.on_regroup_failed)
        self._regroup_thread = regroup  # The signals must outlive run() until they're delivered
        self._threadpool.start(regroup)

    def on_regroup_progress(self, done: int, total: int):
        """Show how far the regroup has got"""
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)
        self.statusBar().showMessage(f"{self._regroup_message}... {done:,} of {total:,} images")

    def cancel_regroup(self):
        """The Cancel button was pressed during a regroup"""
        if self._regroup_thread is not None:
            self._regroup_thread.cancel()
            self._cancel_regroup_btn.setText("Cancelling...")
            self._cancel_regroup_btn.setEnabled(False)

    def on_regroup_cancelled(self):
        """The regroup was cancelled, so the groups are as they were: put the controls back to match them"""
        self._end_regroup()
        # Setting them back mustn't start another regroup
        with QtCore.QSignalBlocker(self.precision_slider), QtCore.QSignalBlocker(self.rotation_chkbx):
            self.precision_slider.setValue(self.processor.strength)
            self.rotation_chkbx.setChecked(self.processor.rotation)

        # The Hash match checkbox goes with the strength (see on_precision_adjust)
        self.hash_match_chkbx.setEnabled(self.processor.strength == 10)
        self.update_labels()
        self.statusBar().showMessage(
            f"Regroup cancelled, still grouped at strength {self.processor.strength}"
            f"{' with tilted copies' if self.processor.rotation else ''}.",
        )

    def _end_regroup(self):
        """Give the GUI back after a regroup"""
        self._regrouping = False
        self._regroup_thread = None
        self._cancel_regroup_btn.hide()
        self._progress_bar.setRange(0, 100)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.centralWidget().setEnabled(True)
        self.menuBar().setEnabled(True)

    def on_regroup_done(self, seconds: float):
        """The regroup finished, so show the new groups"""
        self._end_regroup()

        # Some files lost their duplicates at this strength
        self._drop_marks_without_duplicates()

        self.current_page = 1
        self.update_group_list()
        self.update_labels()
        tilted = " with tilted copies" if self.processor.rotation else ""
        self.statusBar().showMessage(
            f"Regrouped at strength {self.processor.strength}{tilted}: "
            f"{len(self.processor.matches)} groups in {seconds:.1f}s.",
        )

        # The slider may have moved again while this ran
        if not self._scan_matches_settings():
            self.start_regroup()

    def on_regroup_failed(self, error: str):
        """The regroup failed, which leaves the groups half built, so it's the next scan's turn"""
        self._end_regroup()
        self._regroup_failed = True
        self.statusBar().showMessage(f"Regrouping failed ({error}), press start to scan again.")

    def confirm_new_scan(self) -> bool:
        """
        Ask before starting a new scan that would throw out the loaded one.

        Returns:
            True if the user wants the new scan.
        """
        answer = QtWidgets.QMessageBox.question(
            self,
            "New scan?",
            "The loaded images can't be regrouped with these settings, so they need to be scanned again.\n"
            "This clears the current groups and any marked files. Continue?",
            defaultButton=QtWidgets.QMessageBox.StandardButton.No,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def reset_scan(self):
        """Throw out the loaded scan, its groups and its markings"""
        self.processor = None
        self._page_seq = 0  # The next scan's events are numbered from the start again
        self._regroup_failed = False
        self.file_states = {}
        self.current_page = 1
        self._page_starts = [0]
        self._shown_from = 0
        self._elapsed_secs = 0
        self._timer_label.setText("00:00:00")
        self.image_view_area.clear()
        self.duplicate_group_list.set_groups([])
        self.duplicate_group_list.update_page_indicator(1, 1)
        self.statusBar().clearMessage()

    def on_cpu_adjust(self, value: int):
        """The CPU slider has been adjusted, so show its new value"""
        self._cpu_label.setText(f"CPUs to use: {value} of {self.cpu_slider.maximum()}")

    def on_pause(self, e):
        """Pause button has been pressed, so pause"""
        if not e:
            return

        if not self.processor:
            return

        self.processor.pause()
        self._run_timer.stop()
        self._label_timer.stop()

    def on_stop(self, e):
        """Stop button has been pressed, so stop processing. on_finish updates the GUI once it has stopped"""
        if not e:
            return

        if self.processor and not self.processor.is_finished():
            self.pause_btn.setEnabled(False)
            self.processor.stop()

    def on_start(self, e):
        """Start button has been pressed, so start or resume"""
        if not e:
            return

        target_paths = self.file_paths_selected()
        if not target_paths and (not self.processor or self.processor.is_finished()):
            # This click would start a scan, and there's nothing to scan. It checked Start though, so check Stop again
            self.stop_btn.setChecked(True)
            return

        if self.processor and self.processor.is_finished() and (
                self._regroup_failed or self.processor.exact_match != self.hash_match_chkbx.isChecked()):
            # The loaded hashes can't be used with these settings, so this has to be a whole new scan
            if not self.confirm_new_scan():
                self.stop_btn.setChecked(True)
                return

            self.reset_scan()

        if not self.processor:
            # This is the first time running!
            self.processor = ImageMatcher(
                strength=self.precision_slider.value(),
                exact_match=self.hash_match_chkbx.isChecked(),
                rotation=self.rotation_wanted(),
                group_filters=self.group_filters(),
            )

        elif self.processor.is_paused() and not self.processor.is_finished():
            # Already started so we just need to resume
            self.processor.resume()
            self.pause_btn.setEnabled(True)
            self._run_timer.start()
            self._label_timer.start()
            return

        elif not self.processor.is_finished():
            # Already running. Start stays checked when it's clicked again, but the click still gets here
            return

        # Each scan makes its own pool of processes when it starts, so this is when a new CPU setting takes effect
        self.processor.processes = self.cpu_slider.value()

        self._thread = ProcessorThread(self.processor, target_paths)
        self._thread.signals.new_group.connect(self.on_new_match_group_found)
        self._thread.signals.new_match.connect(self.on_new_match_found)
        self._thread.signals.finish.connect(self.on_finish)
        self._threadpool.start(self._thread)
        self.pause_btn.setEnabled(True)
        self.precision_slider.setEnabled(False)
        self.hash_match_chkbx.setEnabled(False)
        self.rotation_chkbx.setEnabled(False)
        self.cpu_slider.setEnabled(False)
        self._run_timer.start()
        self._label_timer.start()
        return

    def on_delete(self, *_):
        """Delete button pressed, so delete every marked file"""
        self.process_file_states(FileAction.DELETE)

    def on_move(self, *_):
        """Move button pressed, so ask for a folder and move every marked file there"""
        self.process_file_states(FileAction.MOVE)

    @property
    def total_pages(self):
        """How many pages are there in the duplicate group list"""
        if not self.processor:
            # Nothing has been scanned yet, so there's only the one empty page
            return 1

        return len(self._page_starts)

    def on_finish(self):
        """Finish callback, to update GUI when processing has completed"""
        self.stop_btn.setChecked(True)
        self.pause_btn.setEnabled(False)
        self.cpu_slider.setEnabled(True)
        self._run_timer.stop()
        self._label_timer.stop()
        self._thread = None
        self.update_labels()

        # The strength and rotation can be changed again now, which regroups what was loaded
        self.precision_slider.setEnabled(True)
        self.rotation_chkbx.setEnabled(not self.hash_match_chkbx.isChecked())
        self.on_precision_adjust(self.precision_slider.value())

    def on_new_match_group_found(self, _event: NewGroup):
        """A new group was found. It goes after the others, so if it fits on the page shown, refresh_pages adds it"""
        self._pages_timer.start()
        self.set_duplicate_groups_label(len(self.processor.matches))

    def on_new_match_found(self, event: NewMatch):
        """New match found callback, update the GUI with new match"""
        row_this_is = event.group.match_i - self._shown_from

        # If the page was rebuilt after this event was sent, it already has this image
        rows = self.duplicate_group_list._rows
        if event.seq > self._page_seq and 0 <= row_this_is < len(rows):
            rows[row_this_is].add_tile(event.path)
            self.duplicate_group_list.relayout()

        # The group may not fit on its page anymore, or (on the page shown) push the last group off it
        self._pages_timer.start()
        self.set_duplicate_images_label(self.processor.duplicate_images)

    def refresh_pages(self):
        """
        Work out the pages again, for groups found or grown since, or a different size of list.

        The page shown is rebuilt only if it would show different groups now, so the preview isn't cleared for nothing.
        """
        if not self.processor:
            return

        self._repaginate()
        page = self.current_page
        shown = (self._shown_from, len(self.duplicate_group_list._rows))
        now = (self._page_starts[page - 1], self._page_end(page) - self._page_starts[page - 1])
        if shown != now:
            self.update_group_list()
            return

        self.duplicate_group_list.relayout()  # The same groups, but maybe a different width to wrap them to
        self.duplicate_group_list.update_page_indicator(self.current_page, self.total_pages)

    def _page_end(self, page: int) -> int:
        """The index after the last group on this page"""
        return self._page_starts[page] if page < len(self._page_starts) else len(self.processor.matches)

    def _repaginate(self, sizes: list[int] | None = None):
        """
        Work out which groups are on each page, staying on the page with the group the shown page starts with.

        Args:
            sizes (list[int] | None): How many images each group has, if they've just been counted.
        """
        if sizes is None:
            with self.processor.paused():
                sizes = [len(group.matches) for group in self.processor.matches]

        first = self._page_starts[min(self.current_page, len(self._page_starts)) - 1]
        self._page_starts = self.duplicate_group_list.page_starts(sizes)
        self.current_page = bisect.bisect_right(self._page_starts, first)

    def change_hidden_folders_state(self, checked: bool):  # ruff: ignore[boolean-type-hint-positional-argument]
        """Set the file_system_model filter based on if we should show hidden folders"""
        state = (QtCore.QDir.Filter.Dirs
                 | QtCore.QDir.Filter.Drives
                 | QtCore.QDir.Filter.NoDotAndDotDot)

        if checked:
            state |= QtCore.QDir.Filter.Hidden

        self.file_system_model.setFilter(state)

    def change_recycle_state(self, checked: bool):  # ruff: ignore[boolean-type-hint-positional-argument]
        """Change the 'Use recycle bin' state"""
        self.recycle = checked

    def group_filters(self) -> list:
        """
        The group filters to use.

        Returns:
            list: Those of the options that hide groups (in the Options menu) which are checked, and the one that hides
                groups whose images are all ignored (unless they're to be shown).
        """
        filters = [group_filter for option, group_filter in self._group_filter_options if option.isChecked()]
        if not self.show_ignored_option.isChecked():
            filters.append(self.ignored.keeps_group)

        return filters

    def change_group_filters(self, *_):
        """An option that hides groups was checked or unchecked, so show the groups that the options don't hide"""
        if not self.processor:
            return  # The scan gets them when it starts

        self.processor.set_group_filters(self.group_filters())
        # Delete and Move would act on marked files in groups that are hidden now, without them being seen
        self._drop_marks_without_duplicates()
        # If there are fewer pages now, stay on the last one
        self.current_page = min(self.current_page, self.total_pages)
        self.update_group_list()
        self.update_labels()

    def set_duplicate_groups_label(self, duplicate_groups: int):
        """Set the duplicate groups count label"""
        self._dup_groups_label.setText(f"{duplicate_groups:,}")

    def set_duplicate_images_label(self, duplicate_images: int):
        """Set the duplicate images count label"""
        self._dup_pictures_label.setText(f"{duplicate_images:,}")

    def set_remaining_files_label(self, remaining_files: int):
        """Set the remaining files count label"""
        self._remaining_files_label.setText(f"{remaining_files:,}")

    def set_loaded_pictures_label(self, loaded_pictures: int):
        """Set the loaded pictures count label"""
        self._loaded_pictures_label.setText(f"{loaded_pictures:,}")

    def set_deleted_files_label(self, deleted_files: int):
        """Set the deleted files count label"""
        self._deleted_files_count_label.setText(f"{deleted_files:,}")

    def set_deleted_size_label(self, deleted_size: int):
        """Set the deleted file size label"""
        self._deleted_files_size_label.setText(human_bytes(deleted_size))

    def update_labels(self):
        """Update all of the boring labels that need to be regularly updated, and the progress bar"""
        if not self.processor:
            return

        if self.processor.found_images:
            self._progress_bar.setMaximum(self.processor.found_images)
            self._progress_bar.setValue(self.processor.processed_images)
        else:
            self._progress_bar.setMaximum(100)
            self._progress_bar.setValue(0)

        self.set_remaining_files_label(self.processor.left_to_process)
        self.set_loaded_pictures_label(self.processor.processed_images)
        self.set_duplicate_groups_label(len(self.processor.matches))
        self.set_duplicate_images_label(self.processor.duplicate_images)
        self.set_deleted_files_label(self.deleted_files_count)
        self.set_deleted_size_label(self.deleted_files_size)

    def on_page_jump_request(self):
        """
        Prompt for a page number and jump there.
        Uses a numeric-only dialog with range [1, total_pages].
        """
        total = self.total_pages
        if total == 1:
            return

        val, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Go to page",
            f"Enter a page number (1-{total}):",
            value=self.current_page,
            minValue=1,
            maxValue=total,
        )
        if not ok:
            return

        if val != self.current_page:
            self.current_page = val
            self.update_group_list()

    def on_page_down(self, *_):
        """The page down button has been pressed"""
        if self.total_pages == 1:
            # Theres only one page so do nothing
            return

        # Make the page system rotate around the beginning
        if self.current_page == 1:
            self.current_page = self.total_pages
        else:
            self.current_page -= 1

        self.update_group_list()

    def on_page_up(self, *_):
        """The page up button has been pressed"""
        if self.total_pages == 1:
            # Theres only one page....
            return

        # Make the page system rotate around the end
        if self.current_page == self.total_pages:
            self.current_page = 1
        else:
            self.current_page += 1

        self.update_group_list()

    def on_page_first(self):
        """Go to first page button pressed"""
        if self.current_page != 1:
            self.current_page = 1
            self.update_group_list()

    def on_page_last(self):
        """Go to last page button pressed"""
        last_page = self.total_pages
        if self.current_page != last_page:
            self.current_page = last_page
            self.update_group_list()

    # region "mark" methods
    #   These mark the currently open file in the view area to something, typically used by the status bar items
    def mark_unmark(self, target_path: ZipPath | None = None):
        """Unmark a file (mark as keep)"""
        self.change_file_state(target_path or self.image_view_area.current_path, SelectionValues.KEEP)

    def mark_file(self, target_path: ZipPath | None = None):
        """Mark a file (for the Delete and Move actions)"""
        self.change_file_state(target_path or self.image_view_area.current_path, SelectionValues.MARK)

    def mark_ignore(self, target_path: ZipPath | None = None):
        """Mark a file as ignore"""
        self.change_file_state(target_path or self.image_view_area.current_path, SelectionValues.IGNORE)

    def choose_move_folder(self, title: str, near: Path) -> Path | None:
        """
        Ask which folder to move files to, starting in the last folder chosen (even in an earlier session).

        Args:
            title: The folder picker's title.
            near: Where to start if no folder has been chosen before (or it's gone).

        Returns:
            Path | None: The folder, or None if none was chosen.
        """
        start = self._settings.value("move/last_folder", "", str)
        if not start or not Path(start).is_dir():
            start = str(near)

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, title, start)
        if not folder:
            return None

        self._settings.setValue("move/last_folder", folder)
        return Path(folder)

    def confirm(self, title: str, text: str) -> bool:
        """
        Ask a yes or no question, with no as the default.

        Returns:
            True if the user said yes.
        """
        answer = QtWidgets.QMessageBox.question(
            self, title, text, defaultButton=QtWidgets.QMessageBox.StandardButton.No,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def set_file_state(self, path: ZipPath, state: SelectionValue):
        """
        Set the state of a single file. Every state change should go through here.

        Files are only marked (for the Delete and Move actions) while they are in a match group, i.e. they have a
            duplicate and their group isn't hidden, so bulk actions like "Ignore zip" can't mark files that have no
            duplicate, or that are hidden.
        Files in zips are read-only, so they are never marked:
            deleting or moving them would act on the zip itself, not the file inside it.
        Ignoring a file is saved straight away (see IgnoredImages), and when every file in its group is ignored, the
            group is hidden straight away. Use mark_paths for several files, which saves and hides them in one go.
        """
        self._set_file_states([path], state)

    def _set_file_states(self, paths: Iterable[ZipPath], state: SelectionValue):
        """Set the state of several files, see set_file_state"""
        ignore, unignore = [], []
        for path in paths:
            if state.state == SelectionState.IGNORE:
                self.file_states.pop(path, None)
                ignore.append(path)
                continue

            if state.state == SelectionState.MARK and (path.is_zip or not self._has_duplicate(path)):
                continue

            self.file_states[path] = state
            if path in self.ignored:
                unignore.append(path)

        self.ignored.add(ignore)
        self.ignored.discard(unignore)
        if self.processor and (ignore or unignore) and not self.show_ignored_option.isChecked():
            # Groups that are all ignored now are hidden (and ones that aren't anymore are shown)
            before = len(self.processor.matches)
            self.processor.recheck_groups(ignore + unignore)
            if len(self.processor.matches) != before:
                self._drop_marks_without_duplicates()
                self.update_group_list()
                self.update_labels()

    def _drop_marks_without_duplicates(self):
        """
        A file is only marked while it has a duplicate (see set_file_state), so drop the markings of files that don't
            anymore: after regrouping, or after hiding groups.
        """
        with self.processor.paused():
            self.file_states = {
                path: state for path, state in self.file_states.items()
                if state.state != SelectionState.MARK or self._has_duplicate(path)
            }

    def _has_duplicate(self, path: ZipPath) -> bool:
        """Is this file in a match group?"""
        hash_ = self.processor._reverse_hashes.get(path)
        return hash_ is not None and self.processor._hashes[hash_].match_i is not None

    def mark_paths(self, paths: Iterable[ZipPath], state: SelectionValue):
        """
        Mark several files as a particular state, then update the tiles.

        Processing is paused while `paths` is iterated, so it can be one of the processor's lists. A generator over the
            processor's data must be made inside a `with self.processor.paused():` block too, since a generator
            expression takes its iterator over that data as soon as it's made (and it could change before the pause).
        """
        if not self.processor:
            return

        with self.processor.paused():
            paths = list(paths)

        self._set_file_states(paths, state)
        self.update_selection_states()

    def mark_group(self, path, selection: SelectionValue):
        """Mark all files in a group with path as a particular state"""
        if not path:
            return

        hash_ = self.processor._reverse_hashes[path]
        self.mark_paths(self.processor._hashes[hash_].matches, selection)

    def mark_ignore_group(self, target_path: ZipPath | None = None):
        """Mark all files in a group as ignore"""
        self.mark_group(target_path or self.image_view_area.current_path, SelectionValues.IGNORE)

    def mark_unmark_group(self, target_path: ZipPath | None = None):
        """Mark all files in a group as keep"""
        self.mark_group(target_path or self.image_view_area.current_path, SelectionValues.KEEP)

    def mark_zip(self, target_path: ZipPath, state: SelectionValue):
        """Mark all files in a zip"""
        if not target_path or target_path.path_obj.suffix.lower() != '.zip':
            return

        self.mark_paths(self.processor._processed_zips[target_path.path], state)

    def mark_ignore_zip(self, target_path: ZipPath | None = None):
        """Mark all files in a zip as ignore"""
        current_zip_path = target_path or self.image_view_area.current_path
        self.mark_zip(current_zip_path, SelectionValues.IGNORE)

    def mark_unmark_zip(self, target_path: ZipPath | None = None):
        """Mark all files in a zip as keep"""
        current_zip_path = target_path or self.image_view_area.current_path
        self.mark_zip(current_zip_path, SelectionValues.KEEP)
    # endregion

    # region Right click actions
    def on_tile_menu_action(self, action: str, path: ZipPath):
        """A tile's right click menu action that the tile can't do itself was chosen"""
        handlers = {
            TileAction.IGNORE_GROUP: self.mark_ignore_group,
            TileAction.UNMARK_GROUP: self.mark_unmark_group,
            TileAction.IGNORE_ZIP: self.mark_ignore_zip,
            TileAction.UNMARK_ZIP: self.mark_unmark_zip,
            TileAction.KEEP_DELETE_OTHERS: self.keep_this_delete_others,
            TileAction.MOVE_LEAVE_OTHERS: lambda p: self.move_this(p, delete_others=False),
            TileAction.MOVE_DELETE_OTHERS: lambda p: self.move_this(p, delete_others=True),
        }
        handlers[action](path)

    def _group_others(self, path: ZipPath) -> list[ZipPath]:
        """The other files in a file's group"""
        hash_ = self.processor._reverse_hashes.get(path) if self.processor else None
        if hash_ is None:
            return []

        return [p for p in self.processor._hashes[hash_].matches if p != path]

    def _confirm_delete(self, paths: list[ZipPath], *, extra: str = "") -> bool:
        """
        Ask before deleting files right away.

        Returns:
            True if the user said yes.
        """
        where = "to the Recycle Bin" if self.recycle else "permanently"
        names = "\n".join(f"    {p.path_obj.name}" for p in paths[:10])
        if len(paths) > 10:
            names += f"\n    ...and {len(paths) - 10} more"
        return self.confirm("Delete?", f"{extra}Delete {len(paths)} file(s) {where}?\n\n{names}")

    def keep_this_delete_others(self, path: ZipPath):
        """Delete the rest of a file's group right away (except files in zips, which are read-only)"""
        others = [p for p in self._group_others(path) if not p.is_zip]
        if not others:
            QtWidgets.QMessageBox.information(self, "Delete", "There are no other files in this group to delete.")
            return

        if self._confirm_delete(others):
            self.act_now(delete=others)

    def move_this(self, path: ZipPath, *, delete_others: bool):
        """
        Move a file to a folder right away, which takes it out of its group, and maybe delete the rest of the group.

        The group stays if it still has two or more files, otherwise it goes away.
        """
        source = path.path_obj
        if path.is_zip or not source.exists():
            return

        folder = self.choose_move_folder(f"Move {source.name} to", source.parent)
        if folder is None or folder == source.parent:
            return

        others = [p for p in self._group_others(path) if not p.is_zip] if delete_others else []
        if others and not self._confirm_delete(others, extra=f"Move {source.name} to {folder}.\n"):
            return

        self.act_now(move=(path, folder), delete=others)

    def act_now(self, *, move: tuple[ZipPath, Path] | None = None, delete: Iterable[ZipPath] = ()):
        """
        Move and delete files right away, then take them out of the results.

        If the move fails, nothing is deleted. Files that fail to delete are left as they are, and listed.

        Args:
            move: A file and the folder to move it to.
            delete: Files to delete (or send to the recycle bin).
        """
        if not self.processor:
            return

        self.image_view_area.clear()
        # Previews and thumbnails are read in the background with their files open, and Windows won't delete or move an
        #   open file
        self.image_view_area.release_files()
        self.duplicate_group_list.release_files()
        failed = []
        try:
            if move:
                path, folder = move
                try:
                    self.move_file(path.path_obj, folder)
                except OSError as e:
                    logger.warning("Failed to move %s to %s: %s", path, folder, e)
                    QtWidgets.QMessageBox.warning(self, "Move", f"Couldn't move {path.path_obj}:\n{e}")
                    return

                self._forget(path)

            for path in delete:
                try:
                    self.deleted_files_size += self.delete_file(path.path_obj)
                    self.deleted_files_count += 1
                except FileNotFoundError:
                    logger.info("File already deleted: %s", path)
                except OSError as e:
                    logger.warning("Failed to delete %s: %s", path, e)
                    failed.append(f"{path.path_obj.name}: {e}")
                    continue

                self._forget(path)
        finally:
            self.current_page = min(self.current_page, self.total_pages)
            self.update_labels()
            self.update_group_list()

        if failed:
            QtWidgets.QMessageBox.warning(self, "Delete", "Couldn't delete:\n" + "\n".join(failed))

    def _forget(self, path: ZipPath):
        """Take a file that was moved or deleted out of the results, along with any marking and preview it had"""
        self.file_states.pop(path, None)
        self.ignored.discard([path])
        self.image_view_area.forget(path)
        with self.processor.paused():
            if path in self.processor._reverse_hashes:
                self.processor.remove(path)
    # endregion

    def update_group_list(self):
        """Work out the pages again, and show the current one"""
        self.image_view_area.clear()
        self._pages_timer.stop()

        with self.processor.paused():
            # The page is built from the processor's data as it is now, which has the changes of every event sent so
            #   far. So events that are still queued are skipped when they arrive, by their seq. Later events are
            #   added to the page, and their groups' numbers (match_i) are current: groups are only renumbered by
            #   removing files, and that always rebuilds the page after.
            #   The groups are copied, since they keep changing after this
            self._page_seq = self.processor.events_sent
            self._repaginate([len(group.matches) for group in self.processor.matches])
            start, end = self._page_starts[self.current_page - 1], self._page_end(self.current_page)
            groups = [list(m.matches) for m in self.processor.matches[start:end]]

        self._shown_from = start
        self.duplicate_group_list.set_groups(groups)
        self.update_selection_states()
        self.duplicate_group_list.update_page_indicator(self.current_page, self.total_pages)

    def change_file_state(self, path: ZipPath, state: SelectionValue):
        """A tile has been clicked and the match state was changed"""
        if not path:
            # The Edit menu acts on the image being previewed, and there isn't one: no image has been hovered over
            #   since the app started or the preview was last cleared (e.g. by changing page)
            return
        self.set_file_state(path, state)
        self.update_selection_states()

    def on_tile_hover(self, path: ZipPath):
        """A tile has been hovered over, so handle that logic"""
        self.image_view_area.set_image(path)
        # Its duplicates are what's most likely to be looked at next, so read them in the background too
        self.image_view_area.prefetch(self.duplicate_group_list.group_around(path))
        self.mark_menu.setEnabled(not path.is_zip)
        self.mark_ignore_zip_menu.setEnabled(path.is_zip)

    def on_match_state_changed(self, path: ZipPath, state):
        """A tile has been clicked and the match state was changed"""
        self.set_file_state(path, state)

    def update_selection_states(self, selections: dict | None = None):
        """Update the states of the selection in the duplicate list view"""
        selections = selections or self.file_states

        for group in self.duplicate_group_list._rows:
            for tile in group.tiles():
                set_state = selections.get(tile.path)
                if tile.path in self.ignored:
                    set_state = SelectionValues.IGNORE
                if set_state:
                    tile.silent_set_state(set_state)

    # region File Path Selection display
    def build_file_path_selection_display(self):
        """
        Build the file path selection display,
            which shows the selected file paths and the controls to add/remove them and re-order them

        Returns:
            QtWidgets.QHBoxLayout: The file path selection display with associated controls in a layout
        """
        # Small square buttons at the top and bottom of each side of the list, with icons like VisiPics': add and
        #   remove on the left, move up and down on the right
        color = self.palette().color(QtGui.QPalette.ColorRole.ButtonText)

        def button(kind: FolderButton, tip: str, clicked) -> QtWidgets.QToolButton:
            btn = QtWidgets.QToolButton(toolTip=tip, autoRaise=False)
            btn.setIcon(folder_button_icon(kind, color))
            btn.setIconSize(QtCore.QSize(20, 20))
            btn.setFixedSize(30, 30)
            btn.clicked.connect(clicked)
            return btn

        def column(top: QtWidgets.QToolButton, bottom: QtWidgets.QToolButton) -> QtWidgets.QWidget:
            layout = QtWidgets.QVBoxLayout()
            layout.setContentsMargins(NO_MARGIN)
            layout.addWidget(top)
            layout.addStretch(1)
            layout.addWidget(bottom)
            return QtWidgets.QWidget(layout=layout)

        add_remove = column(
            button(FolderButton.ADD, "Add the folders selected in the tree (or drag them into the list)",
                   self.file_path_in_clicked),
            button(FolderButton.REMOVE, "Remove the selected folders (or drag them out of the list)",
                   self.file_path_out_clicked),
        )
        up_down = column(
            button(FolderButton.UP, "Move the selected folders up (or drag them)", self.file_path_up_clicked),
            button(FolderButton.DOWN, "Move the selected folders down (or drag them)", self.file_path_down_clicked),
        )

        self.selected_file_path_display = FolderListWidget()
        self.selected_file_path_display.foldersDropped.connect(self.add_folders)
        self.selected_file_path_display.foldersDraggedOut.connect(self.remove_folders)
        # Folders dragged out of the list are removed when they're dropped anywhere else in the window
        self.setAcceptDrops(True)

        file_path_controls = QtWidgets.QHBoxLayout()
        file_path_controls.setContentsMargins(NO_MARGIN)
        file_path_controls.addWidget(add_remove)
        file_path_controls.addWidget(self.selected_file_path_display)
        file_path_controls.addWidget(up_down)
        return file_path_controls

    def file_path_up_clicked(self, _):
        """Move the selected folders up in the ordering"""
        # TODO: If the path has already been loaded for processing in the processor, then this won't do much
        self._move_selected_folders(-1)

    def file_path_down_clicked(self, _):
        """Move the selected folders down in the ordering"""
        self._move_selected_folders(1)

    def _move_selected_folders(self, step: int):
        """Move the selected folders one row up (-1) or down (1), stopping at the top or bottom"""
        view = self.selected_file_path_display
        items = sorted(view.selectedItems(), key=view.row, reverse=step > 0)  # The one nearest the end it goes to first
        blocked = -1 if step < 0 else view.count()  # A row they can't move into
        for item in items:
            row = view.row(item)
            if row + step == blocked:
                blocked = row  # Stuck, so the next one can't move into its row either
                continue

            view.takeItem(row)
            view.insertItem(row + step, item)

        view.select(items[::-1] if step > 0 else items)

    def file_path_in_clicked(self, _):
        """Add the folders selected in the folder tree"""
        model = self.file_system_view.model()
        self.add_folders([model.fileInfo(index).filePath() for index in self.file_system_view.selectedIndexes()])

    def file_path_out_clicked(self, _):
        """Remove the selected folders"""
        self.remove_folders(self.selected_file_path_display.selectedItems())

    def add_folders(self, folders: list[str], row: int | None = None):
        """
        Add folders to the list (any already in it are skipped), and to a scan in progress.

        Args:
            folders (list[str]): The folders' paths.
            row (int | None): Where to put them in the list, or None for the end.
        """
        view = self.selected_file_path_display
        listed = {same_folder(folder) for folder in self.file_paths_selected()}
        row = view.count() if row is None else row
        for folder in folders:
            if same_folder(folder) in listed:
                continue

            listed.add(same_folder(folder))
            view.insertItem(row, folder)
            row += 1

            # A scan that is in progress is told about the folder, even a paused one (it gets to the folder once it's
            #   resumed). With no scan in progress, the next one starts with every folder listed
            if self.processor and not self.processor.is_finished():
                self.processor.add_path(folder)

    def remove_folders(self, items: list[QtWidgets.QListWidgetItem]):
        """Remove folders from the list, and their images from the scan"""
        view = self.selected_file_path_display
        for item in items:
            view.takeItem(view.row(item))
            if self.processor:
                self.processor.remove_path(item.text())

        if items and self.processor:
            self.update_group_list()
            self.update_labels()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        """Folders dragged out of the folder list can be dropped anywhere else in the window, to remove them"""
        if event.source() is self.selected_file_path_display and event.mimeData().hasFormat(LISTED_MIME):
            event.setDropAction(QtCore.Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        """Folders dragged out of the folder list were dropped somewhere else in the window: remove them"""
        if event.source() is self.selected_file_path_display and event.mimeData().hasFormat(LISTED_MIME):
            event.setDropAction(QtCore.Qt.DropAction.MoveAction)
            event.accept()
            self.remove_folders(self.selected_file_path_display.selectedItems())
        else:
            event.ignore()

    def file_paths_selected(self):
        """Get the file paths selected in the selected file path display"""
        return [
            self.selected_file_path_display.item(i).text()
            for i in range(self.selected_file_path_display.count())
        ]
    # endregion

    # region Image View area (bottom-right)
    def build_image_view_area(self):
        """Build the image view area. Fairly simple now that the widget does most of the work"""
        self.preview_resized.changed.connect(self.preview_resized_changed)
        self.image_view_area = ImageViewPane()
        return self.image_view_area

    def preview_resized_changed(self):
        """The preview sizing option has changed"""
        self.image_view_area.set_index(int(not self.preview_resized.isChecked()))
    # endregion

    def build_statusbar(self) -> None:
        """Creates a simple status bar for hints and counts."""
        sb = QtWidgets.QStatusBar()
        self.setStatusBar(sb)
        self.count_label = QtWidgets.QLabel("No groups loaded")
        sb.addPermanentWidget(self.count_label)

        # Only shown while regrouping. The status bar stays usable then, when the rest of the window isn't
        self._cancel_regroup_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_regroup_btn.clicked.connect(self.cancel_regroup)
        self._cancel_regroup_btn.hide()
        sb.addPermanentWidget(self._cancel_regroup_btn)

    def delete_file(self, path: Path) -> int:
        """
        Delete a file, or send it to the recycle bin if that option is set.

        Returns:
            int: The size of the file that was deleted.
        """
        file_size = path.stat().st_size
        if self.recycle:
            logger.info("Moving to recycle: %s", path)
            send2trash(path)
        else:
            logger.info("Deleting %s", path)
            path.unlink()

        return file_size

    @staticmethod
    def move_file(source: Path, folder: Path) -> Path:
        """
        Move a file into a folder. It never replaces a file: if the name is taken, it gets " (2)", " (3)"... added.

        Returns:
            Path: Where the file was moved to.
        """
        destination = folder / source.name
        number = 2
        while destination.exists():
            destination = folder / f"{source.stem} ({number}){source.suffix}"
            number += 1

        folder.mkdir(parents=True, exist_ok=True)
        logger.info("Moving %s to %s", source, destination)
        shutil.move(str(source), str(destination))
        return destination

    def process_file_states(self, action: "FileAction"):
        """
        Do an action (Delete or Move) to every MARK file.

        Move asks for the folder first. Moved files never replace anything (see move_file).
        """
        if not self.processor:
            return

        to_process = [path for path, state in self.file_states.items() if state.state == SelectionState.MARK]
        if not to_process:
            QtWidgets.QMessageBox.information(self, "Nothing to do", "No files are marked.")
            return

        folder = None
        if action == FileAction.MOVE:
            folder = self.choose_move_folder(f"Move {len(to_process)} marked file(s) to", to_process[0].path_obj.parent)
            if folder is None:
                return

        is_paused = self.processor.conditional_pause()
        self.image_view_area.clear()
        # Previews and thumbnails are read in the background with their files open, and Windows won't delete or move an
        #   open file
        self.image_view_area.release_files()
        self.duplicate_group_list.release_files()

        file_size_deleted = 0
        file_count_deleted = 0
        file_count_moved = 0
        file_count_renamed = 0
        processed_files = set()
        failed_files = []
        skipped_files = []

        try:
            # One paused() block for every file, so the processor renumbers its groups once, not once per file
            with self.processor.paused():
                for file in to_process:
                    if file.is_zip:
                        # set_file_state never allows this, but acting on file.path_obj would delete or move the WHOLE
                        #   zip
                        logger.warning("Refusing to %s %s, files in zips are read-only", action.value, file)
                        processed_files.add(file)
                        continue

                    if file not in self.processor._reverse_hashes:
                        # Its folder was removed from the scan after it was marked, so leave the file alone
                        logger.info("Skipping %s, it is no longer part of the scan", file)
                        skipped_files.append(file)
                        processed_files.add(file)
                        continue

                    try:
                        if action == FileAction.DELETE:
                            try:
                                file_size_deleted += self.delete_file(file.path_obj)
                                file_count_deleted += 1
                            except FileNotFoundError:
                                logger.info("File already deleted: %s", file)
                            self.processor.remove(file)
                            self.image_view_area.forget(file)
                        else:  # FileAction.MOVE
                            destination = self.move_file(file.path_obj, folder)
                            file_count_renamed += destination.name != file.path_obj.name
                            # The file isn't at its old path anymore (moves never replace a file, see move_file)
                            self.image_view_area.forget(file)

                            if any(_is_under(f, destination) for f in self.file_paths_selected()):
                                self.processor.rename(file, ZipPath(str(destination)))
                            else:
                                # This file is no longer in a selected folder! So remove it from the matching
                                self.processor.remove(file)
                            file_count_moved += 1
                    except OSError as e:
                        # e.g. the file is open in another program. Leave it marked so it can be tried again
                        logger.warning("Failed to %s %s: %s", action.value, file, e)
                        failed_files.append(file)
                    else:
                        processed_files.add(file)
        finally:
            # Clear the states which we have processed
            self.file_states = {k: v for k, v in self.file_states.items() if k not in processed_files}

            # If total pages has decreased passed the current page, then make sure to set the current page
            self.current_page = min(self.current_page, self.total_pages)

            # Update the GUI:
            self.deleted_files_size += file_size_deleted
            self.deleted_files_count += file_count_deleted
            self.update_labels()
            self.update_group_list()

            # Resume processing
            self.processor.conditional_resume(is_paused)

        # region Final status popup
        popup_text = ""

        if file_count_deleted:
            popup_text += f"Deleted {file_count_deleted} files for a savings of {human_bytes(file_size_deleted)}.\n"

        if file_count_moved:
            popup_text += f"Moved {file_count_moved} file(s) to {folder}.\n"

        if file_count_renamed:
            popup_text += f"Renamed {file_count_renamed} of them, since a file with the same name was already there.\n"

        if failed_files:
            popup_text += f"Failed to process {len(failed_files)} files (see the log), they are still marked.\n"

        if skipped_files:
            popup_text += f"Skipped {len(skipped_files)} files which are no longer part of the scan.\n"

        if popup_text:
            dlg = QtWidgets.QMessageBox(self)
            dlg.setWindowTitle("Result")
            dlg.setText(popup_text)
            dlg.exec()
        # endregion

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Stop the processor if closing, and confirm close if there is data loaded"""
        if self.processor:
            if len(self.processor.matches) and not self.confirm_close():
                # There IS data loaded and user said NO to exiting
                event.ignore()
                return

            # Stop the scan (even a paused one), and wait for its thread to stop. Otherwise the thread pool would wait
            #   for that thread as Python shuts down, and by then the thread can't run to finish, so the app would hang
            self.processor.finish()
            self._threadpool.waitForDone()

        # Opened here next time, see restore_position
        self._settings.setValue("window/geometry", self.saveGeometry())
        event.accept()

    def confirm_close(self) -> bool:
        """
        Show a confirmation dialog asking if the user really wants to close.

        Returns:
            True if the user confirmed closing; False to keep the app open.
        """
        return self.exit_warning.exec() == QtWidgets.QMessageBox.StandardButton.Yes
