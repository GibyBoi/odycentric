"""The Odycentric window: drop photos in, pick settings, get cutouts out.

All the heavy work happens in a separate worker process (worker.py) inside a
Windows job (guard.py). This window queues photos, shows progress, previews
and scores, and can stop the worker instantly.
"""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QSize, QStandardPaths, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QColorDialog, QComboBox, QDialog, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout,
    QWidget)

from odycentric import __version__, guard, history
from odycentric.engine import MODELS, OUTPUT_SUFFIX, PHOTO_TYPES, ROOT

APP = "Odycentric"
ICON = ROOT / "assets" / "odycentric.ico"
GB = guard.GB
CPUS = os.cpu_count() or 4

# Everything but this much is the AI's to use by default; Windows keeps the rest.
RESERVED_FOR_WINDOWS = 3
AUTO_MEMORY = max(4, guard.total_memory() // GB - RESERVED_FOR_WINDOWS)
# Stop if free memory falls below this while working. Past it, Windows starts
# swapping hard and the whole PC can lock up, which helps nobody finish faster.
LOW_MEMORY = GB // 2
IDLE_UNLOAD_MS = 60_000
MAX_PHOTOS_PER_DROP = 2000

PRIORITIES = {
    "high": ("High: takes over the PC", guard.HIGH_PRIORITY),
    "normal": ("Normal: shares fairly", guard.NORMAL_PRIORITY),
    "low": ("Low: spare time only", guard.IDLE_PRIORITY),
}
THREADS = [(0, "Auto (one per core)")] + [
    (n, "All threads" if n == CPUS else str(n)) for n in sorted({CPUS, 16, 12, 8, 6, 4, 2, 1}, reverse=True)
    if n <= CPUS]
EDGES = [("soft", "Soft (natural)"), ("sharp", "Sharp (no see-through)")]
OUTPUT_SIZES = [(0, "Original"), (4096, "Up to 4096 px"), (2048, "Up to 2048 px"), (1024, "Up to 1024 px")]
MEMORY_LIMITS = [(0, f"Auto ({AUTO_MEMORY} GB)")] + [(g, f"{g} GB") for g in (8, 12, 16, 24) if g < AUTO_MEMORY]
BACKGROUNDS = [("transparent", "Transparent"), ("white", "White"), ("black", "Black"), ("custom", "Custom colour…")]
BACKGROUND_COLOURS = {"transparent": None, "white": (255, 255, 255), "black": (0, 0, 0)}

TIPS = {
    "model": "Which AI model cuts the photo out. Hover each choice for what it is good at.",
    "resolution": "How many pixels the AI studies. At the model's own size it makes one pass over the whole "
                  "photo. Above that it makes extra passes along the subject's outline for finer edges, which "
                  "takes several times longer. It never goes past the photo's own size.",
    "edges": "Soft keeps natural see-through edges like hair and fur. Sharp makes every pixel either fully "
             "solid or fully clear, which suits stickers and logos.",
    "cleanup": "Removes the old background's tint from see-through edge pixels so hair doesn't carry a halo. "
               "Fully solid pixels are never changed.",
    "size": "Shrinks the photo before cutting it out: faster, smaller files. Your original file is never touched.",
    "threads": "How many processor threads the AI uses. More is not always faster: on some processors the "
               "extra threads land on slower cores. The scores will show what works on this PC.",
    "priority": "High: Odycentric gets the processor before anything else until your photos are done, so "
                "other programs may stall. Normal: shares fairly. Low: only uses time nothing else wants.",
    "memory": "The most memory the AI may use. If a photo needs more, that photo fails instead of the PC "
              "freezing. The standard models need about 7 GB, the HD models about 22 GB.",
    "goal": "Every finished photo gets a score from 0 to 100 that mixes quality and speed.\n\n"
            "Quality is your star rating if you gave one (5 stars = 100). Otherwise it is edge clarity: "
            "how cleanly the AI separated the subject, judged by how thin the see-through band around its "
            "outline is.\n\nSpeed is 100 at 10 seconds per photo or less, and halves each time the time "
            "doubles.\n\nThis setting decides the mix: Quality counts quality 80% and speed 20%, "
            "Balanced 60/40, Speed 30/70.",
}


def worker_python():
    """pythonw.exe next to the one running the app, so the worker uses the same venv with no console."""
    exe = Path(sys.executable)
    windowless = exe.with_name("pythonw.exe")
    return str(windowless if windowless.exists() else exe)


def scaled_image(path, longest):
    """Read an image already shrunk to fit `longest` pixels. Cheap for big JPEGs."""
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > longest:
        reader.setScaledSize(size.scaled(longest, longest, Qt.KeepAspectRatio))
    image = reader.read()
    return None if image.isNull() else image


def checkerboard(painter, rect, cell=10):
    painter.fillRect(rect, QColor(236, 236, 236))
    dark = QColor(200, 200, 200)
    for y in range(rect.top(), rect.bottom() + 1, cell):
        for x in range(rect.left() + ((y - rect.top()) // cell % 2) * cell, rect.right() + 1, cell * 2):
            painter.fillRect(x, y, min(cell, rect.right() + 1 - x), min(cell, rect.bottom() + 1 - y), dark)


def thumbnail(path, size=56, checker=False):
    image = scaled_image(path, size)
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    if image is not None:
        painter = QPainter(pixmap)
        x, y = (size - image.width()) // 2, (size - image.height()) // 2
        if checker:
            checkerboard(painter, pixmap.rect().adjusted(x, y, -x, -y), 7)
        painter.drawImage(x, y, image)
        painter.end()
    return QIcon(pixmap)


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def stars(n):
    return "★" * n + "☆" * (5 - n) if n else ""


class Combo(QComboBox):
    """Ignores the mouse wheel until clicked, so scrolling the settings panel can't change a setting."""

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        # let the settings panel be narrower than the longest choice
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(8)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class Worker(QObject):
    """One worker process inside its own guard.Job. Emits (worker, event) for each JSON event."""

    event = Signal(object, object)

    def __init__(self, model, threads, priority, memory_limit):
        super().__init__()
        self.model, self.threads, self.priority, self.memory_limit = model, threads, priority, memory_limit
        self.ready = False
        priority_class = PRIORITIES[priority][1]
        # A job can only pin Normal or lower; for High the worker raises itself.
        self.job = guard.Job(memory_limit, None if priority_class == guard.HIGH_PRIORITY else priority_class)
        self.proc = self.job.start([worker_python(), "-u", "-m", "odycentric.worker", self.job.name], ROOT)
        self.stderr_tail = deque(maxlen=40)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _read_stdout(self):
        for line in self.proc.stdout:
            try:
                self.event.emit(self, json.loads(line))
            except ValueError:
                pass
        self.event.emit(self, {"event": "exited", "code": self.proc.wait()})

    def _drain_stderr(self):
        # Always read stderr, or a chatty worker could fill the pipe and hang.
        for line in self.proc.stderr:
            self.stderr_tail.append(line.decode(errors="replace").rstrip())

    def verify(self, pid):
        """Only a worker that is provably inside the job's limits gets its settings."""
        if not self.job.contains(pid):
            return False
        self._send({"model": str(self.model.path), "threads": self.threads,
                    "priority": PRIORITIES[self.priority][1]})
        return True

    def process(self, job):
        self._send(job)

    def _send(self, message):
        try:
            self.proc.stdin.write((json.dumps(message) + "\n").encode())
            self.proc.stdin.flush()
        except OSError:
            pass  # it has died; the exited event will follow

    def stop(self):
        self.job.close()  # kill-on-close ends the worker immediately
        try:
            self.proc.kill()  # and its launcher
        except OSError:
            pass


class Downloader(QThread):
    progress = Signal(int, int)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cancelled = False

    def run(self):
        model = self.model
        partial = model.path.with_name(model.path.name + ".part")
        try:
            model.path.parent.mkdir(parents=True, exist_ok=True)
            digest, done = hashlib.sha256(), 0
            with urllib.request.urlopen(model.url, timeout=30) as response, open(partial, "wb") as out:
                while chunk := response.read(1 << 20):
                    if self.cancelled:
                        raise InterruptedError
                    out.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    self.progress.emit(done, model.size)
            if done != model.size or digest.hexdigest() != model.sha256:
                raise ValueError("the download did not match the expected file, so it was thrown away.")
            partial.replace(model.path)
            self.finished_ok.emit()
        except InterruptedError:
            partial.unlink(missing_ok=True)
            self.failed.emit("")
        except Exception as error:
            partial.unlink(missing_ok=True)
            self.failed.emit(str(error))


class PhotoList(QListWidget):
    def __init__(self):
        super().__init__()
        self.setIconSize(QSize(56, 56))
        self.setSpacing(2)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setAcceptDrops(False)  # the window handles drops
        self.setUniformItemSizes(True)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.count() == 0:
            painter = QPainter(self.viewport())
            painter.setPen(self.palette().placeholderText().color())
            font = painter.font()
            font.setPointSizeF(font.pointSizeF() * 1.3)
            painter.setFont(font)
            painter.drawText(self.viewport().rect(), Qt.AlignCenter,
                             "Drop photos or folders here\n\nor click Add photos")


class Preview(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.message = "Select a finished photo to preview it"
        self.setMinimumSize(260, 260)

    def show_image(self, image, message=""):
        self.image, self.message = image, message
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.image is None:
            painter.setPen(self.palette().placeholderText().color())
            painter.drawText(self.rect().adjusted(12, 12, -12, -12), Qt.AlignCenter | Qt.TextWordWrap, self.message)
            return
        size = self.image.size().scaled(self.size(), Qt.KeepAspectRatio)
        x, y = (self.width() - size.width()) // 2, (self.height() - size.height()) // 2
        target = self.rect().adjusted(x, y, 0, 0)
        target.setSize(size)
        checkerboard(painter, target, 12)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(target, self.image)


class Stars(QWidget):
    rated = Signal(int)

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(QLabel("Your rating "))
        self.buttons = []
        for n in range(1, 6):
            button = QToolButton(text="☆", autoRaise=True)
            button.setToolTip(f"{n} star{'s' if n > 1 else ''}. Your rating replaces edge clarity as this "
                              "photo's quality score. Click the same star again to clear it.")
            button.clicked.connect(lambda _=False, n=n: self.rated.emit(0 if n == self.value else n))
            layout.addWidget(button)
            self.buttons.append(button)
        self.value = 0

    def set_value(self, value, enabled):
        self.value = value or 0
        for n, button in enumerate(self.buttons, 1):
            button.setText("★" if n <= self.value else "☆")
        self.setEnabled(enabled)


class HistoryDialog(QDialog):
    HEADERS = ["When", "Photo", "Subject", "AI res.", "Threads", "Priority", "Edges", "Power",
               "Seconds", "Passes", "Clarity", "Rating", "Score"]

    def __init__(self, parent, runs, goal):
        super().__init__(parent)
        self.setWindowTitle(f"{APP}: score history")
        self.resize(1100, 520)
        table = QTableWidget(len(runs), len(self.HEADERS))
        table.setHorizontalHeaderLabels(self.HEADERS)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for row, run in enumerate(runs):
            values = [
                datetime.fromtimestamp(run["ts"]).strftime("%b %d %H:%M"), Path(run["photo"]).name,
                MODELS[run["model"]].label if run["model"] in MODELS else run["model"], run["resolution"],
                dict(THREADS).get(run["threads"], run["threads"]), PRIORITIES.get(run["priority"], ("?",))[0],
                run["edges"], run["power"], round(run["seconds"], 1), run["passes"], run["clarity"],
                stars(run["rating"]), history.score(run, goal)]
            for col, value in enumerate(values):
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, value)
                table.setItem(row, col, item)
        table.setSortingEnabled(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.cleared = False
        clear = QPushButton("Clear history…")
        clear.clicked.connect(self.clear)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(QLabel(f"{len(runs)} photos scored. Scores use the current “Optimize for” setting."))
        buttons.addStretch()
        buttons.addWidget(clear)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(table)
        layout.addLayout(buttons)

    def clear(self):
        if QMessageBox.question(self, APP, "Delete every score? Your photos and cutouts are not touched.") \
                == QMessageBox.Yes:
            self.cleared = True
            self.accept()


class MainWindow(QMainWindow):
    WORKER_SETTINGS = {"model", "threads", "priority", "memory"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP)
        if ICON.exists():
            self.setWindowIcon(QIcon(str(ICON)))
        self.setAcceptDrops(True)
        self.resize(1380, 820)
        self.settings = QSettings(APP, APP)
        self.history = history.History()

        self.items = {}  # id -> dict(src, status, dst, message, item, run_id, opts, power)
        self.queue = deque()
        self.next_id = 0
        self.current = None
        self.worker = None
        self.downloader = None
        self.crashes = 0
        self.followed = None
        self.suggested = None
        self.batch_total = self.batch_done = 0
        self.recent_seconds = deque(maxlen=5)
        self.custom_colour = QColor(self.settings.value("custom_colour", "#2b6cb0"))

        self.idle_timer = QTimer(self, singleShot=True, interval=IDLE_UNLOAD_MS, timeout=self.unload)
        self.memory_timer = QTimer(self, interval=2000, timeout=self.check_memory)

        self._build()
        self.refresh_suggestions()
        self.set_status_idle()

    # ---- layout ----

    def _combo(self, options, key, default, tip=None, item_tips=None):
        box = Combo()
        for value, label in options:
            box.addItem(label, value)
        saved = str(self.settings.value(key, default))
        for i in range(box.count()):
            if str(box.itemData(i)) == saved:
                box.setCurrentIndex(i)
        for i, text in enumerate(item_tips or []):
            box.setItemData(i, text, Qt.ToolTipRole)
        if tip:
            box.setToolTip(tip)
        box.currentIndexChanged.connect(lambda: self.setting_changed(key, box))
        return box

    def _build(self):
        models = list(MODELS.values())
        self.model_box = self._combo([(m.key, m.label) for m in models], "model", "people", TIPS["model"],
                                     [m.blurb for m in models])
        self.resolution_box = Combo()
        self.resolution_box.setToolTip(TIPS["resolution"])
        self.fill_resolutions()
        self.resolution_box.currentIndexChanged.connect(
            lambda: self.setting_changed("resolution", self.resolution_box))
        self.edges_box = self._combo(EDGES, "edges", "soft", TIPS["edges"])
        self.cleanup_box = QCheckBox("Clean edge colours")
        self.cleanup_box.setToolTip(TIPS["cleanup"])
        self.cleanup_box.setChecked(str(self.settings.value("cleanup", "1")) == "1")
        self.cleanup_box.toggled.connect(self.cleanup_toggled)

        self.background_box = self._combo(BACKGROUNDS, "background", "transparent")
        self.background_box.activated.connect(self.background_chosen)
        self.size_box = self._combo(OUTPUT_SIZES, "max_side", 0, TIPS["size"])
        self.output_label = QLineEdit(readOnly=True)
        change = QPushButton("Change…")
        change.clicked.connect(self.choose_output)
        open_folder = QPushButton("Open folder")
        open_folder.clicked.connect(self.open_output)
        self.update_output_label()

        self.threads_box = self._combo(THREADS, "threads", 0, TIPS["threads"])
        self.priority_box = self._combo([(k, v[0]) for k, v in PRIORITIES.items()], "priority", "high",
                                        TIPS["priority"])
        self.memory_box = self._combo(MEMORY_LIMITS, "memory", 0, TIPS["memory"])

        self.goal_box = self._combo([(k, v[0]) for k, v in history.GOALS.items()], "goal", "balanced",
                                    TIPS["goal"])
        self.suggestion_label = QLabel()
        self.suggestion_label.setWordWrap(True)
        self.suggestion_label.setTextFormat(Qt.RichText)
        self.suggestion_label.setToolTip(TIPS["goal"])
        self.apply_button = QPushButton("Use suggested settings")
        self.apply_button.clicked.connect(self.apply_suggestion)
        history_button = QPushButton("History…")
        history_button.clicked.connect(self.show_history)

        cutout = QFormLayout()
        cutout.addRow("Subject", self.model_box)
        cutout.addRow("AI resolution", self.resolution_box)
        cutout.addRow("Edges", self.edges_box)
        cutout.addRow("", self.cleanup_box)
        output = QFormLayout()
        output.addRow("Background", self.background_box)
        output.addRow("Size", self.size_box)
        output.addRow("Save to", self.output_label)
        folder_buttons = QHBoxLayout()
        folder_buttons.addWidget(change)
        folder_buttons.addWidget(open_folder)
        output.addRow("", folder_buttons)
        performance = QFormLayout()
        performance.addRow("Threads", self.threads_box)
        performance.addRow("Priority", self.priority_box)
        performance.addRow("Memory limit", self.memory_box)
        suggestions = QFormLayout()
        suggestions.addRow("Optimize for", self.goal_box)
        suggestions.addRow(self.suggestion_label)
        suggestion_buttons = QHBoxLayout()
        suggestion_buttons.addWidget(self.apply_button)
        suggestion_buttons.addWidget(history_button)
        suggestions.addRow(suggestion_buttons)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        for title, form in (("Cutout", cutout), ("Output", output), ("Performance", performance),
                            ("Suggestions", suggestions)):
            group = QGroupBox(title)
            group.setLayout(form)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            panel_layout.addWidget(group)
        panel_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)

        add = QPushButton("Add photos…")
        add.clicked.connect(self.choose_photos)
        self.list = PhotoList()
        self.list.currentItemChanged.connect(self.show_preview)
        self.list.itemDoubleClicked.connect(lambda item: self.open_result(item))
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self.list_menu)
        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.addWidget(add)
        middle_layout.addWidget(self.list, 1)

        self.preview = Preview()
        self.show_original = QCheckBox("Show original")
        self.show_original.toggled.connect(lambda: self.show_preview(self.list.currentItem()))
        self.stars = Stars()
        self.stars.rated.connect(self.rate_current)
        self.stars.set_value(0, False)
        open_button = QPushButton("Open")
        open_button.clicked.connect(lambda: self.open_result(self.list.currentItem()))
        reveal_button = QPushButton("Show in folder")
        reveal_button.clicked.connect(lambda: self.reveal(self.list.currentItem()))
        preview_bar = QHBoxLayout()
        preview_bar.addWidget(self.show_original)
        preview_bar.addStretch()
        preview_bar.addWidget(self.stars)
        preview_bar.addStretch()
        preview_bar.addWidget(open_button)
        preview_bar.addWidget(reveal_button)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.preview, 1)
        right_layout.addLayout(preview_bar)

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(middle)
        split.addWidget(right)
        split.setSizes([420, 330, 630])
        split.setChildrenCollapsible(False)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(220)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.go = QPushButton("Start")
        self.go.setMinimumWidth(90)
        self.go.clicked.connect(self.go_clicked)
        bottom = QHBoxLayout()
        bottom.addWidget(self.progress)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.go)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(split, 1)
        layout.addLayout(bottom)
        self.setCentralWidget(root)

    # ---- settings ----

    def setting_changed(self, key, box):
        self.settings.setValue(key, str(box.currentData()))
        if key == "model":
            self.fill_resolutions()
        if key == "goal":
            self.refresh_scores()
        if key in self.WORKER_SETTINGS and self.worker and not self.busy():
            self.unload()  # the next photo starts a worker with the new settings
        self.refresh_suggestions()

    def cleanup_toggled(self, on):
        self.settings.setValue("cleanup", "1" if on else "0")
        self.refresh_suggestions()

    def fill_resolutions(self):
        model = self.model()
        wanted = int(self.settings.value("resolution", model.native))
        self.resolution_box.blockSignals(True)
        self.resolution_box.clear()
        for r in model.resolutions():
            self.resolution_box.addItem(f"{r} px" + (" (one pass)" if r == model.native else " (detail passes)"), r)
        index = self.resolution_box.findData(wanted)
        self.resolution_box.setCurrentIndex(max(index, 0))
        self.resolution_box.blockSignals(False)

    def model(self):
        return MODELS[self.model_box.currentData()]

    def memory_limit(self):
        return (int(self.memory_box.currentData()) or AUTO_MEMORY) * GB

    def background_colour(self):
        key = self.background_box.currentData()
        if key == "custom":
            c = self.custom_colour
            return [c.red(), c.green(), c.blue()]
        colour = BACKGROUND_COLOURS[key]
        return list(colour) if colour else None

    def background_chosen(self, index):
        if self.background_box.itemData(index) == "custom":
            colour = QColorDialog.getColor(self.custom_colour, self, "Background colour")
            if colour.isValid():
                self.custom_colour = colour
                self.settings.setValue("custom_colour", colour.name())

    def output_dir(self):
        default = Path(QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)) / APP
        return Path(self.settings.value("output_dir", str(default)))

    def update_output_label(self):
        self.output_label.setText(str(self.output_dir()))
        self.output_label.setCursorPosition(0)
        self.output_label.setToolTip(str(self.output_dir()))

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Save cutouts to", str(self.output_dir()))
        if folder:
            self.settings.setValue("output_dir", folder)
            self.update_output_label()

    def photo_options(self):
        """Per-photo settings, captured when the photo is handed to the worker."""
        return {"resolution": self.resolution_box.currentData(), "max_side": self.size_box.currentData() or None,
                "sharp": self.edges_box.currentData() == "sharp", "cleanup": self.cleanup_box.isChecked(),
                "background": self.background_colour(), "out_dir": str(self.output_dir())}

    # ---- adding photos ----

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.add_paths([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])

    def choose_photos(self):
        pattern = " ".join(f"*{ext}" for ext in sorted(PHOTO_TYPES))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add photos", self.settings.value("last_folder", ""), f"Photos ({pattern})")
        if files:
            self.settings.setValue("last_folder", str(Path(files[0]).parent))
            self.add_paths(files)

    def collect(self, paths):
        out_dir = self.output_dir().resolve()
        found = []

        def wanted(path):
            return path.suffix.lower() in PHOTO_TYPES and OUTPUT_SUFFIX not in path.stem

        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                for folder, dirs, files in os.walk(path):
                    dirs[:] = [d for d in dirs if (Path(folder) / d).resolve() != out_dir]
                    found += [Path(folder) / f for f in sorted(files) if wanted(Path(f))]
                    if len(found) > MAX_PHOTOS_PER_DROP:
                        break
            elif path.is_file() and wanted(path):
                found.append(path)
        if len(found) > MAX_PHOTOS_PER_DROP:
            QMessageBox.information(self, APP, f"That is a lot of photos. Adding the first {MAX_PHOTOS_PER_DROP}.")
            found = found[:MAX_PHOTOS_PER_DROP]
        return found

    def add_paths(self, paths):
        pending = {str(self.items[i]["src"]) for i in self.queue}
        added = 0
        for path in self.collect(paths):
            if str(path) in pending:
                continue
            item_id, self.next_id = self.next_id, self.next_id + 1
            item = QListWidgetItem(thumbnail(path), "")
            item.setData(Qt.UserRole, item_id)
            self.list.addItem(item)
            self.items[item_id] = {"src": path, "status": "Waiting", "dst": None, "item": item, "run_id": None}
            self.refresh_item(item_id)
            self.queue.append(item_id)
            added += 1
        if not added:
            if paths:
                self.status.setText("No photos found there. Odycentric reads JPG, PNG, WEBP, HEIC, BMP and TIFF.")
            return
        if not self.busy():
            self.batch_total = self.batch_done = 0
        self.batch_total += added
        self.update_progress()
        self.pump()

    def refresh_item(self, item_id):
        entry = self.items[item_id]
        status = entry["status"]
        run = self.history.get(entry["run_id"]) if entry["run_id"] else None
        if run:
            goal = self.goal_box.currentData()
            status = f"Done in {run['seconds']:.0f} s · score {history.score(run, goal)}  {stars(run['rating'])}"
        entry["item"].setText(f"{entry['src'].name}\n{status}")
        tip = entry.get("message") or status
        if run:
            tip = (f"Edge clarity {run['clarity']} · {run['passes']} AI pass{'es' if run['passes'] > 1 else ''} · "
                   f"{run['seconds']:.1f} s\n{entry['dst']}")
        entry["item"].setToolTip(f"{entry['src']}\n{tip}")
        failed = entry["status"].startswith("Failed")
        entry["item"].setForeground(QColor(200, 40, 40) if failed else self.list.palette().text().color())

    def refresh_scores(self):
        for item_id in self.items:
            self.refresh_item(item_id)

    # ---- processing ----

    def busy(self):
        """Working on a photo, loading the model or downloading one. An idle loaded model is not busy."""
        loading = self.worker is not None and not self.worker.ready
        return self.current is not None or self.downloader is not None or loading

    def pump(self):
        """Hand the next waiting photo to the worker, starting it if needed."""
        try:
            self._pump()
        finally:
            self.update_controls()

    def _pump(self):
        if self.current is not None or self.downloader or not self.queue:
            return
        self.idle_timer.stop()
        model = self.model()
        if not model.is_downloaded():
            self.offer_download(model)
            return
        if self.worker is None:
            self.start_worker()
            return
        if not self.worker.ready:
            return
        item_id = self.queue.popleft()
        entry = self.items[item_id]
        self.current = item_id
        entry["status"] = "Removing background…"
        entry["opts"] = self.photo_options()
        entry["power"] = "battery" if guard.on_battery() else "plugged in"
        self.refresh_item(item_id)
        self.update_progress()
        self.worker.process({"id": item_id, "src": str(entry["src"]), **entry["opts"]})

    def start_worker(self):
        model, limit = self.model(), self.memory_limit()
        free = guard.free_memory()
        problems = []
        if model.memory > limit:
            problems.append(f"The {model.label} model needs about {model.memory // GB} GB, but the memory limit "
                            f"is {limit // GB} GB, so it will probably fail. Raise it under Performance.")
        if model.memory > free:
            problems.append(f"The {model.label} model needs about {model.memory // GB} GB and your PC has "
                            f"{free / GB:.1f} GB free. Windows will push other programs out of memory to make "
                            "room, and the whole PC may be very slow until Odycentric finishes.")
        if problems:
            box = QMessageBox(QMessageBox.Warning, APP, "\n\n".join(problems), parent=self)
            start = box.addButton("Start anyway", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() is not start:
                self.pause("Not started. Close some programs or change the settings, then click Start.")
                return
        self.worker = Worker(model, int(self.threads_box.currentData()), self.priority_box.currentData(), limit)
        self.worker.event.connect(self.on_event)
        if self.worker.priority == "high":
            guard.set_own_priority(guard.HIGH_PRIORITY)  # keep this window responsive alongside it
        self.memory_timer.start()
        self.status.setText(f"Loading the {model.label} model…")
        self.progress.setRange(0, 0)

    def on_event(self, worker, event):
        if worker is not self.worker:
            return  # a worker we already stopped
        kind = event["event"]
        if kind == "hello":
            if not worker.verify(event["pid"]):
                self.stop_worker()
                self.pause("Stopped: Windows did not apply the safety limits to the worker, so it was not "
                           "allowed to run.")
        elif kind == "ready":
            worker.ready = True
            self.update_progress()
            self.pump()
        elif kind in ("done", "error"):
            self.finish_current(worker, event)
        elif kind == "failed":
            self.stop_worker()
            self.pause(f"Could not start: {event['message']}")
        elif kind == "exited":
            self.worker_died(worker)

    def finish_current(self, worker, event):
        item_id = event["id"]
        entry = self.items[item_id]
        if event["event"] == "done":
            self.crashes = 0
            opts = entry["opts"]
            entry.update(status="Done", dst=Path(event["dst"]), message="")
            entry["run_id"] = self.history.add({
                "ts": time.time(), "photo": str(entry["src"]), "output": event["dst"],
                "megapixels": event["megapixels"], "model": worker.model.key, "resolution": opts["resolution"],
                "threads": worker.threads, "priority": worker.priority,
                "edges": "sharp" if opts["sharp"] else "soft", "cleanup": int(opts["cleanup"]),
                "max_side": opts["max_side"], "power": entry["power"], "seconds": event["seconds"],
                "ai_seconds": event["ai_seconds"], "passes": event["passes"], "clarity": event["clarity"]})
            entry["item"].setIcon(thumbnail(entry["dst"], checker=True))
            self.recent_seconds.append(event["seconds"])
            self.refresh_suggestions()
        else:
            entry.update(status=f"Failed: {event['message']}", message=event["message"])
        self.refresh_item(item_id)
        self.current = None
        self.batch_done += 1
        selected = self.list.currentItem()
        if selected is None or selected.data(Qt.UserRole) == self.followed:
            self.followed = item_id
            self.list.setCurrentItem(entry["item"])
        self.update_progress()
        if self.queue:
            self.pump()
        else:
            self.idle_timer.start()
            self.set_status_idle(finished=True)

    def worker_died(self, worker):
        tail = "\n".join(worker.stderr_tail)
        self.stop_worker()
        if self.current is not None:
            entry = self.items[self.current]
            entry["status"] = "Failed: the worker stopped unexpectedly (it may have hit its memory limit)."
            entry["message"] = tail[-500:]
            self.refresh_item(self.current)
            self.current = None
            self.batch_done += 1
            self.crashes += 1
        if self.crashes >= 2:
            self.pause("Stopped after the worker failed twice in a row. Hover a failed photo for details.")
        else:
            self.pump()

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        guard.set_own_priority(guard.NORMAL_PRIORITY)
        self.memory_timer.stop()
        self.update_controls()

    def unload(self):
        """Free the model's memory once there is nothing left to do."""
        if self.current is None and not self.queue:
            self.stop_worker()
            self.set_status_idle()

    def check_memory(self):
        free = guard.free_memory()
        if self.worker and free < LOW_MEMORY:
            self.stop_all(f"Stopped because your PC was down to {free / GB:.1f} GB of free memory, the point "
                          "where Windows starts to lock up. Close some programs, then click Start.")

    def go_clicked(self):
        if self.busy():
            self.stop_all("Stopped.")
        elif self.queue:
            self.crashes = 0
            self.pump()

    def stop_all(self, message):
        """Kill the worker immediately. Waiting photos stay in the list for Start."""
        if self.downloader:
            self.downloader.cancelled = True
        if self.current is not None:
            entry = self.items[self.current]
            entry["status"] = "Waiting"
            self.refresh_item(self.current)
            self.queue.appendleft(self.current)
            self.current = None
        self.stop_worker()
        self.pause(message)

    def pause(self, message):
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status.setText(message)
        self.update_controls()

    def update_controls(self):
        busy = self.busy()
        self.go.setText("Stop" if busy else "Start")
        self.go.setEnabled(busy or bool(self.queue))
        # these need a fresh worker, so only between photos
        for box in (self.model_box, self.threads_box, self.priority_box, self.memory_box):
            box.setEnabled(not busy)
        self.apply_button.setEnabled(not busy and self.suggested is not None
                                     and self.suggested != self.current_key())

    def update_progress(self):
        if self.worker and not self.worker.ready:
            return
        self.progress.setRange(0, max(1, self.batch_total))
        self.progress.setValue(self.batch_done)
        if self.current is not None:
            name = self.items[self.current]["src"].name
            text = f"Removing background from {name} ({self.batch_done + 1} of {self.batch_total})"
            if self.recent_seconds:
                left = (self.batch_total - self.batch_done) * sum(self.recent_seconds) / len(self.recent_seconds)
                text += f" · about {left / 60:.0f} min left" if left >= 90 else f" · about {left:.0f} s left"
            self.status.setText(text)

    def set_status_idle(self, finished=False):
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if finished else 0)
        done = f"Finished {self.batch_done} photo{'s' if self.batch_done != 1 else ''}. " if finished else ""
        threads = self.threads_box.currentText().lower()
        priority = self.priority_box.currentData()
        self.status.setText(f"{done}Ready. Photos run on your CPU ({threads} threads, {priority} priority), "
                            f"using up to {self.memory_limit() // GB} GB of memory.")
        self.update_controls()

    # ---- scores and suggestions ----

    def describe(self, settings):
        model = MODELS.get(settings["model"])
        threads = dict(THREADS).get(settings["threads"], settings["threads"])
        return ", ".join([
            model.label if model else settings["model"], f"{settings['resolution']} px",
            f"threads: {str(threads).lower()}", f"{settings['priority']} priority", f"{settings['edges']} edges",
            "colour cleanup " + ("on" if settings["cleanup"] else "off")])

    def current_key(self):
        return {"model": self.model_box.currentData(), "resolution": self.resolution_box.currentData(),
                "threads": int(self.threads_box.currentData()), "priority": self.priority_box.currentData(),
                "edges": self.edges_box.currentData(), "cleanup": int(self.cleanup_box.isChecked())}

    def refresh_suggestions(self):
        goal = self.goal_box.currentData()
        power = "battery" if guard.on_battery() else "plugged in"
        best, situation = history.suggest(self.history.all(), goal, power)
        self.suggested = None
        if not best:
            self.suggestion_label.setText("No scores yet. Every photo you process gets a score, and the best "
                                          "settings you have tried show up here.")
        else:
            where = {"battery": "on battery", "plugged in": "when plugged in"}.get(
                situation, f"across all situations (fewer than {history.MIN_SAME_SITUATION} photos {power} so far)")
            top = best[0]
            lines = [f"<b>Best for {history.GOALS[goal][0].lower()} {where}:</b><br>{self.describe(top.settings)}"
                     f"<br>Expected score {top.score:.0f}, about {top.seconds:.0f} s per photo.",
                     f"<small>Speed from {plural(top.runs, 'photo')} with exactly these settings; quality from "
                     f"{plural(top.quality_runs, 'photo')} with this subject, resolution and edges (threads and "
                     "priority never change the cutout).</small>"]
            for other in best[1:]:
                lines.append(f"<small>Next: {self.describe(other.settings)} (expected {other.score:.0f}, "
                             f"{other.seconds:.0f} s)</small>")
            lines.append("<small>Suggestions only know settings you have tried, so try new ones now and then."
                         "</small>")
            self.suggestion_label.setText("<br>".join(lines))
            if top.settings["model"] in MODELS:
                self.suggested = top.settings
        self.update_controls()

    def apply_suggestion(self):
        if not self.suggested or self.busy():
            return
        s = self.suggested
        self.model_box.setCurrentIndex(self.model_box.findData(s["model"]))
        self.settings.setValue("resolution", s["resolution"])
        self.fill_resolutions()
        self.setting_changed("resolution", self.resolution_box)
        for box, value in ((self.threads_box, s["threads"]), (self.priority_box, s["priority"]),
                           (self.edges_box, s["edges"])):
            index = next((i for i in range(box.count()) if str(box.itemData(i)) == str(value)), -1)
            if index >= 0:
                box.setCurrentIndex(index)
        self.cleanup_box.setChecked(bool(s["cleanup"]))
        self.refresh_suggestions()

    def rate_current(self, value):
        entry = self.entry_for(self.list.currentItem())
        if not entry or not entry["run_id"]:
            return
        self.history.rate(entry["run_id"], value)
        self.stars.set_value(value, True)
        self.refresh_item(entry["item"].data(Qt.UserRole))
        self.refresh_suggestions()

    def show_history(self):
        dialog = HistoryDialog(self, self.history.all(), self.goal_box.currentData())
        dialog.exec()
        if dialog.cleared:
            self.history.clear()
            for entry in self.items.values():
                entry["run_id"] = None
            self.refresh_scores()
            self.refresh_suggestions()

    # ---- model download ----

    def offer_download(self, model):
        answer = QMessageBox.question(
            self, "Download model",
            f"The {model.label} model is a one-time {model.size / 1e6:.0f} MB download from GitHub "
            f"(the rembg project). It is checked against a known fingerprint before use.\n\nDownload it now?")
        if answer != QMessageBox.Yes:
            self.pause(f"The {model.label} model is needed first. Click Start to download it, "
                       "or choose a different subject.")
            return
        self.downloader = Downloader(model)
        self.downloader.progress.connect(self.download_progress)
        self.downloader.finished_ok.connect(self.download_done)
        self.downloader.failed.connect(self.download_failed)
        self.downloader.start()

    def download_progress(self, done, total):
        self.progress.setRange(0, total // 1_000_000)
        self.progress.setValue(done // 1_000_000)
        self.status.setText(f"Downloading the {self.downloader.model.label} model… "
                            f"{done / 1e6:.0f} of {total / 1e6:.0f} MB")

    def download_done(self):
        self.downloader = None
        self.pump()

    def download_failed(self, message):
        self.downloader = None
        self.pause(f"Download failed: {message}" if message else "Download cancelled.")

    # ---- preview and files ----

    def entry_for(self, item):
        return self.items.get(item.data(Qt.UserRole)) if item else None

    def show_preview(self, item, _previous=None):
        entry = self.entry_for(item)
        run = self.history.get(entry["run_id"]) if entry and entry["run_id"] else None
        self.stars.set_value(run["rating"] if run else 0, run is not None)
        if entry is None:
            self.preview.show_image(None, "Select a finished photo to preview it")
            return
        path = entry["src"] if self.show_original.isChecked() or not entry["dst"] else entry["dst"]
        image = scaled_image(path, 1400)
        message = entry["status"] if not entry["dst"] else "Can't preview this file here, but it was saved."
        self.preview.show_image(image, message)

    def open_result(self, item):
        entry = self.entry_for(item)
        if entry and entry["dst"]:
            self.open_path(entry["dst"])

    def reveal(self, item):
        entry = self.entry_for(item)
        if entry:
            subprocess.Popen(f'explorer /select,"{entry["dst"] or entry["src"]}"')

    def open_path(self, path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_output(self):
        self.output_dir().mkdir(parents=True, exist_ok=True)
        self.open_path(self.output_dir())

    def list_menu(self, pos):
        item = self.list.itemAt(pos)
        entry = self.entry_for(item)
        if not entry:
            return
        menu = QMenu(self)
        if entry["dst"]:
            menu.addAction("Open cutout", lambda: self.open_result(item))
        menu.addAction("Open original", lambda: self.open_path(entry["src"]))
        menu.addAction("Show in folder", lambda: self.reveal(item))
        menu.addSeparator()
        remove = QAction("Remove from list", menu)
        remove.setEnabled(item.data(Qt.UserRole) != self.current)
        remove.triggered.connect(self.remove_selected)
        menu.addAction(remove)
        menu.exec(self.list.viewport().mapToGlobal(pos))

    def remove_selected(self):
        for item in self.list.selectedItems():
            item_id = item.data(Qt.UserRole)
            if item_id == self.current:
                continue
            if item_id in self.queue:
                self.queue.remove(item_id)
                self.batch_total -= 1
            self.items.pop(item_id, None)
            self.list.takeItem(self.list.row(item))
        self.update_controls()

    def closeEvent(self, event):
        self.stop_all("Closing.")
        super().closeEvent(event)


def main():
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"{APP}.{APP}")
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setApplicationVersion(__version__)
    if ICON.exists():
        app.setWindowIcon(QIcon(str(ICON)))
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        window.add_paths(sys.argv[1:])
    return app.exec()
