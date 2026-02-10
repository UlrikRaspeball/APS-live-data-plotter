import os
import sys
import glob
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg


@dataclass
class Config:
    watch_folder: str = ""
    file_glob: str = "*.txt"          # adjust if needed: "*.A21" etc.
    poll_ms: int = 1000

    delimiter: str = "\t"
    metadata_lines: int = 7

    date_col: str = "Date"
    start_time_col: str = "Start Time"
    concentration_col: str = "Total Conc."
    median_col: str = "Median(µm)"    # normalization handles Median(um)/Median(μm)/Median(Âµm) too

    date_format: str = "%m/%d/%y"
    time_format: str = "%H:%M:%S"

    # Font sizes
    ui_font_pt: int = 11
    axis_tick_font_pt: int = 11
    axis_label_font_pt: int = 12


def read_text_with_fallback(path: str):
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read(), enc
        except Exception:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(), "utf-8(replace)"


def normalize_col_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("\ufeff", "")      # BOM
    s = s.replace("Âµ", "u").replace("Âμ", "u")
    s = s.replace("µ", "u").replace("μ", "u")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def parse_number_with_units(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    low = s.lower()
    if "ind" in low or "nan" in low:
        return None

    if "," in s and "." not in s:
        s = s.replace(",", ".")

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_datetime(date_s: str, time_s: str, cfg: Config):
    try:
        d = datetime.strptime(date_s, cfg.date_format).date()
        t = datetime.strptime(time_s, cfg.time_format).time()
        return datetime.combine(d, t)
    except Exception:
        return None


def find_latest_file(cfg: Config):
    if not cfg.watch_folder:
        return None
    pattern = os.path.join(cfg.watch_folder, cfg.file_glob)
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def format_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    total = seconds
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class DualTimeAxis(pg.AxisItem):
    """
    Bottom axis shows:
      Line 1: absolute time (HH:MM:SS)
      Line 2: elapsed since start (HH:MM:SS)
    Axis values are POSIX timestamps (float seconds).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_ts: float | None = None

    def set_base_timestamp(self, base_ts: float | None):
        self.base_ts = base_ts
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            try:
                abs_str = datetime.fromtimestamp(v).strftime("%H:%M:%S")
            except Exception:
                abs_str = ""
            if self.base_ts is None:
                out.append(abs_str)
            else:
                out.append(f"{abs_str}\n{format_hms(v - self.base_ts)}")
        return out


def load_full_run(path: str, cfg: Config):
    if not path or not os.path.exists(path):
        return [], [], [], {"error": "File not found"}, None

    text, enc_used = read_text_with_fallback(path)
    lines = text.splitlines()

    if len(lines) <= cfg.metadata_lines:
        return [], [], [], {"error": "File too short", "encoding": enc_used}, None

    data_lines = lines[cfg.metadata_lines:]

    header_line = data_lines[0]
    header = [h.strip() for h in header_line.split(cfg.delimiter)]
    norm_to_idx = {normalize_col_name(h): i for i, h in enumerate(header)}

    def idx_for(name):
        return norm_to_idx.get(normalize_col_name(name))

    date_i = idx_for(cfg.date_col)
    time_i = idx_for(cfg.start_time_col)
    conc_i = idx_for(cfg.concentration_col)
    med_i = idx_for(cfg.median_col)

    info = {
        "encoding": enc_used,
        "median_found": med_i is not None,
        "median_header_match": header[med_i] if med_i is not None else None,
        "required_found": (time_i is not None and conc_i is not None),
    }

    if time_i is None or conc_i is None:
        info["error"] = "Missing required columns (Start Time / Total Conc.)"
        return [], [], [], info, None

    x_ts, conc_vals, median_vals = [], [], []
    base_dt = None

    for line in data_lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(cfg.delimiter)]

        if conc_i >= len(parts) or time_i >= len(parts):
            continue

        conc = parse_number_with_units(parts[conc_i])
        if conc is None:
            continue

        med = None
        if med_i is not None and med_i < len(parts):
            med = parse_number_with_units(parts[med_i])

        dt = None
        if date_i is not None and date_i < len(parts):
            dt = parse_datetime(parts[date_i], parts[time_i], cfg)

        if dt is None:
            if base_dt is None:
                base_dt = datetime.now()
            dt = base_dt + timedelta(seconds=len(x_ts))

        if base_dt is None:
            base_dt = dt

        x_ts.append(dt.timestamp())
        conc_vals.append(conc)
        median_vals.append(float("nan") if med is None else med)

    return x_ts, conc_vals, median_vals, info, base_dt


class LiveTailReader:
    def __init__(self, path: str, cfg: Config):
        self.path = path
        self.cfg = cfg

        self._offset = 0
        self._buffer = ""
        self._last_size = 0

        self.header = []
        self.col_index = {}
        self.encoding_used = None

        self._init_header_and_offset()

    def _init_header_and_offset(self):
        if not os.path.exists(self.path):
            return

        _, enc_used = read_text_with_fallback(self.path)
        self.encoding_used = enc_used

        with open(self.path, "r", encoding=enc_used.replace("(replace)", ""), errors="replace") as f:
            for _ in range(self.cfg.metadata_lines):
                f.readline()
            header_line = f.readline()
            self._offset = f.tell()

        self._last_size = os.path.getsize(self.path)

        header = [h.strip() for h in header_line.strip().split(self.cfg.delimiter)]
        self.header = header
        self.col_index = {normalize_col_name(h): i for i, h in enumerate(header)}

    def _check_truncation(self):
        try:
            size = os.path.getsize(self.path)
        except FileNotFoundError:
            return

        if size < self._last_size:
            self._offset = 0
            self._buffer = ""
            self.header = []
            self.col_index = {}
            self._init_header_and_offset()

        self._last_size = size

    def read_new_rows(self):
        self._check_truncation()
        if not os.path.exists(self.path):
            return []

        enc = (self.encoding_used or "utf-8").replace("(replace)", "")

        try:
            with open(self.path, "r", encoding=enc, errors="replace") as f:
                f.seek(self._offset)
                chunk = f.read()
                self._offset = f.tell()
        except Exception:
            return []

        if not chunk:
            return []

        text = self._buffer + chunk
        lines = text.splitlines(True)

        complete = []
        self._buffer = ""

        for line in lines:
            if line.endswith("\n") or line.endswith("\r\n"):
                complete.append(line.strip())
            else:
                self._buffer = line

        rows = []
        for ln in complete:
            if not ln:
                continue
            parts = [p.strip() for p in ln.split(self.cfg.delimiter)]
            rows.append(parts)

        return rows


class LivePlotWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.settings = QtCore.QSettings("YourLab", "ConcentrationMonitor")

        # Bigger UI font
        app_font = QtGui.QFont()
        app_font.setPointSize(self.cfg.ui_font_pt)
        QtWidgets.QApplication.instance().setFont(app_font)

        self.setWindowTitle("Concentration + Median Monitor")
        self.resize(1180, 780)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)

        self.status_label = QtWidgets.QLabel("No folder selected.")
        top.addWidget(self.status_label, 1)

        self.select_folder_btn = QtWidgets.QPushButton("Select data folder")
        self.select_folder_btn.clicked.connect(self.select_folder)
        top.addWidget(self.select_folder_btn)

        self.full_mode_chk = QtWidgets.QCheckBox("Full run (load once)")
        top.addWidget(self.full_mode_chk)

        self.load_full_btn = QtWidgets.QPushButton("Load full run now")
        self.load_full_btn.clicked.connect(self.load_full_now)
        top.addWidget(self.load_full_btn)

        self.follow_btn = QtWidgets.QPushButton("Follow")
        self.follow_btn.clicked.connect(self.follow_only)
        top.addWidget(self.follow_btn)

        self.reset_follow_btn = QtWidgets.QPushButton("Reset + Follow")
        self.reset_follow_btn.clicked.connect(self.reset_and_follow)
        top.addWidget(self.reset_follow_btn)

        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self.on_pause_toggled)
        top.addWidget(self.pause_btn)

        self.autoscale_btn = QtWidgets.QPushButton("Autoscale Y: ON")
        self.autoscale_btn.setCheckable(True)
        self.autoscale_btn.setChecked(True)
        self.autoscale_btn.toggled.connect(self.on_autoscale_toggled)
        top.addWidget(self.autoscale_btn)

        # Plot with custom time axis
        self.time_axis = DualTimeAxis(orientation="bottom")
        self.plot = pg.PlotWidget(axisItems={"bottom": self.time_axis})
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Time (Start / Elapsed)")
        self.plot.setLabel("left", "Total Conc.", units="#/cm³")
        layout.addWidget(self.plot)

        # Bigger axis fonts
        tick_font = QtGui.QFont()
        tick_font.setPointSize(self.cfg.axis_tick_font_pt)

        label_font = QtGui.QFont()
        label_font.setPointSize(self.cfg.axis_label_font_pt)
        label_font.setBold(True)

        for name in ("left", "bottom", "right"):
            ax = self.plot.getAxis(name)
            try:
                ax.setTickFont(tick_font)
            except Exception:
                try:
                    ax.setStyle(tickFont=tick_font)
                except Exception:
                    pass
            try:
                ax.label.setFont(label_font)
            except Exception:
                pass

        # Left curve
        self.conc_curve = self.plot.plot([], [], pen=pg.mkPen(width=2))

        # Right axis + ViewBox (median, blue)
        self.plot.showAxis("right")
        right_axis = self.plot.getAxis("right")
        right_axis.setLabel("Median", units="um")
        right_axis.setPen(pg.mkPen("b"))
        right_axis.setTextPen(pg.mkPen("b"))

        self.median_vb = pg.ViewBox()
        self.plot.scene().addItem(self.median_vb)
        right_axis.linkToView(self.median_vb)
        self.median_vb.setXLink(self.plot.getViewBox())

        self.median_curve = pg.PlotCurveItem([], [], pen=pg.mkPen("b", width=2))
        self.median_vb.addItem(self.median_curve)

        # Keep viewboxes aligned
        main_vb = self.plot.getViewBox()
        main_vb.sigResized.connect(self.update_views)
        main_vb.sigRangeChanged.connect(lambda *_: self.update_views())
        QtCore.QTimer.singleShot(0, self.update_views)

        # Follow logic
        self.follow_enabled = True
        if hasattr(main_vb, "sigRangeChangedManually"):
            main_vb.sigRangeChangedManually.connect(self._on_user_changed_view)

        # Data
        self.x_ts, self.conc, self.median = [], [], []
        self.current_file = None
        self.reader = None
        self.base_dt = None

        self.paused = False
        self.autoscale_y = True

        last_folder = self.settings.value("watch_folder", "")
        if isinstance(last_folder, str) and last_folder and os.path.isdir(last_folder):
            self.cfg.watch_folder = last_folder
            self.status_label.setText(f"Using saved folder: {last_folder}")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(self.cfg.poll_ms)

    def _on_user_changed_view(self, *args, **kwargs):
        self.follow_enabled = False

    def update_views(self):
        vb = self.plot.getViewBox()
        self.median_vb.setGeometry(vb.sceneBoundingRect())
        self.median_vb.linkedViewChanged(vb, self.median_vb.XAxis)

    def on_pause_toggled(self, checked: bool):
        self.paused = checked
        self.pause_btn.setText("Resume" if checked else "Pause")

    def on_autoscale_toggled(self, checked: bool):
        self.autoscale_y = checked
        self.autoscale_btn.setText(f"Autoscale Y: {'ON' if checked else 'OFF'}")
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=checked)
        self.median_vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=checked)

    def select_folder(self):
        start_dir = self.cfg.watch_folder if self.cfg.watch_folder else os.path.expanduser("~")
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder containing instrument data", start_dir
        )
        if not folder:
            return
        self.cfg.watch_folder = folder
        self.settings.setValue("watch_folder", folder)
        self.reset_state()
        self.status_label.setText(f"Selected folder: {folder}")

    def reset_state(self):
        self.current_file = None
        self.reader = None
        self.base_dt = None
        self.time_axis.set_base_timestamp(None)
        self.x_ts.clear(); self.conc.clear(); self.median.clear()
        self.conc_curve.setData([], [])
        self.median_curve.setData([], [])
        self.follow_enabled = True
        self.update_views()

    def _ensure_file(self):
        if not self.cfg.watch_folder:
            self.status_label.setText("No folder selected.")
            return False

        if self.current_file and os.path.exists(self.current_file):
            return True

        path = find_latest_file(self.cfg)
        if not path:
            self.current_file = None
            self.reader = None
            self.status_label.setText(f"No matching files in: {self.cfg.watch_folder}")
            return False

        self.current_file = path
        self.reader = LiveTailReader(path, self.cfg)
        self.base_dt = None
        self.time_axis.set_base_timestamp(None)

        self.x_ts.clear(); self.conc.clear(); self.median.clear()
        self.conc_curve.setData([], [])
        self.median_curve.setData([], [])

        self.follow_enabled = True
        self.status_label.setText(f"Monitoring: {os.path.basename(path)}")
        return True

    def _current_x_window_width(self) -> float | None:
        vb = self.plot.getViewBox()
        (x0, x1), _ = vb.viewRange()
        width = float(x1 - x0)
        if width <= 0 or width > 1e12:
            return None
        return width

    def apply_follow_view(self, force: bool = False, keep_window: bool = False):
        """
        Follow mode behavior:
          - keep_window=False: show full run so far (x from first to latest)
          - keep_window=True: keep current x-window width and pan to latest point

        FIXES:
          - keep_window=True uses padding=0.0 to avoid creeping growth.
        """
        if not self.x_ts:
            return
        if not (self.follow_enabled or force):
            return

        latest = self.x_ts[-1]
        vb = self.plot.getViewBox()
        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)

        if keep_window:
            width = self._current_x_window_width()
            if width is None:
                x0 = self.x_ts[0]
                x1 = latest
            else:
                x1 = latest
                x0 = x1 - width
                if x0 < self.x_ts[0]:
                    x0 = self.x_ts[0]
                    x1 = x0 + width

            # IMPORTANT: no padding for fixed-window follow
            vb.setXRange(x0, x1, padding=0.0)

        else:
            x0 = self.x_ts[0]
            x1 = latest
            if x1 <= x0:
                x1 = x0 + 1.0
            vb.setXRange(x0, x1, padding=0.02)

    def follow_only(self):
        self.follow_enabled = True
        self.apply_follow_view(force=True, keep_window=True)
        self.update_views()

    def reset_and_follow(self):
        self.follow_enabled = True
        if self.autoscale_y:
            self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            self.median_vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.apply_follow_view(force=True, keep_window=False)
        self.update_views()

    def load_full_now(self):
        if not self._ensure_file():
            return

        x_ts, conc_vals, median_vals, info, base_dt = load_full_run(self.current_file, self.cfg)

        self.x_ts = x_ts
        self.conc = conc_vals
        self.median = median_vals
        self.base_dt = base_dt

        if self.base_dt is not None:
            self.time_axis.set_base_timestamp(self.base_dt.timestamp())

        self.conc_curve.setData(self.x_ts, self.conc)
        self.median_curve.setData(self.x_ts, self.median)

        if self.autoscale_y:
            self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            self.median_vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        self.apply_follow_view(force=True, keep_window=False)
        self.update_views()

        if self.conc:
            latest_abs = datetime.fromtimestamp(self.x_ts[-1]).strftime("%H:%M:%S")
            latest_elapsed = format_hms(self.x_ts[-1] - self.x_ts[0])
            self.status_label.setText(
                f"FULL RUN | {os.path.basename(self.current_file)} | "
                f"Latest: {latest_abs} ({latest_elapsed}) | "
                f"Conc: {self.conc[-1]:.4g} | Median: {self.median[-1]:.4g}"
            )

    def tick(self):
        if self.paused:
            return
        if not self._ensure_file():
            return

        if self.full_mode_chk.isChecked():
            if not self.conc:
                self.load_full_now()
            return

        new_rows = self.reader.read_new_rows()
        if not new_rows:
            # FIX: No new data -> do nothing. Prevents x-range drifting.
            return

        idx = self.reader.col_index
        date_i = idx.get(normalize_col_name(self.cfg.date_col))
        time_i = idx.get(normalize_col_name(self.cfg.start_time_col))
        conc_i = idx.get(normalize_col_name(self.cfg.concentration_col))
        med_i = idx.get(normalize_col_name(self.cfg.median_col))

        if time_i is None or conc_i is None:
            self.status_label.setText("Missing required columns (Start Time / Total Conc.).")
            return

        added = 0
        for parts in new_rows:
            if conc_i >= len(parts) or time_i >= len(parts):
                continue

            conc_val = parse_number_with_units(parts[conc_i])
            if conc_val is None:
                continue

            med_val = None
            if med_i is not None and med_i < len(parts):
                med_val = parse_number_with_units(parts[med_i])

            dt = None
            if date_i is not None and date_i < len(parts) and parts[date_i]:
                dt = parse_datetime(parts[date_i], parts[time_i], self.cfg)
            if dt is None:
                dt = datetime.now()

            if self.base_dt is None:
                self.base_dt = dt
                self.time_axis.set_base_timestamp(self.base_dt.timestamp())

            self.x_ts.append(dt.timestamp())
            self.conc.append(conc_val)
            self.median.append(float("nan") if med_val is None else med_val)
            added += 1

        if added:
            self.conc_curve.setData(self.x_ts, self.conc)
            self.median_curve.setData(self.x_ts, self.median)

            if self.autoscale_y:
                self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                self.median_vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

            if self.follow_enabled:
                self.apply_follow_view(keep_window=True)

            self.update_views()

            latest_abs = datetime.fromtimestamp(self.x_ts[-1]).strftime("%H:%M:%S")
            latest_elapsed = format_hms(self.x_ts[-1] - self.x_ts[0])
            self.status_label.setText(
                f"LIVE | {os.path.basename(self.current_file)} | points={len(self.conc)} | "
                f"Latest: {latest_abs} ({latest_elapsed}) | "
                f"Conc: {self.conc[-1]:.4g} | Median: {self.median[-1]:.4g}"
            )


def main():
    cfg = Config()
    app = QtWidgets.QApplication(sys.argv)
    win = LivePlotWindow(cfg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
