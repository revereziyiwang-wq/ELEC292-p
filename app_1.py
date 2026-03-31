"""
ELEC 292 – Activity Classifier Desktop App
GUI: Tkinter  |  Plot: Matplotlib embedded  |  Classifier: logistic regression (.pkl)

Tab 1 – Offline:  Load CSV → classify → export labels CSV + timeline plot
Tab 2 – Live:     Connect to Phyphox over WiFi → stream accelerometer → classify in real-time

Requirements:
    pip install pandas numpy scipy matplotlib scikit-learn requests
"""

import os
import pickle
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import deque

import numpy as np
import pandas as pd
import requests
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import resample


# ─────────────────────────────────────────────────────────────────────────────
#  Shared constants
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SEC  = 5          # seconds per classification window (must match training)
TARGET_HZ   = 100        # resample target for live mode
WIN_SAMPLES = WINDOW_SEC * TARGET_HZ   # 500 samples per window after resampling
POLL_SEC    = 0.25       # how often to poll Phyphox (seconds)
LABEL_MAP   = {1: "jumping", 0: "walking"}
COLORS      = {"walking": "#2563eb", "jumping": "#dc2626"}


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction  (33 features — MUST match training notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(window: pd.DataFrame) -> list:
    data = window[["x", "y", "z"]].values.astype(float)
    feats = []
    for axis in ("x", "y", "z"):
        s = window[axis].values.astype(float)
        feats += [
            np.mean(s),
            np.std(s),
            np.min(s),
            np.max(s),
            np.max(s) - np.min(s),
        ]

    mag = np.sqrt(window["x"] ** 2 + window["y"] ** 2 + window["z"] ** 2).values
    feats += [np.mean(mag), np.std(mag)]
    return feats   # 17 features total


# ─────────────────────────────────────────────────────────────────────────────
#  CSV reader — handles Phyphox column names
# ─────────────────────────────────────────────────────────────────────────────
_AXIS_ALIASES = {
    "x": ("x", "acc x", "accx", "ax", "acceleration x", "linear acceleration x (m/s^2)"),
    "y": ("y", "acc y", "accy", "ay", "acceleration y", "linear acceleration y (m/s^2)"),
    "z": ("z", "acc z", "accz", "az", "acceleration z", "linear acceleration z (m/s^2)"),
}

def read_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    col_map = {}
    for axis, aliases in _AXIS_ALIASES.items():
        for alias in aliases:
            if alias in raw.columns:
                col_map[axis] = alias
                break
    if len(col_map) < 3:
        numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        if len(numeric_cols) >= 3:
            col_map = {"x": numeric_cols[-3], "y": numeric_cols[-2], "z": numeric_cols[-1]}
        else:
            raise ValueError(f"Could not find x, y, z columns.\nCSV columns: {list(raw.columns)}")
    df = pd.DataFrame({
        "x": pd.to_numeric(raw[col_map["x"]], errors="coerce"),
        "y": pd.to_numeric(raw[col_map["y"]], errors="coerce"),
        "z": pd.to_numeric(raw[col_map["z"]], errors="coerce"),
    })
    return df.ffill().bfill()


# ─────────────────────────────────────────────────────────────────────────────
#  Phyphox JSON poller
#  GET http://<ip>/get?accX=full&accY=full&accZ=full&acc_time=full
#  Returns {"buffer": {"accX": {"buffer": [...]}, ...}}
# ─────────────────────────────────────────────────────────────────────────────
PHYPHOX_URL       = "http://{ip}/get?accX=full&accY=full&accZ=full&acc_time=full"
PHYPHOX_CLEAR_URL = "http://{ip}/control?cmd=clear"

def fetch_phyphox(ip: str, timeout: float = 2.0):
    """Returns {"t": [...], "x": [...], "y": [...], "z": [...]} or None on failure."""
    try:
        r = requests.get(PHYPHOX_URL.format(ip=ip), timeout=timeout)
        r.raise_for_status()
        j = r.json()["buffer"]
        return {
            "t": j["acc_time"]["buffer"],
            "x": j["accX"]["buffer"],
            "y": j["accY"]["buffer"],
            "z": j["accZ"]["buffer"],
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Main application
# ─────────────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.root.title("Activity Classifier — ELEC 292")
        self.root.resizable(True, True)

        # Shared model state
        self.model  = None
        self.scaler = None

        # Offline tab state
        self.csv_path      = None
        self.results_df    = None
        self.sampling_rate = tk.IntVar(value=100)

        # Live tab state
        self._live_running  = False
        self._live_thread   = None
        self._sample_buf    = deque()   # resampled samples waiting to be windowed
        self._last_seen_len = 0         # how many Phyphox samples already consumed
        self._live_labels   = []        # classification results so far
        self._live_lock     = threading.Lock()

        self._build_ui()

    # ── Build tabbed UI ───────────────────────
    def _build_ui(self):
        # Shared model bar at the very top
        top = tk.Frame(self.root, bg="#1e293b", pady=6, padx=10)
        top.pack(fill=tk.X)
        tk.Button(top, text="📂  Load Model (.pkl)", width=20,
                  command=self.load_model, bg="#334155", fg="white",
                  relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 8))
        self.lbl_model = tk.Label(top, text="No model loaded", fg="#f87171",
                                   bg="#1e293b", anchor="w")
        self.lbl_model.pack(side=tk.LEFT)

        # Notebook
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab_offline = tk.Frame(nb, bg="#fafafa")
        self.tab_live    = tk.Frame(nb, bg="#fafafa")
        nb.add(self.tab_offline, text="  📁  Offline (CSV)  ")
        nb.add(self.tab_live,    text="  📡  Live (Phyphox)  ")

        self._build_offline_tab()
        self._build_live_tab()

        # Global status bar
        self.status = tk.StringVar(value="Ready.")
        tk.Label(self.root, textvariable=self.status, anchor="w",
                 relief=tk.SUNKEN, padx=6, bg="#e2e8f0").pack(fill=tk.X, side=tk.BOTTOM)

    # ─────────────────────────────────────────
    #  TAB 1 – OFFLINE
    # ─────────────────────────────────────────
    def _build_offline_tab(self):
        bar = tk.Frame(self.tab_offline, bg="#f1f5f9", pady=8, padx=10)
        bar.pack(fill=tk.X)

        tk.Button(bar, text="📂  Load Input CSV", width=20,
                  command=self.load_csv).grid(row=0, column=0, padx=4, pady=3, sticky="w")
        self.lbl_csv = tk.Label(bar, text="No file selected", fg="#6b7280",
                                 bg="#f1f5f9", anchor="w")
        self.lbl_csv.grid(row=0, column=1, padx=4, sticky="w")

        tk.Label(bar, text="Sample rate (Hz):", bg="#f1f5f9").grid(row=0, column=2, padx=(30, 4))
        tk.Spinbox(bar, from_=10, to=1000, textvariable=self.sampling_rate,
                   width=6).grid(row=0, column=3, padx=4)

        tk.Button(bar, text="▶  Run Classification", bg="#2563eb", fg="white",
                  font=("Helvetica", 10, "bold"), width=22,
                  command=self.run_offline).grid(row=0, column=4, padx=8)

        self.btn_export = tk.Button(bar, text="💾  Export CSV", width=14,
                                     state=tk.DISABLED, command=self.export_csv)
        self.btn_export.grid(row=0, column=5, padx=4)

        self.fig_off, self.ax_off = plt.subplots(figsize=(9, 3.4))
        self.fig_off.patch.set_facecolor("#fafafa")
        self.canvas_off = FigureCanvasTkAgg(self.fig_off, master=self.tab_offline)
        self.canvas_off.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self.lbl_summary = tk.Label(self.tab_offline, text="", font=("Helvetica", 10),
                                     bg="#fafafa", pady=4)
        self.lbl_summary.pack()

        self._placeholder(self.ax_off, self.canvas_off,
                          "Load a CSV and click  ▶  Run Classification")

    # ─────────────────────────────────────────
    #  TAB 2 – LIVE
    # ─────────────────────────────────────────
    def _build_live_tab(self):
        cfg = tk.Frame(self.tab_live, bg="#f1f5f9", pady=10, padx=12)
        cfg.pack(fill=tk.X)

        tk.Label(cfg, text="Phyphox IP:Port", bg="#f1f5f9",
                 font=("Helvetica", 10)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.entry_ip = tk.Entry(cfg, width=20, font=("Courier", 11))
        self.entry_ip.insert(0, "192.168.x.x:8080")
        self.entry_ip.grid(row=0, column=1, padx=4)

        tk.Button(cfg, text="🔗  Test Connection",
                  command=self._test_connection).grid(row=0, column=2, padx=8)

        self.btn_live = tk.Button(cfg, text="▶  Start Live", bg="#16a34a", fg="white",
                                   font=("Helvetica", 10, "bold"), width=16,
                                   command=self._toggle_live)
        self.btn_live.grid(row=0, column=3, padx=4)

        tk.Button(cfg, text="🗑  Clear", width=8,
                  command=self._clear_live).grid(row=0, column=4, padx=4)

        # Big current-label display
        badge = tk.Frame(self.tab_live, bg="#fafafa")
        badge.pack(pady=(10, 0))
        tk.Label(badge, text="Current activity:", bg="#fafafa",
                 font=("Helvetica", 11)).pack()
        self.lbl_live_activity = tk.Label(badge, text="—",
                                           font=("Helvetica", 36, "bold"),
                                           fg="#6b7280", bg="#fafafa")
        self.lbl_live_activity.pack()
        self.lbl_live_count = tk.Label(badge, text="0 windows classified",
                                        bg="#fafafa", fg="#6b7280")
        self.lbl_live_count.pack()

        # Live timeline
        self.fig_live, self.ax_live = plt.subplots(figsize=(9, 2.4))
        self.fig_live.patch.set_facecolor("#fafafa")
        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=self.tab_live)
        self.canvas_live.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self._placeholder(self.ax_live, self.canvas_live,
                          "Connect to Phyphox and press  ▶  Start Live")

    # ─────────────────────────────────────────
    #  Shared helpers
    # ─────────────────────────────────────────
    def _placeholder(self, ax, canvas, msg):
        ax.clear()
        ax.set_facecolor("#fafafa")
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                fontsize=11, color="#9ca3af", transform=ax.transAxes)
        ax.set_axis_off()
        canvas.draw()

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Select trained model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                self.model  = obj.get("model")
                self.scaler = obj.get("scaler")
            else:
                self.model  = obj
                self.scaler = None
            name = os.path.basename(path)
            self.lbl_model.config(text=f"✓  {name}", fg="#86efac")
            self.status.set(f"Model loaded: {name}")
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load model:\n{e}")

    def _classify_window(self, window_df: pd.DataFrame) -> str:
        X = np.array([extract_features(window_df)])
        if self.scaler is not None:
            X = self.scaler.transform(X)
        pred = self.model.predict(X)[0]
        return LABEL_MAP.get(int(pred), str(pred))

    def _draw_timeline(self, ax, canvas, labels: list):
        ax.clear()
        ax.set_facecolor("#f8fafc")
        for i, label in enumerate(labels):
            ax.barh(0, WINDOW_SEC, left=i * WINDOW_SEC, height=0.55,
                    color=COLORS[label], edgecolor="white", linewidth=0.8)
        handles = [Patch(facecolor=COLORS["walking"], label="Walking"),
                   Patch(facecolor=COLORS["jumping"], label="Jumping")]
        ax.legend(handles=handles, loc="upper right", framealpha=0.9)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_title("Activity Timeline", fontsize=12, pad=8)
        ax.set_yticks([])
        ax.set_xlim(0, max(len(labels) * WINDOW_SEC, WINDOW_SEC))
        ax.set_ylim(-0.5, 0.5)
        ax.set_axis_on()
        canvas.figure.tight_layout(pad=1.5)
        canvas.draw()

    # ─────────────────────────────────────────
    #  TAB 1 – OFFLINE logic
    # ─────────────────────────────────────────
    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Select input CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        self.csv_path = path
        self.lbl_csv.config(text=f"✓  {os.path.basename(path)}", fg="#15803d")
        self.status.set(f"CSV loaded: {os.path.basename(path)}")

    def run_offline(self):
        if self.model is None:
            messagebox.showwarning("No model", "Load a model first.")
            return
        if self.csv_path is None:
            messagebox.showwarning("No file", "Load a CSV file first.")
            return
        try:
            self.status.set("Reading CSV…"); self.root.update()
            df = read_csv(self.csv_path)
            sr       = self.sampling_rate.get()
            win_samp = sr * WINDOW_SEC
            n        = len(df)
            if n < win_samp:
                messagebox.showerror("Too short",
                    f"Need at least {win_samp} samples for one window at {sr} Hz. Got {n}.")
                return
            self.status.set("Extracting features & classifying…"); self.root.update()
            labels = []
            for start in range(0, n - win_samp + 1, win_samp):
                labels.append(self._classify_window(df.iloc[start:start + win_samp]))

            self.results_df = pd.DataFrame({
                "window_index": range(len(labels)),
                "start_time_s": [i * WINDOW_SEC for i in range(len(labels))],
                "end_time_s":   [(i + 1) * WINDOW_SEC for i in range(len(labels))],
                "label":        labels,
            })
            self._draw_timeline(self.ax_off, self.canvas_off, labels)
            self.btn_export.config(state=tk.NORMAL)
            n_w = labels.count("jumping")
            n_j = labels.count("walking")
            self.lbl_summary.config(
                text=f"{len(labels)} windows  |  "
                     f"Walking: {n_w} ({n_w/len(labels)*100:.0f}%)  "
                     f"Jumping: {n_j} ({n_j/len(labels)*100:.0f}%)  |  "
                     f"Total: {len(labels)*WINDOW_SEC}s")
            self.status.set(f"Done — {len(labels)} windows classified.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_csv(self):
        if self.results_df is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save output labels", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")], initialfile="output_labels.csv")
        if path:
            self.results_df.to_csv(path, index=False)
            self.status.set(f"Saved → {os.path.basename(path)}")
            messagebox.showinfo("Saved", f"Output written to:\n{path}")

    # ─────────────────────────────────────────
    #  TAB 2 – LIVE logic
    # ─────────────────────────────────────────
    def _test_connection(self):
        ip = self.entry_ip.get().strip()
        result = fetch_phyphox(ip)
        if result is None:
            messagebox.showerror("Connection failed",
                "Could not reach Phyphox.\n\n"
                "• Phone and laptop must be on the same WiFi network\n"
                "• In Phyphox: open 'Acceleration (without g)' → ⋮ → Remote Access → Start\n"
                "• Enter the IP address shown on screen (e.g. 192.168.1.42:8080)")
        else:
            messagebox.showinfo("Connected ✓",
                f"Phyphox is reachable.\nBuffer currently has {len(result['x'])} samples.")

    def _toggle_live(self):
        if not self._live_running:
            self._start_live()
        else:
            self._stop_live()

    def _start_live(self):
        if self.model is None:
            messagebox.showwarning("No model", "Load a model before starting live mode.")
            return
        ip = self.entry_ip.get().strip()
        # Clear Phyphox buffer on phone so we start fresh
        try:
            requests.get(PHYPHOX_CLEAR_URL.format(ip=ip), timeout=2)
        except Exception:
            pass

        with self._live_lock:
            self._sample_buf.clear()
            self._last_seen_len = 0
            self._live_labels   = []

        self._live_running = True
        self.btn_live.config(text="⏹  Stop Live", bg="#dc2626")
        self.status.set("Live mode running…")

        self._live_thread = threading.Thread(
            target=self._poll_loop, args=(ip,), daemon=True)
        self._live_thread.start()
        self._live_ui_loop()

    def _stop_live(self):
        self._live_running = False
        self.btn_live.config(text="▶  Start Live", bg="#16a34a")
        self.status.set("Live mode stopped.")

    def _poll_loop(self, ip: str):
        """
        Background thread — runs independently of Tkinter.

        Phyphox keeps a cumulative buffer of ALL samples since the experiment
        started. On each poll we fetch the full buffer, look at only the NEW
        samples (beyond _last_seen_len), resample them to TARGET_HZ, and push
        them into _sample_buf. Whenever the buffer holds a full window's worth
        of samples (WIN_SAMPLES), we pop them out and classify.
        """
        while self._live_running:
            data = fetch_phyphox(ip)
            if data is not None:
                total_now = len(data["x"])
                if total_now > self._last_seen_len:
                    new_start = self._last_seen_len
                    new_x = data["x"][new_start:]
                    new_y = data["y"][new_start:]
                    new_z = data["z"][new_start:]
                    new_t = data["t"][new_start:]

                    # Resample new chunk to TARGET_HZ using timestamps
                    native_n = len(new_x)
                    if len(new_t) > 1:
                        duration = new_t[-1] - new_t[0]
                    else:
                        duration = native_n / TARGET_HZ
                    target_n = max(1, int(round(duration * TARGET_HZ)))

                    rx = resample(new_x, target_n)
                    ry = resample(new_y, target_n)
                    rz = resample(new_z, target_n)

                    with self._live_lock:
                        for sx, sy, sz in zip(rx, ry, rz):
                            self._sample_buf.append((sx, sy, sz))
                        self._last_seen_len = total_now

                    # Drain complete windows from the buffer
                    while True:
                        with self._live_lock:
                            if len(self._sample_buf) < WIN_SAMPLES:
                                break
                            window_samples = [self._sample_buf.popleft()
                                              for _ in range(WIN_SAMPLES)]
                        window_df = pd.DataFrame(window_samples, columns=["x", "y", "z"])
                        label = self._classify_window(window_df)
                        with self._live_lock:
                            self._live_labels.append(label)

            time.sleep(POLL_SEC)

    def _live_ui_loop(self):
        """Runs on the Tkinter main thread every 500 ms to refresh live UI."""
        if not self._live_running:
            return
        with self._live_lock:
            labels = list(self._live_labels)
        if labels:
            current = labels[-1]
            self.lbl_live_activity.config(text=current.upper(), fg=COLORS[current])
            self.lbl_live_count.config(
                text=f"{len(labels)} window{'s' if len(labels) != 1 else ''} classified  "
                     f"({len(labels) * WINDOW_SEC}s total)")
            self._draw_timeline(self.ax_live, self.canvas_live, labels)
        self.root.after(500, self._live_ui_loop)

    def _clear_live(self):
        self._stop_live()
        with self._live_lock:
            self._sample_buf.clear()
            self._last_seen_len = 0
            self._live_labels   = []
        self.lbl_live_activity.config(text="—", fg="#6b7280")
        self.lbl_live_count.config(text="0 windows classified")
        self._placeholder(self.ax_live, self.canvas_live,
                          "Connect to Phyphox and press  ▶  Start Live")
        self.status.set("Cleared.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("980x560")
    App(root)
    root.mainloop()
