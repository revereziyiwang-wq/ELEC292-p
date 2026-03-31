"""
ELEC 292 – Activity Classifier Desktop App
GUI: Tkinter  |  Plot: Matplotlib embedded  |  Classifier: logistic regression (.pkl)

Usage:
    python app.py

Requirements:
    pip install pandas numpy scipy matplotlib scikit-learn
"""

import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from scipy import stats


# ─────────────────────────────────────────────
#  Feature extraction
#  Extracts 33 features per 5-second window:
#  10 per axis (x, y, z) + 3 magnitude features
#  *** Must match the feature extraction used during training ***
# ─────────────────────────────────────────────
def extract_features(window: pd.DataFrame) -> list:
    feats = []
    for axis in ("x", "y", "z"):
        s = window[axis].values.astype(float)
        feats += [
            np.mean(s),                     # 1. Mean
            np.std(s),                      # 2. Standard deviation
            np.var(s),                      # 3. Variance
            np.min(s),                      # 4. Minimum
            np.max(s),                      # 5. Maximum
            np.max(s) - np.min(s),          # 6. Range
            np.median(s),                   # 7. Median
            stats.skew(s),                  # 8. Skewness
            stats.kurtosis(s),              # 9. Kurtosis
            np.sqrt(np.mean(s ** 2)),       # 10. Root mean square
        ]
    # Resultant magnitude features (cross-axis)
    mag = np.sqrt(window["x"] ** 2 + window["y"] ** 2 + window["z"] ** 2).values
    feats += [
        np.mean(mag),                       # 11. Magnitude mean
        np.std(mag),                        # 12. Magnitude std
        np.max(mag) - np.min(mag),          # 13. Magnitude range
    ]
    return feats  # 33 features total


# ─────────────────────────────────────────────
#  CSV reader — tolerant of Phyphox column names
# ─────────────────────────────────────────────
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

    # Positional fallback if aliases don't match
    if len(col_map) < 3:
        numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        if len(numeric_cols) >= 3:
            # Assume last 3 numeric columns are x, y, z (skip time if present)
            col_map = {"x": numeric_cols[-3], "y": numeric_cols[-2], "z": numeric_cols[-1]}
        else:
            raise ValueError(
                "Could not find x, y, z columns.\n"
                f"CSV columns: {list(raw.columns)}"
            )

    df = pd.DataFrame({
        "x": pd.to_numeric(raw[col_map["x"]], errors="coerce"),
        "y": pd.to_numeric(raw[col_map["y"]], errors="coerce"),
        "z": pd.to_numeric(raw[col_map["z"]], errors="coerce"),
    })
    df = df.ffill().bfill()  # fill missing values
    return df


# ─────────────────────────────────────────────
#  Main application
# ─────────────────────────────────────────────
WINDOW_SEC = 5   # seconds per window (must match training)
LABEL_MAP  = {1: "walking", 0: "jumping"}   # adjust if teammates flip the encoding
COLORS     = {"walking": "#2563eb", "jumping": "#dc2626"}


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Activity Classifier — ELEC 292")
        self.root.resizable(True, True)

        self.model        = None
        self.scaler       = None
        self.csv_path     = None
        self.results_df   = None
        self.sampling_rate = tk.IntVar(value=100)

        self._build_ui()

    # ── UI construction ──────────────────────
    def _build_ui(self):
        # ── Top control bar ──
        bar = tk.Frame(self.root, bg="#f5f5f5", pady=8, padx=10)
        bar.pack(fill=tk.X)

        # Row 0 – model
        tk.Button(bar, text="📂  Load Model (.pkl)", width=20,
                  command=self.load_model).grid(row=0, column=0, padx=4, pady=3, sticky="w")
        self.lbl_model = tk.Label(bar, text="No model loaded", fg="#b91c1c", bg="#f5f5f5", anchor="w")
        self.lbl_model.grid(row=0, column=1, padx=4, sticky="w")

        # Sampling rate entry
        tk.Label(bar, text="Sample rate (Hz):", bg="#f5f5f5").grid(row=0, column=2, padx=(30, 4))
        tk.Spinbox(bar, from_=10, to=1000, textvariable=self.sampling_rate,
                   width=6).grid(row=0, column=3, padx=4)

        # Row 1 – CSV
        tk.Button(bar, text="📂  Load Input CSV", width=20,
                  command=self.load_csv).grid(row=1, column=0, padx=4, pady=3, sticky="w")
        self.lbl_csv = tk.Label(bar, text="No file selected", fg="#6b7280", bg="#f5f5f5", anchor="w")
        self.lbl_csv.grid(row=1, column=1, padx=4, sticky="w")

        # Run + Export buttons
        tk.Button(bar, text="▶  Run Classification", bg="#2563eb", fg="white",
                  font=("Helvetica", 11, "bold"), width=22,
                  command=self.run).grid(row=1, column=2, columnspan=2, padx=4)

        self.btn_export = tk.Button(bar, text="💾  Export CSV", width=16,
                                    state=tk.DISABLED, command=self.export_csv)
        self.btn_export.grid(row=0, column=4, padx=(20, 4))

        # ── Matplotlib canvas ──
        self.fig, self.ax = plt.subplots(figsize=(9, 3.6))
        self.fig.patch.set_facecolor("#fafafa")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 0))

        # ── Result summary label ──
        self.lbl_summary = tk.Label(self.root, text="", font=("Helvetica", 10), pady=4)
        self.lbl_summary.pack()

        # ── Status bar ──
        self.status = tk.StringVar(value="Ready — load a model and CSV to begin.")
        tk.Label(self.root, textvariable=self.status, anchor="w",
                 relief=tk.SUNKEN, padx=6, bg="#e5e7eb").pack(fill=tk.X, side=tk.BOTTOM)

        self._placeholder()

    # ── Placeholder plot ─────────────────────
    def _placeholder(self):
        self.ax.clear()
        self.ax.set_facecolor("#fafafa")
        self.ax.text(0.5, 0.5,
                     "Load a model and CSV file, then click  ▶  Run Classification",
                     ha="center", va="center", fontsize=12, color="#9ca3af",
                     transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.canvas.draw()

    # ── Load model ───────────────────────────
    def load_model(self):
        path = filedialog.askopenfilename(
            title="Select trained model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                self.model  = obj.get("model")
                self.scaler = obj.get("scaler")   # optional
            else:
                self.model  = obj
                self.scaler = None
            name = os.path.basename(path)
            self.lbl_model.config(text=f"✓  {name}", fg="#15803d")
            self.status.set(f"Model loaded: {name}")
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load model:\n{e}")

    # ── Load CSV ─────────────────────────────
    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Select input CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        self.csv_path = path
        self.lbl_csv.config(text=f"✓  {os.path.basename(path)}", fg="#15803d")
        self.status.set(f"CSV loaded: {os.path.basename(path)}")

    # ── Run classification ────────────────────
    def run(self):
        if self.model is None:
            messagebox.showwarning("No model", "Please load a trained model (.pkl) first.")
            return
        if self.csv_path is None:
            messagebox.showwarning("No file", "Please load an input CSV file first.")
            return

        try:
            self.status.set("Reading CSV…"); self.root.update()
            df = read_csv(self.csv_path)

            self.status.set("Windowing & extracting features…"); self.root.update()
            sr       = self.sampling_rate.get()
            win_samp = sr * WINDOW_SEC
            n        = len(df)

            if n < win_samp:
                messagebox.showerror(
                    "Too short",
                    f"Signal has only {n} samples but needs at least {win_samp} "
                    f"for one {WINDOW_SEC}-second window at {sr} Hz."
                )
                return

            feature_matrix = []
            for start in range(0, n - win_samp + 1, win_samp):
                feature_matrix.append(extract_features(df.iloc[start:start + win_samp]))

            X = np.array(feature_matrix)
            if self.scaler is not None:
                X = self.scaler.transform(X)

            self.status.set("Classifying…"); self.root.update()
            preds  = self.model.predict(X)
            labels = [LABEL_MAP.get(int(p), str(p)) for p in preds]

            self.results_df = pd.DataFrame({
                "window_index": range(len(labels)),
                "start_time_s": [i * WINDOW_SEC for i in range(len(labels))],
                "end_time_s":   [(i + 1) * WINDOW_SEC for i in range(len(labels))],
                "label":        labels,
            })

            self._plot_timeline(labels)
            self.btn_export.config(state=tk.NORMAL)

            n_walk = labels.count("walking")
            n_jump = labels.count("jumping")
            total_s = len(labels) * WINDOW_SEC
            self.lbl_summary.config(
                text=f"{len(labels)} windows  |  "
                     f"Walking: {n_walk} ({n_walk/len(labels)*100:.0f}%)  "
                     f"Jumping: {n_jump} ({n_jump/len(labels)*100:.0f}%)  |  "
                     f"Total duration: {total_s}s"
            )
            self.status.set(f"Done — {len(labels)} windows classified.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.set("Classification failed.")

    # ── Timeline plot ─────────────────────────
    def _plot_timeline(self, labels: list):
        self.ax.clear()
        self.ax.set_facecolor("#f8fafc")

        for i, label in enumerate(labels):
            self.ax.barh(
                0, WINDOW_SEC, left=i * WINDOW_SEC,
                height=0.55, color=COLORS[label],
                edgecolor="white", linewidth=0.8
            )

        legend_handles = [
            Patch(facecolor=COLORS["walking"], label="Walking"),
            Patch(facecolor=COLORS["jumping"], label="Jumping"),
        ]
        self.ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
        self.ax.set_xlabel("Time (s)", fontsize=11)
        self.ax.set_title("Activity Classification Timeline", fontsize=13, pad=10)
        self.ax.set_yticks([])
        self.ax.set_xlim(0, len(labels) * WINDOW_SEC)
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_axis_on()
        self.fig.tight_layout(pad=2)
        self.canvas.draw()

    # ── Export output CSV ─────────────────────
    def export_csv(self):
        if self.results_df is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save output labels",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="output_labels.csv",
        )
        if path:
            self.results_df.to_csv(path, index=False)
            self.status.set(f"Saved → {os.path.basename(path)}")
            messagebox.showinfo("Saved", f"Output written to:\n{path}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("960x520")
    App(root)
    root.mainloop()
