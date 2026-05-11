"""
ICG Black-Spot Player  --  standalone
======================================
Opens a pre-computed session file produced by icg_inflow_analysis.py and
launches the interactive player directly, without re-running any analysis.

Usage
-----
    python icg_player.py                          # uses SESSION_PATH below
    python icg_player.py path/to/session.npz      # or pass as argument

The session file is saved automatically at the end of every pipeline run:
    <output_dir>/FIM{vid_nr}_analysis/FIM{vid_nr}_session.npz

The player reads video frames directly from the original .mp4 file on
demand -- the session file contains only the analysis results (arrival map,
curves, ROI masks, fit parameters), not the video itself.

Player controls
---------------
    SPACE         play / pause
    LEFT / RIGHT  step -1 / +1 frame   (a / d also work)
    UP   / DOWN   step +10 / -10 frames
    + / -         playback speed up / slow down
    h             toggle arrival-time heat-map overlay
    b             toggle black-spot contour overlay
    p             toggle curve display: fraction (%) / absolute pixels
    1 / 2         cycle which method's contours/curves are shown
                  (both -> Method A only -> Method B only -> both)
    r             toggle ROI boundaries
    w             toggle wound boundary
    s             save current composite frame as PNG (to session output_dir)
    0             jump to frame 0
    q / ESC       quit
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Disable tqdm's background monitor thread at import time to prevent
# GIL conflicts with OpenCV/matplotlib event loops on Windows.
try:
    from tqdm import TMonitor
    TMonitor.monitor_interval = 0
except (ImportError, AttributeError):
    pass
try:
    import tqdm as _tqdm_mod
    _tqdm_mod.tqdm.monitor_interval = 0
except (ImportError, AttributeError):
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# =====================================================
# CONFIG -- set the path to your session file here
# =====================================================
vidnr = "013"
SESSION_PATH: Path = Path(f"C:/Users/verhu/M2-2 stage/Analysed_videos/FIM{vidnr}_analysis/FIM{vidnr}_session.npz")


# =====================================================
# SESSION LOADER
# =====================================================
def load_session(path: Path) -> Dict:
    """
    Load and un-flatten a session .npz file written by save_session()
    in icg_inflow_analysis.py.

    Returns a dict with keys:
        vid_nr, fps, method_labels, method_label_A, method_label_B,
        contour_smooth_radius, contour_approx_eps,
        player_width, plot_panel_height, output_dir,
        video        : uint8 (T, H, W)
        arrival      : float32 (H, W)
        time_axis    : float32 (T,)
        wound_mask   : bool (H, W) or None
        roi_sets     : {method_label: {roi_label: bool (H, W)}}
        results      : {method_label: {blackspots_raw, blackspots_smooth, fits}}
    """
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")

    raw = np.load(str(path), allow_pickle=False)

    # Un-flatten: replace __SLASH__ back to /
    flat: Dict[str, np.ndarray] = {
        k.replace("__SLASH__", "/"): raw[k] for k in raw.files
    }

    def _int(key: str) -> int:
        return int(flat[key])

    def _float(key: str) -> float:
        return float(flat[key])

    session: Dict = {
        "vid_nr":                str(flat["_meta/vid_nr"]),
        "fps":                   _float("_meta/fps"),
        "method_labels":         list(flat["_meta/method_labels"].astype(str)),
        "method_label_A":        str(flat["_meta/method_label_A"]),
        "method_label_B":        str(flat["_meta/method_label_B"]),
        "contour_smooth_radius":      _int("_meta/contour_smooth_radius"),
        "contour_approx_eps":         _float("_meta/contour_approx_eps"),
        "wound_contour_sigma":      _float("_meta/wound_contour_sigma")
                                    if "_meta/wound_contour_sigma" in flat
                                    else 2.0,
        "wound_contour_approx_eps": _float("_meta/wound_contour_approx_eps")
                                    if "_meta/wound_contour_approx_eps" in flat
                                    else 1.0,
        "player_width":          _int("_meta/player_width"),
        "plot_panel_height":     _int("_meta/plot_panel_height"),
        "output_dir":            Path(str(flat["_meta/output_dir"])),
        "video_path":            Path(str(flat["_meta/video_path"])),
        "player_curve_mode":     str(flat["_meta/player_curve_mode"])
                                 if "_meta/player_curve_mode" in flat
                                 else "fraction",
        "arrival":               flat["arrival"],       # float32
        "time_axis":             flat["time_axis"],     # float32
    }

    session["wound_mask"] = (flat["wound_mask"].astype(bool)
                             if "wound_mask" in flat else None)

    # ROI sets
    roi_sets: Dict[str, Dict[str, np.ndarray]] = {}
    roi_prefix = "roi_sets/"
    for k, v in flat.items():
        if k.startswith(roi_prefix):
            parts = k[len(roi_prefix):].split("/", 1)   # method / roi_label
            ml, rl = parts[0], parts[1]
            roi_sets.setdefault(ml, {})[rl] = v.astype(bool)
    session["roi_sets"] = roi_sets

    # Results (blackspots_raw, blackspots_smooth, fits)
    results: Dict[str, Dict] = {}
    method_labels = session["method_labels"]

    for ml in method_labels:
        bsr: Dict[str, np.ndarray] = {}
        bss: Dict[str, np.ndarray] = {}
        fits: Dict[str, Tuple] = {}

        raw_pfx  = f"results/{ml}/blackspots_raw/"
        smth_pfx = f"results/{ml}/blackspots_smooth/"
        fit_pfx  = f"results/{ml}/fits/"

        for k, v in flat.items():
            if k.startswith(raw_pfx):
                bsr[k[len(raw_pfx):]] = v
            elif k.startswith(smth_pfx):
                bss[k[len(smth_pfx):]] = v

        # Collect fit scalars and arrays per ROI
        fit_data: Dict[str, Dict] = {}
        for k, v in flat.items():
            if k.startswith(fit_pfx):
                rest = k[len(fit_pfx):]          # roi_label/fi_A  or  roi_label/t_dec
                roi_label, field = rest.rsplit("/", 1)
                fit_data.setdefault(roi_label, {})[field] = v

        for roi_label, fdict in fit_data.items():
            fi = {
                "A":                   float(fdict["fi_A"]),
                "k_hill":              float(fdict["fi_k_hill"]),
                "n_hill":              float(fdict["fi_n_hill"]),
                "B":                   float(fdict["fi_B"]),
                "B_ci95":              float(fdict.get("fi_B_ci95",
                                               np.array(np.nan))),
                "t_half_fit_s":        float(fdict["fi_t_half_fit_s"]),
                "idx_onset":           int(fdict["fi_idx_onset"]),
                "t_onset":             float(fdict["fi_t_onset"]),
                "plateau":    float(fdict.get("fi_plateau",
                                               np.array(np.nan))),
                "plateau_lo": float(fdict.get("fi_plateau_lo",
                                               np.array(np.nan))),
                "plateau_hi": float(fdict.get("fi_plateau_hi",
                                               np.array(np.nan))),
            }
            t_dec     = fdict["t_dec"]
            fit_curve = fdict["fit_curve"]
            fits[roi_label] = (fi, t_dec, fit_curve)

        # Pixel-count smooth curves
        bsp: Dict[str, np.ndarray] = {}
        px_pfx = f"results/{ml}/blackspots_pixels_smooth/"
        for k, v in flat.items():
            if k.startswith(px_pfx):
                bsp[k[len(px_pfx):]] = v

        # n_pixels per ROI
        n_px_pfx = f"results/{ml}/n_pixels/"
        n_pixels_dict: Dict[str, int] = {}
        for k, v in flat.items():
            if k.startswith(n_px_pfx):
                n_pixels_dict[k[len(n_px_pfx):]] = int(v)
        met = {lbl: {"n_pixels": n_pixels_dict.get(lbl, 0)}
               for lbl in bss}

        results[ml] = {
            "blackspots_raw":           bsr,
            "blackspots_smooth":        bss,
            "blackspots_pixels_smooth": bsp,
            "fits":                     fits,
            "metrics":                  met,
        }

    session["results"] = results
    return session



def extract_wound_contours(binary_mask: np.ndarray,
                            sigma: float,
                            approx_eps: float) -> List[np.ndarray]:
    """Gaussian-blurred wound contours. See icg_inflow_analysis.py."""
    from scipy import ndimage as _ndi
    blurred = _ndi.gaussian_filter(
        np.array(binary_mask, dtype=np.float32), sigma=sigma)
    m = (blurred > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(
        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if approx_eps > 0:
        return [cv2.approxPolyDP(c, approx_eps, closed=True)
                for c in contours if len(c) >= 5]
    return [c for c in contours if len(c) >= 5]


# =====================================================
# VIDEO FRAME READER  (loads frames on demand from mp4)
# =====================================================
class VideoFrameReader:
    """
    Wraps a cv2.VideoCapture and provides random-access reads by frame index.

    Frames at the target_fps are decoded lazily and cached in a small LRU
    cache so sequential playback never re-decodes the same frame twice,
    while memory use stays bounded regardless of video length.

    The cache holds `cache_size` frames (default 128). At 1080p uint8
    grayscale that is ~256 MB, which is comfortable for most machines.
    To reduce memory use, lower cache_size. To eliminate seeking overhead
    during fast-forward, increase it.
    """
    def __init__(self, video_path: Path, target_fps: float,
                 cache_size: int = 128):
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self._path = video_path
        self._cap  = cv2.VideoCapture(str(video_path))
        orig_fps   = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._step = max(int(round(orig_fps / target_fps)), 1)
        self._eff_fps = orig_fps / self._step

        # Count total player frames
        total_orig = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.T = max(total_orig // self._step, 1)

        # Read first frame to get H, W
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"Could not read first frame of {video_path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.H, self.W = gray.shape

        self._cache: Dict[int, np.ndarray] = {}   # player_idx -> uint8 (H,W)
        self._cache_order: List[int] = []
        self._cache_size = cache_size
        self._last_orig_pos = -1

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.T, self.H, self.W

    def __len__(self) -> int:
        return self.T

    def __getitem__(self, player_idx: int) -> np.ndarray:
        if player_idx in self._cache:
            return self._cache[player_idx]
        orig_idx = player_idx * self._step
        # Seek only if not sequential (avoids expensive backward seeks)
        if orig_idx != self._last_orig_pos + self._step:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
        ok, frame = self._cap.read()
        if not ok:
            # Return last cached frame or zeros on read failure
            if self._cache_order:
                return self._cache[self._cache_order[-1]]
            return np.zeros((self.H, self.W), dtype=np.uint8)
        self._last_orig_pos = orig_idx
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._store(player_idx, gray)
        return gray

    def _store(self, idx: int, frame: np.ndarray):
        if idx not in self._cache:
            if len(self._cache_order) >= self._cache_size:
                evict = self._cache_order.pop(0)
                del self._cache[evict]
            self._cache_order.append(idx)
        self._cache[idx] = frame

    def release(self):
        self._cap.release()


# =====================================================
# COLOUR HELPERS  (duplicated from icg_inflow_analysis.py so this
#                  script is fully self-contained)
# =====================================================
def make_palette(n: int) -> List[Tuple[int, int, int]]:
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [(int(b * 255), int(g * 255), int(r * 255))
            for r, g, b, _ in (cmap(i % cmap.N) for i in range(n))]


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)).astype(bool)


def arrival_heatmap(arrival: np.ndarray) -> np.ndarray:
    h, w = arrival.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    finite = np.isfinite(arrival)
    if not finite.any():
        return out
    a_min = float(arrival[finite].min())
    span = float(arrival[finite].max()) - a_min
    if span < 1e-6:
        out[finite] = (0, 255, 255)
        return out
    norm = np.clip((arrival[finite] - a_min) / span, 0, 1)
    out[finite] = cv2.applyColorMap(
        (norm * 255).astype(np.uint8), cv2.COLORMAP_JET).reshape(-1, 3)
    return out


def extract_smooth_contours(binary_mask: np.ndarray,
                             smooth_radius: int,
                             approx_eps: float) -> List[np.ndarray]:
    m = binary_mask.astype(np.uint8)
    if smooth_radius > 0:
        m = cv2.medianBlur(m, 2 * smooth_radius + 1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if approx_eps > 0:
        return [cv2.approxPolyDP(c, approx_eps, closed=True)
                for c in contours if len(c) >= 5]
    return [c for c in contours if len(c) >= 5]


# =====================================================
# CURVES PANEL
# =====================================================
def render_curves_panel(time_axis: np.ndarray,
                        results: Dict[str, Dict],
                        palettes: Dict[str, List],
                        width_px: int,
                        height_px: int,
                        active_methods: Optional[List[str]] = None,
                        mode: str = "fraction",
                        ) -> Tuple[np.ndarray, Tuple[int, int]]:
    if active_methods is None:
        active_methods = list(results.keys())
    use_pixels = (mode == "pixels")
    smo_key = "blackspots_pixels_smooth" if use_pixels \
              else "blackspots_smooth"

    dpi = 100
    fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#101010")
    fig.patch.set_facecolor("#101010")

    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    for mi, ml in enumerate(active_methods):
        if ml not in results:
            continue
        res = results[ml]
        pal = palettes[ml]
        ls = linestyles[mi % len(linestyles)]
        curves = res.get(smo_key, res["blackspots_smooth"])
        for i, (label, bs_s) in enumerate(curves.items()):
            b, g, r = pal[i % len(pal)]
            col = (r / 255, g / 255, b / 255)
            roi_num = label.split("ROI_")[-1]
            n_px = res.get("metrics", {}).get(label, {}).get("n_pixels", 1)
            scale = 1 if use_pixels else 100
            ax.plot(time_axis, bs_s * scale, color=col, lw=1.6, ls=ls,
                    label=f"{ml} ROI {roi_num}")
            fi, t_dec, fit_curve = res["fits"].get(label, ({}, np.array([]),
                                                             np.array([])))
            if t_dec.size and not np.isnan(fi.get("k_hill", np.nan)):
                ax.plot(t_dec, fit_curve * (n_px if use_pixels else 100),
                        color=col, lw=0.9, ls="--", alpha=0.75)
            # Plateau: horizontal dashed line + shaded CI band
            pl    = fi.get("plateau", np.nan)
            pl_lo = fi.get("plateau_lo", np.nan)
            pl_hi = fi.get("plateau_hi", np.nan)
            if not np.isnan(pl):
                pl_sc = pl * (n_px if use_pixels else 100)
                ax.axhline(pl_sc, color=col, lw=0.9,
                           ls=(0, (4, 2)), alpha=0.6)
                if not np.isnan(pl_lo):
                    ax.axhspan(pl_lo * (n_px if use_pixels else 100),
                               pl_hi * (n_px if use_pixels else 100),
                               color=col, alpha=0.07)

    ax.set_xlabel("Time (s)", color="white", fontsize=9)
    ylabel = "Black-spot  S(t)  [px]" if use_pixels \
             else "Black-spot  S(t)  [%]"
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    if not use_pixels:
        ax.set_ylim(-2, 105)
    ax.set_xlim(float(time_axis[0]), float(time_axis[-1]))
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#666")
    ax.grid(True, color="#333", lw=0.5)
    n_curves = sum(len(results[m].get(smo_key, results[m]["blackspots_smooth"]))
                   for m in active_methods if m in results)
    ax.legend(fontsize=7, ncol=min(4, n_curves), loc="upper right",
              facecolor="#202020", labelcolor="white", framealpha=0.7)
    fig.tight_layout(pad=0.6)
    canvas.draw()
    bbox = ax.get_window_extent(canvas.get_renderer())
    bgr = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    return bgr, (int(bbox.x0), int(bbox.x1))


# =====================================================
# INTERACTIVE PLAYER
# =====================================================
class Player:
    WINDOW = "ICG Black-Spot Player"

    def __init__(self, session: Dict):
        s = session
        self.reader     = VideoFrameReader(s["video_path"], s["fps"])
        self.fps        = s["fps"]
        self.arrival    = s["arrival"]
        self.time_axis  = s["time_axis"]
        self.roi_sets   = s["roi_sets"]
        self.wound_mask = s["wound_mask"]
        self.results    = s["results"]
        self.output_dir = s["output_dir"]
        self.vid_nr     = s["vid_nr"]

        self.contour_smooth_radius      = s["contour_smooth_radius"]
        self.contour_approx_eps         = s["contour_approx_eps"]
        self.wound_contour_sigma    = s["wound_contour_sigma"]
        self.wound_contour_approx_eps = s["wound_contour_approx_eps"]
        self.player_width          = s["player_width"]
        self.plot_panel_height     = s["plot_panel_height"]

        self.T = self.reader.T
        self.H = self.reader.H
        self.W = self.reader.W
        self.method_labels = s["method_labels"]

        # Shared palette: same ROI index = same colour across methods
        max_rois = max(len(m) for m in self.roi_sets.values()) if self.roi_sets else 1
        shared_pal = make_palette(max_rois)
        self.palettes = {ml: shared_pal for ml in self.method_labels}

        # Pre-compute boundaries and heatmap
        self.roi_boundaries = {
            ml: {k: boundary_mask(v) for k, v in masks.items()}
            for ml, masks in self.roi_sets.items()
        }
        self.wound_mask_img = self.wound_mask   # kept for smooth contour drawing
        self.heatmap_img = arrival_heatmap(self.arrival)

        self.disp_w = self.player_width
        self.disp_h = int(self.H * self.player_width / self.W)

        # State
        self.t = 0
        self.playing = False
        self.speed = 1.0
        self.show_heatmap   = False
        self.show_blackspot = True
        self.show_rois      = True
        self.show_wound     = True
        self.active_method_idx = -1   # -1 = both; 0 = A; 1 = B
        self.curve_mode = s.get("player_curve_mode", "fraction")

        self._rebuild_panel()

    def _active_methods(self) -> List[str]:
        if self.active_method_idx < 0:
            return self.method_labels
        return [self.method_labels[self.active_method_idx
                                   % len(self.method_labels)]]

    def _rebuild_panel(self):
        self.curves_panel, self.plot_bounds = render_curves_panel(
            self.time_axis, self.results, self.palettes,
            self.player_width, self.plot_panel_height,
            active_methods=self._active_methods(),
            mode=self.curve_mode)

    def _composite_frame(self, t: int) -> np.ndarray:
        base = self.reader[t]   # decoded on demand, LRU cached
        bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        # Arrival heat-map
        if self.show_heatmap:
            arrived = (self.arrival <= t) & np.isfinite(self.arrival)
            if arrived.any():
                hm = self.heatmap_img.copy()
                hm[~arrived] = 0
                bgr = cv2.addWeighted(bgr, 0.6, hm, 0.4, 0)

        # Black-spot contours
        unperfused = ~(self.arrival <= t)
        if self.show_blackspot:
            thicknesses = [3, 1]
            for mi, ml in enumerate(self._active_methods()):
                masks = self.roi_sets.get(ml, {})
                pal = self.palettes[ml]
                thick = thicknesses[mi % len(thicknesses)]
                for ri, (_, roi_mask) in enumerate(masks.items()):
                    black = (roi_mask & unperfused).astype(np.uint8)
                    if black.sum() < 5:
                        continue
                    for c in extract_smooth_contours(
                            black,
                            self.contour_smooth_radius,
                            self.contour_approx_eps):
                        cv2.polylines(bgr, [c], True,
                                      pal[ri % len(pal)], thick, cv2.LINE_AA)

        # ROI boundaries
        if self.show_rois:
            for mi, ml in enumerate(self._active_methods()):
                pal = self.palettes[ml]
                for ri, (_, b) in enumerate(
                        self.roi_boundaries.get(ml, {}).items()):
                    bgr[b] = pal[ri % len(pal)]

        # Wound boundary
        if self.show_wound and self.wound_mask_img is not None:
            for c in extract_wound_contours(
                    self.wound_mask_img,
                    sigma=self.wound_contour_sigma,
                    approx_eps=self.wound_contour_approx_eps):
                cv2.polylines(bgr, [c], True, (0, 0, 255), 2, cv2.LINE_AA)

        bgr = cv2.resize(bgr, (self.disp_w, self.disp_h),
                         interpolation=cv2.INTER_AREA)

        # Time-line on curves panel
        panel = self.curves_panel.copy()
        x_frac = t / max(self.T - 1, 1)
        lp, rp = self.plot_bounds
        xp = int(lp + x_frac * (rp - lp))
        cv2.line(panel, (xp, 0), (xp, panel.shape[0]), (60, 60, 255), 2)

        # HUD
        hud = bgr.copy()
        active_str = ("Both" if self.active_method_idx < 0
                      else self.method_labels[self.active_method_idx
                                              % len(self.method_labels)])
        lines = [
            f"frame {t:>5d}/{self.T-1}   t={t/self.fps:6.2f} s   "
            f"speed x{self.speed:.1f}   {'PLAY' if self.playing else 'PAUSE'}",
            f"[h]eat {'ON' if self.show_heatmap else 'off'}   "
            f"[b]lackspot {'ON' if self.show_blackspot else 'off'}   "
            f"[1/2] showing: {active_str}   "
            f"[p] curve: {self.curve_mode}   "
            f"[r]oi {'ON' if self.show_rois else 'off'}   "
            f"[w]ound {'ON' if self.show_wound else 'off'}",
        ]
        for i, txt in enumerate(lines):
            y = 22 + i * 22
            cv2.putText(hud, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(hud, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (255, 255, 255), 1, cv2.LINE_AA)

        return np.vstack([hud, panel])

    def _on_trackbar(self, val):
        self.t = int(np.clip(val, 0, self.T - 1))

    def run(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("frame", self.WINDOW, 0, self.T - 1,
                           self._on_trackbar)
        save_idx = 0

        freq = cv2.getTickFrequency()
        last_tick = cv2.getTickCount()
        time_accum = 0.0
        frame_period = 1.0 / max(self.fps, 1e-3)

        while True:
            comp = self._composite_frame(self.t)
            cv2.imshow(self.WINDOW, comp)
            cv2.setTrackbarPos("frame", self.WINDOW, self.t)

            wait_ms = 16
            key = cv2.waitKey(wait_ms) & 0xFFFF

            now_tick = cv2.getTickCount()
            elapsed = (now_tick - last_tick) / freq
            last_tick = now_tick

            try:
                if cv2.getWindowProperty(self.WINDOW,
                                         cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                self.playing = not self.playing
                time_accum = 0.0   # avoid burst on resume
            elif key in (ord('h'), ord('H')):
                self.show_heatmap = not self.show_heatmap
            elif key in (ord('b'), ord('B')):
                self.show_blackspot = not self.show_blackspot
            elif key in (ord('r'), ord('R')):
                self.show_rois = not self.show_rois
            elif key in (ord('w'), ord('W')):
                self.show_wound = not self.show_wound
            elif key == ord('1'):
                n = len(self.method_labels)
                self.active_method_idx = (self.active_method_idx + 1) % (n + 1)
                if self.active_method_idx == n:
                    self.active_method_idx = -1
                self._rebuild_panel()
            elif key == ord('2'):
                n = len(self.method_labels)
                if self.active_method_idx == -1:
                    self.active_method_idx = n - 1
                else:
                    self.active_method_idx -= 1
                    if self.active_method_idx < 0:
                        self.active_method_idx = -1
                self._rebuild_panel()
            elif key in (ord('p'), ord('P')):
                self.curve_mode = (
                    "pixels" if self.curve_mode == "fraction"
                    else "fraction")
                self._rebuild_panel()
            elif key in (ord('+'), ord('=')):
                self.speed = min(self.speed * 1.5, 16.0)
            elif key in (ord('-'), ord('_')):
                self.speed = max(self.speed / 1.5, 0.0625)
            elif key == ord('0'):
                self.t = 0
            elif key == ord('s'):
                self.output_dir.mkdir(parents=True, exist_ok=True)
                fn = (self.output_dir
                      / f"snapshot_FIM{self.vid_nr}"
                        f"_t{self.t:05d}_{save_idx:02d}.png")
                cv2.imwrite(str(fn), comp)
                print(f"[i] Snapshot -> {fn}")
                save_idx += 1
            elif key in (81, 2424832, ord('a')):
                self.t = max(self.t - 1, 0); self.playing = False
            elif key in (83, 2555904, ord('d')):
                self.t = min(self.t + 1, self.T - 1); self.playing = False
            elif key in (82, 2490368):
                self.t = min(self.t + 10, self.T - 1); self.playing = False
            elif key in (84, 2621440):
                self.t = max(self.t - 10, 0); self.playing = False

            if self.playing:
                time_accum += elapsed * self.speed
                while time_accum >= frame_period:
                    self.t += 1
                    time_accum -= frame_period
                    if self.t >= self.T:
                        self.t = 0
                        self.playing = False
                        break
            else:
                time_accum = 0.0

        cv2.destroyAllWindows()
        self.reader.release()


# =====================================================
# ENTRY POINT
# =====================================================
def main():
    # Resolve session path: command-line arg > SESSION_PATH constant
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    elif SESSION_PATH != Path(""):
        path = SESSION_PATH
    else:
        print("Usage:  python icg_player.py  path/to/FIMxxx_session.npz")
        print("        or set SESSION_PATH at the top of this file.")
        sys.exit(1)

    print(f"[i] Loading session: {path}")
    session = load_session(path)
    print(f"[i] FIM{session['vid_nr']}  "
          f"@ {session['fps']:.2f} fps")
    print(f"[i] Video: {session['video_path']}")
    for ml, masks in session["roi_sets"].items():
        print(f"[i] {ml}: {len(masks)} ROI(s)")

    player = Player(session)

    print("\n[i] Player ready.")
    print("    SPACE=play  arrows/a/d=step  1/2=cycle method  "
          "h/b/r/w=overlays  s=snapshot  q=quit")
    player.run()


if __name__ == "__main__":
    main()