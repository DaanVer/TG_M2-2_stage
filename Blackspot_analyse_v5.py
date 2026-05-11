"""
ICG Angiography Black-Spot Analysis -- Per-ROI shrinking-deficit curves and
Interactive Playback
=========================================================================

GOAL OF THE ANALYSIS
--------------------
The ROIs delimit regions of suspected perfusion deficit ("black spots") in
ICG angiography. Each ROI starts entirely dark; as ICG inflow progresses,
pixels inside it become bright (perfused) and the dark region shrinks. For
every ROI we want a curve

        S(t) = fraction of ROI pixels still dark at time t

which is monotonically non-increasing and plateaus at the residual deficit.
The analysis is run on TWO sets of ROI masks (produced by different
segmentation methods) so their results can be compared directly.

Per-method, per-ROI outputs:
  - S(t) raw and smoothed curves saved as .npy (fraction and pixel count)
  - Two curves figures saved as .png (fraction % and absolute pixel count)
  - Metrics table printed to console and saved as .txt
  - Hill model fit: S(t) = B + A / (1 + ((t - t0) / k)^n)
      A : drop amplitude (initial value minus plateau)
      B : residual plateau (fraction that never perfuses)
      k : half-perfusion time in seconds after onset (primary clinical metric)
      n : Hill exponent (steepness; ~2 for typical ICG perfusion fronts)
      t0: onset time (fitted, not hard-coded from a threshold)
  - Arrival-time heat-map visible in the interactive player
  - Interactive player: video + live black-spot contour + curves panel

Key design decisions
--------------------
1. ICG is BRIGHT -- `invert_signal = False`.  The previous notebook used
   `frame < THRESH`, which detects *darkening* events and was therefore
   always backwards for bright ICG.

2. Global brightness threshold derived from the full video.  A pixel is
   considered perfused when its intensity exceeds:
       threshold = arrival_fraction x p95(all non-zero pixels, 60-frame sample)
   This anchors the threshold to what bright ICG actually looks like in
   each video, making it robust to zero-baseline pixels and comparable
   across videos.  `arrival_fraction` (default 0.40) is the main tuning
   knob -- raise it to be more selective, lower it to be more sensitive.
   A pixel must stay above the threshold for `stable_seconds` consecutive
   seconds (converted to analysis frames automatically) before arrival is
   recorded, filtering transient noise spikes.

3. Two frame rates: `player_fps` for smooth playback, `analysis_fps` for
   arrival detection.  The analysis video is a fraction of the size of the
   player video, keeping RAM use manageable.  Arrival frame indices are
   rescaled back to player-fps coordinates so overlays stay accurate.

4. One global arrival map shared across both ROI sets -- computed in a
   single pass over the analysis video.  Signal is zeroed outside the
   union of all ROIs so pixels outside the regions of interest cannot
   trigger spurious arrivals.

5. Memory-efficient design: the full player video is never loaded as a
   float32 array.  The threshold is computed from a 60-frame uint8
   subsample.  The analysis signal is freed after the arrival map is
   computed.  The interactive player reads video frames on demand from
   the original .mp4 file via an LRU-cached VideoFrameReader.

6. Contour smoothing is configurable via `contour_smooth_radius` (median-
   filter radius on the binary mask before contour extraction) and
   `contour_approx_eps` (Douglas-Peucker epsilon in pixels).

7. Outputs saved to <output_dir>/FIM{vid_nr}_analysis/:
     FIM{vid_nr}_<method>_curves.npy    -- raw, smooth, pixel curves + n_pixels
     FIM{vid_nr}_curves_fraction.png    -- S(t) as % of ROI
     FIM{vid_nr}_curves_pixels.png      -- S(t) as absolute pixel count
     FIM{vid_nr}_metrics.txt            -- full metrics table + fit summary
     FIM{vid_nr}_session.npz            -- session file for standalone player

Player controls (with the player window focused):
    SPACE         play / pause
    LEFT  / RIGHT step -1 / +1 frame  (a / d also work)
    UP   / DOWN   step +10 / -10 frames
    + / -         playback speed up / slow down
    h             toggle arrival-time heat-map overlay
    b             toggle black-spot contour overlay
    p             toggle curve display: fraction (%) / absolute pixels
    1 / 2         cycle which method's contours/curves are shown
    r             toggle ROI boundaries
    w             toggle wound boundary
    s             save current composite frame as PNG
    0             jump to frame 0
    q / ESC       quit
"""

# =====================================================
# IMPORTS
# =====================================================
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from tqdm import tqdm
# Disable tqdm's background monitor thread immediately at import time.
# On Windows the monitor thread fights OpenCV and matplotlib for the GIL
# whenever their event loops run, causing a fatal crash (0xC0000409).
# Setting the monitor interval to 0 prevents the thread from ever starting.
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
# "Agg" is used only for the curves panel rendered inside the player.
# The interactive figure window uses the default backend (TkAgg on most
# systems). We switch backend only when rendering to numpy arrays.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


# =====================================================
# CONFIG -- FILL IN THE PATHS, THEN RUN
# =====================================================
@dataclass
class Config:
    # ------ I/O -- FILL THESE IN ------
    vid_nr: str = "013"
    base_dir: Path = Path("C:/Users/verhu/M2-2 stage")

    # Video file (derived from base_dir + vid_nr if left as None).
    video_path: Optional[Path] = None

    # Two ROI mask files -- one per segmentation method.
    # Leave as None to derive paths from base_dir + vid_nr, or set directly.
    roi_path_A: Optional[Path] = None   # No peak method
    roi_path_B: Optional[Path] = None   # Relative intensity method

    # Human-readable labels shown in all plots and filenames.
    method_label_A: str = "No peak method"
    method_label_B: str = "Relative intensity method"

    # Optional wound mask (display only).
    wound_mask_path: Optional[Path] = None

    # Output directory for .npy curves, .png figure, overlay video.
    output_dir: Optional[Path] = None

    # ------ Video sampling ------
    # The video is loaded at two different frame rates:
#
#   player_fps   : fps at which frames are stored for the interactive
#                  player. Higher = smoother playback but more memory
#                  and a larger .npz session file.
#
#   analysis_fps : fps used for arrival detection and curve computation.
#                  ICG perfusion is a slow process; 2 fps captures it
#                  accurately while reducing the arrival-map workload
#                  by ~15x compared to 30 fps. The resulting arrival
#                  times are rescaled back to player_fps coordinates
#                  so the player heatmap and contours remain accurate.
#                  Recommended range: 1 – 5 fps.
    player_fps: int = 30
    analysis_fps: int = 10
    max_frames: Optional[int] = None   # e.g. 300 for a quick test run

    # ------ Signal direction ------
    # ICG is BRIGHT in these videos (signal rises with arrival).
    # Set True only if you process a video where ICG genuinely darkens.
    invert_signal: bool = False

    # ------ Arrival detection ------
    # Threshold is derived from the brightest pixels in the whole video:
    #
    #   threshold = arrival_fraction × np.percentile(bright_pixels, 95)
    #
    # where bright_pixels = all non-zero ROI pixels across all frames.
    # This anchors the threshold to what 'actually bright with ICG' looks
    # like in this specific video, so it is robust to zero-baseline pixels
    # and does not require a separate baseline window.
    #
    # stable_seconds : a pixel must stay above the threshold for this many
    #     seconds of VIDEO TIME before arrival is recorded.  The equivalent
    #     number of analysis frames is computed automatically as:
    #
    #         stable_frames = max(3, round(stable_seconds * analysis_fps))
    #
    #     so the real-time stability window stays the same regardless of
    #     which analysis_fps you choose.  A minimum of 3 frames is always
    #     enforced to suppress single-frame noise spikes.
    #
    #     Recommended range: 0.5 s (lenient) – 3.0 s (strict).
    arrival_fraction: float = 0.40
    stable_seconds: float = 1.0

    # ------ Curve smoothing ------
    smooth_window_s: float = 1.5       # Savitzky-Golay window (seconds)
    poly_order: int = 2

    # ------ ROI processing ------
    min_roi_area: int = 50             # drop connected components smaller than this

    # ------ Contour smoothing ------
    # Two independent controls for how smooth the live black-spot contour looks:
    #
    #   contour_smooth_radius : median-filter kernel radius (px) applied to
    #       the binary unperfused mask before contour extraction.
    #       0 = off.  3 = slight smoothing.  5-7 = clearly smoother edges.
    #       Higher values may merge very thin protrusions into the background.
    #
    #   contour_approx_eps : Douglas-Peucker simplification tolerance (px).
    #       0 = off.  0.5 = nearly lossless.  2-4 = fewer vertices, rounder
    #       corners.  Does not affect how closely the contour follows blobs,
    #       only how many polygon vertices are used to represent it.
    #
    # Recommended starting point: radius=5, eps=2.0
    contour_smooth_radius: int = 5
    contour_approx_eps: float = 2.0

    # Wound mask contour smoothing.
    # Uses Gaussian blur (same technique as the ROI selection script)
    # rather than median blur, because Gaussian preserves narrow ridges
    # while still producing smooth organic-looking boundaries.
    # wound_contour_sigma : Gaussian sigma in pixels.
    #     2.0 = smooth edges, preserves fine structure (recommended).
    #     4.0 = softer/rounder boundary.
    #     0.5 = almost no smoothing.
    # wound_contour_approx_eps : Douglas-Peucker simplification (0=off).
    wound_contour_sigma: float = 2.0
    wound_contour_approx_eps: float = 1.0

    # ------ Run mode ------
    show_curves: bool = True       # always show the curves figure
                                   # (matplotlib window, non-blocking)
    run_player: bool = True        # open the interactive video player
    save_overlay_video: bool = False  # write overlay .mp4 (slow!)

    # ------ Player ------
    player_width: int = 900
    plot_panel_height: int = 340
    initial_speed: float = 1.0
    # Curve display mode in the interactive player.
    # "fraction" : S(t) as % of ROI area  (0-100)
    # "pixels"   : S(t) as absolute pixel count
    # Press [p] in the player to toggle between modes at runtime.
    player_curve_mode: str = "fraction"  # "fraction" or "pixels"


    def __post_init__(self):
        if self.base_dir == Path(""):
            return   # paths must be set manually
        if self.video_path is None:
            self.video_path = (self.base_dir / "Masked_videos"
                               / f"FIM{self.vid_nr}_ICG_filtered_masked.mp4")
        if self.roi_path_A is None:
            self.roi_path_A = (self.base_dir / "ROI_masks/No_peak"
                               / f"FIM{self.vid_nr}_mask_no_peak.npy")
        if self.roi_path_B is None:
            self.roi_path_B = (self.base_dir / "ROI_masks/Relative_intensity"
                               / f"FIM{self.vid_nr}_mask_relative_intensity.npy")
        if self.wound_mask_path is None:
            candidate = self.base_dir / "Wound_masks" / f"FIM{self.vid_nr}_mask.jpg"
            self.wound_mask_path = candidate if candidate.exists() else None
        if self.output_dir is None:
            self.output_dir = (self.base_dir / "Analysed_videos"
                               / f"FIM{self.vid_nr}_analysis")


CFG = Config()


# =====================================================
# I/O
# =====================================================
def _load_video_at_fps(video_path: Path, target_fps: int,
                        dtype: np.dtype,
                        max_frames: Optional[int] = None
                        ) -> Tuple[np.ndarray, float]:
    """Internal helper: load grayscale frames at a given fps."""
    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(orig_fps / target_fps)), 1)
    eff_fps = orig_fps / step
    frames: List[np.ndarray] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(dtype)
            frames.append(gray)
            if max_frames is not None and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack(frames), eff_fps


def load_video(video_path: Path,
               player_fps: int,
               analysis_fps: int,
               max_frames: Optional[int] = None
               ) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """
    Load the video at two frame rates.

    Returns
    -------
    player_video  : uint8  (T_player,  H, W)  -- for display
    player_fps    : effective player fps
    analysis_video: float32 (T_analysis, H, W) -- for arrival detection
    analysis_fps  : effective analysis fps

    Storing the player video as uint8 (not float32) cuts memory use by 4x.
    The analysis video is float32 but has ~15x fewer frames at 2 fps.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    p_video, p_fps = _load_video_at_fps(
        video_path, player_fps,  np.uint8,   max_frames)
    a_video, a_fps = _load_video_at_fps(
        video_path, analysis_fps, np.float32, max_frames)
    return p_video, p_fps, a_video, a_fps


def load_wound_mask(path: Optional[Path], h: int, w: int) -> Optional[np.ndarray]:
    if path is None or not Path(path).exists():
        print("[i] No wound mask -- skipping wound overlay.")
        return None
    ext = Path(path).suffix.lower()
    m = np.load(path) if ext == ".npy" else cv2.imread(str(path),
                                                        cv2.IMREAD_GRAYSCALE)
    if m is None:
        print(f"[!] Could not read wound mask {path}")
        return None
    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)


def load_rois(roi_path: Path, h: int, w: int,
              min_area: int, label_prefix: str) -> Dict[str, np.ndarray]:
    """Load ROI mask file, label connected components, return {label: bool mask}."""
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")
    roi = np.load(roi_path)
    if roi.shape != (h, w):
        roi = cv2.resize(roi.astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST)
    labeled, n = ndimage.label(roi.astype(bool), structure=np.ones((3, 3)))
    masks: Dict[str, np.ndarray] = {}
    kept = 0
    for i in range(1, n + 1):
        m = labeled == i
        if int(m.sum()) < min_area:
            continue
        kept += 1
        masks[f"{label_prefix}_ROI_{kept}"] = m
    print(f"[i] {label_prefix}: {kept} ROI(s) (area >= {min_area} px)")
    return masks


# =====================================================
# SIGNAL PREP
# =====================================================
def prepare_signal(video: np.ndarray, invert: bool,
                   roi_union: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Signal stack where higher values = more ICG.
    Pixels outside roi_union are zeroed so they can't contaminate baseline
    statistics or trigger spurious arrivals.
    """
    signal = (float(video.max()) - video) if invert else video.copy()
    if roi_union is not None:
        signal[:, ~roi_union] = 0.0
    return signal


def compute_arrival_threshold(video: np.ndarray,
                              arrival_fraction: float,
                              cfg_invert: bool = False) -> float:
    """
    Compute a single global brightness threshold from the full raw video.

    Strategy
    --------
    Collect every non-zero pixel value across the entire video (all frames,
    full frame -- not restricted to the ROI), compute the 95th percentile,
    and multiply by `arrival_fraction`.

    Why the full frame?
    - The ROIs are the *dark* (unperfused) regions. Their own pixel values
      are inherently low, so taking the percentile only over ROI pixels
      would produce a threshold that is far too low and inconsistent across
      videos with different ROI placements.
    - The bright ICG signal is visible in the *perfused* tissue outside the
      ROI. Using the full frame anchors the threshold to what ICG actually
      looks like in this video, making it comparable across videos.

    Why non-zero pixels only?
    - Filtered/masked ICG videos often have large black borders or masked
      regions with value exactly 0. Including those would artificially
      deflate the percentile. Excluding zeros means we only consider
      pixels that contain real tissue signal.

    Why the 95th percentile?
    - It captures the brightness of well-perfused tissue while ignoring
      the top 5% of specular highlights or noise peaks.

    arrival_fraction (default 0.20) sets the bar at 20% of that peak
    brightness, matching the logic of the original notebook's
    `0.2 * np.percentile(video, 99.5)`.
    """
    # Subsample up to 60 evenly-spaced frames for the percentile.
    # Works directly on uint8 video -- no float32 conversion needed.
    # This avoids allocating a 30 GB float32 copy of the player video.
    t_len = len(video)
    idx = np.linspace(0, t_len - 1, min(60, t_len), dtype=int)
    sample = video[idx].ravel().astype(np.float32)
    if cfg_invert:
        sample = float(video.max()) - sample
    nonzero = sample[sample > 0]
    if nonzero.size == 0:
        raise RuntimeError("Video contains no non-zero pixels. "
                           "Check that the video file loaded correctly.")
    p95 = float(np.percentile(nonzero, 95))
    threshold = arrival_fraction * p95
    print(f"[i] Arrival threshold: {arrival_fraction:.2f} × p95 "
          f"({p95:.1f}) = {threshold:.1f} gray levels "
          f"(from {len(idx)}-frame subsample of full video)")
    return threshold


# =====================================================
# ARRIVAL MAP  (single pass -- shared by both ROI sets)
# =====================================================
def compute_arrival_map(signal: np.ndarray,
                        threshold: float,
                        roi_union: np.ndarray,
                        stable_frames: int,
                        analysis_fps: float,
                        player_fps: float) -> np.ndarray:
    """
    (H, W) array of per-pixel arrival times in PLAYER-fps frame indices.

    Extracts only the ROI columns from one frame at a time, keeping the
    working set to O(N_roi) floats regardless of video length.
    Once a pixel has arrived it is removed from the active set so the
    per-frame work shrinks as the video progresses.
    """
    t_frames, h_sig, w_sig = signal.shape
    roi_flat = roi_union.ravel()
    roi_idx  = np.where(roi_flat)[0]
    n_roi    = roi_idx.size
    scale    = player_fps / analysis_fps

    arrival_roi = np.full(n_roi, np.inf, dtype=np.float32)
    counter     = np.zeros(n_roi, dtype=np.int16)
    pending     = np.ones(n_roi,  dtype=bool)
    sig_flat    = signal.reshape(t_frames, -1)   # view, no copy

    for t in tqdm(range(t_frames), desc="Arrival map"):
        if not pending.any():
            break
        frame_roi = sig_flat[t][roi_idx]
        above = pending & (frame_roi > threshold)
        counter[above]            += 1
        counter[~above & pending]  = 0
        newly = pending & (counter >= stable_frames)
        if newly.any():
            arrival_roi[newly] = (t - stable_frames + 1) * scale
            pending[newly] = False

    arrival = np.full(h_sig * w_sig, np.inf, dtype=np.float32)
    arrival[roi_idx] = arrival_roi
    return arrival.reshape(h_sig, w_sig)


# =====================================================
# BLACK-SPOT CURVES
# =====================================================
def compute_spread(arrival: np.ndarray,
                   roi_mask: np.ndarray, T: int) -> np.ndarray:
    """Cumulative fraction of ROI pixels perfused by frame t."""
    a = arrival[roi_mask]
    finite = a[np.isfinite(a)].astype(np.int64)
    total = int(roi_mask.sum())
    if finite.size == 0 or total == 0:
        return np.zeros(T, dtype=np.float32)
    hist = np.bincount(finite, minlength=T + 1)[:T]
    return np.cumsum(hist).astype(np.float32) / total


def blackspot_curve(spread: np.ndarray) -> np.ndarray:
    """S(t) = 1 - spread(t)  -- fraction of ROI still dark."""
    return 1.0 - spread


def smooth_curve(x: np.ndarray, window_s: float,
                 fps: float, poly: int) -> np.ndarray:
    w = max(int(window_s * fps), poly + 2)
    w = w + 1 if w % 2 == 0 else w
    if w >= len(x):
        w = len(x) - (1 - len(x) % 2)
        if w < poly + 2:
            return x.copy()
    return savgol_filter(x, w, poly)


def compute_tic(signal: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Mean signal over ROI pixels per frame (used as a sanity check)."""
    return signal[:, roi_mask].mean(axis=1)


# =====================================================
# DECAY FIT  (Hill / generalised hyperbola)
# =====================================================
def fit_blackspot_decay(bs_smooth: np.ndarray,
                        time_axis: np.ndarray,
                        onset_drop: float = 0.05
                        ) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Fit the Hill (generalised hyperbola) model to the post-onset segment:

        S(t) = B + A / (1 + ((t - t_onset) / k) ^ n)

    Returns (fit_dict, t_dec, fitted_curve). fit_dict keys:
      A         : drop amplitude  (S at onset minus residual plateau)
      B         : residual plateau  (black-spot fraction that never perfuses)
      k_hill    : half-decay time in seconds after onset
      n_hill    : Hill steepness exponent  (typical range 1 -- 4)
      B_ci95    : 95 %% CI half-width on B from fit covariance
      t_half_fit_s : same as k_hill (convenience alias)
      t_onset   : fitted onset time (seconds)

    Why Hill instead of exponential
    --------------------------------
    The exponential forces the fastest decay at t_onset and a
    monotonically decreasing rate, which often misses the initially
    linear descent visible in ICG curves (see FIM018).  The Hill
    model has a tunable steepness (n) that can match both an abrupt
    onset and the gradual flattening toward the residual, without
    extrapolating above S=1 before t_onset.

    onset_drop : S must fall this far below its initial value before
                 t_onset is declared (default 5 %%).
    """
    nan_r = ({"A": np.nan, "k_hill": np.nan, "n_hill": np.nan,
               "B": np.nan, "B_ci95": np.nan, "t_half_fit_s": np.nan,
               "idx_onset": -1, "t_onset": np.nan},
              np.array([]), np.array([]))
    if bs_smooth.size < 5:
        return nan_r

    initial = float(bs_smooth[:5].mean())
    cands = np.where(bs_smooth < initial - onset_drop)[0]
    if not cands.size or cands[0] >= len(bs_smooth) - 5:
        return nan_r
    onset_idx = int(cands[0])

    # Fit over the FULL curve with t_onset as a free parameter.
    # This avoids the systematic bias that occurs when fitting only
    # post-onset data: the onset detector lags the true onset because
    # the 5 %% threshold is crossed after the curve has already started
    # moving, so a fixed-onset fit systematically underestimates k and n.
    #
    # Model:  S(t) = B + A / (1 + max(0, (t - t0) / k)^n)
    # The max(0, ...) ensures S=B+A (plateau) for t < t0.

    B0  = float(np.clip(bs_smooth[-5:].mean(), 0.0, 0.99))
    A0  = float(np.clip(initial - B0, 1e-3, 1.0))
    t0_0 = float(time_axis[onset_idx])   # rough onset from threshold

    # k0: half-decay time -- estimate from post-onset data
    y_post = bs_smooth[onset_idx:]
    t_post = time_axis[onset_idx:] - t0_0
    half_level = B0 + 0.5 * A0
    below_half = np.where(y_post <= half_level)[0]
    k0 = float(t_post[below_half[0]]) if below_half.size else float(t_post[-1] / 2)
    k0 = max(k0, 1.0)

    def hill_full(t, amp, k_h, n_h, b_h, t0):
        t_rel = np.maximum(t - t0, 0.0)
        ratio = np.where(t_rel > 0, np.clip(t_rel / k_h, 0, 1e6), 0.0)
        return b_h + amp / (1.0 + ratio ** n_h)

    t_min = float(time_axis[0])
    t_max = float(time_axis[-1])
    # Allow t0 to vary +/- 10 s around the detected onset
    t0_lo = max(t_min, t0_0 - 10.0)
    t0_hi = min(t_max * 0.8, t0_0 + 10.0)

    try:
        popt, pcov = curve_fit(
            hill_full, time_axis, bs_smooth,
            p0=[A0, k0, 2.0, B0, t0_0],
            bounds=([0,    1.0,  0.5, 0,   t0_lo],
                    [2.0,  np.inf, 20, 1.0, t0_hi]),
            maxfev=60000)
        A, k, n, B, t0_fit = (float(popt[0]), float(popt[1]),
                               float(popt[2]), float(popt[3]),
                               float(popt[4]))
        t_half_fit = k
        fitted = hill_full(time_axis, *popt)
        # 95 %% CI on B (index 3 in popt) from the fit covariance.
        # pcov[3,3] is the variance of B; sqrt gives 1-sigma, *1.96 = 95%%.
        # If the covariance is infinite (poor fit), report nan.
        B_var = float(pcov[3, 3])
        B_ci95 = 1.96 * np.sqrt(B_var) if np.isfinite(B_var) else np.nan
    except (RuntimeError, ValueError):
        A = k = n = B = t0_fit = t_half_fit = np.nan
        B_ci95 = np.nan
        fitted = np.full(len(time_axis), np.nan, dtype=np.float32)
        t0_fit = t0_0

    t_onset_final = t0_fit if not np.isnan(t0_fit) else float(time_axis[onset_idx])

    # t_dec covers the post-onset portion for plotting
    plot_mask = time_axis >= t_onset_final
    t_dec = time_axis[plot_mask]
    fitted_plot = fitted[plot_mask] if not np.any(np.isnan(fitted)) \
                  else np.full(plot_mask.sum(), np.nan, dtype=np.float32)

    return ({"A": A, "k_hill": k, "n_hill": n, "B": B,
              "B_ci95": B_ci95,
              "t_half_fit_s": t_half_fit,
              "idx_onset": onset_idx,
              "t_onset": t_onset_final},
             t_dec, fitted_plot)


# =====================================================
# METRICS
# =====================================================
def extract_metrics(label: str, bs_raw: np.ndarray, bs_smooth: np.ndarray,
                    arrival: np.ndarray, roi_mask: np.ndarray,
                    fit_info: Dict, time_axis: np.ndarray, fps: float,
                    tic_smooth: Optional[np.ndarray] = None) -> Dict:
    initial = float(bs_smooth[:5].mean())
    final = float(bs_smooth[-5:].mean())
    total_red = max(initial - final, 0.0)

    def _t_cross(level: float) -> float:
        idx = np.where(bs_smooth <= level)[0]
        return float(time_axis[idx[0]]) if idx.size else np.nan

    t_half = _t_cross(initial - 0.5 * total_red) if total_red > 1e-6 else np.nan
    t_90 = _t_cross(initial - 0.9 * total_red) if total_red > 1e-6 else np.nan

    finite_arr = arrival[roi_mask]
    finite_arr = finite_arr[np.isfinite(finite_arr)]
    med_arr = float(np.median(finite_arr) / fps) if finite_arr.size else np.nan

    if tic_smooth is not None and tic_smooth.size:
        n_base = max(int(fps), 1)
        peak_tic = float(tic_smooth.max() - tic_smooth[:n_base].mean())
    else:
        peak_tic = np.nan

    n_pixels = int(roi_mask.sum())
    return {
        "label": label,
        "n_pixels": n_pixels,
        "initial_blackspot": initial,
        "final_blackspot": final,
        "final_blackspot_pixels": round(final * n_pixels),
        "total_reduction": total_red,
        "t_onset_s": fit_info.get("t_onset", np.nan),
        "t_half_s": t_half,
        "t_90_s": t_90,
        "fit_k_half_s": fit_info.get("t_half_fit_s", np.nan),
        "fit_n": fit_info.get("n_hill", np.nan),
        "fit_B": fit_info.get("B", np.nan),
        "fit_B_ci95": fit_info.get("B_ci95", np.nan),
        "plateau": fit_info.get("plateau", np.nan),
        "plateau_lo": fit_info.get("plateau_lo", np.nan),
        "plateau_hi": fit_info.get("plateau_hi", np.nan),
        "median_pixel_arrival_s": med_arr,
        "peak_tic_above_base": peak_tic,
    }


# =====================================================
# COLOUR HELPERS
# =====================================================
def make_palette(n: int) -> List[Tuple[int, int, int]]:
    """n distinct BGR colours (matplotlib tab10/tab20)."""
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [(int(b * 255), int(g * 255), int(r * 255))
            for r, g, b, _ in (cmap(i % cmap.N) for i in range(n))]


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8)).astype(bool)


def arrival_heatmap(arrival: np.ndarray) -> np.ndarray:
    """BGR heatmap, colourmap spans the range of finite arrival times."""
    H, W = arrival.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
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


# =====================================================
# CONTOUR EXTRACTION (with smoothing)
# =====================================================
def extract_smooth_contours(binary_mask: np.ndarray,
                             smooth_radius: int,
                             approx_eps: float) -> List[np.ndarray]:
    """
    Extract contours from a binary mask with configurable smoothing.

    smooth_radius : radius of the median-blur pre-filter (0 = skip).
        Removes single-pixel jaggies before contour tracing.
        Recommended: 3-7.  Higher = smoother but loses fine structure.

    approx_eps : Douglas-Peucker polygon simplification in pixels (0 = off).
        Reduces vertex count for a rounder-looking outline.
        Recommended: 1.0-3.0.  Has no effect on blob position/size.
    """
    m = binary_mask.astype(np.uint8)
    if smooth_radius > 0:
        m = cv2.medianBlur(m, 2 * smooth_radius + 1)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if approx_eps > 0:
        contours = [cv2.approxPolyDP(c, approx_eps, closed=True)
                    for c in contours if len(c) >= 5]
    else:
        contours = [c for c in contours if len(c) >= 5]
    return contours


def extract_wound_contours(binary_mask: np.ndarray,
                            sigma: float,
                            approx_eps: float) -> List[np.ndarray]:
    """
    Extract smooth contours from a wound mask using Gaussian blur
    before contour tracing -- the same technique used in the ROI
    selection script (Automatic_ROI_selection.py).

    Unlike median blur (used for black-spot contours), Gaussian blur
    preserves thin ridges while still producing smooth, organic-looking
    boundary lines.  The mask is blurred as a float, re-thresholded
    at 0.5, then contours are traced.  This closely matches what
    matplotlib's ax.contour() produces internally.

    sigma     : Gaussian sigma in pixels.  2-4 gives smooth edges
                while preserving narrow features.  The ROI script
                uses sigma=2 for final mask smoothing.
    approx_eps: Douglas-Peucker simplification (0 = off).
    """
    # Gaussian-blur the float mask, then re-threshold
    blurred = ndimage.gaussian_filter(
        binary_mask.astype(np.float32), sigma=sigma)
    m = (blurred > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(
        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if approx_eps > 0:
        contours = [cv2.approxPolyDP(c, approx_eps, closed=True)
                    for c in contours if len(c) >= 5]
    else:
        contours = [c for c in contours if len(c) >= 5]
    return contours


# =====================================================
# SAVE CURVES
# =====================================================
def save_curves_npy(out_dir: Path, vid_nr: str, method_label: str,
                    time_axis: np.ndarray,
                    res: Dict):
    """
    Save curves to a .npy file (use np.load(..., allow_pickle=True).item()
    to reload as a dict).

    Structure:
        {
          "time_s":        np.ndarray (T,),
          "raw":           {roi_label: np.ndarray (T,), ...},
          "smooth":        {roi_label: np.ndarray (T,), ...},
          "pixels_smooth": {roi_label: np.ndarray (T,), ...},
          "n_pixels":      {roi_label: int, ...},
        }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = method_label.replace(" ", "_")
    path = out_dir / f"FIM{vid_nr}_{safe}_curves.npy"
    bss = res["blackspots_smooth"]
    np.save(str(path), {
        "time_s":        time_axis,
        "raw":           res["blackspots_raw"],
        "smooth":        bss,
        "pixels_smooth": res["blackspots_pixels_smooth"],
        "n_pixels":      {lbl: res["metrics"][lbl]["n_pixels"]
                           for lbl in bss},
    })
    print(f"[i] Curves (.npy) -> {path}")


def _save_one_curves_figure(out_dir: Path, vid_nr: str,
                            time_axis: np.ndarray,
                            results: Dict[str, Dict],
                            palettes: Dict[str, List],
                            mode: str) -> Path:
    """
    Save a two-panel figure for one display mode.
    mode: "fraction" (0-100 %) or "pixels" (absolute pixel count).
    """
    use_pixels = (mode == "pixels")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5),
                             sharey=False)  # axes are independent
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#101010")

    for ax, (ml, res) in zip(axes, results.items()):
        ax.set_facecolor("#101010")
        pal = palettes[ml]
        for i, (label, bs_s) in enumerate(res["blackspots_smooth"].items()):
            b, g, r = pal[i % len(pal)]
            col = (r / 255, g / 255, b / 255)
            short = label.split("ROI_")[-1]
            n_px = res["metrics"][label]["n_pixels"]
            if use_pixels:
                scale = n_px
                raw_key = "blackspots_raw"
                smo_key = "blackspots_pixels_smooth"
            else:
                scale = 100
                raw_key = "blackspots_raw"
                smo_key = "blackspots_smooth"
            y_raw = res[raw_key][label] * scale
            y_smo = res[smo_key][label] * scale if not use_pixels \
                    else res[smo_key][label]
            ax.plot(time_axis, y_raw,
                    color=col, lw=0.8, alpha=0.3)
            ax.plot(time_axis, y_smo,
                    color=col, lw=1.8, label=f"ROI {short}")
            fi, t_dec, fit_curve = res["fits"][label]
            if t_dec.size and not np.isnan(fi.get("k_hill", np.nan)):
                fc = fit_curve * scale
                ax.plot(t_dec, fc,
                        color=col, lw=1.2, ls="--", alpha=0.9)
            if not np.isnan(fi.get("t_onset", np.nan)):
                ax.axvline(fi["t_onset"], color=col, lw=0.8, ls=":", alpha=0.6)
            # Plateau: draw a horizontal dashed line at B with a
            # shaded 95 %% CI band, so the long-term residual is
            # immediately visible relative to the current curve.
            pl    = fi.get("plateau", np.nan)
            pl_lo = fi.get("plateau_lo", np.nan)
            pl_hi = fi.get("plateau_hi", np.nan)
            if not np.isnan(pl):
                t0_plt = fi.get("t_onset", float(time_axis[0]))
                ax.axhline(pl * scale, color=col, lw=1.0,
                           ls="--", alpha=0.6)
                if not np.isnan(pl_lo):
                    ax.axhspan(pl_lo * scale, pl_hi * scale,
                               color=col, alpha=0.08)

        ax.set_title(ml, color="white", fontsize=11)
        ax.set_xlabel("Time (s)", color="white", fontsize=9)
        ylabel = ("Black-spot size  S(t)  [pixels]"
                  if use_pixels else "Black-spot size  S(t)  [% of ROI]")
        ax.set_ylabel(ylabel, color="white", fontsize=9)
        if not use_pixels:
            ax.set_ylim(-2, 105)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#555")
        ax.grid(True, color="#333", lw=0.5)
        ax.legend(fontsize=8, facecolor="#202020",
                  labelcolor="white", framealpha=0.7)

    suffix = "pixels" if use_pixels else "fraction"
    fig.suptitle(
        f"FIM{vid_nr} -- Black-spot decay ({suffix})",
        color="white", fontsize=12)
    fig.tight_layout()
    path = out_dir / f"FIM{vid_nr}_curves_{suffix}.png"
    fig.savefig(str(path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_curves_figure(out_dir: Path, vid_nr: str,
                       time_axis: np.ndarray,
                       results: Dict[str, Dict],
                       palettes: Dict[str, List]):
    """Save both fraction and pixel figures."""
    for mode in ("fraction", "pixels"):
        path = _save_one_curves_figure(
            out_dir, vid_nr, time_axis, results, palettes, mode)
        print(f"[i] Curves figure (.png) -> {path.name}")


# =====================================================
# PLAYER CURVES PANEL
# =====================================================
def render_curves_panel(time_axis: np.ndarray,
                        results: Dict[str, Dict],
                        palettes: Dict[str, List],
                        width_px: int,
                        height_px: int,
                        active_methods: Optional[List[str]] = None,
                        mode: str = "fraction",
                        ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Render the live curves panel for the player.
    mode: "fraction" (0-100%) or "pixels" (absolute pixel count).
    Returns (BGR image, (left_px, right_px)) for exact time-line placement.
    """
    if active_methods is None:
        active_methods = list(results.keys())
    use_pixels = (mode == "pixels")

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
        smo_key: str = ("blackspots_pixels_smooth" if use_pixels
                        else "blackspots_smooth")
        for i, (label, bs_s) in enumerate(res[smo_key].items()):
            b, g, r = pal[i % len(pal)]
            col = (r / 255, g / 255, b / 255)
            roi_num = label.split("ROI_")[-1]
            n_px = res["metrics"][label]["n_pixels"]
            scale = 1 if use_pixels else 100
            ax.plot(time_axis, bs_s * scale, color=col, lw=1.6, ls=ls,
                    label=f"{ml} ROI {roi_num}")
            fi, t_dec, fit_curve = res["fits"][label]
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
    ylabel = ("Black-spot  S(t)  [px]"
              if use_pixels else "Black-spot  S(t)  [%]")
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    if not use_pixels:
        ax.set_ylim(-2, 105)
    t0, t_end = float(time_axis[0]), float(time_axis[-1])
    ax.set_xlim(t0, t_end)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#666")
    ax.grid(True, color="#333", lw=0.5)
    n_curves = sum(len(results[m][smo_key])
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
# =====================================================
# VIDEO FRAME READER  (shared with icg_player.py)
# =====================================================
class VideoFrameReader:
    """Decode video frames on demand with LRU cache. See icg_player.py."""
    def __init__(self, video_path: Path, target_fps: float,
                 cache_size: int = 128):
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self._cap = cv2.VideoCapture(str(video_path))
        orig_fps  = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._step = max(int(round(orig_fps / target_fps)), 1)
        self._eff_fps = orig_fps / self._step
        total_orig = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.T = max(total_orig // self._step, 1)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"Could not read first frame of {video_path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.H, self.W = gray.shape
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_order: List[int] = []
        self._cache_size = cache_size
        self._last_orig_pos = -1

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.T, self.H, self.W)

    def __len__(self) -> int:
        return self.T

    def __getitem__(self, player_idx: int) -> np.ndarray:
        if player_idx in self._cache:
            return self._cache[player_idx]
        orig_idx = player_idx * self._step
        if orig_idx != self._last_orig_pos + self._step:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
        ok, frame = self._cap.read()
        if not ok:
            return (self._cache[self._cache_order[-1]]
                    if self._cache_order
                    else np.zeros((self.H, self.W), dtype=np.uint8))
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


class InteractivePlayer:
    WINDOW = "ICG Black-Spot Player"

    def __init__(self, video: Path, fps: float, cfg: Config,
                 roi_sets: Dict[str, Dict[str, np.ndarray]],
                 wound_mask: Optional[np.ndarray],
                 arrival: np.ndarray,
                 time_axis: np.ndarray,
                 results: Dict[str, Dict],
                 palettes: Dict[str, List]):
        self.reader = VideoFrameReader(video, fps)
        self.fps = fps
        self.cfg = cfg
        self.roi_sets = roi_sets           # {method_label: {roi_label: mask}}
        self.wound_mask = wound_mask
        self.arrival = arrival
        self.time_axis = time_axis
        self.results = results
        self.palettes = palettes

        self.T = self.reader.T
        self.H = self.reader.H
        self.W = self.reader.W
        self.method_labels = list(roi_sets.keys())

        # Pre-compute per-method ROI boundaries and shared heatmap.
        self.roi_boundaries = {
            ml: {k: boundary_mask(v) for k, v in masks.items()}
            for ml, masks in roi_sets.items()
        }
        self.wound_mask_img = wound_mask   # kept for smooth contour drawing
        self.heatmap_img = arrival_heatmap(arrival)

        self.disp_w = cfg.player_width
        self.disp_h = int(self.H * cfg.player_width / self.W)

        # State
        self.t = 0
        self.playing = False
        self.speed = cfg.initial_speed
        self.show_heatmap = False
        self.show_blackspot = True
        self.show_rois = True
        self.show_wound = True
        # -1 = both methods; 0 = method A only; 1 = method B only
        self.active_method_idx = -1
        self.curve_mode = cfg.player_curve_mode   # "fraction" or "pixels"
        self._rebuild_panel()

    def _active_methods(self) -> List[str]:
        if self.active_method_idx < 0:
            return self.method_labels
        return [self.method_labels[self.active_method_idx % len(self.method_labels)]]

    def _rebuild_panel(self):
        self.curves_panel, self.plot_bounds = render_curves_panel(
            self.time_axis, self.results, self.palettes,
            self.cfg.player_width, self.cfg.plot_panel_height,
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
            # Method A = thick line (3 px), Method B = thin (1 px)
            thicknesses = [3, 1]
            for mi, ml in enumerate(self._active_methods()):
                masks = self.roi_sets.get(ml, {})
                pal = self.palettes[ml]
                thick = thicknesses[mi % len(thicknesses)]
                for ri, (_, roi_mask) in enumerate(masks.items()):
                    black = (roi_mask & unperfused).astype(np.uint8)
                    if black.sum() < 5:
                        continue
                    contours = extract_smooth_contours(
                        black,
                        self.cfg.contour_smooth_radius,
                        self.cfg.contour_approx_eps)
                    col = pal[ri % len(pal)]
                    for c in contours:
                        cv2.polylines(bgr, [c], True, col, thick, cv2.LINE_AA)

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
                    sigma=self.cfg.wound_contour_sigma,
                    approx_eps=self.cfg.wound_contour_approx_eps):
                cv2.polylines(bgr, [c], True, (0, 0, 255), 2, cv2.LINE_AA)

        bgr = cv2.resize(bgr, (self.disp_w, self.disp_h),
                         interpolation=cv2.INTER_AREA)

        # Live time-line on curves panel
        panel = self.curves_panel.copy()
        x_frac = t / max(self.T - 1, 1)
        lp, rp = self.plot_bounds
        cv2.line(panel,
                 (int(lp + x_frac * (rp - lp)), 0),
                 (int(lp + x_frac * (rp - lp)), panel.shape[0]),
                 (60, 60, 255), 2)

        # HUD overlay
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

        # Accumulator-based timing: we track how many seconds of "video
        # time" have accumulated since the last frame advance.  Each
        # iteration of the loop adds the real elapsed wall time (scaled
        # by playback speed) to the accumulator; when it exceeds one
        # frame period (1/fps) we advance the frame counter and subtract
        # one period.  This is immune to the jitter caused by measuring
        # last_t before vs after waitKey.
        freq = cv2.getTickFrequency()
        last_tick = cv2.getTickCount()
        time_accum = 0.0           # accumulated video-time seconds
        frame_period = 1.0 / max(self.fps, 1e-3)

        while True:
            comp = self._composite_frame(self.t)
            cv2.imshow(self.WINDOW, comp)
            cv2.setTrackbarPos("frame", self.WINDOW, self.t)

            # Always wait ~16 ms (≈60 Hz UI refresh) regardless of fps.
            # Frame advancement is handled by the accumulator below.
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
                # Cycle forward: both -> A -> B -> both
                n = len(self.method_labels)
                self.active_method_idx = (self.active_method_idx + 1) % (n + 1)
                if self.active_method_idx == n:
                    self.active_method_idx = -1
                self._rebuild_panel()
            elif key == ord('2'):
                # Cycle backward
                n = len(self.method_labels)
                if self.active_method_idx == -1:
                    self.active_method_idx = n - 1
                else:
                    self.active_method_idx = (self.active_method_idx - 1)
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
                self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
                fn = (self.cfg.output_dir
                      / f"snapshot_FIM{self.cfg.vid_nr}"
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
                        time_accum = 0.0
                        break
            else:
                time_accum = 0.0   # reset so we don't burst on resume

        cv2.destroyAllWindows()
        self.reader.release()


# =====================================================
# SESSION SAVE  (everything the standalone player needs)
# =====================================================
def save_session(out_dir: Path, vid_nr: str,
                 fps: float,
                 arrival: np.ndarray,
                 time_axis: np.ndarray,
                 roi_sets: Dict[str, Dict[str, np.ndarray]],
                 results: Dict[str, Dict],
                 wound_mask,
                 cfg) -> Path:
    """
    Save all data needed to re-open the interactive player without
    re-running the full pipeline.

    Output:  <out_dir>/FIM{vid_nr}_session.npz

    Nested dict keys are flattened with "/" as separator, then "/" is
    replaced by "__SLASH__" because npz keys cannot contain slashes.
    The companion script icg_player.py handles un-flattening automatically.

    Load manually with:
        data = np.load(path, allow_pickle=False)
        # key "roi_sets__SLASH__Method A__SLASH__Method A_ROI_1" -> bool mask
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"FIM{vid_nr}_session.npz"

    arrays = {
        "_meta/vid_nr":                np.array(vid_nr),
        "_meta/fps":                   np.array(fps, dtype=np.float32),
        "_meta/method_labels":         np.array(list(roi_sets.keys())),
        "_meta/method_label_A":        np.array(cfg.method_label_A),
        "_meta/method_label_B":        np.array(cfg.method_label_B),
        "_meta/contour_smooth_radius":        np.array(cfg.contour_smooth_radius),
        "_meta/contour_approx_eps":            np.array(cfg.contour_approx_eps,
                                                         dtype=np.float32),
        "_meta/wound_contour_sigma":      np.array(cfg.wound_contour_sigma,
                                                    dtype=np.float32),
        "_meta/wound_contour_approx_eps": np.array(cfg.wound_contour_approx_eps,
                                                    dtype=np.float32),
        "_meta/player_width":          np.array(cfg.player_width),
        "_meta/plot_panel_height":     np.array(cfg.plot_panel_height),
        "_meta/output_dir":            np.array(str(out_dir)),
        "_meta/video_path":            np.array(str(cfg.video_path)),
        "_meta/player_curve_mode":     np.array(cfg.player_curve_mode),
        "arrival":   arrival,
        "time_axis": time_axis.astype(np.float32),
    }

    if wound_mask is not None:
        arrays["wound_mask"] = wound_mask.astype(np.uint8)

    for ml, masks in roi_sets.items():
        for roi_label, mask in masks.items():
            arrays[f"roi_sets/{ml}/{roi_label}"] = mask.astype(np.uint8)

    for ml, res in results.items():
        for roi_label, bs_raw in res["blackspots_raw"].items():
            arrays[f"results/{ml}/blackspots_raw/{roi_label}"] = \
                bs_raw.astype(np.float32)
        for roi_label, bs_s in res["blackspots_smooth"].items():
            arrays[f"results/{ml}/blackspots_smooth/{roi_label}"] = \
                bs_s.astype(np.float32)
        for roi_label, px_s in res.get("blackspots_pixels_smooth", {}).items():
            arrays[f"results/{ml}/blackspots_pixels_smooth/{roi_label}"] = \
                px_s.astype(np.float32)
        # Save n_pixels per ROI so the player can convert fraction->pixels
        for roi_label, m in res["metrics"].items():
            arrays[f"results/{ml}/n_pixels/{roi_label}"] = \
                np.array(m["n_pixels"], dtype=np.int32)
        for roi_label, (fi, t_dec, fit_curve) in res["fits"].items():
            base = f"results/{ml}/fits/{roi_label}"
            for k, v in fi.items():
                arrays[f"{base}/fi_{k}"] = np.array(v, dtype=np.float32)
            arrays[f"{base}/t_dec"]      = t_dec.astype(np.float32)
            arrays[f"{base}/fit_curve"]  = fit_curve.astype(np.float32)

    safe = {k.replace("/", "__SLASH__"): v for k, v in arrays.items()}
    np.savez_compressed(str(path), **safe)
    print(f"[i] Session saved -> {path.name}  "
          f"({path.stat().st_size / 1e6:.1f} MB)")
    return path



# =====================================================
# OVERLAY VIDEO EXPORT
# =====================================================
def dump_overlay_video(player: InteractivePlayer,
                       out_path: Path, fps: float):
    """Write the composite overlay video frame-by-frame."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample = player._composite_frame(0)
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for t in tqdm(range(player.T), desc=f"Writing {out_path.name}"):
            writer.write(player._composite_frame(t))
    finally:
        writer.release()
    print(f"[i] Overlay video -> {out_path}")

# =====================================================
# CURVES WINDOW  (non-blocking, shown alongside the player)
# =====================================================
def _show_curves_window(out_dir: Path, vid_nr: str,
                        blocking: bool = True):
    """
    Open the saved curves PNG figures in matplotlib windows.
    Both the fraction (%) and pixel-count figures are shown, one after
    the other (each blocks until closed when blocking=True).
    """
    import matplotlib
    import matplotlib.pyplot as _plt
    import matplotlib.image as _mpimg

    # Switch to a GUI backend if needed.
    cur = matplotlib.get_backend()
    if cur.lower() == "agg":
        for backend in ("TkAgg", "Qt5Agg", "QtAgg"):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue

    pngs = [
        out_dir / f"FIM{vid_nr}_curves_fraction.png",
        out_dir / f"FIM{vid_nr}_curves_pixels.png",
    ]
    found = [p for p in pngs if p.exists()]
    if not found:
        print(f"[!] No curves figures found in {out_dir} -- "
              f"skipping display. (Expected: {[p.name for p in pngs]})")
        return

    for png in found:
        try:
            fig, ax = _plt.subplots(figsize=(12, 5))
            ax.imshow(_mpimg.imread(str(png)))
            ax.axis("off")
            fig.tight_layout(pad=0)
            fig.canvas.manager.set_window_title(
                f"FIM{vid_nr} -- {png.stem}")
            if blocking:
                print(f"[i] Showing {png.name} -- "
                      f"close this window to continue.")
                _plt.show(block=True)
            else:
                _plt.show(block=False)
                _plt.pause(0.1)
                print(f"[i] Curves window open ({png.name})")
        except Exception as e:
            print(f"[!] Could not open {png.name}: {e}")
            print(f"    Saved figure is at: {png}")


# =====================================================
# MAIN
# =====================================================
def main(cfg: Config = CFG):
    # ---------- load ----------
    print(f"[i] Loading video: {cfg.video_path}")
    video, p_fps, a_video, a_fps = load_video(
        cfg.video_path, cfg.player_fps, cfg.analysis_fps, cfg.max_frames)
    T,  H, W = video.shape
    Ta, _, _ = a_video.shape
    print(f"[i] Player video:   {T}  frames @ {p_fps:.2f} fps  "
          f"({video.nbytes / 1e6:.0f} MB, uint8)")
    print(f"[i] Analysis video: {Ta} frames @ {a_fps:.2f} fps  "
          f"({a_video.nbytes / 1e6:.0f} MB, float32)")

    wound_mask = load_wound_mask(cfg.wound_mask_path, H, W)

    roi_sets: Dict[str, Dict[str, np.ndarray]] = {
        cfg.method_label_A: load_rois(cfg.roi_path_A, H, W,
                                       cfg.min_roi_area, cfg.method_label_A),
        cfg.method_label_B: load_rois(cfg.roi_path_B, H, W,
                                       cfg.min_roi_area, cfg.method_label_B),
    }

    # Union of ALL ROIs from BOTH methods.
    roi_union = np.zeros((H, W), dtype=bool)
    for masks in roi_sets.values():
        for m in masks.values():
            roi_union |= m
    print(f"[i] Combined ROI union: {roi_union.sum()} px "
          f"({100 * roi_union.mean():.1f} %)")

    # ---------- signal + arrival ----------
    # Threshold: computed directly from the uint8 player video using a
    # 60-frame subsample -- no float32 copy of the full video needed.
    # This saves ~30 GB of RAM for a 2-min 1080p video.
    threshold = compute_arrival_threshold(
        video, cfg.arrival_fraction, cfg_invert=cfg.invert_signal)

    # Arrival map: computed on the low-fps analysis video.
    # prepare_signal zeros pixels outside the ROI union so they cannot
    # trigger spurious arrivals.
    signal = prepare_signal(a_video, cfg.invert_signal, roi_union=roi_union)
    del a_video   # free ~10 GB -- signal is the only copy we need

    # ---------- ONE arrival map shared by both ROI sets ----------
    stable_frames = max(3, round(cfg.stable_seconds * a_fps))
    print(f"[i] Stable frames: max(3, round({cfg.stable_seconds} s × "
          f"{a_fps:.2f} fps)) = {stable_frames} frames "
          f"({stable_frames / a_fps:.2f} s at analysis fps)")
    arrival = compute_arrival_map(
        signal, threshold, roi_union, stable_frames, a_fps, p_fps)
    # signal is still needed for compute_tic in the per-ROI loop below;
    # it will be deleted after that loop completes.

    # ---------- per-method, per-ROI curves + metrics ----------
    time_axis = np.arange(T) / p_fps

    # Use a shared palette so same ROI index = same colour across methods.
    max_rois = max(len(m) for m in roi_sets.values())
    shared_palette = make_palette(max_rois)
    palettes = {ml: shared_palette for ml in roi_sets}

    results: Dict[str, Dict] = {}

    for ml, roi_masks in roi_sets.items():
        bsr: Dict[str, np.ndarray] = {}
        bss: Dict[str, np.ndarray] = {}
        fits: Dict[str, Tuple] = {}
        met: Dict[str, Dict] = {}

        bsp: Dict[str, np.ndarray] = {}   # pixel-count smooth curves
        for label, mask in roi_masks.items():
            spread = compute_spread(arrival, mask, T)
            bs_raw = blackspot_curve(spread)
            bs_s = smooth_curve(bs_raw, cfg.smooth_window_s,
                                 p_fps, cfg.poly_order)
            bsr[label] = bs_raw
            bss[label] = bs_s
            # Absolute pixel count = fraction × ROI size; smooth separately
            n_px = int(mask.sum())
            bs_px_raw = bs_raw * n_px
            bsp[label] = smooth_curve(bs_px_raw, cfg.smooth_window_s,
                                      p_fps, cfg.poly_order)

            fi, t_dec, fit_c = fit_blackspot_decay(bs_s, time_axis)

            # The plateau is B itself (the Hill model's asymptote as t→∞).
            # The CI on B from the fit covariance gives the uncertainty.
            if not np.isnan(fi.get("k_hill", np.nan)):
                ci = fi.get("B_ci95", np.nan)
                fi["plateau"]    = float(fi["B"])
                fi["plateau_lo"] = float(np.clip(fi["B"] - ci, 0.0, 1.0)) \
                                   if not np.isnan(ci) else np.nan
                fi["plateau_hi"] = float(np.clip(fi["B"] + ci, 0.0, 1.0)) \
                                   if not np.isnan(ci) else np.nan
            else:
                fi["plateau"]    = np.nan
                fi["plateau_lo"] = np.nan
                fi["plateau_hi"] = np.nan

            fits[label] = (fi, t_dec, fit_c)

            tic_raw = compute_tic(signal, mask)
            # Subtract rough baseline: mean of first 10 s worth of frames
            n_base = max(int(10.0 * a_fps), 5)
            tic_s = smooth_curve(
                tic_raw - tic_raw[:n_base].mean(),
                cfg.smooth_window_s, a_fps, cfg.poly_order)

            met[label] = extract_metrics(label, bs_raw, bs_s, arrival, mask,
                                         fi, time_axis, a_fps,
                                         tic_smooth=tic_s)

        results[ml] = {"blackspots_raw": bsr, "blackspots_smooth": bss,
                        "blackspots_pixels_smooth": bsp,
                        "fits": fits, "metrics": met}

        # Save per-method curves as .npy
        save_curves_npy(cfg.output_dir, cfg.vid_nr, ml,
                        time_axis, results[ml])

    del signal   # no longer needed -- free analysis video RAM

    # Save combined figure
    save_curves_figure(cfg.output_dir, cfg.vid_nr, time_axis,
                       results, palettes)

    # ---------- print metrics ----------
    fields = ["initial_blackspot", "final_blackspot",
              "final_blackspot_pixels", "total_reduction",
              "t_onset_s", "t_half_s", "t_90_s",
              "fit_k_half_s", "fit_n", "fit_B", "fit_B_ci95",
              "plateau", "plateau_lo", "plateau_hi",
              "median_pixel_arrival_s", "peak_tic_above_base"]

    # Build the metrics text, print it, and save it to a .txt file.
    metrics_lines: List[str] = []

    def _mprint(*args, **kwargs):
        """Print and simultaneously collect lines for the saved file."""
        line = " ".join(str(a) for a in args)
        print(line, **kwargs)
        metrics_lines.append(line)

    for ml, res in results.items():
        _mprint(f"\n==== {ml} -- black-spot metrics ====")
        header = f"{'ROI':<26} " + " ".join(f"{f:>14s}" for f in fields)
        _mprint(header)
        _mprint("-" * len(header))
        for label, m in res["metrics"].items():
            cells = [f"{m[f]:>14.3f}" if not np.isnan(m[f])
                     else f"{'nan':>14s}" for f in fields]
            _mprint(f"{label:<26} " + " ".join(cells))

        _mprint(f"\n  Hill fits:")
        for label, (fi, _, _) in res["fits"].items():
            if np.isnan(fi.get("k_hill", np.nan)):
                _mprint(f"  {label}: fit failed (curve did not drop enough)")
            else:
                ci    = fi.get('B_ci95', np.nan)
                ci_s  = f" \u00b1 {ci:.3f}" if not np.isnan(ci) else ""
                pl    = fi.get('plateau', np.nan)
                pl_lo = fi.get('plateau_lo', np.nan)
                pl_hi = fi.get('plateau_hi', np.nan)
                pl_s  = (f"  plateau: {pl*100:.1f}%"
                         + (f" [{pl_lo*100:.1f}\u2013{pl_hi*100:.1f}%]"
                            if not np.isnan(pl_lo) else "")
                         if not np.isnan(pl) else "")
                _mprint(f"  {label}: "
                        f"k_half={fi['k_hill']:.2f} s  "
                        f"n={fi['n_hill']:.2f}  "
                        f"B={fi['B']:.3f}{ci_s}  "
                        f"t_onset={fi['t_onset']:.2f} s"
                        f"{pl_s}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.output_dir / f"FIM{cfg.vid_nr}_metrics.txt"
    metrics_path.write_text("\n".join(metrics_lines), encoding="utf-8")
    print(f"[i] Metrics saved -> {metrics_path}")

    # ---------- save session (for standalone player) ----------
    save_session(cfg.output_dir, cfg.vid_nr, p_fps,
                 arrival, time_axis, roi_sets, results,
                 wound_mask, cfg)
    # The player loads video directly from the .mp4 file, so we can
    # free the in-memory video array (~7 GB) before opening the player.
    del video

    # ---------- curves figure + player ----------
    # OpenCV and matplotlib cannot safely share the GUI event loop on
    # Windows -- running both simultaneously causes a fatal GIL crash
    # (0xC0000409) when the user interacts with the matplotlib window.
    # The solution is to never have both event loops alive at the same
    # time: show the curves figure first (blocking), then open the player;
    # or show the curves after the player closes.
    if cfg.run_player:
        player = InteractivePlayer(
            video=cfg.video_path, fps=p_fps, cfg=cfg,
            roi_sets=roi_sets,
            wound_mask=wound_mask,
            arrival=arrival,
            time_axis=time_axis,
            results=results,
            palettes=palettes,
        )

        if cfg.save_overlay_video:
            dump_overlay_video(
                player,
                cfg.output_dir / f"FIM{cfg.vid_nr}_overlay.mp4",
                p_fps)

        if cfg.show_curves:
            # Show curves BEFORE the player so only one GUI event loop
            # runs at a time.  User closes this window, then the player
            # opens.
            print("[i] Close the curves window to open the video player.")
            _show_curves_window(cfg.output_dir, cfg.vid_nr,
                                blocking=True)

        print("\n[i] Player ready.")
        print("    SPACE=play  arrows/a/d=step  1/2=cycle method  "
              "h/b/r/w=overlays  s=snapshot  q=quit")
        player.run()

        # Show curves again after the player closes so the user can
        # inspect them at leisure without the player in the way.
        if cfg.show_curves:
            print("[i] Player closed. Showing curves -- close window to exit.")
            _show_curves_window(cfg.output_dir, cfg.vid_nr,
                                blocking=True)
    else:
        if cfg.show_curves:
            print("[i] Showing curves. Close the figure window to exit.")
            _show_curves_window(cfg.output_dir, cfg.vid_nr,
                                blocking=True)
        else:
            print("[i] Done. (run_player=False, show_curves=False)")


if __name__ == "__main__":
    main()