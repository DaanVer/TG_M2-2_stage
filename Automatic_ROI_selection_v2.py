# =============================================================
# ICG FLUORESCENCE ANGIOGRAPHY — PERFUSION ROI DETECTION
# =============================================================
# Two parallel techniques for detecting low-perfusion ROIs,
# followed by an interactive editing pipeline for both outputs.
#
# TECHNIQUE 1 — TIC shape classification
#   Computes per-superpixel time-intensity curves, smooths them,
#   and classifies each superpixel by the shape of its curve
#   (presence of a peak followed by a decline).
#
# TECHNIQUE 2 — Relative intensity at fixed timepoint
#   Averages a temporal window around inflow + offset, normalises
#   per pixel relative to the brightest 5% of the frame, then
#   thresholds to find dim (low-perfusion) regions.
#
# EDITING PIPELINE (runs on both technique outputs)
#   Phase 1 — Select which ROIs to keep
#   Phase 2 — Cut ROIs with multipoint polylines
#   Phase 3 — Add back regions with line + seed tool (per ROI)
#   Final   — Last-chance ROI removal before comparison plot
# =============================================================


# -------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter, gaussian_filter1d,
    binary_fill_holes, binary_opening, binary_closing,
    binary_dilation, binary_propagation
)
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.draw import line
from skimage.morphology import remove_small_objects, disk
from tqdm import tqdm


# =============================================================
# PARAMETERS  — edit these to tune behaviour
# =============================================================

# ---- Video / paths ------------------------------------------
VID_NR     = "013"
VIDEO_PATH = f"C:/Users/verhu/M2-2 stage/Masked_videos/FIM{VID_NR}_ICG_filtered_masked.mp4"
MASK_PATH  = f"C:/Users/verhu/M2-2 stage/Wound_masks/FIM{VID_NR}_mask.jpg"

# ---- Output folders -----------------------------------------
# Each mask is saved as a .npy (boolean array for further analysis)
# and a .png overlay (for visual inspection).
OUT_DIR_T1 = "C:/Users/verhu/M2-2 stage/ROI_masks/No_peak"
OUT_DIR_T2 = "C:/Users/verhu/M2-2 stage/ROI_masks/Relative_intensity"

# ---- Inspection ---------------------------------------------
INSPECT_N_PIXELS = 10   # random pixels shown after Technique 1

# ---- Technique 1 --------------------------------------------
T1_N_SEGMENTS        = 10000  # number of superpixels
T1_SMOOTH_SIGMA      = 100    # Gaussian sigma for TIC smoothing (frames)
T1_BASELINE_FRAMES   = 30     # frames used to estimate pre-inflow baseline
T1_INFLOW_THRESHOLD  = 3      # std-deviations above baseline to declare inflow

# ---- Technique 2 --------------------------------------------
T2_OFFSET_FRAMES     = 1500  # frames after inflow to sample (timepoint X)
T2_WINDOW_HALF       = 10    # ±frames around X used for temporal averaging
T2_BRIGHT_PERCENTILE = 95    # percentile defining "bright" reference population
T2_CUTOFF_THRESHOLD  = 0.4   # relative intensity below this → low perfusion
T2_SPATIAL_SIGMA_1   = 2.0   # Gaussian on the averaged frame (shot noise)
T2_SPATIAL_SIGMA_2   = 5.0   # Gaussian on the relative intensity map (edges)
T2_FINAL_SIGMA       = 5.0   # final mask edge softening

# ---- Region-adding tool (EnclosedRegionTool) ----------------
# The perfusion and wound boundaries are dilated by this many pixels
# before region growing, to prevent the seed from leaking back into
# the existing mask. Reduce toward 0 if the region needs to grow
# through a very narrow gap and stops too early.
REGION_BOUNDARY_DILATION = 1


# =============================================================
# LOADING
# =============================================================

def load_video(video_path):
    """Read an MP4 and return a float32 array of shape (T, H, W)."""
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    print("Video loaded")
    return np.stack(frames).astype(np.float32)


def load_mask(mask_path, shape):
    """
    Load a binary wound mask and resize it cleanly to the video frame size.

    JPEG compression leaves noisy intermediate grey values at edges.
    Strategy:
      1. Gaussian blur at full resolution — dissolves JPEG artifacts into
         smooth gradients without destroying thin mask features.
      2. Threshold → clean binary at full resolution.
      3. Minimal cleanup: fill internal holes and remove isolated noise
         blobs only. NO binary_opening/closing with large disks —
         those destroy thin parts of the mask.
      4. Resize the float image with bilinear interpolation (smooth edges).
      5. Light post-resize smooth + re-threshold.
    """
    mask = cv2.imread(mask_path, 0).astype(np.float32) / 255.0

    # Gaussian blur removes JPEG artifacts without destroying thin regions
    mask      = gaussian_filter(mask, sigma=3)
    mask_bool = mask > 0.5

    # fill holes and remove tiny isolated noise blobs only —
    # remove_small_objects acts on connected components so thin features
    # within the main mask are never removed
    mask_bool = binary_fill_holes(mask_bool)
    mask_bool = remove_small_objects(mask_bool, min_size=100)

    # resize as float, not binary — bilinear gives smooth sub-pixel edges
    mask_float = cv2.resize(mask_bool.astype(np.float32),
                            (shape[1], shape[0]),
                            interpolation=cv2.INTER_LINEAR)

    # light post-resize smooth then re-threshold
    mask_float = gaussian_filter(mask_float, sigma=2)

    print("Mask loaded")
    return mask_float > 0.5


# =============================================================
# SHARED UTILITIES
# =============================================================

def apply_wound_exclusion(mask, wound_mask):
    """Remove wound pixels from a boolean mask."""
    wound_resized = cv2.resize(
        wound_mask.astype(np.uint8),
        (mask.shape[1], mask.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    return mask & (~wound_resized)


def clean_mask(mask):
    """
    Standard morphological cleanup:
    fill holes → remove small objects → close gaps → smooth borders.
    """
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=200)
    mask = binary_closing(mask, disk(3))
    mask = binary_opening(mask, disk(2))
    return mask


# =============================================================
# TECHNIQUE 1 — TIC SHAPE CLASSIFICATION
# =============================================================

def detect_inflow_frame(video,
                        baseline=T1_BASELINE_FRAMES,
                        threshold=T1_INFLOW_THRESHOLD):
    """
    Find the first frame where mean intensity exceeds
    baseline + threshold * sigma (detects contrast agent arrival).
    """
    curve = video.mean(axis=(1, 2))
    mu    = curve[:baseline].mean()
    sigma = curve[:baseline].std()
    for i in range(baseline, len(curve)):
        if curve[i] > mu + threshold * sigma:
            return i
    return baseline


def get_superpixels(video, n_segments=T1_N_SEGMENTS):
    """Compute SLIC superpixel labels on the temporal mean frame."""
    mean_frame = video.mean(axis=0)
    return slic(
        img_as_float(mean_frame),
        n_segments=n_segments,
        compactness=10,
        start_label=0,
        channel_axis=None
    )


def compute_tics(video, segments):
    """Extract a mean time-intensity curve (TIC) per superpixel."""
    T = video.shape[0]
    n = segments.max() + 1
    tics = np.zeros((n, T), dtype=np.float32)
    for i in tqdm(range(n), desc="TIC extraction"):
        mask    = (segments == i)
        tics[i] = video[:, mask].mean(axis=1)
    return tics


def smooth_tics(tics, sigma=T1_SMOOTH_SIGMA):
    """Apply a 1-D Gaussian smooth along the time axis of all TICs."""
    return gaussian_filter1d(tics, sigma=sigma, axis=1)


def classify_tics(tics_smooth, inflow):
    """
    Label each superpixel TIC as perfused (True) or low-perfusion (False).
    A TIC is perfused when:
      - the peak rises > 10% above baseline, and
      - intensity declines after the peak (negative post-peak slope).
    """
    n      = tics_smooth.shape[0]
    labels = np.zeros(n, dtype=bool)
    for i in tqdm(range(n), desc="Classifying TIC shapes"):
        tic      = tics_smooth[i]
        baseline = tic[:inflow].mean()
        peak     = np.argmax(tic)
        rise     = tic[peak] - baseline
        if rise < 0.1 * baseline:
            continue
        if peak < len(tic) - 30:
            post_slope = np.mean(np.diff(tic[peak:peak + 200]))
            labels[i]  = post_slope <= 0
    return labels


def build_mask(segments, labels):
    """Build a boolean mask of low-perfusion pixels from superpixel labels."""
    mask = np.zeros_like(segments, dtype=bool)
    for i, perfused in enumerate(labels):
        if not perfused:
            mask[segments == i] = True
    return mask


def upscale(mask, shape):
    """Upscale a boolean mask to target shape using Gaussian-blurred kron."""
    zf = (shape[0] / mask.shape[0], shape[1] / mask.shape[1])
    up = gaussian_filter(
        np.kron(mask.astype(float),
                np.ones((int(zf[0]), int(zf[1])))),
        sigma=2
    )
    return up > 0.5


def technique1_pipeline(video, wound_mask):
    """
    Full Technique 1 pipeline:
    superpixels → TICs → inflow detection → smoothing
    → classification → mask → upscale → clean → wound exclusion.

    Returns mask (H, W bool), inflow frame index.
    """
    print("Video shape:", video.shape)
    segments    = get_superpixels(video)
    tics        = compute_tics(video, segments)
    inflow      = detect_inflow_frame(video)
    tics_smooth = smooth_tics(tics)
    labels      = classify_tics(tics_smooth, inflow)
    mask        = build_mask(segments, labels)
    mask        = upscale(mask, video.shape[1:])
    mask        = clean_mask(mask)
    mask        = gaussian_filter(mask.astype(float), sigma=10) > 0.5
    mask        = apply_wound_exclusion(mask, wound_mask)
    return mask, inflow


# =============================================================
# TECHNIQUE 1 — INSPECTION HELPERS
# =============================================================

def classify_single_tic(tic, inflow):
    """Classify one smoothed TIC; returns (label_string, peak_index)."""
    baseline = tic[:inflow].mean()
    peak     = np.argmax(tic)
    rise     = tic[peak] - baseline
    if rise < 0.1 * baseline:
        return "no_rise", peak
    if peak < len(tic) - 30:
        post_slope = np.mean(np.diff(tic[peak:peak + 200]))
        return ("peak" if post_slope <= 0 else "still_rising"), peak
    return "still_rising", peak


def plot_pixel(video, y, x, inflow):
    """Plot raw and smoothed TIC for pixel (y, x) with classification."""
    tic_raw     = video[:, y, x]
    tic_smooth  = gaussian_filter1d(tic_raw, sigma=T1_SMOOTH_SIGMA)
    label, peak = classify_single_tic(tic_smooth, inflow)

    plt.figure(figsize=(10, 4))
    plt.plot(tic_raw,    alpha=0.25, label="Raw")
    plt.plot(tic_smooth, linewidth=3, label="Smoothed")
    plt.axvline(inflow, linestyle="--", label="Inflow")
    plt.scatter(peak, tic_smooth[peak], color="red", label="Peak")
    plt.title(f"Pixel ({y},{x}) → {label}")
    plt.xlabel("Frame")
    plt.ylabel("Intensity")
    plt.legend()
    plt.draw()
    plt.pause(0.5)


def inspect_random(video, inflow, n=INSPECT_N_PIXELS):
    """Plot TICs for n randomly sampled pixels (away from borders)."""
    H, W = video.shape[1:]
    for _ in range(n):
        y = np.random.randint(20, H - 20)
        x = np.random.randint(20, W - 20)
        plot_pixel(video, y, x, inflow)


# =============================================================
# TECHNIQUE 2 — RELATIVE INTENSITY AT FIXED TIMEPOINT
# =============================================================

def technique2_pipeline(video, wound_mask, inflow,
                        offset_frames=T2_OFFSET_FRAMES,
                        window_half=T2_WINDOW_HALF,
                        bright_pct=T2_BRIGHT_PERCENTILE,
                        cutoff=T2_CUTOFF_THRESHOLD,
                        spatial_sigma_1=T2_SPATIAL_SIGMA_1,
                        spatial_sigma_2=T2_SPATIAL_SIGMA_2,
                        final_sigma=T2_FINAL_SIGMA):
    """
    Full Technique 2 pipeline:
      1. Average frames in window [inflow + offset ± window_half]
      2. Spatial Gaussian denoise
      3. Normalise each pixel relative to the mean of the top
         (100 - bright_pct)% brightest pixels
      4. Spatial smooth of the relative map
      5. Threshold: pixels below cutoff → low perfusion
      6. Clean → wound exclusion (applied last so clean_mask cannot
         reconnect regions across the wound gap)

    Returns mask (H, W bool), smoothed relative intensity map (H, W float).
    """
    T = video.shape[0]

    # temporal window average around inflow + offset
    centre  = inflow + offset_frames
    t_start = max(0, centre - window_half)
    t_end   = min(T, centre + window_half + 1)
    print(f"[T2] Averaging frames {t_start}–{t_end - 1} (centre {centre})")

    averaged = video[t_start:t_end].mean(axis=0)
    smoothed = gaussian_filter(averaged, sigma=spatial_sigma_1)

    # reference max: mean of top (100-bright_pct)% pixels
    pct_val = np.percentile(smoothed, bright_pct)
    ref_max = smoothed[smoothed >= pct_val].mean()
    print(f"[T2] Reference max (mean of top "
          f"{100 - bright_pct:.0f}% pixels): {ref_max:.2f}")

    # relative intensity map
    rel_map        = np.clip(smoothed / ref_max, 0.0, 1.0)
    rel_map_smooth = gaussian_filter(rel_map, sigma=spatial_sigma_2)

    # threshold → clean → wound exclusion last
    mask = rel_map_smooth < cutoff
    mask = clean_mask(mask)
    mask = gaussian_filter(mask.astype(float), sigma=final_sigma) > 0.5
    mask = apply_wound_exclusion(mask, wound_mask)

    return mask, rel_map_smooth


# =============================================================
# ROI SPLITTING
# =============================================================

def split_and_visualise_rois(mask, video, min_size=50, connectivity=3):
    """
    Label connected components of a boolean mask and show their
    outlines on the mid-frame. Returns dict {name: bool mask}.
    """
    structure  = np.ones((connectivity, connectivity), dtype=bool)
    labeled, n = ndimage.label(mask.astype(bool), structure=structure)

    roi_masks = {}
    kept_id   = 1
    for i in range(1, n + 1):
        roi = labeled == i
        if roi.sum() < min_size:
            continue
        roi_masks[f"ROI_{kept_id}"] = roi.astype(bool)
        kept_id += 1

    print(f"Detected {len(roi_masks)} ROI(s) (raw: {n})")

    frame = video[len(video) // 2]
    plt.figure(figsize=(8, 8))
    plt.imshow(frame, cmap="gray")
    for name, roi in roi_masks.items():
        contour = ndimage.binary_dilation(roi) ^ roi
        y, x    = np.where(contour)
        plt.scatter(x, y, s=1)
        cy, cx  = np.column_stack(np.where(roi)).mean(axis=0)
        plt.text(cx, cy, name, color="yellow", fontsize=10, ha="center")
    plt.title("Detected ROIs")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.5)

    return roi_masks


# =============================================================
# REGION-ADDING TOOL  (Phase 3 of the editing pipeline)
# =============================================================

class EnclosedRegionTool:
    """
    Interactive tool to add back a region between the perfusion mask
    and the wound mask.

    Usage
    -----
    Step 1 — click 4 points: p1 (perf edge) → w1 (wound edge) →
             p2 (perf edge) → w2 (wound edge).
             These define two barrier lines that close off the region.
    ENTER  → switch to seed mode  (or ENTER / ESC with 0 points to skip)
    Step 2 — click one or more seeds inside the region to add.
    ENTER  → grow seeds and add region.
    ESC    → skip this ROI entirely at any stage.
    BACKSPACE → undo last point / seed.
    """

    def __init__(self, image, perf_region, wound_mask):
        self.image  = image
        self.perf   = perf_region.astype(bool)
        self.wound  = wound_mask.astype(bool)
        self.points = []
        self.seeds  = []
        self.region = None
        self.stage  = "lines"

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(image, cmap="gray")
        self.ax.contour(self.perf,  colors="lime", linewidths=1)
        self.ax.contour(self.wound, colors="red",  linewidths=1)
        self.ax.set_title(
            "Step 1: 4 clicks  (p1 → w1 → p2 → w2)\n"
            "ENTER → seed mode  |  ESC or ENTER with 0 points → skip"
        )
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("key_press_event",    self.onkey)
        plt.show()

    def onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)

        if self.stage == "lines":
            if len(self.points) >= 4:
                return
            self.points.append((x, y))
            self.ax.plot(x, y, "bo")
            if len(self.points) == 2:
                self._draw_line(self.points[0], self.points[1])
            elif len(self.points) == 4:
                self._draw_line(self.points[2], self.points[3])
            self.fig.canvas.draw()

        elif self.stage == "seed":
            self.seeds.append((y, x))
            self.ax.plot(x, y, "go")
            self.fig.canvas.draw()
            print(f"Seed added ({len(self.seeds)})")

    def _draw_line(self, pA, pB):
        self.ax.plot([pA[0], pB[0]], [pA[1], pB[1]], "cyan")

    def onkey(self, event):
        if event.key == "escape":
            # skip this ROI entirely
            self.region = None
            plt.close(self.fig)
            return

        if event.key == "backspace":
            if self.stage == "lines" and self.points:
                self.points.pop()
                self._redraw()
            elif self.stage == "seed" and self.seeds:
                self.seeds.pop()
                self._redraw()

        elif event.key == "enter":
            if self.stage == "lines":
                if len(self.points) == 0:
                    # no points placed — skip
                    self.region = None
                    plt.close(self.fig)
                elif len(self.points) == 4:
                    self.stage = "seed"
                    self.ax.set_title(
                        "Step 2: click seeds (multiple allowed)\n"
                        "ENTER → finish  |  BACKSPACE → undo  |  ESC → skip"
                    )
                    self.fig.canvas.draw()
            elif self.stage == "seed":
                if not self.seeds:
                    self.region = None
                    plt.close(self.fig)
                    return
                self._build_region()
                plt.close(self.fig)

    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.image, cmap="gray")
        self.ax.contour(self.perf,  colors="lime")
        self.ax.contour(self.wound, colors="red")
        for p in self.points:
            self.ax.plot(p[0], p[1], "bo")
        for s in self.seeds:
            self.ax.plot(s[1], s[0], "go")
        if len(self.points) >= 2:
            self._draw_line(self.points[0], self.points[1])
        if len(self.points) == 4:
            self._draw_line(self.points[2], self.points[3])
        self.fig.canvas.draw()

    def _build_region(self):
        p1, w1, p2, w2 = self.points

        # rasterise the two barrier lines
        barrier = np.zeros_like(self.perf, dtype=bool)
        for (r0, c0), (r1, c1) in [
            ((p1[1], p1[0]), (w1[1], w1[0])),
            ((p2[1], p2[0]), (w2[1], w2[0]))
        ]:
            rr, cc = line(r0, c0, r1, c1)
            barrier[rr, cc] = True

        # thicken barrier and existing boundaries.
        # REGION_BOUNDARY_DILATION controls how much the perf/wound edges
        # are expanded before growing — reduce it if growing stops too early
        # because the gap to fill is narrower than 2 × dilation pixels.
        barrier  = binary_dilation(barrier, iterations=1)
        barrier |= binary_dilation(self.perf,  iterations=REGION_BOUNDARY_DILATION)
        barrier |= binary_dilation(self.wound, iterations=REGION_BOUNDARY_DILATION)

        # flood-fill from all seeds within allowed area
        seed_mask = np.zeros_like(self.perf, dtype=bool)
        for sy, sx in self.seeds:
            seed_mask[sy, sx] = True
        grown = binary_propagation(seed_mask, mask=~barrier)

        # exclude existing perfusion and wound regions; dilate to close seam
        raw_region  = grown & (~self.perf) & (~self.wound)
        self.region = binary_dilation(raw_region, iterations=1)

        plt.figure(figsize=(6, 6))
        plt.imshow(self.image, cmap="gray")
        plt.contour(self.perf,       colors="lime")
        plt.contour(self.wound,      colors="red")
        plt.contour(self.region,     colors="yellow")
        plt.title(f"Added region ({len(self.seeds)} seed(s))")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.5)
        print("Region added")


def annotate_regions(image, perfusion_mask, wound_mask, roi_masks):
    """
    Run EnclosedRegionTool for each ROI in roi_masks.
    Returns a list of boolean region masks to add back.
    """
    outputs = []
    for i, (name, region) in enumerate(roi_masks.items()):
        print(f"\nAnnotating {name} ({i + 1}/{len(roi_masks)})")
        tool = EnclosedRegionTool(image, region, wound_mask)
        if tool.region is not None:
            outputs.append(tool.region)
    return outputs


# =============================================================
# ROI EDITOR — PHASE 1: SELECTOR
# =============================================================

class ROISelectorTool:
    """
    Toggle which ROIs to keep before further editing.

    Controls
    --------
    Left-click : toggle ROI between KEEP (bright) and SKIP (dim)
    ENTER      : confirm selection
    """

    def __init__(self, image, roi_masks, wound_mask):
        self.image     = image
        self.roi_names = list(roi_masks.keys())
        self.roi_masks = list(roi_masks.values())
        self.wound     = wound_mask.astype(bool)
        self.n         = len(self.roi_masks)
        self.kept      = [True] * self.n

        cmap        = cm.get_cmap("tab20", max(self.n, 1))
        self.colors = [cmap(i)[:3] for i in range(self.n)]

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        self.fig.canvas.mpl_connect("key_press_event",    self._onkey)
        self._draw()
        plt.show()

    def _draw(self):
        self.ax.clear()
        self.ax.imshow(self.image, cmap="gray")
        for i, mask in enumerate(self.roi_masks):
            r, g, b = self.colors[i]
            alpha   = 0.55 if self.kept[i] else 0.12
            overlay = np.zeros((*mask.shape, 4), dtype=float)
            overlay[mask] = [r, g, b, alpha]
            self.ax.imshow(overlay)
            coords = np.column_stack(np.where(mask))
            if coords.size > 0:
                cy, cx = coords.mean(axis=0)
                self.ax.text(
                    cx, cy,
                    f"{self.roi_names[i]}\n{'KEEP' if self.kept[i] else 'SKIP'}",
                    color="white", fontsize=8, ha="center", va="center",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="black", alpha=0.4)
                )
        # always show wound boundary for orientation
        self.ax.contour(self.wound, colors="red", linewidths=1.5)
        self.ax.set_title(
            f"Phase 1 — ROI Selection  ({sum(self.kept)}/{self.n} kept)\n"
            "Left-click to toggle  |  ENTER to confirm  |  red = wound boundary",
            fontsize=11
        )
        self.ax.axis("off")
        self.fig.canvas.draw()

    def _find_roi(self, x, y):
        """Return topmost ROI index at pixel (y, x), or None."""
        H, W = self.roi_masks[0].shape
        if not (0 <= y < H and 0 <= x < W):
            return None
        for i in range(self.n - 1, -1, -1):
            if self.roi_masks[i][y, x]:
                return i
        return None

    def _onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return
        i = self._find_roi(int(event.xdata), int(event.ydata))
        if i is not None:
            self.kept[i] = not self.kept[i]
            self._draw()

    def _onkey(self, event):
        if event.key == "enter":
            plt.close(self.fig)

    def get_kept_masks(self):
        return {self.roi_names[i]: self.roi_masks[i]
                for i in range(self.n) if self.kept[i]}


# =============================================================
# ROI EDITOR — PHASE 2: CUTTER
# =============================================================

class ROICutterTool:
    """
    Draw multipoint polylines to cut an ROI into pieces, then
    click the piece to remove. Repeat for multiple cuts.

    Controls
    --------
    Left-click  : add vertex to current polyline
    Right-click : finish polyline and split ROI
                  (line must fully cross the ROI)
    Left-click  : (after split) click the region to REMOVE
    BACKSPACE   : undo last vertex (drawing) / redo line (exclude)
    ENTER       : done with this ROI (drawing stage only)
    """

    def __init__(self, image, roi_mask, roi_name, wound_mask):
        self.image        = image
        self.current_mask = roi_mask.copy().astype(bool)
        self.roi_name     = roi_name
        self.wound        = wound_mask.astype(bool)
        self.vertices     = []
        self.labeled      = None
        self.stage        = "drawing"

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        self.fig.canvas.mpl_connect("key_press_event",    self._onkey)
        self._draw()
        plt.show()

    def _draw(self):
        self.ax.clear()
        self.ax.imshow(self.image, cmap="gray")

        overlay = np.zeros((*self.current_mask.shape, 4), dtype=float)
        overlay[self.current_mask] = [1.0, 0.3, 0.3, 0.45]
        self.ax.imshow(overlay)

        # wound boundary always visible as reference
        self.ax.contour(self.wound, colors="red", linewidths=1.5)

        if len(self.vertices) >= 2:
            xs = [v[1] for v in self.vertices]
            ys = [v[0] for v in self.vertices]
            self.ax.plot(xs, ys, color="cyan", linewidth=2)
        if self.vertices:
            self.ax.scatter([v[1] for v in self.vertices],
                            [v[0] for v in self.vertices],
                            color="cyan", s=30, zorder=5)

        if self.stage == "drawing":
            self.ax.set_title(
                f"Phase 2 — Cutting [{self.roi_name}]  "
                f"({len(self.vertices)} vertex/vertices)\n"
                "L-click: add vertex  |  R-click: finish line  |  "
                "BACKSPACE: undo  |  ENTER: done  |  red = wound boundary",
                fontsize=10
            )
        else:
            self.ax.set_title(
                f"Phase 2 — Cutting [{self.roi_name}]\n"
                "Click the region to REMOVE  |  BACKSPACE: redo line",
                fontsize=10
            )
        self.ax.axis("off")
        self.fig.canvas.draw()

    def _onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)

        if self.stage == "drawing":
            if event.button == 1:
                self.vertices.append((y, x))
                self._draw()
            elif event.button == 3:
                if len(self.vertices) < 2:
                    print("[Cutter] Need at least 2 vertices.")
                    return
                self._split()

        elif self.stage == "exclude" and event.button == 1:
            self._apply_exclusion(y, x)

    def _split(self):
        """Rasterise polyline as a barrier and label connected components."""
        barrier = np.zeros_like(self.current_mask, dtype=bool)
        for i in range(len(self.vertices) - 1):
            r0, c0 = self.vertices[i]
            r1, c1 = self.vertices[i + 1]
            rr, cc = line(r0, c0, r1, c1)
            valid  = ((rr >= 0) & (rr < barrier.shape[0]) &
                      (cc >= 0) & (cc < barrier.shape[1]))
            barrier[rr[valid], cc[valid]] = True

        barrier = binary_dilation(barrier, iterations=2)
        labeled, n_comp = ndimage.label(self.current_mask & ~barrier)

        if n_comp < 2:
            print("[Cutter] Line does not fully divide the ROI — "
                  "extend it to cross both edges and try again.")
            self.vertices = []
            self.stage    = "drawing"
            self._draw()
            return

        self.labeled = labeled

        # show each component in a distinct colour
        self.ax.clear()
        self.ax.imshow(self.image, cmap="gray")
        cmap_c = cm.get_cmap("Set1", n_comp)
        for c in range(1, n_comp + 1):
            comp     = labeled == c
            r, g, b  = cmap_c(c - 1)[:3]
            ov       = np.zeros((*comp.shape, 4), dtype=float)
            ov[comp] = [r, g, b, 0.55]
            self.ax.imshow(ov)

        self.ax.plot([v[1] for v in self.vertices],
                     [v[0] for v in self.vertices],
                     color="white", linewidth=2)
        self.ax.contour(self.wound, colors="red", linewidths=1.5)
        self.ax.set_title(
            f"Phase 2 — Cutting [{self.roi_name}]  ({n_comp} regions)\n"
            "Click the region to REMOVE  |  BACKSPACE: redo line  "
            "|  red = wound boundary",
            fontsize=10
        )
        self.ax.axis("off")
        self.fig.canvas.draw()
        self.stage = "exclude"

    def _apply_exclusion(self, y, x):
        H, W = self.labeled.shape
        if not (0 <= y < H and 0 <= x < W):
            print("[Cutter] Click outside image bounds.")
            return
        comp_label = self.labeled[y, x]
        if comp_label == 0:
            print("[Cutter] Clicked on barrier — try again.")
            return

        self.current_mask &= ~(self.labeled == comp_label)
        self.vertices = []
        self.labeled  = None
        self.stage    = "drawing"
        print("[Cutter] Region removed. Draw another line or ENTER to finish.")
        self._draw()

    def _onkey(self, event):
        if event.key == "enter" and self.stage == "drawing":
            plt.close(self.fig)
        elif event.key == "backspace":
            if self.stage == "drawing" and self.vertices:
                self.vertices.pop()
                self._draw()
            elif self.stage == "exclude":
                self.stage   = "drawing"
                self.labeled = None
                self._draw()

    def get_mask(self):
        return self.current_mask


# =============================================================
# EDITING PIPELINE  (runs on both technique outputs)
# =============================================================

def edit_rois(mask, video, wound_mask, technique_name):
    """
    Three-phase interactive editing pipeline:
      Phase 1 — Select which ROIs to keep  (ROISelectorTool)
      Phase 2 — Cut ROIs with polylines    (ROICutterTool)
      Phase 3 — Add back excluded regions  (EnclosedRegionTool, per ROI)

    Returns the final edited boolean mask.
    """
    mid_frame = video[len(video) // 2]

    print(f"\n{'='*55}")
    print(f"  Editing: {technique_name}")
    print(f"{'='*55}")

    roi_masks = split_and_visualise_rois(mask, video)
    if not roi_masks:
        print("No ROIs detected — returning original mask.")
        return mask

    # ---- Phase 1: select ----
    print(f"\n[{technique_name}] Phase 1: ROI selection")
    selector   = ROISelectorTool(mid_frame, roi_masks, wound_mask)
    kept_masks = selector.get_kept_masks()
    print(f"  → {len(kept_masks)}/{len(roi_masks)} ROI(s) kept.")

    if not kept_masks:
        print("  → No ROIs kept.")
        return np.zeros_like(mask, dtype=bool)

    # ---- Phase 2: cut ----
    print(f"\n[{technique_name}] Phase 2: ROI cutting")
    cut_masks = {}
    for name, roi_mask in kept_masks.items():
        print(f"  Cutting {name} ...")
        cutter          = ROICutterTool(mid_frame, roi_mask, name, wound_mask)
        cut_masks[name] = cutter.get_mask()

    # combine and re-apply wound exclusion: dilation in the cutter
    # can push pixels back into wound territory
    combined = np.zeros_like(mask, dtype=bool)
    for m in cut_masks.values():
        combined |= m
    combined = apply_wound_exclusion(combined, wound_mask)

    # ---- Phase 3: add back regions, one ROI at a time ----
    # Each ROI is shown separately so its boundary is unambiguous
    # and never visually connected through the wound area.
    print(f"\n[{technique_name}] Phase 3: Add regions back (per ROI)")
    print("  (ESC or ENTER with 0 points to skip an ROI)")
    total_added = 0
    for name, roi_mask in cut_masks.items():
        print(f"  Adding back near {name} ...")
        added = annotate_regions(
            image          = mid_frame,
            perfusion_mask = roi_mask,
            wound_mask     = wound_mask,
            roi_masks      = {name: roi_mask},
        )
        for region in added:
            combined |= region
        total_added += len(added)

    # final wound exclusion: EnclosedRegionTool dilation can bleed
    # into wound pixels near the boundary
    combined = apply_wound_exclusion(combined, wound_mask)
    print(f"  → {total_added} region(s) added back in total.")

    return combined


# =============================================================
# FINAL ROI REMOVAL  (last chance before comparison plot)
# =============================================================

def final_roi_removal(mask, video, wound_mask, technique_name):
    """
    Re-split the edited mask and open ROISelectorTool one last time
    so any unwanted ROIs can be removed before the final comparison.
    Wound exclusion is re-applied after removal.
    """
    print(f"\n[{technique_name}] Final ROI check — remove any unwanted ROIs")
    roi_masks = split_and_visualise_rois(mask, video)
    if not roi_masks:
        return mask

    selector = ROISelectorTool(video[len(video) // 2], roi_masks, wound_mask)
    kept     = selector.get_kept_masks()

    result = np.zeros_like(mask, dtype=bool)
    for m in kept.values():
        result |= m

    result = apply_wound_exclusion(result, wound_mask)
    print(f"  → {len(kept)}/{len(roi_masks)} ROI(s) kept after final check.")
    return result


# =============================================================
# VISUALISATION
# =============================================================

def plot_raw_masks(video, mask1, mask2, rel_map,
                   inflow, offset_frames=T2_OFFSET_FRAMES,
                   cutoff=T2_CUTOFF_THRESHOLD):
    """
    3-panel comparison of both technique outputs before editing,
    plus the Technique 2 relative intensity map.
    """
    frame = video[-1]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].imshow(frame, cmap="gray")
    axes[0].imshow(mask1, alpha=0.4, cmap="Reds")
    axes[0].set_title("Technique 1 — TIC shape", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(frame, cmap="gray")
    axes[1].imshow(mask2, alpha=0.4, cmap="Blues")
    axes[1].set_title(
        f"Technique 2 — Relative intensity\n"
        f"(inflow + {offset_frames} frames,  cutoff = {cutoff})",
        fontsize=12
    )
    axes[1].axis("off")

    im = axes[2].imshow(rel_map, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Technique 2 — Relative intensity map", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04,
                 label="Relative intensity")

    plt.suptitle("Raw technique outputs (before editing)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


def plot_edited_masks(video, mask1_orig, mask1_ed, mask2_orig, mask2_ed):
    """
    2×2 grid: original (top) vs edited (bottom) for each technique.
    """
    frame = video[-1]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    configs = [
        (axes[0, 0], mask1_orig, "Reds",  "Technique 1  [original]"),
        (axes[0, 1], mask2_orig, "Blues", "Technique 2  [original]"),
        (axes[1, 0], mask1_ed,   "Reds",  "Technique 1  [edited]"),
        (axes[1, 1], mask2_ed,   "Blues", "Technique 2  [edited]"),
    ]
    for ax, m, cmap_name, title in configs:
        ax.imshow(frame, cmap="gray")
        ax.imshow(m, alpha=0.45, cmap=cmap_name)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.suptitle("ROI Detection — Before vs After Editing",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


# =============================================================
# SAVE MASKS
# =============================================================

def save_mask(mask, video, out_dir, filename):
    """
    Save a final boolean mask to out_dir under the given filename.

    Two files are written:
      <filename>.npy  — boolean numpy array, ready for further analysis
      <filename>.png  — RGB overlay of the mask on the last video frame,
                        for quick visual inspection

    The output folder is created automatically if it does not exist.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    npy_path = os.path.join(out_dir, f"{filename}.npy")
    png_path = os.path.join(out_dir, f"{filename}.png")

    # save boolean array
    np.save(npy_path, mask)

    # build an RGB overlay on the last frame and save as PNG
    frame = video[-1]
    frame_norm = ((frame - frame.min()) /
                  (frame.max() - frame.min()) * 255).astype(np.uint8)
    frame_rgb  = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)

    # red overlay where mask is True
    overlay             = frame_rgb.copy()
    overlay[mask, 2]    = np.clip(overlay[mask, 2].astype(int) + 120, 0, 255).astype(np.uint8)
    blended             = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)

    cv2.imwrite(png_path, blended)

    print(f"  Saved: {npy_path}")
    print(f"  Saved: {png_path}")


# =============================================================
# MAIN
# =============================================================

# ---- Load ----
video      = load_video(VIDEO_PATH)
wound_mask = load_mask(MASK_PATH, video.shape[1:])

# ---- Technique 1 ----
print("\nRunning Technique 1...")
mask1, inflow = technique1_pipeline(video, wound_mask)
print(f"Inflow frame: {inflow}")

# Sample random pixels to verify TIC classification visually
inspect_random(video, inflow, n=INSPECT_N_PIXELS)

# ---- Technique 2 ----
print("\nRunning Technique 2...")
mask2, rel_map = technique2_pipeline(video, wound_mask, inflow)

# ---- Pre-edit comparison ----
plot_raw_masks(video, mask1, mask2, rel_map, inflow)

# ---- Interactive editing ----
mask1_edited = edit_rois(mask1, video, wound_mask, "Technique 1 — TIC shape")
mask2_edited = edit_rois(mask2, video, wound_mask, "Technique 2 — Relative intensity")

# ---- Final ROI removal (before comparison) ----
mask1_final = final_roi_removal(mask1_edited, video, wound_mask, "Technique 1 — TIC shape")
mask2_final = final_roi_removal(mask2_edited, video, wound_mask, "Technique 2 — Relative intensity")

# ---- Final comparison ----
plot_edited_masks(video, mask1, mask1_final, mask2, mask2_final)

# ---- Save final masks ----
print("\nSaving masks...")
save_mask(mask1_final, video,
          out_dir  = OUT_DIR_T1,
          filename = f"FIM{VID_NR}_mask_no_peak")
save_mask(mask2_final, video,
          out_dir  = OUT_DIR_T2,
          filename = f"FIM{VID_NR}_mask_relative_intensity")

# ---- Save raw masks (before editing, for reference) ----
save_mask(mask1, video,
          out_dir  = OUT_DIR_T1,
          filename = f"FIM{VID_NR}_mask_no_peak_raw")
save_mask(mask2, video,
          out_dir  = OUT_DIR_T2,
          filename = f"FIM{VID_NR}_mask_relative_intensity_raw")
print("Done.")

plt.show()