# ICG Fluorescence Angiography — Perfusion Analysis Pipeline

This repository contains the analysis pipeline developed for the objective quantification of mastectomy skin flap perfusion using intraoperative ICG fluorescence angiography, as part of the FLUID study at the Fluorescence Imaging Lab of Medisch Spectrum Twente (MST).

The pipeline consists of three sequential stages: preprocessing, automatic ROI selection, and dark spot analysis. Each stage produces output that feeds directly into the next. The breathing filter and wound mask scripts support both single-case and bulk processing. The ROI selection and dark spot analysis scripts are designed to be run one case at a time.

---

## Repository Structure

```
├── Breathingfilter.ipynb         # Preprocessing step 1: breathing motion filter
├── Mask_video.ipynb              # Preprocessing step 2: apply wound bed mask to video
├── Automatic_ROI_selection.py    # Stage 1: automatic ROI detection + interactive editing
├── Blackspot_analyse_v5.py       # Stage 2: dark spot analysis + interactive player
├── icg_player.py                 # Standalone interactive player (no reanalysis needed)
```

---

## Requirements

- Python 3.11 (all packages up to date as of March 2026)
- The following Python libraries must be installed:

```
numpy
opencv-python
scipy
scikit-image
matplotlib
tqdm
```

Install them with:

```bash
pip install numpy opencv-python scipy scikit-image matplotlib tqdm
```

The ROI selection script (`Automatic_ROI_selection.py`) uses an interactive matplotlib window and requires a display. It uses the `QtAgg` backend, so you will also need:

```bash
pip install PyQt5
```

---

## Input Data

Before running any script, prepare the following for each case:

1. **Raw ICG video** — recorded with the Karl Storz Rubina system, trimmed in Microsoft Clipchamp to start approximately 10 seconds before visual ICG inflow, and exported at 1280×720 pixels. Name the file `FIM{NR}_ICG.mp4` (e.g. `FIM018_ICG.mp4`).

2. **Wound bed mask** — a binary JPG image drawn manually in ImageJ on the white light image captured immediately before ICG administration. The wound bed region should be drawn in white on a black background. Name the file `FIM{NR}_mask.jpg`.

### Expected folder structure

```
M2-2 stage/
├── ICG_video's/                  # Raw trimmed videos
│   └── FIM018_ICG.mp4
├── ICG_video's_filtered/         # Output of breathing filter
├── Masked_videos/                # Output of mask application
├── Masks/                        # Wound bed masks
│   └── FIM018_mask.jpg
├── ROI_masks/
│   ├── No_peak/                  # Output of Technique 1 ROI selection
│   └── Relative_intensity/       # Output of Technique 2 ROI selection
└── Results/                      # Output of dark spot analysis
```

---

## Step-by-step Workflow

### Step 0 — Prepare the wound bed mask

Draw the wound bed region manually in ImageJ on the white light image from the recording. Export as a JPG and place it in the `Masks/` folder with the correct filename (`FIM{NR}_mask.jpg`).

---

### Step 1 — Apply the breathing motion filter

**Script:** `Breathingfilter.ipynb`

**What it does:** Detects and removes the periodic intensity oscillation caused by mechanical ventilation during surgery, which causes the breast to move within the frame and introduces rhythmic noise into the fluorescence signal. A second-order Butterworth bandstop filter is applied along the temporal axis of every pixel, targeting the dominant ventilator frequency detected via FFT (expected range: 0.18–0.35 Hz).

**How to run:**

Open the notebook in Jupyter. To process a **single video**, update the path in the single-case cell:

```python
input_video = "/path/to/FIM018_ICG.mp4"
output_dir  = "/path/to/ICG_video's_filtered/"
filter_breathing_fast(input_video, output_dir)
```

To **bulk-process** all videos in a folder at once, use the loop cell:

```python
input_folder  = "/path/to/ICG_video's/"
output_folder = "/path/to/ICG_video's_filtered/"
for file_path in Path(input_folder).glob("*.mp4"):
    print(f"Processing: {file_path.name}")
    filter_breathing_fast(str(file_path), output_folder)
```

**Output:** A filtered video saved as `FIM{NR}_ICG_filtered.mp4` in the output folder.

**Key parameter:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bw` | 0.05 Hz | Bandwidth of the bandstop filter around the detected breathing frequency |

---

### Step 2 — Apply the wound bed mask

**Script:** `Mask_video.ipynb`

**What it does:** Applies the wound bed mask to the filtered video by setting all wound bed pixels to zero in every frame. This ensures the open wound area is excluded from all subsequent analysis.

**How to run:**

Open the notebook in Jupyter. To process a **single video**, set the case number in the single-case cell:

```python
vid_nr = "018"
```

Verify the paths are correct and run the cell.

To **bulk-process** all available cases at once, run the loop cell, which automatically matches each filtered video to its corresponding mask file and skips cases where the mask is missing.

**Output:** A masked video saved as `FIM{NR}_ICG_filtered_masked.mp4` in the `Masked_videos/` folder.

---

### Step 3 — Automatic ROI selection

**Script:** `Automatic_ROI_selection.py`

**What it does:** Detects regions of potentially reduced perfusion using two independent techniques, followed by an interactive editing pipeline to refine the outputs.

- **Technique 1 (TIC shape classification):** Divides the frame into superpixels, computes a time-intensity curve (TIC) per superpixel, and classifies superpixels as hypoperfused if their TIC does not show the expected rise-then-decline pattern after ICG inflow.
- **Technique 2 (Relative intensity):** Averages a temporal window around a fixed timepoint after inflow and normalises each pixel relative to the brightest 5% of pixels. Pixels below a set threshold are labelled as hypoperfused.

Both techniques are followed by an **interactive three-phase editing pipeline**:

- **Phase 1 — Select ROIs:** The detected mask is split into connected components, each displayed as a coloured overlay. Click on a component to toggle it on or off. Close the window when done.
- **Phase 2 — Cut ROIs:** Draw a multipoint polyline across a component to split it; select which side to discard. Useful for removing frame-border artefacts that are spatially connected to a clinically relevant region. Right-click or press Enter to finish a line. Close the window when done.
- **Phase 3 — Add regions:** Draw a closed boundary near the wound bed and click inside to add back regions that were incorrectly excluded by the automated detection. Press Escape or close the window to skip.
- **Final check:** One last opportunity to remove any remaining unwanted components before saving.

**How to run:**

Set the case number at the top of the parameters block:

```python
VID_NR = "018"
```

Verify that the `VIDEO_PATH`, `MASK_PATH`, `OUT_DIR_T1`, and `OUT_DIR_T2` paths match your folder structure, then run the script from the command line:

```bash
python Automatic_ROI_selection.py
```

The script will run Technique 1, show a sample of random pixel TICs for visual verification, then run Technique 2, display a three-panel comparison of both outputs before editing, and then launch the interactive editing pipeline for each technique in sequence.

**Output:** For each technique, two files are saved:
- `FIM{NR}_mask_no_peak.npy` / `FIM{NR}_mask_relative_intensity.npy` — boolean NumPy array for use in Stage 2
- `FIM{NR}_mask_no_peak.png` / `FIM{NR}_mask_relative_intensity.png` — RGB overlay on the last video frame for visual inspection

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T1_N_SEGMENTS` | 10000 | Number of superpixels for SLIC segmentation |
| `T1_SMOOTH_SIGMA` | 100 frames | Gaussian smoothing sigma for TIC curves |
| `T1_BASELINE_FRAMES` | 30 | Frames used to estimate pre-inflow baseline |
| `T1_INFLOW_THRESHOLD` | 3 | Standard deviations above baseline to declare inflow |
| `T2_OFFSET_FRAMES` | 1500 | Frames after inflow at which relative intensity is sampled |
| `T2_WINDOW_HALF` | 10 | Half-window (frames) around the offset timepoint for temporal averaging |
| `T2_BRIGHT_PERCENTILE` | 95 | Percentile defining the bright reference population |
| `T2_CUTOFF_THRESHOLD` | 0.4 | Relative intensity below which a pixel is labelled hypoperfused |
| `T2_SPATIAL_SIGMA_1` | 2.0 | Spatial Gaussian sigma applied before thresholding (shot noise) |
| `T2_SPATIAL_SIGMA_2` | 5.0 | Spatial Gaussian sigma applied to the relative intensity map |

---

### Step 4 — Dark spot analysis

**Script:** `Blackspot_analyse_v5.py`

**What it does:** Quantifies the temporal dynamics of the dark spots within the ROIs defined in Step 3. For each ROI, it computes the black-spot decay curve S(t) — the fraction of ROI pixels not yet perfused at time t — fits a Hill model to the curve, and derives a set of clinical metrics. An interactive video player is launched automatically at the end for visual verification of the results.

At the end of the analysis, a session file is saved automatically. This session file contains all computed results and can be used to reopen the interactive player later without rerunning the full analysis (see Step 5).

The Hill model fitted to S(t) is:

```
S(t) = B + A / (1 + ((t - t0) / k)^n)
```

Where:
- `A` = drop amplitude (initial value minus plateau)
- `B` = residual plateau (fraction of ROI that never perfuses)
- `k` = half-perfusion time in seconds after onset
- `n` = Hill exponent (steepness of the perfusion front)
- `t0` = onset time

**How to run:**

Set the case number at the top of the Config block:

```python
vid_nr: str = "018"
```

Verify that `base_dir` points to your root folder, then run:

```bash
python Blackspot_analyse_v5.py
```

The script will compute the arrival map, generate the S(t) curves and Hill fits, print the metrics table to the console, save all outputs, and launch the interactive player.

**Output:** Saved to `Results/FIM{NR}_analysis/`:
- `FIM{NR}_curves_fraction.png` — S(t) curves as percentage of ROI area
- `FIM{NR}_curves_pixels.png` — S(t) curves as absolute pixel count
- `FIM{NR}_metrics.txt` — full metrics table and Hill fit summary
- `FIM{NR}_session.npz` — session file for reopening the player without recomputing

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `arrival_fraction` | 0.40 | Threshold as fraction of the 95th percentile of bright pixels |
| `stable_seconds` | 1.0 s | Minimum time a pixel must stay above threshold before arrival is recorded |
| `analysis_fps` | 10 fps | Frame rate used for arrival detection (downsampled from 30 fps) |
| `smooth_window_s` | 1.5 s | Savitzky-Golay smoothing window for S(t) curves |

---

### Step 5 — Standalone interactive player

**Script:** `icg_player.py`

**What it does:** Reopens the interactive player for a previously analysed case without rerunning the full pipeline. It loads the session file saved by `Blackspot_analyse_v5.py` and reads video frames directly from the original masked video file on demand. This is useful for reviewing results, inspecting specific timepoints, or comparing the outputs of both techniques after the analysis has already been completed.

**How to run:**

Either pass the session file path as a command-line argument:

```bash
python icg_player.py /path/to/Results/FIM018_analysis/FIM018_session.npz
```

Or set the `SESSION_PATH` variable at the top of the script and run without arguments:

```python
SESSION_PATH = "C:/path/to/Results/FIM018_analysis/FIM018_session.npz"
```

```bash
python icg_player.py
```

**Note:** The session file does not contain the video itself. The player reads frames directly from the original masked video file (`FIM{NR}_ICG_filtered_masked.mp4`). Make sure this file is still accessible at the path it was at when the analysis was run, or update the path inside the session file accordingly.

**Interactive player controls:**

| Key | Action |
|-----|--------|
| `SPACE` | Play / pause |
| `LEFT` / `RIGHT` or `A` / `D` | Step −1 / +1 frame |
| `UP` / `DOWN` | Step +10 / −10 frames |
| `+` / `−` | Speed up / slow down playback |
| `h` | Toggle arrival-time heatmap overlay |
| `b` | Toggle black-spot contour overlay |
| `p` | Toggle curve display: fraction (%) / absolute pixels |
| `1` / `2` | Cycle between Technique 1 and Technique 2 results |
| `r` | Toggle ROI boundaries |
| `w` | Toggle wound boundary |
| `s` | Save current composite frame as PNG |
| `0` | Jump to frame 0 |
| `Q` / `ESC` | Quit |

---

## Notes

- All parameters listed above were set through visual inspection of results across the available dataset. They are not formally optimised and may need adjustment when applied to new cases or a larger dataset.
- The interactive editing step in Stage 1 introduces a degree of observer dependence. Editing decisions should be documented per case.
- The breathing filter loads the full video into memory as a float32 array. For long recordings, this can require substantial RAM. A laptop with at least 16 GB of RAM is recommended.
- The dark spot analysis script uses a two-stage frame rate: 30 fps for the interactive player and 10 fps for arrival detection. This keeps memory usage manageable while maintaining temporal accuracy.

---

## Contact

Developed by D.A. Verhulst as part of the M2-2 internship, Technical Medicine, University of Twente, 2026.
Fluorescence Imaging Lab, Medisch Spectrum Twente.
