# neon-screen-gaze

A Python pipeline for mapping Neon eye tracker gaze data onto a screen surface using AprilTag markers, with improved detection stability over the built-in Neon Player surface tracker.

---

## The Problem

The Neon eye tracker uses AprilTag markers placed at the corners of a display to estimate where on the screen a participant is looking. The built-in Neon Player surface tracker detects these markers with a 2x image downsampling setting (`quad_decimate=2.0`), which makes detection unstable when markers are small or slightly blurry. In some frames, the estimated surface can jump wildly, producing gaze coordinates that are completely wrong even though the recording looks fine visually.

## The Solution

This pipeline replaces the Neon Player surface tracker with a custom Python implementation that:

- Runs AprilTag detection at **full image resolution** (`quad_decimate=1.0`)
- **Rejects frames** where fewer than 3 of 4 markers are visible
- Applies **temporal median filtering** to smooth marker corner positions over time
- Performs **outlier rejection** based on surface area and aspect ratio
- **Caches detection results** to disk so re-runs are instant
- Outputs a clean CSV with per-gaze-sample screen coordinates and quality flags


## Requirements

- Neon eye tracker (Pupil Labs)
- AprilTag markers (family: `tag36h11`, IDs 0–3) rendered at the four corners of your stimulus screen
- Python 3.11, conda environment

## Installation

```bash
conda create -n neon_gaze python=3.11 -y
conda activate neon_gaze
pip install -r requirements.txt
```

## Usage

### Step 1 — Confirm tag ID to corner mapping

```bash
python neon_gaze_pipeline.py --recording "path/to/recording" --report-only
```

This runs detection on all frames, caches the results, and prints which tag ID is at which screen corner. Check the output against your setup. If the mapping is wrong, use `--tag-ids` to reassign (see below).

### Step 2 — Run the full pipeline

```bash
python neon_gaze_pipeline.py --recording "path/to/recording"
```

Outputs saved to `<recording>/gaze_pipeline_output/`:
- `gaze_on_surface.csv` — main output, one row per gaze sample (200 Hz)
- `surface_quality.csv` — per scene-frame QA diagnostics
- `markers_cache.npy` — cached detection results (re-used on subsequent runs)

### Step 3 — Generate QA video

```bash
python surface_qa_video.py --recording "path/to/recording"
```

Renders an annotated video showing the detected surface polygon, marker corners, and gaze point on every frame. Green overlay = valid frame, red border = rejected frame.

To render only a clip:
```bash
python surface_qa_video.py --recording "path/to/recording" --start 60 --end 180
```

## Configuration

All parameters are set via command-line arguments:

| Argument | Default | Description |
|---|---|---|
| `--recording` | required | Path to Neon recording folder |
| `--screen-w` | 1920 | Screen width in pixels |
| `--screen-h` | 1080 | Screen height in pixels |
| `--min-markers` | 3 | Minimum markers for a valid frame |
| `--smooth-win` | 7 | Temporal median filter window (frames) |
| `--report-only` | False | Run detection report only |
| `--start` | None | QA video clip start (seconds) |
| `--end` | None | QA video clip end (seconds) |

## Marker Setup

For best results:

- Use `tag36h11` family AprilTags (IDs 0, 1, 2, 3)
- Render markers **on-screen** as part of your stimulus (not physical printouts)
- Place one tag in each corner of the display window
- Use a tag size of at least (waiting to get a response from Pupil Lab) screen pixels with a white quiet zone around each tag
- Keep the same tag IDs across all recordings so the surface definition is consistent

## Output CSV columns

| Column | Description |
|---|---|
| `gaze_timestamp_ns` | Nanosecond timestamp |
| `gaze_timestamp_s` | Timestamp in seconds |
| `scene_frame_idx` | Nearest scene camera frame index |
| `gaze_x_norm` | Raw gaze x in scene-camera space (pixels) |
| `gaze_y_norm` | Raw gaze y in scene-camera space (pixels) |
| `gaze_x_screen_px` | Gaze x on screen in pixels |
| `gaze_y_screen_px` | Gaze y on screen in pixels |
| `gaze_x_screen_norm` | Gaze x normalised [0, 1] |
| `gaze_y_screen_norm` | Gaze y normalised [0, 1] |
| `surface_valid` | True if gaze is within screen bounds and frame passed QA |

## How It Works

1. **AprilTag detection** — `pupil_apriltags.Detector` runs on every scene camera frame at full resolution. Only the four known tag IDs are kept.
2. **Temporal smoothing** — for each tag, the center position is linearly interpolated across undetected frames and then median-filtered over a 7-frame window. The smoothed delta is applied to all four corners of each tag.
3. **Homography estimation** — for frames with 3 or more detected tags, `cv2.findHomography` (RANSAC) maps the known screen-space tag positions to the detected scene-camera positions.
4. **Outlier rejection** — frames where the projected surface area or aspect ratio deviates more than 5 median absolute deviations from the recording median are rejected.
5. **Gaze mapping** — each gaze sample (already in scene-camera pixel coordinates) is undistorted using the Neon scene camera calibration, then projected onto the screen using the inverse homography.


## License

MIT License. See LICENSE file.
