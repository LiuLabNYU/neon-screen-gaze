# 3_fixation_saccade

Prepares gaze data for ClusterFix and runs fixation and saccade detection.

## Scripts

- `prepare_for_clusterfix.py` — converts pipeline outputs to ClusterFix input format
- `run_clusterfix.m` — runs ClusterFix on the prepared gaze data
- `ClusterFix.m` — ClusterFix algorithm (Mack et al., 2017)

---

## Step 1: prepare_for_clusterfix.py

### What it does

Takes the outputs of the surface mapping pipeline (`gaze_on_surface.csv`,
`surface_quality.csv`) and the blink detection pipeline (`blinks.csv`) and
produces two files ready for `run_clusterfix.m`:

- `gaze_ready.csv` — gaze in the format expected by ClusterFix, with blink
  periods marked as invalid
- `annotations.csv` — recording start and end timestamps

### How recording start and end are determined

The script uses `surface_quality.csv` to find the first and last scene camera
frame where the screen surface was successfully detected (i.e. at least 3 of
4 AprilTag markers were visible and the surface geometry passed quality
filters). These frame indices are then mapped to gaze timestamps using the
`scene_frame_idx` column in `gaze_on_surface.csv`.

In practice this means:

- `recording.begin` = timestamp of the first gaze sample whose nearest scene
  frame had a valid surface detection
- `recording.end` = timestamp of the last gaze sample whose nearest scene
  frame had a valid surface detection

Everything before the participant sat down in front of the screen or after
they looked away is excluded automatically. No manual annotation of start
and end times is needed.

### Blink masking

Gaze samples that fall within a detected blink window are marked as
`detected on surface = False`, adding them to the invalid mask that
ClusterFix uses to exclude those periods from fixation and saccade detection.
A configurable buffer (default 50 ms each side) is applied around each blink
to catch the eye movement artifacts at blink onset and offset.

### Usage

```bash
python prepare_for_clusterfix.py --recording "path/to/recording"
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--recording` | required | Path to Neon recording folder |
| `--blink-buffer-ms` | 50 | Buffer around each blink in ms (each side) |

### Output columns (gaze_ready.csv)

| Column | Description |
|---|---|
| `timestamp` | Nanoseconds since epoch |
| `detected on surface` | False during blinks, off-surface, or rejected frames |
| `position on surface x` | Normalised [0, 1] |
| `position on surface y` | Normalised [0, 1] |

---

## Step 2: run_clusterfix.m

Run in MATLAB after `prepare_for_clusterfix.py` completes:

```matlab
results = run_clusterfix( ...
    'path/to/recording/clusterfix_input/gaze_ready.csv', ...
    'path/to/recording/clusterfix_input/annotations.csv');
```

---

## Dependencies

- Python: `numpy`, `pandas` (already in `neon_gaze` conda environment)
- MATLAB: ClusterFix (Mack et al., 2017) — not included in this repo.
  Download `ClusterFix.m` from https://github.com/mackdc/clusterfix
  and place it in this folder before running `run_clusterfix.m`.

## Reference

Mack, D. J., Belfanti, S., & Schwartz, S. (2017). The effect of sampling rate
and lowpass filters on saccades and microsaccades detection in a
ground-truth data set. *Behavior Research Methods*, 49, 2115–2128.
