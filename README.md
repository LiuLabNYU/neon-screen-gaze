# neon-screen-gaze

A Python and MATLAB pipeline for processing Neon eye tracker recordings from screen-based experiments. Covers gaze-to-screen mapping, blink detection, and fixation and saccade detection.

**Author:** Forouzan Farahani, Liu Lab, NYU Langone Health

---

## Pipeline Overview

```
Neon recording
      |
      v
1_surface_mapping/        Map raw gaze to screen coordinates using AprilTags
      |
      v
2_blink_detection/        Detect blinks using optical flow and XGBoost
      |
      v
3_fixation_saccade/       Prepare data and run ClusterFix
      |
      v
gaze_on_surface.csv       Screen-space gaze (200 Hz)
blinks.csv                Blink timestamps and durations
ClusterFix_Results.mat    Fixations and saccades
```

---

## Quick Start

### Installation

```bash
conda create -n neon_gaze python=3.11 -y
conda activate neon_gaze
pip install -r requirements.txt
```

### Step 1: Surface mapping

```bash
# Confirm tag ID to corner mapping (first run only)
python 1_surface_mapping/neon_gaze_pipeline.py --recording "path/to/recording" --report-only

# Run full pipeline
python 1_surface_mapping/neon_gaze_pipeline.py --recording "path/to/recording"

# Optional: generate QA video
python 1_surface_mapping/surface_qa_video.py --recording "path/to/recording" --start 60 --end 180
```

Output: `<recording>/gaze_pipeline_output/gaze_on_surface.csv`

### Step 2: Blink detection

Requires [real-time-blink-detection](https://github.com/pupil-labs/real-time-blink-detection) and a NVIDIA GPU. See `2_blink_detection/PATCHES.md` for required changes to the blink detector repo.

```bash
python 2_blink_detection/blink_detect.py --recording "path/to/recording" --blink-repo "path/to/real-time-blink-detection"
```

Output: `<recording>/blinks.csv`

### Step 3: Fixation and saccade detection

```bash
# Prepare input files for ClusterFix
python 3_fixation_saccade/prepare_for_clusterfix.py --recording "path/to/recording"
```

Then in MATLAB (requires [ClusterFix](https://github.com/mackdc/clusterfix)):

```matlab
cd('path/to/recording/clusterfix_input')
results = run_clusterfix('gaze_ready.csv', 'annotations.csv');
```

Output: `ClusterFix_Results.mat`

---

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA support (for blink detection)
- MATLAB (for ClusterFix)
- Neon eye tracker (Pupil Labs)
- AprilTag markers rendered at the four corners of your stimulus display

See each subfolder's README for detailed documentation.

---

## Citation

If you use this pipeline in your research, please cite it as:

> Farahani, F. (2026). neon-screen-gaze: Gaze-to-screen mapping, blink detection, and fixation and saccade detection for the Neon eye tracker. GitHub. https://github.com/forouzanfarahani/neon-screen-gaze

## License

MIT License. See LICENSE file.
