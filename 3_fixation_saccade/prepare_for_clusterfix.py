"""
Prepare Gaze Data for ClusterFix
==================================
Converts neon-screen-gaze pipeline outputs to the CSV format expected by
run_clusterfix.m, incorporating blink detection results as additional
invalid samples.

Inputs:
    gaze_on_surface.csv     from neon_gaze_pipeline.py
    surface_quality.csv     from neon_gaze_pipeline.py
    blinks.csv              from blink_detect.py

Outputs (saved to <recording>/clusterfix_input/):
    gaze_ready.csv          gaze in run_clusterfix.m format
    annotations.csv         recording.begin and recording.end timestamps

Usage:
    python prepare_for_clusterfix.py --recording "path/to/recording"

Output CSV columns (gaze_ready.csv):
    timestamp                   nanoseconds since epoch
    detected on surface         True/False (False during blinks or off-surface)
    position on surface x       normalised [0, 1]
    position on surface y       normalised [0, 1]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare gaze data for ClusterFix."
    )
    p.add_argument("--recording", required=True,
                   help="Path to Neon recording folder")
    p.add_argument("--blink-buffer-ms", type=float, default=50.0,
                   help="Buffer around each blink to mark invalid (ms each side, default 50)")
    return p.parse_args()


def main():
    args       = parse_args()
    rec_dir    = Path(args.recording)
    pipeline_dir = rec_dir / "gaze_pipeline_output"
    output_dir   = rec_dir / "clusterfix_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Recording : {rec_dir}")
    print(f"Output    : {output_dir}")

    # ── Load gaze ─────────────────────────────────────────────────────────────
    gaze_path = pipeline_dir / "gaze_on_surface.csv"
    if not gaze_path.exists():
        raise FileNotFoundError(
            f"gaze_on_surface.csv not found in {pipeline_dir}\n"
            "Run neon_gaze_pipeline.py first."
        )
    gaze = pd.read_csv(gaze_path)
    print(f"Gaze samples loaded  : {len(gaze):,}")

    # ── Load surface quality ──────────────────────────────────────────────────
    quality_path = pipeline_dir / "surface_quality.csv"
    if not quality_path.exists():
        raise FileNotFoundError(
            f"surface_quality.csv not found in {pipeline_dir}\n"
            "Run neon_gaze_pipeline.py first."
        )
    quality = pd.read_csv(quality_path)

    # First and last frame with valid surface detection
    valid_frames = quality[quality["has_H"]]
    if valid_frames.empty:
        raise ValueError("No valid surface frames found in surface_quality.csv")

    first_valid_frame = valid_frames["frame_idx"].min()
    last_valid_frame  = valid_frames["frame_idx"].max()

    # Map frame indices to timestamps via gaze
    first_ts = gaze[gaze["scene_frame_idx"] >= first_valid_frame]["gaze_timestamp_ns"].min()
    last_ts  = gaze[gaze["scene_frame_idx"] <= last_valid_frame]["gaze_timestamp_ns"].max()

    print(f"First valid frame    : {first_valid_frame}  (t={first_ts})")
    print(f"Last valid frame     : {last_valid_frame}  (t={last_ts})")

    # Trim gaze to valid surface period
    gaze = gaze[
        (gaze["gaze_timestamp_ns"] >= first_ts) &
        (gaze["gaze_timestamp_ns"] <= last_ts)
    ].copy()
    print(f"Gaze after trimming  : {len(gaze):,}")

    # ── Load blinks ───────────────────────────────────────────────────────────
    blink_path = rec_dir / "blinks.csv"
    if not blink_path.exists():
        print("WARNING: blinks.csv not found — blink masking skipped.")
        print("         Run blink_detect.py first for best results.")
        blinks = pd.DataFrame(columns=["start_timestamp", "end_timestamp"])
    else:
        blinks = pd.read_csv(blink_path)
        print(f"Blinks loaded        : {len(blinks):,}")

    # ── Build valid mask ──────────────────────────────────────────────────────
    # Start with surface_valid flag from pipeline
    on_surface = gaze["surface_valid"].astype(bool).copy()

    # Mark blink periods as invalid (with buffer)
    buffer_ns = int(args.blink_buffer_ms * 1e6)
    n_blink_masked = 0

    if len(blinks) > 0:
        ts = gaze["gaze_timestamp_ns"].values
        for _, blink in blinks.iterrows():
            t0 = int(blink["start_timestamp"]) - buffer_ns
            t1 = int(blink["end_timestamp"])   + buffer_ns
            mask = (ts >= t0) & (ts <= t1)
            n_blink_masked += mask.sum()
            on_surface[mask] = False

        print(f"Blink buffer         : +/- {args.blink_buffer_ms:.0f} ms")
        print(f"Samples masked (blink): {n_blink_masked:,}")

    print(f"Final on-surface     : {on_surface.sum():,} / {len(on_surface):,} "
          f"({on_surface.mean()*100:.1f}%)")

    # ── Fill NaN coordinates with nearest valid value ─────────────────────────
    # Samples with no homography have NaN x/y. MATLAB's interp1 does not skip
    # NaN so we forward/backward fill to give placeholder coordinates.
    # The invalid mask already excludes these samples from fixation/saccade detection.
    gaze["gaze_x_screen_norm"] = gaze["gaze_x_screen_norm"].ffill().bfill()
    gaze["gaze_y_screen_norm"] = gaze["gaze_y_screen_norm"].ffill().bfill()

    # ── Build output gaze CSV ─────────────────────────────────────────────────
    gaze_out = pd.DataFrame({
        "timestamp"              : gaze["gaze_timestamp_ns"].values,
        "detected on surface"    : on_surface.values,
        "position on surface x"  : gaze["gaze_x_screen_norm"].values,
        "position on surface y"  : gaze["gaze_y_screen_norm"].values,
    })

    gaze_out_path = output_dir / "gaze_ready.csv"
    gaze_out.to_csv(gaze_out_path, index=False)
    print(f"\nSaved: {gaze_out_path}")

    # ── Build annotations CSV ─────────────────────────────────────────────────
    annotations = pd.DataFrame({
        "timestamp" : [first_ts, last_ts],
        "label"     : ["recording.begin", "recording.end"],
    })

    ann_path = output_dir / "annotations.csv"
    annotations.to_csv(ann_path, index=False)
    print(f"Saved: {ann_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    duration_s = (last_ts - first_ts) / 1e9
    print(f"\n{'='*55}")
    print("READY FOR CLUSTERFIX")
    print(f"{'='*55}")
    print(f"  Duration           : {duration_s/60:.1f} min")
    print(f"  Total gaze samples : {len(gaze_out):,}")
    print(f"  On-surface samples : {on_surface.sum():,} ({on_surface.mean()*100:.1f}%)")
    print(f"  Blinks masked      : {len(blinks):,}")
    print(f"\n  Outputs in: {output_dir}")
    print("    gaze_ready.csv")
    print("    annotations.csv")
    print(f"{'='*55}")
    print(f"\nRun in MATLAB:")
    print(f"  results = run_clusterfix(")
    print(f"    '{output_dir / 'gaze_ready.csv'}', ...")
    print(f"    '{output_dir / 'annotations.csv'}');")


if __name__ == "__main__":
    main()
