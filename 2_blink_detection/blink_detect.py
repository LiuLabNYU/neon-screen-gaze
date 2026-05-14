"""
Neon Blink Detection
====================
Runs Pupil Labs' real-time blink detector on a Neon recording and exports
blinks to CSV for downstream use (e.g. as an invalid mask in ClusterFix).

Requires the real-time-blink-detection repo:
    https://github.com/pupil-labs/real-time-blink-detection

GPU acceleration (recommended):
    The blink detector uses XGBoost for classification. For significantly
    faster processing, patch helper.py in the blink_detector repo:

    Find:   clf = XGBClassifier()
    Change: clf = XGBClassifier(device="cuda", tree_method="hist")

    Also patch blink_detector.py to use batched prediction (see README).

Usage:
    python blink_detect.py --recording "path/to/recording"
    python blink_detect.py --recording "path/to/recording" --blink-repo "path/to/real-time-blink-detection"

Output CSV columns (all timestamps in nanoseconds since epoch):
    start_timestamp           eyelid begins closing
    onset_end_timestamp       eyelid finished closing (eye fully closed)
    offset_start_timestamp    eyelid begins opening
    end_timestamp             eye fully open again
    eyelid_closing_duration_s duration of closing phase
    eyelid_opening_duration_s duration of opening phase
    blink_duration_s          total blink duration

The 4 timestamps mark the four phase boundaries of one blink:
    start_ts --> onset_end_ts --> (eye closed) --> offset_start_ts --> end_ts
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Detect blinks in a Neon recording and export to CSV."
    )
    p.add_argument("--recording", required=True,
                   help="Path to Neon recording folder")
    p.add_argument("--blink-repo",
                   default=r"C:\Users\Forouzan\real-time-blink-detection",
                   help="Path to real-time-blink-detection repo clone")
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: <recording>/blinks.csv)")
    return p.parse_args()


def get_attr(ev, *names, default=None):
    """Try several attribute names; return default if none found."""
    for n in names:
        if hasattr(ev, n):
            return getattr(ev, n)
        if isinstance(ev, dict) and n in ev:
            return ev[n]
    if default is not None:
        return default
    raise AttributeError(f"None of {names} on {ev!r}")


def main():
    args = parse_args()

    rec_path  = Path(args.recording)
    out_csv   = Path(args.output) if args.output else rec_path / "blinks.csv"
    blink_repo = Path(args.blink_repo)

    if not blink_repo.exists():
        raise FileNotFoundError(
            f"Blink detector repo not found: {blink_repo}\n"
            "Clone it from: https://github.com/pupil-labs/real-time-blink-detection\n"
            "Then pass its path with --blink-repo"
        )

    # Make blink_detector importable
    if str(blink_repo) not in sys.path:
        sys.path.insert(0, str(blink_repo))

    from blink_detector import blink_detection_pipeline
    from blink_detector.helper import preprocess_recording

    # Preprocess
    print(f"Recording : {rec_path}")
    print(f"Output    : {out_csv}")
    print("Preprocessing eye video...")
    left, right, ts = preprocess_recording(str(rec_path), is_neon=True)
    print(f"  {len(ts):,} frames loaded.")

    # Detect
    print("Running blink detection...")
    blink_events = []
    for event in blink_detection_pipeline(left, right, ts):
        blink_events.append(event)
        if len(blink_events) % 50 == 0:
            t_sec = (get_attr(event, "start_time", "start_timestamp") - ts[0]) / 1e9
            print(f"  {len(blink_events)} blinks so far  "
                  f"(last at {t_sec:.1f}s into recording)", flush=True)

    print(f"  Done. {len(blink_events)} blinks detected.")

    # Export
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "start_timestamp",
            "onset_end_timestamp",
            "offset_start_timestamp",
            "end_timestamp",
            "eyelid_closing_duration_s",
            "eyelid_opening_duration_s",
            "blink_duration_s",
        ])
        for ev in blink_events:
            t0   = int(get_attr(ev, "start_time", "start_timestamp", "start_ts"))
            t1   = int(get_attr(ev, "end_time",   "end_timestamp",   "end_ts"))
            clos = float(get_attr(ev, "eyelid_closing_duration_s",
                                      "closing_duration_s", default=0.0) or 0.0)
            opng = float(get_attr(ev, "eyelid_opening_duration_s",
                                      "opening_duration_s", default=0.0) or 0.0)
            bdur = float(get_attr(ev, "blink_duration_s",
                                      "duration_s", default=(t1 - t0) / 1e9))
            onset_end_ts    = t0 + int(round(clos * 1e9))
            offset_start_ts = t1 - int(round(opng * 1e9))
            # Clamp degenerate cases
            if onset_end_ts > offset_start_ts:
                mid = (t0 + t1) // 2
                onset_end_ts    = mid
                offset_start_ts = mid
            w.writerow([t0, onset_end_ts, offset_start_ts, t1, clos, opng, bdur])

    print(f"  CSV written: {out_csv}")

    # Summary
    if blink_events:
        dur_s = (ts[-1] - ts[0]) / 1e9
        rate  = len(blink_events) / (dur_s / 60)
        print(f"\nSummary:")
        print(f"  Recording duration : {dur_s/60:.1f} min")
        print(f"  Total blinks       : {len(blink_events)}")
        print(f"  Blink rate         : {rate:.1f} blinks/min")


if __name__ == "__main__":
    main()
