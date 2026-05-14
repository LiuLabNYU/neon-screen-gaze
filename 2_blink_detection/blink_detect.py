"""
Run Pupil Labs' real-time blink detector on a Neon recording and export
blinks to CSV for downstream use (e.g. as an invalid mask in ClusterFix).

The blink_detector package has no setup.py / pyproject.toml, so we add its
parent folder to sys.path manually. Edit BLINK_REPO if your clone lives
somewhere else.

CSV columns (all timestamps are ns since epoch):
    start_timestamp           = event1.start_time   (eyelid begins closing)
    onset_end_timestamp       = start_timestamp + eyelid_closing_duration
                                (eyelid finished closing -> eye fully closed)
    offset_start_timestamp    = end_timestamp - eyelid_opening_duration
                                (eyelid begins opening)
    end_timestamp             = event2.end_time     (eye fully open again)
    eyelid_closing_duration_s = duration of the onset run (closing phase)
    eyelid_opening_duration_s = duration of the offset run (opening phase)
    blink_duration_s          = end_timestamp - start_timestamp  (in seconds)

The 4 timestamps mark the four phase boundaries of one blink:
    start_ts -- ONSET RUN --> onset_end_ts -- (eye closed) --> offset_start_ts -- OFFSET RUN --> end_ts
"""

import sys
from pathlib import Path

# --- Make the blink_detector package importable from any CWD ---------------
BLINK_REPO = Path(r"C:\Users\Forouzan\real-time-blink-detection")
if str(BLINK_REPO) not in sys.path:
    sys.path.insert(0, str(BLINK_REPO))

import csv
from blink_detector import blink_detection_pipeline
from blink_detector.helper import preprocess_recording, visualize_blink_events


# --- Settings --------------------------------------------------------------
RECORDING_PATH = r"D:\Data\Temp-Neon\2026-05-11-17-27-07"
OUT_CSV        = Path(RECORDING_PATH) / "blinks.csv"
VISUALIZE      = False
START_S, END_S = 0, 120   # seconds of viz window


# --- Detect blinks ---------------------------------------------------------
print(f"Preprocessing recording: {RECORDING_PATH}")
left, right, ts = preprocess_recording(RECORDING_PATH, is_neon=True)

print("Running blink detection pipeline...")
blink_events = list(blink_detection_pipeline(left, right, ts))
print(f"  {len(blink_events)} blink events detected")


# --- Export to CSV ---------------------------------------------------------
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


with open(OUT_CSV, "w", newline="") as f:
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
        # Clamp degenerate cases (closing+opening longer than blink span)
        if onset_end_ts > offset_start_ts:
            mid = (t0 + t1) // 2
            onset_end_ts    = mid
            offset_start_ts = mid
        w.writerow([t0, onset_end_ts, offset_start_ts, t1, clos, opng, bdur])
print(f"  CSV written: {OUT_CSV}")


# --- Visualise -------------------------------------------------------------
if VISUALIZE:
    print(f"Visualising {START_S}-{END_S} s ...")
    visualize_blink_events(blink_events, ts, START_S, END_S)