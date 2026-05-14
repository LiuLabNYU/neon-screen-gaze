"""
neon-screen-gaze: Surface Detection QA Video
=============================================
Author: Forouzan Farahani, Liu Lab, NYU Langone Health

Renders an annotated scene-camera video showing:
  - Detected AprilTag corners (one color per tag ID)
  - Projected surface polygon (green=valid, red=rejected)
  - Gaze point in scene-camera space
  - Per-frame status and marker count

Usage:
    python surface_qa_video.py --recording "path/to/recording"
    python surface_qa_video.py --recording "path/to/recording" --start 60 --end 180
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Generate surface detection QA video.")
    p.add_argument("--recording", required=True,
                   help="Path to Neon recording folder")
    p.add_argument("--screen-w",  type=int,   default=1920)
    p.add_argument("--screen-h",  type=int,   default=1080)
    p.add_argument("--scale",     type=float, default=0.6,
                   help="Output video scale (default: 0.6)")
    p.add_argument("--fps",       type=int,   default=30,
                   help="Output video fps (default: 30)")
    p.add_argument("--start",     type=float, default=None,
                   help="Clip start in seconds")
    p.add_argument("--end",       type=float, default=None,
                   help="Clip end in seconds")
    return p.parse_args()


TAG_COLORS = {
    0: (0,   200, 255),
    1: (255, 100,   0),
    2: (0,   255, 100),
    3: (100,   0, 255),
}

TAG_ID_TO_SCREEN_BBOX = {
    0: np.float32([[20,   20],  [100,  20],  [100,  100],  [20,  100]]),
    1: np.float32([[1820, 20],  [1900, 20],  [1900, 100],  [1820, 100]]),
    2: np.float32([[20,   980], [100,  980], [100,  1060], [20,  1060]]),
    3: np.float32([[1820, 980], [1900, 980], [1900, 1060], [1820, 1060]]),
}


def main():
    args       = parse_args()
    rec_dir    = Path(args.recording)
    output_dir = rec_dir / "gaze_pipeline_output"
    output_video = output_dir / "surface_qa.mp4"

    scene_vids = sorted(rec_dir.glob("Neon Scene Camera v1*.mp4"))
    if not scene_vids:
        raise FileNotFoundError("No scene video found")
    scene_video = scene_vids[0]
    print(f"Scene video: {scene_video.name}")

    cap     = cv2.VideoCapture(str(scene_video))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Scene: {FRAME_W}x{FRAME_H} @ {fps:.1f} fps  ({total} frames)")

    ts_files = sorted(rec_dir.glob("Neon Scene Camera v1*.time"))
    scene_times_ns = np.fromfile(str(ts_files[0]), dtype=np.int64)

    cache_path = output_dir / "markers_cache.npy"
    if not cache_path.exists():
        raise FileNotFoundError(
            "markers_cache.npy not found. Run neon_gaze_pipeline.py first."
        )
    markers_by_frame = list(np.load(str(cache_path), allow_pickle=True))
    print(f"Marker cache: {len(markers_by_frame)} frames")

    qdf = pd.read_csv(output_dir / "surface_quality.csv").set_index("frame_idx")

    gaze_raw_path   = next(p for p in [rec_dir/"gaze ps1.raw",  rec_dir/"gaze.raw"]  if p.exists())
    gaze_time_path  = next(p for p in [rec_dir/"gaze ps1.time", rec_dir/"gaze.time"] if p.exists())
    gaze_dtype_path = rec_dir / "gaze.dtype"
    dtype      = np.dtype(eval(gaze_dtype_path.read_text().strip()))
    gaze_data  = np.fromfile(str(gaze_raw_path),  dtype=dtype)
    gaze_ts_ns = np.fromfile(str(gaze_time_path), dtype=np.int64)
    print(f"Gaze: {len(gaze_ts_ns)} samples")

    t0_ns = scene_times_ns[0]
    start_frame = (int(np.searchsorted(scene_times_ns, t0_ns + int(args.start * 1e9)))
                   if args.start is not None else 0)
    end_frame   = (int(np.searchsorted(scene_times_ns, t0_ns + int(args.end * 1e9)))
                   if args.end   is not None else total)
    end_frame   = min(end_frame, total, len(markers_by_frame))
    n_frames    = end_frame - start_frame
    print(f"Rendering frames {start_frame} to {end_frame} ({n_frames} frames)")

    out_w  = int(FRAME_W * args.scale)
    out_h  = int(FRAME_H * args.scale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, args.fps, (out_w, out_h))
    print(f"Output: {output_video}  ({out_w}x{out_h} @ {args.fps} fps)")

    gaze_for_frame = {}
    for fi in range(start_frame, end_frame):
        ts = scene_times_ns[fi]
        gi = int(np.searchsorted(gaze_ts_ns, ts))
        gi = min(gi, len(gaze_ts_ns) - 1)
        if gi > 0 and abs(gaze_ts_ns[gi-1]-ts) < abs(gaze_ts_ns[gi]-ts):
            gi -= 1
        gaze_for_frame[fi] = (float(gaze_data["x"][gi]), float(gaze_data["y"][gi]))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for fi in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        fm     = markers_by_frame[fi] if fi < len(markers_by_frame) else {}
        has_H  = bool(qdf.loc[fi, "has_H"]) if fi in qdf.index else False
        status = str(qdf.loc[fi, "status"]) if fi in qdf.index else "unknown"
        n_det  = len(fm)

        for tid, corners in fm.items():
            color  = TAG_COLORS.get(tid, (255, 255, 255))
            center = corners.mean(axis=0).astype(int)
            for corner in corners.astype(int):
                cv2.circle(frame, tuple(corner), 5, color, -1)
            cv2.circle(frame, tuple(center), 8, color, 2)
            cv2.putText(frame, f"ID{tid}", tuple(center + np.array([8, -8])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if has_H:
            src, dst = [], []
            for tid, scene_corners in fm.items():
                src.append(TAG_ID_TO_SCREEN_BBOX[tid])
                dst.append(scene_corners)
            src = np.vstack(src).astype(np.float32)
            dst = np.vstack(dst).astype(np.float32)
            Hmat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if Hmat is not None:
                sc   = np.float32([
                    [0,             0            ],
                    [args.screen_w, 0            ],
                    [args.screen_w, args.screen_h],
                    [0,             args.screen_h],
                ]).reshape(-1, 1, 2)
                proj = cv2.perspectiveTransform(sc, Hmat).reshape(-1, 2).astype(np.int32)
                cv2.polylines(frame, [proj], isClosed=True, color=(0, 255, 0), thickness=2)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [proj], (0, 180, 0))
                frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
        else:
            cv2.rectangle(frame, (4, 4), (FRAME_W-4, FRAME_H-4), (0, 0, 220), 3)

        gx, gy = gaze_for_frame[fi]
        if not (np.isnan(gx) or np.isnan(gy)):
            gpt = (int(np.clip(float(gx), 0, FRAME_W-1)),
                   int(np.clip(float(gy), 0, FRAME_H-1)))
            cv2.circle(frame, gpt, 12, (255, 255, 255), 2)
            cv2.circle(frame, gpt, 4,  (0,   0,   255), -1)

        bar_color = (0, 160, 0) if has_H else (0, 0, 180)
        cv2.rectangle(frame, (0, 0), (FRAME_W, 28), bar_color, -1)
        txt = (f"Frame {fi:05d}  |  markers: {n_det}/4  |  "
               f"status: {status}  |  {'VALID' if has_H else 'REJECTED'}")
        cv2.putText(frame, txt, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if args.scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

        if (fi - start_frame) % 500 == 0 and fi > start_frame:
            done = fi - start_frame
            print(f"  {done}/{n_frames} ({done/n_frames*100:.1f}%)")

    cap.release()
    writer.release()
    print(f"\nDone. Video saved to:\n  {output_video}")
    print("\nKey:")
    print("  Green polygon + green bar = valid surface frame")
    print("  Red border  + red bar    = rejected frame")
    print("  Colored dots             = detected tag corners (one color per ID)")
    print("  White circle + red dot   = gaze point")


if __name__ == "__main__":
    main()
