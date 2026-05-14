"""
neon-screen-gaze: Improved gaze-to-screen mapping for the Neon eye tracker
===========================================================================
Author: Forouzan Farahani, Liu Lab, NYU Langone Health

Usage:
    # Step 1: confirm tag ID to corner mapping
    python neon_gaze_pipeline.py --recording "path/to/recording" --report-only

    # Step 2: run full pipeline
    python neon_gaze_pipeline.py --recording "path/to/recording"

See README.md for full documentation.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Defaults (overridden by command-line arguments)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    screen_w              = 1920,
    screen_h              = 1080,
    tag_family            = "tag36h11",
    quad_decimate         = 1.0,
    decode_sharpening     = 0.25,
    nthreads              = 4,
    min_markers           = 3,
    smooth_win            = 7,
    area_mad_threshold    = 5.0,
    aspect_mad_threshold  = 5.0,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Map Neon gaze to screen coordinates using AprilTag surface tracking."
    )
    p.add_argument("--recording",      required=True,
                   help="Path to Neon recording folder")
    p.add_argument("--screen-w",       type=int,   default=DEFAULTS["screen_w"],
                   help="Screen width in pixels (default: 1920)")
    p.add_argument("--screen-h",       type=int,   default=DEFAULTS["screen_h"],
                   help="Screen height in pixels (default: 1080)")
    p.add_argument("--min-markers",    type=int,   default=DEFAULTS["min_markers"],
                   help="Minimum markers for a valid frame (default: 3)")
    p.add_argument("--smooth-win",     type=int,   default=DEFAULTS["smooth_win"],
                   help="Temporal median filter window in frames (default: 7)")
    p.add_argument("--report-only",    action="store_true",
                   help="Run detection and print report only, do not map gaze")
    p.add_argument("--force-redetect", action="store_true",
                   help="Ignore cached detections and re-run from scratch")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Tag layout — edit this if your tag IDs or screen positions differ
# ---------------------------------------------------------------------------

def build_tag_layout(screen_w, screen_h):
    """
    Returns:
      TAG_ID_TO_SCREEN_CENTER  dict: tag_id -> (cx, cy) in screen pixels
      TAG_ID_TO_SCREEN_BBOX    dict: tag_id -> (4,2) array of tag corners in screen pixels

    Default layout: 80x80 px tags at the four screen corners.
    Edit this function if your tag size or positions are different.
    """
    tag_size = 80
    margin   = 20

    centers = {
        0: (margin + tag_size // 2,           margin + tag_size // 2),           # TL
        1: (screen_w - margin - tag_size // 2, margin + tag_size // 2),           # TR
        2: (margin + tag_size // 2,           screen_h - margin - tag_size // 2), # BL
        3: (screen_w - margin - tag_size // 2, screen_h - margin - tag_size // 2),# BR
    }

    bboxes = {}
    for tid, (cx, cy) in centers.items():
        x1, y1 = cx - tag_size // 2, cy - tag_size // 2
        x2, y2 = cx + tag_size // 2, cy + tag_size // 2
        bboxes[tid] = np.float32([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ])

    return centers, bboxes


# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------

def load_neon_recording(rec_dir: Path):
    from pupil_labs.neon_recording import NeonRecording

    log.info("Loading recording...")
    neon = NeonRecording(str(rec_dir))

    scene_vids = sorted(rec_dir.glob("Neon Scene Camera v1*.mp4"))
    if not scene_vids:
        raise FileNotFoundError(f"No scene video found in {rec_dir}")
    scene_video = scene_vids[0]
    log.info(f"Scene video : {scene_video.name}")

    ts_files = sorted(rec_dir.glob("Neon Scene Camera v1*.time"))
    if not ts_files:
        raise FileNotFoundError("Scene .time file not found")
    scene_times_ns = np.fromfile(str(ts_files[0]), dtype=np.int64)
    log.info(f"Scene frames: {len(scene_times_ns)}")

    gaze_raw_path   = _find_file(rec_dir, ["gaze ps1.raw",  "gaze.raw"])
    gaze_time_path  = _find_file(rec_dir, ["gaze ps1.time", "gaze.time"])
    gaze_dtype_path = rec_dir / "gaze.dtype"
    if not gaze_dtype_path.exists():
        raise FileNotFoundError("gaze.dtype not found")

    dtype      = np.dtype(eval(gaze_dtype_path.read_text().strip()))
    gaze_data  = np.fromfile(str(gaze_raw_path),  dtype=dtype)
    gaze_ts_ns = np.fromfile(str(gaze_time_path), dtype=np.int64)
    log.info(f"Gaze samples: {len(gaze_ts_ns)}  fields: {dtype.names}")

    calib = neon.calibration
    K = np.array(calib.scene_camera_matrix,           dtype=np.float64)
    D = np.array(calib.scene_distortion_coefficients, dtype=np.float64).flatten()
    log.info(f"Camera: fx={K[0,0]:.2f} fy={K[1,1]:.2f} "
             f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")

    return dict(
        scene_video    = scene_video,
        scene_times_ns = scene_times_ns,
        gaze_data      = gaze_data,
        gaze_ts_ns     = gaze_ts_ns,
        K              = K,
        D              = D,
    )


def _find_file(rec_dir, candidates):
    for name in candidates:
        p = rec_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {candidates} found in {rec_dir}")


# ---------------------------------------------------------------------------
#  Step 1 — Detection (with cache)
# ---------------------------------------------------------------------------

def get_markers(scene_video, output_dir, known_ids,
                force_redetect=False,
                quad_decimate=1.0, decode_sharpening=0.25, nthreads=4,
                tag_family="tag36h11"):
    cache_path = output_dir / "markers_cache.npy"

    if cache_path.exists() and not force_redetect:
        log.info(f"Loading marker cache: {cache_path}")
        markers = list(np.load(str(cache_path), allow_pickle=True))
        log.info(f"Cache loaded: {len(markers)} frames")
        return markers

    log.info("No cache found — running detection (this may take ~20 min)...")
    markers = _detect_markers(
        scene_video, known_ids, quad_decimate, decode_sharpening,
        nthreads, tag_family
    )
    np.save(str(cache_path), np.array(markers, dtype=object))
    log.info(f"Cache saved: {cache_path}")
    return markers


def _detect_markers(scene_video, known_ids, quad_decimate,
                    decode_sharpening, nthreads, tag_family):
    import pupil_apriltags

    detector = pupil_apriltags.Detector(
        families          = tag_family,
        nthreads          = nthreads,
        quad_decimate     = quad_decimate,
        decode_sharpening = decode_sharpening,
    )
    log.info(f"Detector: quad_decimate={quad_decimate}, "
             f"decode_sharpening={decode_sharpening}, nthreads={nthreads}")

    cap   = cv2.VideoCapture(str(scene_video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info(f"Processing {total} frames...")

    markers = []
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(gray)
        fm   = {det.tag_id: np.array(det.corners, dtype=np.float32)
                for det in dets if det.tag_id in known_ids}
        markers.append(fm)

        if fi % 500 == 0 and fi > 0:
            n_det = sum(1 for m in markers if m)
            log.info(f"  {fi}/{total} ({fi/total*100:.1f}%)  "
                     f"frames with detections: {n_det}")

    cap.release()
    n_det = sum(1 for m in markers if m)
    log.info(f"Detection complete. Frames with tags: {n_det}/{len(markers)} "
             f"({n_det/len(markers)*100:.1f}%)")
    return markers


# ---------------------------------------------------------------------------
#  Detection report
# ---------------------------------------------------------------------------

def run_detection_report(markers, output_dir, scene_cam_w=1600, scene_cam_h=1200):
    n_total    = len(markers)
    det_frames = [fi for fi, fm in enumerate(markers) if fm]
    n_det      = len(det_frames)

    print("\n" + "=" * 65)
    print("DETECTION REPORT")
    print("=" * 65)
    print(f"Total frames     : {n_total:,}")
    print(f"Frames with tags : {n_det:,} ({n_det/n_total*100:.1f}%)")

    if n_det == 0:
        print("\nNo tags detected. Check that:")
        print("  1. TAG_FAMILY is correct (default: tag36h11)")
        print("  2. Your tags are from the tag36h11 family")
        print("  3. The scene video shows the display with markers")
        return

    step   = max(1, n_det // 200)
    sample = det_frames[::step]

    id_positions = {}
    id_counts    = Counter()
    for fi in sample:
        for tid, corners in markers[fi].items():
            cx, cy = corners.mean(axis=0)
            vpos   = "top"   if cy < scene_cam_h / 2 else "bottom"
            hpos   = "left"  if cx < scene_cam_w / 2 else "right"
            id_positions.setdefault(tid, []).append(f"{vpos}-{hpos}")
            id_counts[tid] += 1

    print(f"\nTag ID summary (sampled {len(sample)} frames):")
    for tid in sorted(id_positions):
        common = Counter(id_positions[tid]).most_common(1)[0][0]
        pct    = id_counts[tid] / len(sample) * 100
        print(f"  Tag ID {tid} -> scene position: {common:<15s}"
              f"seen in {id_counts[tid]}/{len(sample)} frames ({pct:.0f}%)")

    print(f"\nFirst frame with detections: {det_frames[0]:,}")
    print(f"Last  frame with detections: {det_frames[-1]:,}")
    print("\nIf the scene positions match your expected screen corners,")
    print("the config is correct. Run without --report-only for the full pipeline.")
    print("=" * 65 + "\n")

    report_path = output_dir / "detection_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Total frames: {n_total}\n")
        f.write(f"Frames with tags: {n_det} ({n_det/n_total*100:.1f}%)\n\n")
        for tid in sorted(id_positions):
            common = Counter(id_positions[tid]).most_common(1)[0][0]
            f.write(f"Tag ID {tid} -> {common}\n")
    log.info(f"Report saved: {report_path}")


# ---------------------------------------------------------------------------
#  Step 2 — Temporal smoothing
# ---------------------------------------------------------------------------

def smooth_marker_positions(markers, known_ids, smooth_win):
    n      = len(markers)
    raw_cx = {tid: np.full(n, np.nan) for tid in known_ids}
    raw_cy = {tid: np.full(n, np.nan) for tid in known_ids}

    for fi, fm in enumerate(markers):
        for tid, corners in fm.items():
            raw_cx[tid][fi] = corners[:, 0].mean()
            raw_cy[tid][fi] = corners[:, 1].mean()

    sm_cx = {tid: _smooth_1d(raw_cx[tid], smooth_win) for tid in known_ids}
    sm_cy = {tid: _smooth_1d(raw_cy[tid], smooth_win) for tid in known_ids}

    smoothed = []
    for fi, fm in enumerate(markers):
        new_fm = {}
        for tid, corners in fm.items():
            dx = sm_cx[tid][fi] - raw_cx[tid][fi]
            dy = sm_cy[tid][fi] - raw_cy[tid][fi]
            if np.isnan(dx) or np.isnan(dy):
                new_fm[tid] = corners
            else:
                new_fm[tid] = corners + np.array([dx, dy], dtype=np.float32)
        smoothed.append(new_fm)
    return smoothed


def _smooth_1d(arr, win):
    out  = arr.copy()
    nans = np.isnan(out)
    if nans.all():
        return out
    idx = np.arange(len(out))
    out[nans] = np.interp(idx[nans], idx[~nans], out[~nans])
    pad    = win // 2
    padded = np.pad(out, pad, mode="edge")
    for i in range(len(out)):
        out[i] = np.median(padded[i: i + win])
    return out


# ---------------------------------------------------------------------------
#  Step 3 — Homography + QA
# ---------------------------------------------------------------------------

def compute_homographies(markers, tag_bboxes, screen_w, screen_h,
                         min_markers, area_mad_thr, aspect_mad_thr):
    rows         = []
    homographies = []

    for fi, fm in enumerate(markers):
        n_det  = len(fm)
        H      = None
        area   = np.nan
        aspect = np.nan
        status = "ok"

        if n_det < min_markers:
            status = f"too_few_markers({n_det})"
        else:
            src, dst = [], []
            for tid, scene_corners in fm.items():
                src.append(tag_bboxes[tid])
                dst.append(scene_corners)
            src = np.vstack(src).astype(np.float32)
            dst = np.vstack(dst).astype(np.float32)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)

            if H is None:
                status = "homography_failed"
            else:
                sc   = np.float32([
                    [0,        0       ],
                    [screen_w, 0       ],
                    [screen_w, screen_h],
                    [0,        screen_h],
                ]).reshape(-1, 1, 2)
                proj = cv2.perspectiveTransform(sc, H).reshape(-1, 2)
                x, y = proj[:, 0], proj[:, 1]
                area = 0.5 * abs(
                    np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
                )
                ws   = np.linalg.norm(proj[1] - proj[0])
                hs   = np.linalg.norm(proj[3] - proj[0])
                aspect = ws / hs if hs > 0 else np.nan

        homographies.append(H)
        rows.append(dict(frame_idx=fi, n_detected=n_det, area_px2=area,
                         aspect_ratio=aspect, status=status, has_H=H is not None))

    qdf = pd.DataFrame(rows)

    for col, thr in [("area_px2", area_mad_thr), ("aspect_ratio", aspect_mad_thr)]:
        valid = qdf["has_H"]
        if not valid.any():
            continue
        vals = qdf.loc[valid, col].values
        med  = np.nanmedian(vals)
        mad  = np.nanmedian(np.abs(vals - med))
        if mad == 0:
            continue
        bad    = np.abs(vals - med) > thr * mad
        bad_ix = qdf.index[valid][bad]
        qdf.loc[bad_ix, "status"] = f"outlier_{col}"
        qdf.loc[bad_ix, "has_H"]  = False
        for i in bad_ix:
            homographies[i] = None

    n_valid = qdf["has_H"].sum()
    log.info(f"Valid frames: {n_valid:,}/{len(qdf):,} ({n_valid/len(qdf)*100:.1f}%)")
    return homographies, qdf


# ---------------------------------------------------------------------------
#  Step 4 — Gaze mapping
# ---------------------------------------------------------------------------

def map_gaze_to_screen(gaze_data, gaze_ts_ns, scene_times_ns,
                       homographies, K, D, screen_w, screen_h):
    rows = []
    n    = len(gaze_ts_ns)

    for gi in range(n):
        ts = gaze_ts_ns[gi]
        gx = float(gaze_data["x"][gi])
        gy = float(gaze_data["y"][gi])

        fi = int(np.searchsorted(scene_times_ns, ts))
        fi = min(fi, len(scene_times_ns) - 1)
        if fi > 0 and (abs(scene_times_ns[fi-1]-ts) < abs(scene_times_ns[fi]-ts)):
            fi -= 1

        H = homographies[fi] if fi < len(homographies) else None

        row = dict(
            gaze_timestamp_ns  = int(ts),
            gaze_timestamp_s   = ts * 1e-9,
            scene_frame_idx    = fi,
            gaze_x_norm        = gx,
            gaze_y_norm        = gy,
            gaze_x_screen_px   = np.nan,
            gaze_y_screen_px   = np.nan,
            gaze_x_screen_norm = np.nan,
            gaze_y_screen_norm = np.nan,
            surface_valid      = False,
        )

        if H is not None and not (np.isnan(gx) or np.isnan(gy)):
            pts  = np.array([[[gx, gy]]], dtype=np.float64)
            ud   = cv2.undistortPoints(pts, K, D, P=K).reshape(2)
            H_inv = np.linalg.inv(H)
            pt    = np.array([[[float(ud[0]), float(ud[1])]]], dtype=np.float32)
            scr   = cv2.perspectiveTransform(pt, H_inv).reshape(2)
            sx, sy = float(scr[0]), float(scr[1])

            row["gaze_x_screen_px"]   = sx
            row["gaze_y_screen_px"]   = sy
            row["gaze_x_screen_norm"] = sx / screen_w
            row["gaze_y_screen_norm"] = sy / screen_h
            row["surface_valid"]      = (0 <= sx <= screen_w and
                                         0 <= sy <= screen_h)

        rows.append(row)
        if gi % 10000 == 0:
            log.info(f"  Gaze {gi:,}/{n:,} ({gi/n*100:.1f}%)")

    gaze_df = pd.DataFrame(rows)
    n_valid = gaze_df["surface_valid"].sum()
    log.info(f"On-surface: {n_valid:,}/{n:,} ({n_valid/n*100:.1f}%)")
    return gaze_df


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args       = parse_args()
    rec_dir    = Path(args.recording)
    output_dir = rec_dir / "gaze_pipeline_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Recording : {rec_dir}")
    log.info(f"Output    : {output_dir}")

    # Build tag layout from screen dimensions
    tag_centers, tag_bboxes = build_tag_layout(args.screen_w, args.screen_h)
    known_ids = set(tag_centers.keys())

    # Load recording
    rec = load_neon_recording(rec_dir)

    # Step 1: detection (cached)
    markers_raw = get_markers(
        scene_video      = rec["scene_video"],
        output_dir       = output_dir,
        known_ids        = known_ids,
        force_redetect   = args.force_redetect,
        quad_decimate    = DEFAULTS["quad_decimate"],
        decode_sharpening= DEFAULTS["decode_sharpening"],
        nthreads         = DEFAULTS["nthreads"],
        tag_family       = DEFAULTS["tag_family"],
    )

    # Detection report mode
    if args.report_only:
        run_detection_report(markers_raw, output_dir)
        log.info("Report done. Run without --report-only for the full pipeline.")
        return

    # Step 2: smooth
    log.info(f"Step 2/4: Smoothing (window={args.smooth_win} frames)...")
    markers_smooth = smooth_marker_positions(markers_raw, known_ids, args.smooth_win)

    # Step 3: homographies
    log.info("Step 3/4: Computing homographies + QA...")
    homographies, quality_df = compute_homographies(
        markers_smooth, tag_bboxes,
        args.screen_w, args.screen_h,
        args.min_markers,
        DEFAULTS["area_mad_threshold"],
        DEFAULTS["aspect_mad_threshold"],
    )
    quality_df.to_csv(output_dir / "surface_quality.csv", index=False)
    log.info("Saved surface_quality.csv")

    # Step 4: gaze mapping
    log.info("Step 4/4: Mapping gaze...")
    gaze_df = map_gaze_to_screen(
        gaze_data      = rec["gaze_data"],
        gaze_ts_ns     = rec["gaze_ts_ns"],
        scene_times_ns = rec["scene_times_ns"],
        homographies   = homographies,
        K              = rec["K"],
        D              = rec["D"],
        screen_w       = args.screen_w,
        screen_h       = args.screen_h,
    )
    gaze_df.to_csv(output_dir / "gaze_on_surface.csv", index=False)
    log.info("Saved gaze_on_surface.csv")

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Total gaze samples : {len(gaze_df):,}")
    print(f"  On-surface samples : {gaze_df['surface_valid'].sum():,}")
    print(f"  Surface coverage   : {gaze_df['surface_valid'].mean()*100:.1f}%")
    print(f"  Valid scene frames : {quality_df['has_H'].sum():,}/{len(quality_df):,}")
    print(f"\n  Outputs: {output_dir}")
    print("    gaze_on_surface.csv")
    print("    surface_quality.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
