"""
Diagnostic: read calibration using the neon_recording library.
This shows us the correct camera matrix and distortion coefficients,
and confirms the gaze field names and scene camera resolution.
"""
from pathlib import Path

rec_dir = Path(r"D:\Data\Temp-Neon\2026-05-11-17-27-07")

# ── Try neon_recording library ────────────────────────────────────────────────
print("── Loading via pupil_labs.neon_recording ──")
try:
    from pupil_labs.neon_recording import NeonRecording
    rec = NeonRecording(str(rec_dir))

    print(f"\nRecording loaded OK")
    print(f"Scene: {rec.scene}")

    # Try common attribute names for calibration
    calib = rec.calibration
    print(f"\ncalibration object: {calib}")
    print(f"calibration type  : {type(calib)}")
    print(f"calibration dir   : {[a for a in dir(calib) if not a.startswith('_')]}")

    # Try to get camera matrix
    for attr in ["scene_camera_matrix", "camera_matrix", "K", "intrinsics"]:
        if hasattr(calib, attr):
            print(f"\ncalib.{attr}:\n{getattr(calib, attr)}")

    for attr in ["scene_distortion_coefficients", "distortion_coefficients",
                 "D", "dist_coeffs", "distortion"]:
        if hasattr(calib, attr):
            print(f"\ncalib.{attr}:\n{getattr(calib, attr)}")

except Exception as e:
    print(f"neon_recording failed: {e}")

# ── Also inspect raw binary manually ─────────────────────────────────────────
print("\n\n── Raw float64 parse of calibration.bin ──")
import numpy as np, struct
data = (rec_dir / "calibration.bin").read_bytes()
print(f"File size: {len(data)} bytes")

# Skip byte 0 (version=1) and bytes 1-6 ("117280" serial)
offset = 7
floats = []
while offset + 8 <= len(data):
    val = struct.unpack_from("<d", data, offset)[0]
    floats.append((offset, val))
    offset += 8

print(f"\nAll float64 values (little-endian) starting at byte 7:")
for off, v in floats:
    print(f"  byte {off:4d}: {v:.6f}")

print(f"\nTotal float64 values: {len(floats)}")
print(f"Bytes after last float64: {len(data) - (7 + len(floats)*8)}")
