import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Users\Forouzan\real-time-blink-detection")))
from blink_detector.helper import preprocess_recording, get_classifier

rec = r"D:\Data\Temp-Neon\2026-05-11-17-27-07"

print("Loading recording...")
left, right, ts = preprocess_recording(rec, is_neon=True)
print(f"Loaded {len(ts)} samples")

print("Loading classifier...")
t0 = time.time()
clf = get_classifier(None)
print(f"Classifier loaded in {time.time()-t0:.1f}s")
print(f"Classifier type: {type(clf)}")

print("Running one prediction to test...")
import numpy as np
dummy = np.zeros((1, left.shape[1] if hasattr(left, 'shape') else 10))
print(f"Left data shape: {left.shape if hasattr(left, 'shape') else type(left)}")
print(f"Right data shape: {right.shape if hasattr(right, 'shape') else type(right)}")
print(f"Timestamps shape: {ts.shape if hasattr(ts, 'shape') else len(ts)}")
print("Done.")
