# Patches for real-time-blink-detection

The blink detector requires two patches to the
[real-time-blink-detection](https://github.com/pupil-labs/real-time-blink-detection)
repo for acceptable performance on a full Neon recording.
Without these changes, detection on a 30-minute recording can appear to stall.

---

## Prerequisites

- NVIDIA GPU with CUDA support
- XGBoost 2.0 or later

Install the correct XGBoost version:

```bash
pip uninstall xgboost -y
pip install xgboost --upgrade
```

Verify GPU support:

```bash
python -c "import xgboost as xgb; xgb.train({'device': 'cuda', 'tree_method': 'hist'}, xgb.DMatrix([[1,2],[3,4]], label=[0,1]), num_boost_round=1); print('GPU OK')"
```

---

## Patch 1: GPU acceleration for XGBoost

**File:** `blink_detector/helper.py`

**Why:** The original code runs XGBoost inference on CPU. Moving to GPU
gives a large speedup on any NVIDIA GPU with CUDA support.

**Before:**
```python
def get_classifier(is_neon: bool = True):
    clf_path = get_clf_path(is_neon)
    clf = XGBClassifier()
    clf.load_model(clf_path)
    return clf
```

**After:**
```python
def get_classifier(is_neon: bool = True):
    clf_path = get_clf_path(is_neon)
    clf = XGBClassifier(device="cuda", tree_method="hist")
    clf.load_model(clf_path)
    return clf
```

---

## Patch 2: Batched GPU prediction

**File:** `blink_detector/blink_detector.py`

**Why:** The original `predict_class_probas` function calls
`clf.predict_proba(features)` one sample at a time, resulting in hundreds
of thousands of individual GPU calls for a typical recording. This causes
the GPU to sit mostly idle and the pipeline to appear stuck. Batching
2048 samples per call makes full use of the GPU.

**Before:**
```python
def predict_class_probas(
    optical_flow: T.Generator, clf: XGBClassifier, of_params: OfParams = OfParams()
) -> T.Generator:
    window_length = (of_params.n_layers - 1) * of_params.layer_interval + 1
    window_center = (window_length - 1) // 2

    optical_flow = pad_beginning(optical_flow, window_center)

    indices = np.arange(0, window_length, of_params.layer_interval)

    for window in windowed(optical_flow, n=window_length):
        timestamp = window[window_center][1]
        features = np.hstack([window[i][0] for i in indices])[None, :]
        probas = clf.predict_proba(features)[0]

        yield probas, timestamp
```

**After:**
```python
def predict_class_probas(
    optical_flow: T.Generator, clf: XGBClassifier, of_params: OfParams = OfParams(),
    batch_size: int = 2048
) -> T.Generator:
    window_length = (of_params.n_layers - 1) * of_params.layer_interval + 1
    window_center = (window_length - 1) // 2
    optical_flow = pad_beginning(optical_flow, window_center)
    indices = np.arange(0, window_length, of_params.layer_interval)

    batch_features = []
    batch_timestamps = []

    for window in windowed(optical_flow, n=window_length):
        timestamp = window[window_center][1]
        features = np.hstack([window[i][0] for i in indices])
        batch_features.append(features)
        batch_timestamps.append(timestamp)

        if len(batch_features) >= batch_size:
            X = np.vstack(batch_features)
            probas_batch = clf.predict_proba(X)
            for probas, ts in zip(probas_batch, batch_timestamps):
                yield probas, ts
            batch_features = []
            batch_timestamps = []

    # flush remaining samples
    if batch_features:
        X = np.vstack(batch_features)
        probas_batch = clf.predict_proba(X)
        for probas, ts in zip(probas_batch, batch_timestamps):
            yield probas, ts
```

---

## Note on preprocessing speed

The preprocessing step (loading and decoding the eye video) is limited by
sequential video decoding via PyAV and cannot be easily parallelized.
On a 30-minute Neon recording this typically takes several minutes regardless
of hardware. The two patches above address the detection step which was the
main bottleneck.
