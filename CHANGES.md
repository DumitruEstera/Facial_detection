# Facial recognition refactor — 2026-04-21

Consolidated the face pipeline from three models (YuNet + facenet-pytorch + DeepFace) down to two (InsightFace + FER) to reduce VRAM pressure, simplify threading, and fix bad age/gender predictions.

## Summary of the refactor

| Concern            | Before                         | After                                     |
|--------------------|--------------------------------|-------------------------------------------|
| Face detection     | YuNet (`face_detection.py`)    | InsightFace `FaceAnalysis`                |
| Embeddings (128-d) | facenet-pytorch                | InsightFace `w600k_mbf` (512-d)           |
| Age / gender       | DeepFace (slow, inaccurate)    | InsightFace `genderage` head (same pass)  |
| Emotion            | DeepFace                       | FER (Haar cascade, CPU)                   |
| Faiss dim          | 128                            | 512                                       |

`face_detection.py` and `feature_extraction.py` are now deprecation stubs; all logic lives in `facial_recognition_system.py`.

## Files touched

- `requirements.txt` — removed `deepface`, `facenet-pytorch`; added `insightface==0.7.3`, `onnxruntime-gpu`, `fer==22.5.1`.
- `facial_recognition_system.py` — complete rewrite around `FaceAnalysis` + `FER`.
- `face_detection.py`, `feature_extraction.py` — `DeprecationWarning` on import.
- `app.py` — removed DeepFace/`FaceDemographicsAnalyzer` and the separate demographics queue; upload endpoint now calls `face_system.extract_embedding_from_image`.
- `gui_application.py` — same DeepFace removal.
- `start_enhanced_system.py` — dependency check now probes `insightface`, `onnxruntime`, `fer`, and `CUDAExecutionProvider`.
- `database_manager.py` — `sslmode='disable'` added (see psycopg2 segfault below).

## Problems hit and the fixes

### 1. NumPy 2.x ABI break
Some dependency pulled numpy 2.x transitively, breaking the cv2/torch/onnxruntime wheels (all compiled against 1.x).
**Fix:** `pip install "numpy==1.26.4" "opencv-python==4.10.0.84" "opencv-python-headless==4.10.0.84" --force-reinstall` in a single command (doing it in two kept pulling numpy 2 back in).

### 2. `ModuleNotFoundError: No module named 'moviepy.editor'`
FER 22.5.1 imports `moviepy.editor`, which was removed in moviepy 2.
**Fix:** `pip install "moviepy<2.0"`.

### 3. Segfault during `psycopg2.connect`
`python -X faulthandler` traced the crash to psycopg2 inside the libpq/OpenSSL it bundles, colliding with the OpenSSL already loaded by torch/TF in the same process.
**Fix:** `'sslmode': 'disable'` in `database_manager.py`'s connection params.

### 4. Registration produced 20 faces but 0 embeddings
`load_faces_from_files` was returning tight face crops. `register_person` then re-ran InsightFace on those crops, and the detector needs context around the face — zero detections, zero embeddings.
**Fix:** `load_faces_from_files` and `capture_faces_for_registration` now return **full images**. InsightFace re-detects at natural scale in `register_person`.

### 5. `libcublasLt.so.11` missing (onnxruntime)
`onnxruntime-gpu` 1.17–1.18 on PyPI are built against CUDA 11.8.
**Fix:** bumped to `onnxruntime-gpu==1.19.2` (CUDA 12 default on PyPI).

### 6. `libcudnn.so.9` missing (onnxruntime 1.19)
Installed cuDNN 9 with `pip install "nvidia-cudnn-cu12>=9.1,<10"`, and added `_preload_nvidia_pypi_libs()` at the top of `facial_recognition_system.py` that `ctypes.CDLL(..., RTLD_GLOBAL)`s every relevant `libcudart/libcublas/libcudnn/...` from `site-packages/nvidia/*/lib`. This avoids having to fiddle with `LD_LIBRARY_PATH` — once the soname is in the process's loaded-object set, later `dlopen`s short-circuit the linker search.

### 7. `libcudnn.so.8` missing (torch)
After bumping site-packages cuDNN to 9 for onnxruntime, torch 2.2.2+cu121 broke because its RPATH looks for cuDNN 8 in the same `site-packages/nvidia/cudnn/lib`.
**Fix:** install cuDNN 8 **to a side-car path** so it doesn't clobber cuDNN 9:
```bash
pip install --target ~/.local/cudnn8_pkg --no-deps "nvidia-cudnn-cu12==8.9.2.26"
```
Extended `_preload_nvidia_pypi_libs()` to also scan `~/.local/cudnn8_pkg/nvidia/cudnn/lib` and preload every `libcudnn*.so.8` it finds. Both versions coexist in-process.

### 8. Qt Wayland plugin missing
cv2 ships only the xcb platform plugin.
**Fix:** `export QT_QPA_PLATFORM=xcb` before running.

### 9. `age=None`, `gender=None`
Initially used `name="buffalo_sc"`, which lacks the `genderage.onnx` head.
**Fix:** switched to `name="buffalo_s"` — same recognizer (`w600k_mbf`, so existing embeddings stay valid) plus detection + genderage. First run auto-downloads to `~/.insightface/models/buffalo_s/`.

### 10. Recognition returned Unknown with confidence 0.00
Threshold was tuned for ResNet50 ArcFace (`L2 ≈ 1.10`), but buffalo_s uses MobileFaceNet whose operating distribution is different.
**Fix:**
- Added `FR_DEBUG=1` env flag that logs the actual top-1 L2 distance per face so the threshold can be calibrated empirically.
- Empirical measurement: same-person L2 sits in **0.6 – 0.85**.
- `recognition_threshold` tightened to `1.0`, `min_confidence` set to `0.4`.
- Confidence is now reported as **cosine similarity** (`cos_sim = 1 - L2²/2`), which matches the standard face-ID metric and reads more intuitively (0.7–0.8 for a solid match).

### 11. `emotion=None` almost always
FER's Haar cascade struggles on the tight InsightFace bbox crop.
**Fix:** pad the crop by 30% on each side before passing it to FER.

### 12. `CUDA error: operation not permitted when stream is capturing`
Running InsightFace (onnxruntime) concurrently with torch models (YOLO plate / fire / weapon / SlowFast HAR) trips every parallel torch kernel. Root cause: onnxruntime's default `cudnn_conv_algo_search='EXHAUSTIVE'` puts the CUDA stream into capture mode to benchmark each cuDNN algorithm, and other threads can't submit work to a stream that is being captured.
**Fix:** pass `cudnn_conv_algo_search='HEURISTIC'` via `provider_options` to `FaceAnalysis`. Heuristic selection skips the benchmark (no capture), at a small one-shot latency cost.

### 13. Demographics showed on known faces too
New `process_frame` ran genderage + FER on every face unconditionally.
**Fix:** gated the demographics block on `is_known` — known persons get `age=gender=emotion=None`, unknowns still get the full demographics. FER (the expensive CPU call) is skipped for known faces.

### 14. "Alerts and Logs are empty"
Looked like data loss after the `TRUNCATE persons ... CASCADE` wipe.
**Actual cause:** expired JWT in the frontend — `/api/alarms/stats` returned 401, so the tables came up empty. Re-login restored everything.

## How to bring the env up from scratch

```bash
# Core
pip install -r requirements.txt

# Pin numpy/cv2 to the 1.x-compatible line
pip install "numpy==1.26.4" "opencv-python==4.10.0.84" "opencv-python-headless==4.10.0.84" --force-reinstall

# FER needs moviepy<2
pip install "moviepy<2.0"

# cuDNN 8 side-car (for torch), cuDNN 9 in site-packages (for onnxruntime)
pip install "nvidia-cudnn-cu12>=9.1,<10"
pip install --target ~/.local/cudnn8_pkg --no-deps "nvidia-cudnn-cu12==8.9.2.26"

# Wipe legacy FaceNet embeddings
psql -d facial_recognition -c "TRUNCATE TABLE persons RESTART IDENTITY CASCADE;"

# Re-register one person, then run
export QT_QPA_PLATFORM=xcb
python example_usage.py --mode register --name "..." --employee-id "..." --images-dir ./face_dir
python app.py
```
Changes 2026-04-21
Summary of changes to facial_recognition_system.py:                                                                                                   
                                                                                                                                                      
  - Removed from fer import FER and the FER(mtcnn=False) initialisation.                                                                                
  - Added direct Mini-Xception load via tf.keras.models.load_model(...). The .hdf5 weights still come from the installed fer package                    
  (importlib.resources), so no file needs to be vendored into the repo.                                                                                 
  - Rewrote _classify_emotion to do the preprocessing inline: BGR→gray → resize to (W, H) from model.input_shape → scale to [-1, 1] (FER's v2           
  preprocessing) → (1, H, W, 1) tensor → argmax over the 7 FER-2013 labels. No Haar cascade, no second face-detection pass.                             
  - Removed the 30% padding in process_frame (facial_recognition_system.py:418) — the Mini-Xception CNN is now fed the tight InsightFace bbox directly. 
  If accuracy looks weak on certain angles, add a small ~10% pad back in; Haar's 30% is no longer warranted.                                            
  - Updated the TF-on-CPU comment block, class docstring, and process_frame docstring to reflect the new pipeline.

# Fire & smoke detection overhaul — 2026-04-23

Swapped the fire/smoke model for the HuggingFace **TommyNgx/YOLOv10-Fire-and-Smoke-Detection** (`fire_and_smoke_model_huggingface/best.pt`), removed the 4-level severity system, and added several false-positive filters.

## Why
The previous model (`models/fire_smoke_best.pt`, luminous0219 YOLOv8n) produced frequent false positives on grey walls, reflections, and warm-coloured objects. The severity ladder (`low`/`medium`/`high`/`critical`) mixed confidence and box size into one label, used `low` as a silent-reject signal, and made tuning opaque.

## `fire_detection_system.py` — rewritten
- Default model path: `fire_and_smoke_model_huggingface/best.pt`.
- Class names pulled from `model.names` and normalised to lowercase (`{0: 'fire', 1: 'smoke'}`).
- **Dropped `_calculate_severity` and the `severity` field entirely.** The detection dict now exposes `confirmed` and `alert` as the only gates.
- **Per-class confidence thresholds** — fire `0.40`, smoke `0.55` (smoke is far more FP-prone, so it's stricter).
- **Size sanity filters** — reject boxes smaller than 0.3% or larger than 85% of frame area.
- **HSV colour verification** —
  - fire: ≥12% of ROI pixels must be warm (hue ≤25 or ≥160) **and** saturated (S≥90, V≥120).
  - smoke: ≥55% low-saturation pixels (S≤60) that aren't pitch black (V≥40), plus a non-uniform-brightness check to reject clear sky / flat walls.
- **Multi-frame IoU confirmation** kept; confirmation window reduced from 8 to 6 frames now that the above filters remove most single-frame noise.
- Alert cooldown key coarsened (`class_x//40_y//40`) so a slightly-drifting true-positive box isn't treated as a brand-new alert each frame.
- `draw_detections` only draws **confirmed** boxes; non-confirmed candidates are invisible.
- Statistics extended with `rejected_by_size` and `rejected_by_color` counters for tuning.

## Downstream callsite updates
- `app.py` — fire-processing thread logs fire alerts based on `class == 'fire' and alert`, not severity. `_log_detection_if_needed` and `_create_alarm_if_needed` now require `confirmed` (and `alert` for alarms) and map severity simply: fire → `critical`, smoke → `high`, so the DB column stays populated without resurrecting the ladder. The `has_alarm` banner/snapshot trigger also requires confirmed+alert fire results.
- `gui_application.py` — top status uses `confirmed` + `class == 'fire' && alert` for the red banner; fire log only accepts confirmed detections; the "Severity" column is kept for layout compatibility but now derived (`fire` → `CRITICAL`, `smoke` → `HIGH`).
- `frontend/src/App.js` — **this was the source of the "ghost alerts" bug**: the Recent Activity panel was pushing a "Fire Detection Alert" on *any* non-empty `fire_results`, including unconfirmed single-frame candidates. Since my backend tightening limited drawn boxes and DB alarms to `confirmed && alert`, the frontend was showing alerts that had no matching bounding box and no alarm-dashboard entry. Frontend now filters `fire_results` to `confirmed && alert` before building Recent Activity entries, and uses the same severity mapping as the backend. Requires a frontend rebuild to take effect.

## Files touched
- `fire_detection_system.py` — rewrite.
- `app.py` — three callsites (thread log, detection-log writer, alarm creator) plus `has_alarm` trigger.
- `gui_application.py` — status label, fire-log filter, `add_to_fire_log` severity derivation.
- `frontend/src/App.js` — fire alert gate.

## Follow-up suggestions (not implemented)
1. Per-camera ignore-regions to mask chronic FP spots (signs, radiators, windows).
2. Temporal flicker check for fire (ROI intensity stddev over N frames) to separate real flames from static warm objects.
3. Day/night-aware thresholds — raise the fire threshold in low-light frames (headlights, streetlights look like fire).
4. Higher YOLO `imgsz` (960 or 1280) on high-resolution cameras to cut FPs at distance.
5. Collect ~100–200 hard-negative frames from the real cameras and fine-tune the HuggingFace checkpoint on them — highest-leverage fix if FPs persist.
6. Cross-class agreement bonus: treat co-located fire+smoke as higher confidence than either alone.
           