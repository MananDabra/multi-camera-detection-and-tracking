# Single Camera Multi-Person Tracking

A single-camera multi-person detection and tracking project implementing modular phases from YOLOv11 person detection to DeepSORT identity tracking with OSNet appearance features.

---

## Project Structure

```
├── .gitignore
├── README.md
├── detect_person.py          # Phase 1: YOLOv11 person detection pipeline
├── track_person.py           # Phase 2: YOLO11s + DeepSORT (MobileNet) baseline tracker
├── track_person_osnet.py     # Phase 3: YOLO11s + DeepSORT (OSNet Re-ID) tracker
└── tracking_data.csv         # Exported tracking coordinates (frame, ID, bbox, center)
```

---

## Pipeline Overview

### Phase 1: Person Detection
- **Model**: YOLOv11 (`yolo11n.pt` / `yolo11s.pt`)
- **Class**: COCO Class 0 (`person` only)
- **Configuration**: `conf = 0.25`, `iou = 0.50`
- Preserves original video resolution and frame rate, overlays frame-accurate timestamps.

### Phase 2: Multi-Person Tracking (Baseline DeepSORT)
- Integrates YOLO11s detections with standard DeepSORT Kalman filter and MobileNet appearance embedder.
- Maintains persistent IDs across consecutive frames and exports trajectory coordinates to CSV.

### Phase 3: Stronger Appearance Tracking (OSNet Embedder)
- Upgrades the appearance feature extractor to **OSNet** (`osnet_ain_x1_0`).
- Generates 512-dimensional appearance embedding vectors per person crop.
- Employs `max_age = 50` for robust identity re-association during occlusions.

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- PyTorch
- OpenCV
- Ultralytics
- deep-sort-realtime
- torchreid

### Running Detection (Phase 1)
```bash
python detect_person.py
```

### Running Baseline Tracking (Phase 2)
```bash
python track_person.py
```

### Running OSNet Tracking (Phase 3)
```bash
python track_person_osnet.py
```
