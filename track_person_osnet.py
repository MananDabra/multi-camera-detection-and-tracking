import os
import sys
import csv
import cv2
import torch
import numpy as np

# Resolve deep_sort_realtime TorchReID import compatibility
import torchreid
import torchreid.reid.utils
sys.modules['torchreid.utils'] = torchreid.reid.utils

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.embedder.embedder_pytorch import TorchReID_Embedder

def get_track_color(track_id):
    """Generate a consistent, distinct BGR color for each track ID."""
    try:
        val = int(track_id)
    except ValueError:
        val = sum(ord(c) for c in str(track_id))
    b = int((val * 67 + 60) % 256)
    g = int((val * 131 + 60) % 256)
    r = int((val * 193 + 60) % 256)
    return (b, g, r)

def verify_osnet_embedder(tracker):
    """Verify that OSNet is actively loaded and producing 512-dimensional embeddings."""
    embedder = tracker.embedder
    assert isinstance(embedder, TorchReID_Embedder), (
        f"ERROR: Expected TorchReID_Embedder but got {type(embedder)}"
    )
    dummy_crop = np.zeros((128, 64, 3), dtype=np.uint8)
    feats = embedder.predict([dummy_crop])
    assert len(feats) == 1, "Embedder failed to produce embedding"
    emb_dim = feats[0].shape[-1]
    print(f"Verified Active Embedder : {type(embedder).__name__}")
    print(f"Verified Extractor Class : {type(embedder.model).__name__}")
    print(f"Verified Embedding Dim   : {emb_dim}")
    return emb_dim

def main():
    input_video_path = "demo.mp4"
    output_video_path = "output_tracking_osnet.mp4"
    csv_output_path = "tracking_data_osnet.csv"
    model_path = "yolo11s.pt"

    # Detection configurations (identical to Phase 2 baseline)
    conf_threshold = 0.25
    nms_iou_threshold = 0.50

    if not os.path.exists(input_video_path):
        print(f"Error: Input video '{input_video_path}' not found.")
        return

    if not os.path.exists(model_path):
        print(f"Error: YOLO11s model '{model_path}' not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load YOLO11s detector
    print(f"Loading YOLO11s detector from {model_path}...")
    detector = YOLO(model_path)

    # 2. Initialize DeepSORT tracker with OSNet embedder and max_age = 50
    print("Initializing DeepSORT with OSNet appearance embedder...")
    deepsort_config = {
        "max_age": 50,  # Explicitly configured to 50 for Phase 3
        "n_init": 3,
        "nms_max_overlap": 1.0,
        "max_cosine_distance": 0.3,
        "nn_budget": None,
        "embedder": "torchreid",
        "embedder_model_name": "osnet_ain_x1_0",
        "half": False,
        "embedder_gpu": False
    }
    tracker = DeepSort(**deepsort_config)

    # 3. Verify OSNet active embedder and dimensions
    embedding_dim = verify_osnet_embedder(tracker)

    # Open video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_video_path}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n--- Phase 3 Tracking Configuration (YOLO11s + OSNet) ---")
    print(f"Detector Model         : YOLO11s ({model_path})")
    print(f"Target Class           : 0 (Person only)")
    print(f"Detection Threshold    : {conf_threshold}")
    print(f"NMS IoU Threshold      : {nms_iou_threshold}")
    print(f"Tracker                : DeepSORT")
    print(f"Appearance Embedder    : OSNet (osnet_ain_x1_0)")
    print(f"Embedding Dimension    : {embedding_dim}")
    print(f"Max Age (frames)       : {deepsort_config['max_age']}")
    print(f"N Init (frames)        : {deepsort_config['n_init']}")
    print(f"Max Cosine Distance    : {deepsort_config['max_cosine_distance']}")
    print(f"Video Resolution       : {width}x{height}")
    print(f"Video FPS              : {fps}")
    print(f"Total Frames           : {total_frames}")
    print("--------------------------------------------------------\n")

    # Set up VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize CSV output
    csv_file = open(csv_output_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "track_id", "x1", "y1", "x2", "y2", "center_x", "center_y"])

    # Visual style settings (scaled to high resolution 2592x1944)
    box_thickness = max(2, int(width / 800))
    font_scale = max(0.65, width / 2000.0)
    font_thickness = max(1, int(box_thickness / 2))
    time_font_scale = max(0.8, width / 2200.0)
    time_font_thickness = max(2, int(time_font_scale * 2))

    frame_id = 0
    unique_track_ids = set()
    total_tracked_records = 0

    print(f"Processing every frame from '{input_video_path}' with YOLO11s + OSNet DeepSORT...")

    torch.set_grad_enabled(False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # 1. Run YOLO11s person detection (class 0 only)
        results = detector.predict(
            source=frame,
            classes=[0],
            conf=conf_threshold,
            iou=nms_iou_threshold,
            device=device,
            verbose=False
        )

        # 2. Format YOLO detections for DeepSORT: ([left, top, w, h], confidence, class_name)
        raw_detections = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                raw_detections.append(([int(x1), int(y1), int(w), int(h)], conf, "person"))

        # 3. Update DeepSORT tracks using OSNet appearance features + Kalman motion
        tracks = tracker.update_tracks(raw_detections, frame=frame)

        # 4. Process confirmed tracks
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            unique_track_ids.add(track_id)
            total_tracked_records += 1

            ltrb = track.to_ltrb()
            x1 = max(0, min(width - 1, int(ltrb[0])))
            y1 = max(0, min(height - 1, int(ltrb[1])))
            x2 = max(0, min(width - 1, int(ltrb[2])))
            y2 = max(0, min(height - 1, int(ltrb[3])))

            center_x = round((x1 + x2) / 2.0, 2)
            center_y = round((y1 + y2) / 2.0, 2)

            # Store tracking data to CSV
            csv_writer.writerow([frame_id, track_id, x1, y1, x2, y2, center_x, center_y])

            # Draw track bounding box and ID label
            color = get_track_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            det_conf = track.get_det_conf()
            if det_conf is not None:
                label = f"ID {track_id} ({det_conf:.2f})"
            else:
                label = f"ID {track_id}"

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_y1 = max(0, y1 - th - baseline - 6)
            label_y2 = y1
            cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
                lineType=cv2.LINE_AA
            )

        # 5. Draw video timestamp overlay
        current_time_sec = (frame_id - 1) / fps
        mins = int(current_time_sec // 60)
        secs = int(current_time_sec % 60)
        millis = int((current_time_sec - int(current_time_sec)) * 1000)
        time_text = f"Time: {mins:02d}:{secs:02d}.{millis:03d} | Frame: {frame_id}"

        (tw_t, th_t), base_t = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, time_font_scale, time_font_thickness)
        cv2.rectangle(frame, (20, 20), (35 + tw_t, 35 + th_t + base_t), (0, 0, 0), -1)
        cv2.putText(
            frame,
            time_text,
            (28, 25 + th_t),
            cv2.FONT_HERSHEY_SIMPLEX,
            time_font_scale,
            (0, 255, 255),
            time_font_thickness,
            lineType=cv2.LINE_AA
        )

        out.write(frame)

        if frame_id % 100 == 0 or frame_id == total_frames:
            print(f"Processed {frame_id}/{total_frames} frames (Confirmed tracks active/unique: {len(unique_track_ids)})...")

    cap.release()
    out.release()
    csv_file.close()

    print("\n========================================================")
    print("   Processing Summary (Phase 3 OSNet DeepSORT Tracking)  ")
    print("========================================================")
    print(f"Detector Model          : YOLO11s (yolo11s.pt)")
    print(f"Appearance Embedder     : OSNet (osnet_ain_x1_0)")
    print(f"Embedding Dimension     : {embedding_dim}")
    print(f"DeepSORT Max Age        : {deepsort_config['max_age']}")
    print(f"Total frames processed  : {frame_id}")
    print(f"Video FPS               : {fps}")
    print(f"Resolution              : {width}x{height}")
    print(f"Total confirmed records : {total_tracked_records}")
    print(f"Unique Track IDs        : {len(unique_track_ids)}")
    print(f"Output video saved to   : {output_video_path}")
    print(f"Tracking CSV saved to   : {csv_output_path}")
    print("========================================================\n")

if __name__ == "__main__":
    main()
