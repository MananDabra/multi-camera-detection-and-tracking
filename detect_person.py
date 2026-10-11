import os
import cv2
import torch
import datetime
from ultralytics import YOLO

def main():
    input_video_path = "demo.mp4"
    # Append current time to output filename (using '-' instead of ':' as Windows prohibits ':' in filenames)
    output_video_path = "output_detection_yolo11s.mp4"
    model_path = "yolo11s.pt"

    # Configuration for overlapping person detection testing
    conf_threshold = 0.25  # Lowered from 0.40 to 0.25
    iou_threshold = 0.50   # Explicitly configured NMS IoU threshold

    if not os.path.exists(input_video_path):
        print(f"Error: Input video '{input_video_path}' not found.")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load YOLOv11 model
    print(f"Loading YOLOv11 model from {model_path}...")
    model = YOLO(model_path)

    # Open video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video '{input_video_path}'.")
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n--- Video & Detection Configuration ---")
    print(f"Resolution         : {width}x{height}")
    print(f"FPS                : {fps}")
    print(f"Total Frames       : {total_frames}")
    print(f"Previous Conf Thres: 0.40")
    print(f"Current Conf Thres : {conf_threshold}")
    print(f"NMS IoU Threshold  : {iou_threshold}")
    print("---------------------------------------\n")

    # Diagnostic hook: capture raw person proposals before NMS suppression
    raw_proposals_count = [0]
    _ = model.predict(
        source=torch.zeros((1, 3, 640, 640)),
        conf=conf_threshold,
        iou=iou_threshold,
        classes=[0],
        verbose=False
    )
    orig_postprocess = model.predictor.postprocess

    def diagnostic_postprocess(preds, img, orig_imgs, **kwargs):
        p = preds[0] if isinstance(preds, (list, tuple)) else preds
        # Class 0 (person) raw confidence proposals across all feature map anchor cells
        raw_proposals_count[0] = int((p[:, 4, :] > conf_threshold).sum().item())
        return orig_postprocess(preds, img, orig_imgs, **kwargs)

    model.predictor.postprocess = diagnostic_postprocess

    # Set up VideoWriter to preserve original resolution and FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_person_detections = 0
    overlap_frames_diagnostics = []

    # Visual style settings (proportional to 2592x1944 resolution)
    box_thickness = max(2, int(width / 800))
    font_scale = max(0.6, width / 2000.0)
    font_thickness = max(1, int(box_thickness / 2))
    time_font_scale = max(0.8, width / 2200.0)
    time_font_thickness = max(2, int(time_font_scale * 2))

    print(f"Processing frames from '{input_video_path}'...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Calculate timestamp for current frame
        current_time_sec = (frame_count - 1) / fps
        mins = int(current_time_sec // 60)
        secs = int(current_time_sec % 60)
        millis = int((current_time_sec - int(current_time_sec)) * 1000)
        time_text = f"Time: {mins:02d}:{secs:02d}.{millis:03d} | Frame: {frame_count}"

        # Run YOLOv11 inference for class 0 (person)
        results = model.predict(
            source=frame,
            classes=[0],          # COCO class 0: person
            conf=conf_threshold,  # 0.25
            iou=iou_threshold,    # 0.50 NMS IoU
            device=device,
            verbose=False
        )

        boxes = results[0].boxes
        num_detections = len(boxes) if boxes is not None else 0
        raw_proposals = raw_proposals_count[0]

        # Check for overlapping bounding boxes among detections
        overlap_found = False
        max_iou_in_frame = 0.0
        if boxes is not None and num_detections >= 2:
            box_coords = boxes.xyxy.cpu().numpy()
            for i in range(num_detections):
                for j in range(i + 1, num_detections):
                    b1, b2 = box_coords[i], box_coords[j]
                    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    denom = a1 + a2 - inter
                    iou = inter / denom if denom > 0 else 0
                    if iou > max_iou_in_frame:
                        max_iou_in_frame = iou
                    if iou > 0.15:
                        overlap_found = True

        # Log diagnostic details for significant overlap events
        if overlap_found or (raw_proposals > 20 and num_detections <= 2):
            if len(overlap_frames_diagnostics) < 20 or frame_count % 50 == 0:
                overlap_frames_diagnostics.append(
                    (frame_count, time_text, raw_proposals, num_detections, round(float(max_iou_in_frame), 3))
                )

        # Draw bounding boxes and confidence scores
        if boxes is not None and num_detections > 0:
            for box in boxes:
                total_person_detections += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Green bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

                # Label text
                label = f"Person {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                label_y1 = max(0, y1 - th - baseline - 6)
                label_y2 = y1
                cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 6, label_y2), (0, 255, 0), -1)
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

        # Draw timestamp overlay in the top-left corner
        (tw_time, th_time), base_time = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, time_font_scale, time_font_thickness)
        cv2.rectangle(frame, (20, 20), (35 + tw_time, 35 + th_time + base_time), (0, 0, 0), -1)
        cv2.putText(
            frame,
            time_text,
            (28, 25 + th_time),
            cv2.FONT_HERSHEY_SIMPLEX,
            time_font_scale,
            (0, 255, 255),  # Yellow for high visibility
            time_font_thickness,
            lineType=cv2.LINE_AA
        )

        # Write frame
        out.write(frame)

        if frame_count % 100 == 0 or frame_count == total_frames:
            print(f"Processed {frame_count}/{total_frames} frames (Detections so far: {total_person_detections})...")

    # Release resources
    cap.release()
    out.release()

    # Print processing summary and diagnostics
    print("\n========================================================")
    print("      Processing Summary (Phase 1 Overlap Test)         ")
    print("========================================================")
    print(f"Total frames processed   : {frame_count}")
    print(f"Video FPS                : {fps}")
    print(f"Resolution               : {width}x{height}")
    print(f"Previous Conf Threshold  : 0.40")
    print(f"New Conf Threshold       : {conf_threshold}")
    print(f"NMS IoU Threshold        : {iou_threshold}")
    print(f"Total person detections  : {total_person_detections}")
    print(f"Output saved to          : {output_video_path}")
    print("========================================================")

    if overlap_frames_diagnostics:
        print("\nSample Difficult / Overlapping Frames Diagnostics:")
        print(f"{'Frame':<8} | {'Timestamp':<26} | {'Raw Proposals':<14} | {'After NMS':<10} | {'Max Pair IoU'}")
        print("-" * 75)
        for f, t_str, raw_p, kept_d, max_iou in overlap_frames_diagnostics[:15]:
            print(f"{f:<8} | {t_str:<26} | {raw_p:<14} | {kept_d:<10} | {max_iou}")
        print("========================================================\n")

if __name__ == "__main__":
    main()
