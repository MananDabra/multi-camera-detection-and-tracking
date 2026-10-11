"""
Multi-Camera Person Re-Identification Pipeline with Shared Global Identity Gallery
Supports synchronized multi-camera feeds: lobby2, lobby3, lobby4
"""

import os
import sys
import csv
import time
import math
import argparse
import collections
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
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# Global Configuration & Parameters
# ==============================================================================
DEFAULT_SIMILARITY_THRESHOLD = 0.72   # Cosine similarity threshold for cross-camera Re-ID
DEFAULT_MARGIN_THRESHOLD = 0.04       # Minimum distinctiveness margin (best_sim - runner_up_sim)
MAX_GALLERY_EMBEDDINGS_PER_ID = 25    # Bounded appearance memory per global identity
MIN_CANDIDATE_EMBEDDINGS_EVAL = 3     # Min quality embeddings before Re-ID evaluation
MAX_CANDIDATE_GRACE_FRAMES = 8        # Frames to observe newly confirmed track before forced Re-ID

DIRECTION_WINDOW = 10
MIN_MOVEMENT_PIXELS = 15
ARROW_LENGTH = 40
ARROW_ALPHA = 0.55
ARROW_THICKNESS = 3
ARROW_HEAD_LENGTH = 11

def compute_box_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    denom = a1 + a2 - inter
    return inter / denom if denom > 0 else 0.0

def normalize_vector(v):
    """L2-normalize a feature vector."""
    norm = np.linalg.norm(v)
    if norm > 1e-12:
        return v / norm
    return v

def get_identity_color(global_id_str):
    """Generate a consistent, visually distinct BGR color for each global ID."""
    try:
        val = int(global_id_str.replace("G", ""))
    except Exception:
        val = hash(global_id_str) % 1000
    b = int((val * 79 + 50) % 256)
    g = int((val * 157 + 50) % 256)
    r = int((val * 223 + 50) % 256)
    return (b, g, r)

# ==============================================================================
# Shared Global Identity Gallery
# ==============================================================================
class GlobalIdentityGallery:
    """
    Centralized multi-camera appearance registry and identity gallery.
    Maintains persistent Global IDs (G001, G002, ...) across all camera streams.
    """
    def __init__(self, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                 margin_threshold=DEFAULT_MARGIN_THRESHOLD,
                 max_embeddings_per_id=MAX_GALLERY_EMBEDDINGS_PER_ID,
                 aggregation_mode="top_k_mean",
                 top_k=3):
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.max_embeddings_per_id = max_embeddings_per_id
        self.aggregation_mode = aggregation_mode
        self.top_k = top_k
        self.next_global_idx = 1
        
        # Gallery registry:
        # {
        #   global_id (str): {
        #       'embeddings': [L2-normalized 512-d numpy vectors],
        #       'representative_embedding': L2-normalized centroid vector,
        #       'cameras': set of camera_ids,
        #       'associated_tracks': {camera_id: [track_ids]},
        #       'first_seen': {'camera': str, 'frame': int, 'time_sec': float},
        #       'last_seen': {'camera': str, 'frame': int, 'time_sec': float},
        #       'sightings_count': int,
        #       'dominant_direction': str
        #   }
        # }
        self.gallery = {}
        
        # Cross-camera tracking state: (camera_id, local_track_id) -> global_id
        self.camera_track_to_global = {}
        
        # Audit log of all Re-ID events and decisions
        self.reid_decisions_log = []

    def format_global_id(self, idx):
        return f"G{idx:03d}"

    def compute_similarity(self, cand_embeddings, gallery_embeddings):
        """
        Compute similarity score between candidate track embeddings and gallery embeddings.
        Supports:
          - 'top_k_mean': mean of top-k cosine similarities across all candidate-gallery pairs
          - 'max': single best pairwise cosine similarity
          - 'centroid': dot product of candidate mean and gallery centroid
        """
        if not cand_embeddings or not gallery_embeddings:
            return 0.0

        if self.aggregation_mode == "centroid":
            cand_mean = normalize_vector(np.mean(cand_embeddings, axis=0))
            gal_mean = normalize_vector(np.mean(gallery_embeddings, axis=0))
            return float(np.dot(cand_mean, gal_mean))

        # Pairwise cosine dot products (both sets are L2-normalized)
        sim_matrix = np.dot(cand_embeddings, np.array(gallery_embeddings).T)
        flat_sims = sim_matrix.flatten()

        if self.aggregation_mode == "max":
            return float(np.max(flat_sims))
        else: # top_k_mean
            k = min(self.top_k, len(flat_sims))
            top_k_vals = np.partition(flat_sims, -k)[-k:]
            return float(np.mean(top_k_vals))

    def evaluate_candidate(self, cand_embeddings, camera_id, active_global_ids_in_cam):
        """
        Evaluates a candidate track's embeddings against all eligible global identities.
        Returns:
          (best_id, best_sim, runner_up_id, runner_up_sim, margin)
        """
        if not self.gallery:
            return None, -1.0, None, -1.0, 0.0

        scores = []
        for gid, gdata in self.gallery.items():
            # Anti-collision constraint: Cannot match an identity actively tracked in the SAME camera
            if gid in active_global_ids_in_cam:
                continue

            sim = self.compute_similarity(cand_embeddings, gdata['embeddings'])
            scores.append((gid, sim))

        if not scores:
            return None, -1.0, None, -1.0, 0.0

        # Sort descending by similarity score
        scores.sort(key=lambda x: x[1], reverse=True)
        best_id, best_sim = scores[0]

        if len(scores) > 1:
            runner_up_id, runner_up_sim = scores[1]
            margin = round(best_sim - runner_up_sim, 4)
        else:
            runner_up_id, runner_up_sim = None, -1.0
            margin = round(best_sim, 4)

        return best_id, best_sim, runner_up_id, runner_up_sim, margin

    def register_new_global_id(self, camera_id, track_id, embeddings, frame_id, time_sec):
        """Register a brand new global identity in the gallery."""
        gid = self.format_global_id(self.next_global_idx)
        self.next_global_idx += 1

        norm_embeds = [normalize_vector(e) for e in embeddings]
        centroid = normalize_vector(np.mean(norm_embeds, axis=0))

        self.gallery[gid] = {
            'embeddings': norm_embeds[:self.max_embeddings_per_id],
            'representative_embedding': centroid,
            'cameras': {camera_id},
            'associated_tracks': {camera_id: [track_id]},
            'first_seen': {'camera': camera_id, 'frame': frame_id, 'time_sec': time_sec},
            'last_seen': {'camera': camera_id, 'frame': frame_id, 'time_sec': time_sec},
            'sightings_count': 1,
            'dominant_direction': 'unknown'
        }
        self.camera_track_to_global[(camera_id, track_id)] = gid
        return gid

    def update_global_identity(self, gid, camera_id, track_id, new_embeddings, frame_id, time_sec):
        """Update existing global identity appearance memory with high-quality new embeddings."""
        gdata = self.gallery[gid]
        gdata['cameras'].add(camera_id)
        if camera_id not in gdata['associated_tracks']:
            gdata['associated_tracks'][camera_id] = []
        if track_id not in gdata['associated_tracks'][camera_id]:
            gdata['associated_tracks'][camera_id].append(track_id)

        # Update appearance memory with diverse embeddings
        for e in new_embeddings:
            norm_e = normalize_vector(e)
            # Only add if distinct enough from existing vectors to maintain diversity
            existing_sims = [float(np.dot(norm_e, ex)) for ex in gdata['embeddings']]
            if max(existing_sims, default=0.0) < 0.96:
                gdata['embeddings'].append(norm_e)
                if len(gdata['embeddings']) > self.max_embeddings_per_id:
                    gdata['embeddings'].pop(0)

        # Recompute centroid
        mean_v = np.mean(gdata['embeddings'], axis=0)
        gdata['representative_embedding'] = normalize_vector(mean_v)
        gdata['last_seen'] = {'camera': camera_id, 'frame': frame_id, 'time_sec': time_sec}
        gdata['sightings_count'] += 1
        self.camera_track_to_global[(camera_id, track_id)] = gid

# ==============================================================================
# Single Camera Tracking & Feature Extraction
# ==============================================================================
def process_camera_stream(video_path, camera_id, detector, tracker_config, global_gallery,
                          save_video=False, output_video_path=None, max_frames=None):
    """
    Processes a single camera video stream, associating detections with local tracks
    and coordinating with the shared GlobalIdentityGallery.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None and max_frames > 0:
        total_frames = min(total_frames, max_frames)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n========================================================")
    print(f"  Processing Camera: {camera_id} ({os.path.basename(video_path)})")
    print(f"  Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")
    print(f"========================================================")

    # Initialize DeepSORT tracker for this camera stream
    tracker = DeepSort(**tracker_config)

    video_writer = None
    if save_video and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Track-level candidate management
    # track_candidates: {track_id: {'quality_embeddings': [], 'frames': [], 'boxes': [], 'confs': []}}
    track_buffers = collections.defaultdict(lambda: {
        'quality_embeddings': [],
        'frames': [],
        'boxes': [],
        'confs': [],
        'first_frame': None,
        'last_frame': None,
        'center_points': []
    })

    # Track decision status: track_id -> dict
    track_decisions = {}
    
    # Active tracks in current frame for anti-collision
    frame_id = 0
    t_cam_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if max_frames is not None and frame_id > max_frames:
            break
        current_time_sec = (frame_id - 1) / fps

        # 1. YOLO11s Person Detection
        results = detector.predict(
            source=frame,
            classes=[0],
            conf=0.25,
            iou=0.50,
            device="cpu",
            verbose=False
        )

        raw_detections = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                raw_detections.append(([int(x1), int(y1), int(w), int(h)], conf, "person"))

        # 2. OSNet Feature Generation via DeepSORT embedder
        valid_detections = [d for d in raw_detections if d[0][2] > 0 and d[0][3] > 0]
        if valid_detections:
            embeds = tracker.generate_embeds(frame, valid_detections)
        else:
            embeds = []

        # 3. Update DeepSORT tracks
        tracks = tracker.update_tracks(valid_detections, embeds=embeds)
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]

        # Gather track bounding boxes in current frame for quality & overlap check
        current_track_boxes = {}
        for t in confirmed_tracks:
            ltrb = t.to_ltrb()
            current_track_boxes[t.track_id] = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]

        # Identify which Global IDs are active in this frame (for anti-collision)
        active_gids_in_frame = set()
        for t in confirmed_tracks:
            tid = t.track_id
            if tid in track_decisions:
                active_gids_in_frame.add(track_decisions[tid]['global_id'])

        # 4. Process confirmed tracks and collect quality embeddings
        for t in confirmed_tracks:
            tid = t.track_id
            box = current_track_boxes[tid]
            x1, y1, x2, y2 = box
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            cx = round((x1 + x2) / 2.0, 2)
            cy = round((y1 + y2) / 2.0, 2)
            conf = t.get_det_conf() or 0.0

            tb = track_buffers[tid]
            if tb['first_frame'] is None:
                tb['first_frame'] = frame_id
            tb['last_frame'] = frame_id
            tb['frames'].append(frame_id)
            tb['boxes'].append(box)
            tb['confs'].append(conf)
            tb['center_points'].append((cx, cy))

            # Quality filter for Re-ID appearance vector
            # Must be fresh detection, confidence >= 0.40, adequate size (w>=35, h>=80),
            # aspect ratio between 1.3 and 4.5, and low overlap with other tracks
            aspect_ratio = (h / w) if w > 0 else 0
            is_valid_size = (w >= 35 and h >= 80 and 1.2 <= aspect_ratio <= 4.5)
            
            max_overlap = 0.0
            for otid, obox in current_track_boxes.items():
                if otid != tid:
                    iou = compute_box_iou(box, obox)
                    if iou > max_overlap:
                        max_overlap = iou

            is_isolated = (max_overlap < 0.30)
            is_fresh = (t.time_since_update == 0)

            if is_fresh and conf >= 0.40 and is_valid_size and is_isolated and len(t.features) > 0:
                feat = normalize_vector(t.get_feature())
                tb['quality_embeddings'].append(feat)

            # 5. Global ID assignment evaluation
            if tid not in track_decisions:
                num_quality_feats = len(tb['quality_embeddings'])
                frames_tracked = len(tb['frames'])
                
                # Ready for evaluation if gathered sufficient quality samples OR reached grace frames
                if num_quality_feats >= MIN_CANDIDATE_EMBEDDINGS_EVAL or frames_tracked >= MAX_CANDIDATE_GRACE_FRAMES:
                    # If no quality filtered feats, fall back to whatever confirmed features exist
                    eval_feats = tb['quality_embeddings'] if tb['quality_embeddings'] else [normalize_vector(f) for f in t.features]
                    
                    if not eval_feats:
                        continue # Wait until at least one feature is available

                    # Evaluate against shared gallery
                    best_id, best_sim, runner_up_id, runner_up_sim, margin = global_gallery.evaluate_candidate(
                        eval_feats, camera_id, active_gids_in_frame
                    )

                    # Estimate trajectory direction
                    direction = "unknown"
                    if len(tb['center_points']) >= 3:
                        dx = tb['center_points'][-1][0] - tb['center_points'][0][0]
                        dy = tb['center_points'][-1][1] - tb['center_points'][0][1]
                        if math.hypot(dx, dy) >= MIN_MOVEMENT_PIXELS:
                            if abs(dx) > abs(dy):
                                direction = "right" if dx > 0 else "left"
                            else:
                                direction = "down" if dy > 0 else "up"
                        else:
                            direction = "stationary"

                    # Matching Decision Logic
                    if best_id is not None and best_sim >= global_gallery.similarity_threshold:
                        if runner_up_id is not None and margin < global_gallery.margin_threshold:
                            # Ambiguity detected: Best score is above threshold, but runner up is too close
                            # To guarantee accuracy, do not blindly merge; treat as ambiguous and assign distinct ID
                            assigned_gid = global_gallery.register_new_global_id(
                                camera_id, tid, eval_feats, frame_id, current_time_sec
                            )
                            match_status = "AMBIGUOUS_BELOW_MARGIN"
                            print(f"[{camera_id}] Track {tid:>2} -> {assigned_gid} (NEW) | Ambiguous match with {best_id} (Sim: {best_sim:.3f}, Runner: {runner_up_sim:.3f}, Margin: {margin:.3f} < {global_gallery.margin_threshold:.2f})", flush=True)
                        else:
                            # Confirmed cross-camera / persistent match!
                            assigned_gid = best_id
                            match_status = "REIDENTIFIED"
                            global_gallery.update_global_identity(
                                assigned_gid, camera_id, tid, eval_feats, frame_id, current_time_sec
                            )
                            cams_seen = list(global_gallery.gallery[assigned_gid]['cameras'])
                            print(f"[{camera_id}] Track {tid:>2} -> {assigned_gid} (RE-ID) | Matched (Sim: {best_sim:.3f} >= {global_gallery.similarity_threshold:.2f}, Margin: {margin:.3f}) | Cameras: {cams_seen}", flush=True)
                    else:
                        # Below threshold or empty gallery: create new global identity
                        assigned_gid = global_gallery.register_new_global_id(
                            camera_id, tid, eval_feats, frame_id, current_time_sec
                        )
                        match_status = "NEW_IDENTITY" if best_id is None else "BELOW_THRESHOLD"
                        best_str = f"{best_sim:.3f}" if best_id is not None else "N/A"
                        print(f"[{camera_id}] Track {tid:>2} -> {assigned_gid} (NEW) | Best Sim: {best_str} < {global_gallery.similarity_threshold:.2f}", flush=True)

                    active_gids_in_frame.add(assigned_gid)
                    track_decisions[tid] = {
                        'camera_id': camera_id,
                        'track_id': tid,
                        'identity_id': tid, # Baseline track-level identity id
                        'global_id': assigned_gid,
                        'first_frame': tb['first_frame'],
                        'last_frame': frame_id,
                        'total_frames': frames_tracked,
                        'best_match_global_id': best_id or "NONE",
                        'cosine_similarity': round(best_sim, 4) if best_id else 0.0,
                        'runner_up_similarity': round(runner_up_sim, 4) if runner_up_id else 0.0,
                        'similarity_margin': round(margin, 4),
                        'match_status': match_status,
                        'spatial_direction': direction
                    }

        # 6. Optional visualization overlay (translucent pills, thin borders, anti-collision layout)
        if video_writer is not None:
            overlay = frame.copy()
            track_items = []
            for t in confirmed_tracks:
                tid = t.track_id
                box = current_track_boxes[tid]
                gid = track_decisions[tid]['global_id'] if tid in track_decisions else f"T{tid}"
                color = get_identity_color(gid)
                track_items.append({
                    'track_id': tid,
                    'global_id': gid,
                    'box': box,
                    'color': color
                })

            # Sort tracks horizontally for consistent label placement
            track_items.sort(key=lambda item: item['box'][0])

            # Stagger and anti-collision placement for label pills
            placed_labels = []
            for it in track_items:
                x1, y1, x2, y2 = it['box']
                gid = it['global_id']
                tid = it['track_id']
                color = it['color']

                # 1. Thin 1px bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)

                # 2. Compact label text: Global ID + small track indicator
                lbl = f"{gid} [T{tid}]"
                font_scale = 0.40
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

                bw = tw + 6
                bh = th + 4

                # Candidate placements: above box vs inside top of box vs inside lower
                above_rect = [x1, max(0, y1 - bh - 2), min(width, x1 + bw), y1 - 2]
                inside_rect = [x1 + 2, y1 + 2, min(width, x1 + 2 + bw), y1 + 2 + bh]

                def overlaps(r1, r2):
                    return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

                if above_rect[1] < 10 or any(overlaps(above_rect, p) for p in placed_labels):
                    if not any(overlaps(inside_rect, p) for p in placed_labels):
                        best_rect = inside_rect
                    else:
                        best_rect = [x1 + 2, y1 + bh + 4, min(width, x1 + 2 + bw), y1 + bh * 2 + 4]
                else:
                    best_rect = above_rect

                placed_labels.append(best_rect)
                it['rect'] = best_rect
                it['lbl'] = lbl

            # Render translucent background pills with colored accent bar
            for it in track_items:
                rx1, ry1, rx2, ry2 = it['rect']
                color = it['color']
                cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (18, 18, 18), -1)
                cv2.rectangle(overlay, (rx1, ry1), (rx1 + 3, ry2), color, -1)

            # Blend overlay at 50% opacity
            cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

            # Draw crisp anti-aliased text over blended pills
            for it in track_items:
                rx1, ry1, rx2, ry2 = it['rect']
                lbl = it['lbl']
                cv2.putText(frame, lbl, (rx1 + 5, ry2 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # Top HUD: Camera stream info & timestamp
            mins = int(current_time_sec // 60)
            secs = int(current_time_sec % 60)
            millis = int((current_time_sec - int(current_time_sec)) * 1000)
            hud_text = f"CAM: {camera_id.upper()} | Time: {mins:02d}:{secs:02d}.{millis:03d} | Frame: {frame_id}/{total_frames} | Active IDs: {len(active_gids_in_frame)}"
            (htw, hth), hbase = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (20, 20), (35 + htw, 35 + hth + hbase), (0, 0, 0), -1)
            cv2.putText(frame, hud_text, (28, 25 + hth), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, lineType=cv2.LINE_AA)

            video_writer.write(frame)

        if frame_id % 100 == 0 or frame_id == total_frames:
            elapsed = time.time() - t_cam_start
            cur_fps = frame_id / elapsed if elapsed > 0 else 0
            print(f"[{camera_id}] Frame {frame_id:4d}/{total_frames} ({cur_fps:.1f} FPS) | Tracks so far: {len(track_buffers)} | Resolved: {len(track_decisions)}", flush=True)

    cap.release()
    if video_writer:
        video_writer.release()

    # Finalize any short-lived tracks that were not resolved before video end
    for tid, tb in track_buffers.items():
        if tid not in track_decisions:
            # Fall back to best available features
            feats = tb['quality_embeddings']
            if not feats:
                continue
            best_id, best_sim, runner_up_id, runner_up_sim, margin = global_gallery.evaluate_candidate(
                feats, camera_id, set()
            )
            if best_id is not None and best_sim >= global_gallery.similarity_threshold and margin >= global_gallery.margin_threshold:
                assigned_gid = best_id
                match_status = "REIDENTIFIED"
                global_gallery.update_global_identity(assigned_gid, camera_id, tid, feats, frame_id, frame_id / fps)
            else:
                assigned_gid = global_gallery.register_new_global_id(camera_id, tid, feats, frame_id, frame_id / fps)
                match_status = "NEW_IDENTITY"

            track_decisions[tid] = {
                'camera_id': camera_id,
                'track_id': tid,
                'identity_id': tid,
                'global_id': assigned_gid,
                'first_frame': tb['first_frame'],
                'last_frame': tb['last_frame'],
                'total_frames': len(tb['frames']),
                'best_match_global_id': best_id or "NONE",
                'cosine_similarity': round(best_sim, 4) if best_id else 0.0,
                'runner_up_similarity': round(runner_up_sim, 4) if runner_up_id else 0.0,
                'similarity_margin': round(margin, 4),
                'match_status': match_status,
                'spatial_direction': "unknown"
            }

    # Update final frame spans and directions
    for tid, rec in track_decisions.items():
        tb = track_buffers[tid]
        rec['last_frame'] = tb['last_frame']
        rec['total_frames'] = len(tb['frames'])
        if len(tb['center_points']) >= 3:
            dx = tb['center_points'][-1][0] - tb['center_points'][0][0]
            dy = tb['center_points'][-1][1] - tb['center_points'][0][1]
            if math.hypot(dx, dy) >= MIN_MOVEMENT_PIXELS:
                if abs(dx) > abs(dy):
                    rec['spatial_direction'] = "right" if dx > 0 else "left"
                else:
                    rec['spatial_direction'] = "down" if dy > 0 else "up"
            else:
                rec['spatial_direction'] = "stationary"

    elapsed_cam = time.time() - t_cam_start
    print(f"[{camera_id}] Finished in {elapsed_cam:.1f}s ({total_frames / elapsed_cam:.2f} FPS). Total local tracks: {len(track_decisions)}\n")
    return list(track_decisions.values()), elapsed_cam

# ==============================================================================
# Multi-Camera Sequential Pipeline Runner
# ==============================================================================
def run_multicamera_pipeline(video_paths_dict, model_path="yolo11s.pt",
                             mapping_csv_path="multicamera_global_id_mapping.csv",
                             sim_thresh=DEFAULT_SIMILARITY_THRESHOLD,
                             margin_thresh=DEFAULT_MARGIN_THRESHOLD,
                             save_video=False,
                             max_frames=None):
    """
    Executes the multi-camera person Re-ID pipeline sequentially across camera feeds
    while preserving a single persistent GlobalIdentityGallery across all cameras.
    """
    print("=" * 70)
    print("  ESA PROJECT — MULTI-CAMERA PERSON RE-IDENTIFICATION")
    print("  Pipeline Stage: Shared Global ID Across lobby2, lobby3, lobby4")
    print("=" * 70)
    print(f"YOLO Model                   : {model_path}")
    print(f"OSNet Re-ID Model            : osnet_ain_x1_0 (512-dim)")
    print(f"Similarity Threshold         : {sim_thresh} (Cosine Similarity)")
    print(f"Margin Threshold             : {margin_thresh} (Runner-Up Margin)")
    print(f"Max Frames per Camera        : {max_frames if max_frames else 'Full Video'}")
    print(f"Output Mapping CSV           : {mapping_csv_path}")
    print(f"Input Camera Streams         : {list(video_paths_dict.keys())}")
    print("=" * 70 + "\n")

    # 1. Load YOLO11s detector
    print(f"Loading YOLO11s detector from '{model_path}'...")
    detector = YOLO(model_path)

    # 2. Shared DeepSORT configuration
    deepsort_config = {
        "max_age": 50,
        "n_init": 3,
        "nms_max_overlap": 1.0,
        "max_cosine_distance": 0.3,
        "nn_budget": None,
        "embedder": "torchreid",
        "embedder_model_name": "osnet_ain_x1_0",
        "half": False,
        "embedder_gpu": False
    }

    # 3. Instantiate Persistent Shared Global Identity Gallery
    global_gallery = GlobalIdentityGallery(
        similarity_threshold=sim_thresh,
        margin_threshold=margin_thresh,
        max_embeddings_per_id=MAX_GALLERY_EMBEDDINGS_PER_ID,
        aggregation_mode="top_k_mean",
        top_k=3
    )

    all_camera_mappings = []
    camera_timing_stats = {}
    total_start_time = time.time()

    # Process cameras sequentially: lobby2 -> lobby3 -> lobby4
    for camera_id, video_file in video_paths_dict.items():
        if not os.path.exists(video_file):
            print(f"Error: Video file not found: {video_file}")
            continue

        out_vid = f"output_{camera_id}_global.mp4" if save_video else None
        cam_records, cam_time = process_camera_stream(
            video_path=video_file,
            camera_id=camera_id,
            detector=detector,
            tracker_config=deepsort_config,
            global_gallery=global_gallery,
            save_video=save_video,
            output_video_path=out_vid,
            max_frames=max_frames
        )
        all_camera_mappings.extend(cam_records)
        camera_timing_stats[camera_id] = {
            'elapsed_sec': cam_time,
            'tracks': len(cam_records)
        }

    total_pipeline_time = time.time() - total_start_time

    # 4. Write multicamera_global_id_mapping.csv
    headers = [
        "camera_id",
        "track_id",
        "identity_id",
        "global_id",
        "first_frame",
        "last_frame",
        "total_frames",
        "best_match_global_id",
        "cosine_similarity",
        "runner_up_similarity",
        "similarity_margin",
        "match_status",
        "spatial_direction"
    ]

    with open(mapping_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rec in all_camera_mappings:
            writer.writerow(rec)

    print("\n" + "=" * 75)
    print("      MULTI-CAMERA PERSON RE-IDENTIFICATION SUMMARY REPORT      ")
    print("=" * 75)
    print(f"Total Cameras Processed        : {len(video_paths_dict)}")
    print(f"Total Local Tracks (All Cams)  : {len(all_camera_mappings)}")
    print(f"Total Unique Global IDs Created: {len(global_gallery.gallery)}")
    print(f"Total Cross-Camera Matches     : {sum(1 for r in all_camera_mappings if r['match_status'] == 'REIDENTIFIED')}")
    print(f"Mapping CSV File Saved To      : {mapping_csv_path}")
    print("=" * 75)

    # Per-Camera Track Distribution
    print("\n--- Local Track Breakdown per Camera ---")
    print(f"{'Camera ID':<12} | {'Local Tracks':<14} | {'Processing Time (s)':<22} | {'FPS'}")
    print("-" * 65)
    for cid, stats in camera_timing_stats.items():
        fps_c = 0.0
        cap_temp = cv2.VideoCapture(video_paths_dict[cid])
        f_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames and max_frames > 0:
            f_count = min(f_count, max_frames)
        cap_temp.release()
        fps_c = f_count / stats['elapsed_sec'] if stats['elapsed_sec'] > 0 else 0
        print(f"{cid:<12} | {stats['tracks']:<14} | {stats['elapsed_sec']:>18.1f}s | {fps_c:>6.2f}")
    print("-" * 65)
    print(f"{'Total':<12} | {len(all_camera_mappings):<14} | {total_pipeline_time:>18.1f}s")

    # Cross-Camera Global ID Traversal Breakdown
    print("\n--- Global Identity Cross-Camera Traversal Summary ---")
    print(f"{'Global ID':<10} | {'Cameras Visited':<24} | {'Local Tracks Across Cameras':<32} | {'Sightings'}")
    print("-" * 80)
    for gid, gdata in sorted(global_gallery.gallery.items()):
        cams = ", ".join(sorted(gdata['cameras']))
        track_str_parts = []
        for c, tids in gdata['associated_tracks'].items():
            track_str_parts.append(f"{c}: {tids}")
        track_str = "; ".join(track_str_parts)
        print(f"{gid:<10} | {cams:<24} | {track_str:<32} | {gdata['sightings_count']:>9}")
    print("=" * 80 + "\n")

    # Match Status Breakdown
    status_counts = collections.Counter(r['match_status'] for r in all_camera_mappings)
    print("--- Decision Status Distribution ---")
    for st, count in status_counts.items():
        pct = (count / len(all_camera_mappings) * 100) if all_camera_mappings else 0
        print(f"  {st:<25}: {count:>3} ({pct:>5.1f}%)")
    print("=" * 80 + "\n")

    # 5. Build and save searchable Global ID index
    try:
        from global_identity_index import GlobalIdentityIndex
        indexer = GlobalIdentityIndex(
            mapping_csv_path=mapping_csv_path,
            video_paths=video_paths_dict
        )
        indexer.build_index(save=True)
        print(f"[INDEX] Consolidated searchable index saved to: multicamera_global_identity_index.json & multicamera_track_index.csv\n")
    except Exception as ex:
        print(f"[INDEX WARNING] Could not build index: {ex}\n")

    return all_camera_mappings, global_gallery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Camera Person Re-ID Pipeline")
    parser.add_argument("--lobby2", default="lobby2.mp4", help="Path to lobby2 video")
    parser.add_argument("--lobby3", default="lobby3.mp4", help="Path to lobby3 video")
    parser.add_argument("--lobby4", default="lobby4.mp4", help="Path to lobby4 video")
    parser.add_argument("--model", default="yolo11s.pt", help="Path to YOLO model")
    parser.add_argument("--output_csv", default="multicamera_global_id_mapping.csv", help="Output mapping CSV path")
    parser.add_argument("--sim_thresh", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="Cosine similarity threshold")
    parser.add_argument("--margin_thresh", type=float, default=DEFAULT_MARGIN_THRESHOLD, help="Distinctiveness margin threshold")
    parser.add_argument("--save_video", action="store_true", help="Save annotated output videos")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process per video (optional)")
    args = parser.parse_args()

    video_dict = {
        "lobby2": args.lobby2,
        "lobby3": args.lobby3,
        "lobby4": args.lobby4
    }

    run_multicamera_pipeline(
        video_paths_dict=video_dict,
        model_path=args.model,
        mapping_csv_path=args.output_csv,
        sim_thresh=args.sim_thresh,
        margin_thresh=args.margin_thresh,
        save_video=args.save_video,
        max_frames=args.max_frames
    )
