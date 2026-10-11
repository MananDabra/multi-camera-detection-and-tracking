"""
ESA Multi-Camera Person Tracking & Re-Identification Dashboard Server
Serves baseline dark surveillance web UI, video streams with HTTP Range support,
and zero-inference pre-indexed Global ID search APIs.
"""

import os
import re
import json
import glob
from typing import Dict, Any
from flask import Flask, render_template, jsonify, request, Response, send_file, send_from_directory
from global_identity_index import GlobalIdentityIndex

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configurable Video Directory
VIDEO_DIR = os.environ.get("VIDEO_DIR", os.path.abspath("."))

# Ensure static directories exist
os.makedirs("static/crops", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Initialize data layer
indexer = GlobalIdentityIndex()
indexer.build_index(save=False)

# Camera definitions
CAMERAS_CONFIG = [
    {
        "id": "lobby2",
        "name": "Camera 1 - lobby2.mp4",
        "badge_num": 1,
        "source_video": "lobby2.mp4",
        "annotated_video": "output_lobby2_global.mp4",
        "fps": 30.0,
        "resolution": "1440x1080",
        "duration_seconds": 8.300
    },
    {
        "id": "lobby3",
        "name": "Camera 2 - lobby3.mp4",
        "badge_num": 2,
        "source_video": "lobby3.mp4",
        "annotated_video": "output_lobby3_global.mp4",
        "fps": 30.0,
        "resolution": "1440x1080",
        "duration_seconds": 8.300
    },
    {
        "id": "lobby4",
        "name": "Camera 3 - lobby4.mp4",
        "badge_num": 3,
        "source_video": "lobby4.mp4",
        "annotated_video": "output_lobby4_global.mp4",
        "fps": 30.0,
        "resolution": "1440x1080",
        "duration_seconds": 8.300
    }
]

def stream_video_with_range(file_path: str):
    """
    Streams MP4 video with HTTP 206 Partial Content (Range request) support.
    Essential for browser HTML5 <video> seeking and scrubbing without buffering delays.
    """
    if not os.path.exists(file_path):
        return Response("Video file not found", status=404)

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range", None)

    if not range_header:
        return send_file(file_path, mimetype="video/mp4")

    match = re.search(r"bytes=(\d+)-(\d*)", range_header)
    if not match:
        return Response("Invalid Range Header", status=416)

    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else file_size - 1

    if start >= file_size or end >= file_size or start > end:
        return Response("Requested range not satisfiable", status=416)

    chunk_length = end - start + 1

    def generate_chunks():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = chunk_length
            while remaining > 0:
                chunk = f.read(min(remaining, 1024 * 64))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    rv = Response(generate_chunks(), status=206, mimetype="video/mp4", direct_passthrough=True)
    rv.headers.add("Content-Range", f"bytes {start}-{end}/{file_size}")
    rv.headers.add("Accept-Ranges", "bytes")
    rv.headers.add("Content-Length", str(chunk_length))
    rv.headers.add("Cache-Control", "no-cache")
    return rv

@app.route("/")
def index():
    """Main Dashboard View"""
    return render_template("index.html")

@app.route("/videos/<filename>")
def serve_video(filename: str):
    """Video streaming endpoint with configurable directory & Range seek support"""
    safe_name = os.path.basename(filename)
    video_dir = os.environ.get("VIDEO_DIR", os.path.abspath("."))
    
    candidates = [
        os.path.join(video_dir, safe_name),
        os.path.join(".", safe_name),
        os.path.abspath(safe_name)
    ]

    # Fallback annotated -> raw or raw -> annotated
    if safe_name.startswith("output_") and safe_name.endswith("_global.mp4"):
        raw_name = safe_name.replace("output_", "").replace("_global.mp4", ".mp4")
        candidates.extend([
            os.path.join(video_dir, raw_name),
            os.path.join(".", raw_name)
        ])
    else:
        ann_name = f"output_{safe_name.replace('.mp4', '')}_global.mp4"
        candidates.extend([
            os.path.join(video_dir, ann_name),
            os.path.join(".", ann_name)
        ])

    for path in candidates:
        if os.path.exists(path):
            return stream_video_with_range(path)

    return Response(
        f"Video file '{safe_name}' not found. Expected in directory: {os.path.abspath(video_dir)}", 
        status=404
    )

@app.route("/crops/<filename>")
def serve_crop(filename: str):
    """Person appearance crop thumbnail server"""
    safe_name = os.path.basename(filename)
    return send_from_directory("static/crops", safe_name)

@app.route("/api/config")
def get_config():
    """Returns video configuration, directory, and availability of all cameras"""
    video_dir = os.environ.get("VIDEO_DIR", os.path.abspath("."))
    cams_status = {}

    for cam in CAMERAS_CONFIG:
        # Check annotated first, then raw video
        ann_candidates = [
            os.path.join(video_dir, cam["annotated_video"]),
            os.path.join(".", cam["annotated_video"]),
            os.path.abspath(cam["annotated_video"])
        ]
        raw_candidates = [
            os.path.join(video_dir, cam["source_video"]),
            os.path.join(".", cam["source_video"]),
            os.path.abspath(cam["source_video"])
        ]

        found_path = None
        stream_url = None
        is_annotated = False

        for p in ann_candidates:
            if os.path.exists(p):
                found_path = p
                stream_url = f"/videos/{cam['annotated_video']}"
                is_annotated = True
                break

        if not found_path:
            for p in raw_candidates:
                if os.path.exists(p):
                    found_path = p
                    stream_url = f"/videos/{cam['source_video']}"
                    is_annotated = False
                    break

        cams_status[cam["id"]] = {
            "id": cam["id"],
            "name": cam["name"],
            "source_video": cam["source_video"],
            "annotated_video": cam["annotated_video"],
            "is_available": found_path is not None,
            "resolved_path": os.path.abspath(found_path) if found_path else None,
            "stream_url": stream_url,
            "is_annotated": is_annotated,
            "expected_folder": os.path.abspath(video_dir),
            "expected_file": cam["source_video"],
            "duration": cam["duration_seconds"]
        }

    available_count = sum(1 for c in cams_status.values() if c["is_available"])

    return jsonify({
        "video_dir": os.path.abspath(video_dir),
        "total_cameras": len(CAMERAS_CONFIG),
        "available_cameras": available_count,
        "all_videos_loaded": available_count == len(CAMERAS_CONFIG),
        "cameras": cams_status
    })

@app.route("/api/cameras")
def get_cameras():
    """API: Returns active camera list, video URLs, and metadata"""
    return jsonify(CAMERAS_CONFIG)

@app.route("/api/identities")
def get_identities():
    """API: Returns all discovered Global IDs ordered naturally (G001, G002, ...)"""
    all_gids = indexer.get_all_global_ids()
    results = []

    for gid in all_gids:
        rec = indexer.query_global_id(gid)
        if rec:
            results.append({
                "global_id": gid,
                "cameras_count": rec["cameras_count"],
                "cameras": rec["cameras"],
                "total_sightings": rec["total_sightings"],
                "is_multicamera": rec["is_multicamera"],
                "first_detected": rec["first_detected"],
                "last_detected": rec["last_detected"]
            })

    # Natural numeric sort: G001, G002, G003, G004, G005, ...
    def parse_gid_num(item):
        m = re.search(r'\d+', item["global_id"])
        return int(m.group(0)) if m else 9999

    results.sort(key=parse_gid_num)
    return jsonify(results)

@app.route("/api/identities/<global_id>")
def get_identity_detail(global_id: str):
    """
    API: Returns complete detection records across cameras for the selected Global ID.
    Lightweight zero-inference query from saved metadata index.
    """
    rec = indexer.query_global_id(global_id.upper())
    if not rec:
        return jsonify({"error": f"Global ID '{global_id}' not found"}), 404

    cam_name_map = {
        "lobby2": "Camera 1 - lobby2.mp4",
        "lobby3": "Camera 2 - lobby3.mp4",
        "lobby4": "Camera 3 - lobby4.mp4"
    }

    detections = []
    for item in rec["traversal_timeline"]:
        sighting = {
            "global_id": global_id.upper(),
            "camera_id": item["camera_id"],
            "camera_name": cam_name_map.get(item["camera_id"], item["camera_id"]),
            "source_video": item["source_video"],
            "local_track_id": item["local_track_id"],
            "first_seen_seconds": round(float(item["first_seen_seconds"]), 2),
            "last_seen_seconds": round(float(item["last_seen_seconds"]), 2),
            "duration_seconds": round(float(item["duration_seconds"]), 2),
            "match_status": item.get("match_status", "DETECTED"),
            "match_confidence": round(float(item.get("match_confidence", 1.0)), 4) if item.get("match_confidence") is not None else None
        }
        detections.append(sighting)

    response_data = {
        "global_id": global_id.upper(),
        "cameras_count": rec["cameras_count"],
        "cameras": rec["cameras"],
        "total_sightings": rec["total_sightings"],
        "is_multicamera": rec["is_multicamera"],
        "first_detected": rec["first_detected"],
        "last_detected": rec["last_detected"],
        "detection_records": detections,
        "traversal_timeline": detections
    }
    return jsonify(response_data)

@app.route("/api/statistics")
def get_statistics():
    """API: Returns aggregate multi-camera Re-ID system statistics"""
    val_report = indexer.validate_index()
    all_gids = indexer.get_all_global_ids()
    multicam_gids = [gid for gid in all_gids if indexer.query_global_id(gid).get("is_multicamera", False)]

    stats = {
        "cameras_connected": len(CAMERAS_CONFIG),
        "total_local_tracks": val_report["total_local_tracks"],
        "total_global_ids": val_report["unique_global_ids"],
        "multicamera_identities_count": len(multicam_gids),
        "multicamera_identities": multicam_gids,
        "unresolved_matches_count": val_report["unresolved_matches_count"],
        "tracks_per_camera": val_report["tracks_per_camera"],
        "match_status_distribution": val_report["match_status_distribution"],
        "avg_duration_seconds": round(float(indexer.tracks_df["duration_seconds"].mean()), 2) if indexer.tracks_df is not None else 0.0,
        "similarity_threshold": 0.7200,
        "distinctiveness_margin": 0.0400
    }
    return jsonify(stats)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting ESA Multi-Camera Tracking Baseline Dashboard on http://127.0.0.1:{port}...")
    print(f"Configured Video Directory: {VIDEO_DIR}")
    app.run(host="0.0.0.0", port=port, debug=False)
