"""
Global Identity Indexer & Searchable Data Layer
Provides zero-inference, pre-indexed lookups for Multi-Camera Person Re-ID.
Consolidates lobby2.mp4, lobby3.mp4, and lobby4.mp4 tracking records into a searchable index.
"""

import os
import csv
import json
import cv2
import pandas as pd
from typing import Dict, List, Optional, Any

DEFAULT_VIDEO_PATHS = {
    "lobby2": "lobby2.mp4",
    "lobby3": "lobby3.mp4",
    "lobby4": "lobby4.mp4"
}

class GlobalIdentityIndex:
    """
    Searchable identity index and query engine for multi-camera tracking.
    Pre-indexes all local tracks, relative timestamps, and Global ID assignments.
    Queries execute as O(1) in-memory dictionary lookups with 0 neural network calls.
    """
    def __init__(self,
                 mapping_csv_path: str = "multicamera_global_id_mapping.csv",
                 track_index_csv_path: str = "multicamera_track_index.csv",
                 index_json_path: str = "multicamera_global_identity_index.json",
                 video_paths: Optional[Dict[str, str]] = None):
        self.mapping_csv_path = mapping_csv_path
        self.track_index_csv_path = track_index_csv_path
        self.index_json_path = index_json_path
        self.video_paths = video_paths or DEFAULT_VIDEO_PATHS

        # Cached video metadata (FPS and duration)
        self.video_meta = {}
        self._load_video_metadata()

        # In-memory index structures
        self.tracks_df = None
        self.identity_index = {}

    def _load_video_metadata(self):
        """Read FPS and total frame count per camera video."""
        for cam_id, vpath in self.video_paths.items():
            fps = 30.0
            total_frames = 0
            if os.path.exists(vpath):
                cap = cv2.VideoCapture(vpath)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
            self.video_meta[cam_id] = {
                "source_video": os.path.basename(vpath),
                "fps": float(fps),
                "total_frames": total_frames,
                "duration_seconds": round(total_frames / fps, 3) if fps > 0 else 0.0
            }

    def build_index(self, save: bool = True) -> Dict[str, Any]:
        """
        Builds the consolidated identity index from existing multicamera_global_id_mapping.csv.
        Calculates relative timestamps (seconds from video frame 1) for each camera stream.
        """
        if not os.path.exists(self.mapping_csv_path):
            raise FileNotFoundError(f"Mapping CSV '{self.mapping_csv_path}' not found.")

        df_source = pd.read_csv(self.mapping_csv_path)

        track_records = []
        for _, row in df_source.iterrows():
            cam_id = str(row["camera_id"])
            track_id = int(row["track_id"])
            global_id = str(row["global_id"])
            first_frame = int(row["first_frame"])
            last_frame = int(row["last_frame"])
            total_frames = int(row["total_frames"]) if "total_frames" in row else (last_frame - first_frame + 1)
            match_status = str(row["match_status"])
            direction = str(row["spatial_direction"]) if "spatial_direction" in row else "unknown"

            # Camera metadata
            meta = self.video_meta.get(cam_id, {
                "source_video": f"{cam_id}.mp4",
                "fps": 30.0
            })
            fps = meta["fps"]
            source_video = meta["source_video"]

            # Relative timestamp calculation: Time zero is frame 1 of that specific video
            first_seen_sec = round((first_frame - 1) / fps, 3)
            last_seen_sec = round((last_frame - 1) / fps, 3)
            duration_sec = round((last_frame - first_frame + 1) / fps, 3)

            # Match confidence derivation
            cos_sim = float(row["cosine_similarity"]) if "cosine_similarity" in row and not pd.isna(row["cosine_similarity"]) else 0.0
            margin = float(row["similarity_margin"]) if "similarity_margin" in row and not pd.isna(row["similarity_margin"]) else 0.0
            best_match_id = str(row["best_match_global_id"]) if "best_match_global_id" in row else "NONE"

            if match_status == "REIDENTIFIED":
                match_confidence = round(cos_sim, 4)
            elif match_status == "NEW_IDENTITY":
                match_confidence = 1.0000  # Initial registered reference
            else:  # BELOW_THRESHOLD or AMBIGUOUS_BELOW_MARGIN
                match_confidence = round(cos_sim, 4) if cos_sim > 0 else 0.0

            rec = {
                "global_id": global_id,
                "camera_id": cam_id,
                "source_video": source_video,
                "local_track_id": track_id,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "total_frames": total_frames,
                "first_seen_seconds": first_seen_sec,
                "last_seen_seconds": last_seen_sec,
                "duration_seconds": duration_sec,
                "match_status": match_status,
                "match_confidence": match_confidence,
                "best_match_global_id": best_match_id,
                "similarity_margin": round(margin, 4),
                "spatial_direction": direction
            }
            track_records.append(rec)

        # Create tracks DataFrame
        self.tracks_df = pd.DataFrame(track_records)

        # Build hierarchical identity index keyed by global_id
        self.identity_index = {}
        for gid, group in self.tracks_df.groupby("global_id"):
            cameras_seen = sorted(list(group["camera_id"].unique()))
            sightings = []
            for _, r in group.sort_values(by=["first_seen_seconds", "camera_id"]).iterrows():
                sightings.append({
                    "camera_id": r["camera_id"],
                    "source_video": r["source_video"],
                    "local_track_id": int(r["local_track_id"]),
                    "first_frame": int(r["first_frame"]),
                    "last_frame": int(r["last_frame"]),
                    "total_frames": int(r["total_frames"]),
                    "first_seen_seconds": float(r["first_seen_seconds"]),
                    "last_seen_seconds": float(r["last_seen_seconds"]),
                    "duration_seconds": float(r["duration_seconds"]),
                    "match_status": r["match_status"],
                    "match_confidence": float(r["match_confidence"]),
                    "best_match_global_id": r["best_match_global_id"],
                    "similarity_margin": float(r["similarity_margin"]),
                    "spatial_direction": r["spatial_direction"]
                })

            # Consolidated record for this global identity
            self.identity_index[gid] = {
                "global_id": gid,
                "total_sightings": len(sightings),
                "cameras_count": len(cameras_seen),
                "cameras": cameras_seen,
                "is_multicamera": len(cameras_seen) > 1,
                "first_detected": {
                    "camera_id": sightings[0]["camera_id"],
                    "source_video": sightings[0]["source_video"],
                    "frame": sightings[0]["first_frame"],
                    "seconds": sightings[0]["first_seen_seconds"]
                },
                "last_detected": {
                    "camera_id": sightings[-1]["camera_id"],
                    "source_video": sightings[-1]["source_video"],
                    "frame": sightings[-1]["last_frame"],
                    "seconds": sightings[-1]["last_seen_seconds"]
                },
                "traversal_timeline": sightings
            }

        if save:
            self._save_index_files()

        return self.identity_index

    def _save_index_files(self):
        """Persist track index CSV and consolidated JSON index."""
        # 1. Save Track-Level Index CSV (exact requested schema)
        ordered_cols = [
            "global_id",
            "camera_id",
            "source_video",
            "local_track_id",
            "first_frame",
            "last_frame",
            "first_seen_seconds",
            "last_seen_seconds",
            "match_status",
            "match_confidence",
            "total_frames",
            "duration_seconds",
            "best_match_global_id",
            "similarity_margin",
            "spatial_direction"
        ]
        self.tracks_df[ordered_cols].to_csv(self.track_index_csv_path, index=False)

        # 2. Save Consolidated Global Identity Index JSON
        with open(self.index_json_path, "w", encoding="utf-8") as f:
            json.dump(self.identity_index, f, indent=2)

    def load_index(self):
        """Loads pre-built index from disk (0 ms initialization)."""
        if os.path.exists(self.index_json_path) and os.path.exists(self.track_index_csv_path):
            with open(self.index_json_path, "r", encoding="utf-8") as f:
                self.identity_index = json.load(f)
            self.tracks_df = pd.read_csv(self.track_index_csv_path)
            return True
        return False

    def get_all_global_ids(self) -> List[str]:
        """Returns sorted list of all registered Global IDs."""
        if not self.identity_index:
            self.build_index(save=False)
        return sorted(list(self.identity_index.keys()))

    def query_global_id(self, global_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves complete identity record for a Global ID without re-running inference.
        Returns all associated cameras, local tracks, first/last detection times, and Re-ID confidence.
        """
        if not self.identity_index:
            self.build_index(save=False)
        return self.identity_index.get(global_id, None)

    def get_tracks_by_camera(self, camera_id: str) -> List[Dict[str, Any]]:
        """Retrieve all local tracks detected in a specific camera."""
        if self.tracks_df is None:
            self.build_index(save=False)
        subset = self.tracks_df[self.tracks_df["camera_id"] == camera_id]
        return subset.to_dict(orient="records")

    def get_unresolved_tracks(self) -> List[Dict[str, Any]]:
        """Retrieve all tracks with unresolved/uncertain match status."""
        if self.tracks_df is None:
            self.build_index(save=False)
        unresolved = self.tracks_df[self.tracks_df["match_status"].isin(["BELOW_THRESHOLD", "AMBIGUOUS_BELOW_MARGIN"])]
        return unresolved.to_dict(orient="records")

    def validate_index(self) -> Dict[str, Any]:
        """
        Validates index consistency against original mapping CSV and video streams.
        Returns comprehensive verification report.
        """
        if not self.identity_index or self.tracks_df is None:
            self.build_index(save=False)

        source_df = pd.read_csv(self.mapping_csv_path)
        total_source_rows = len(source_df)
        total_index_rows = len(self.tracks_df)

        assert total_source_rows == total_index_rows, f"Row count mismatch: source {total_source_rows} vs index {total_index_rows}"

        # Per-camera track counts
        cam_counts = self.tracks_df["camera_id"].value_counts().to_dict()

        # Status distribution
        status_counts = self.tracks_df["match_status"].value_counts().to_dict()

        # Multicamera identities
        multicam_ids = [gid for gid, data in self.identity_index.items() if data["is_multicamera"]]

        # Timestamp bounds validation
        for _, r in self.tracks_df.iterrows():
            assert r["first_seen_seconds"] <= r["last_seen_seconds"], f"Timestamp inversion in track {r['local_track_id']}"
            assert r["first_frame"] <= r["last_frame"], f"Frame inversion in track {r['local_track_id']}"

        return {
            "validation_status": "PASSED",
            "total_local_tracks": total_index_rows,
            "unique_global_ids": len(self.identity_index),
            "tracks_per_camera": cam_counts,
            "match_status_distribution": status_counts,
            "unresolved_matches_count": len(self.get_unresolved_tracks()),
            "multicamera_identities_count": len(multicam_ids),
            "multicamera_identities": multicam_ids,
            "output_files": {
                "track_index_csv": os.path.abspath(self.track_index_csv_path),
                "identity_index_json": os.path.abspath(self.index_json_path),
                "source_mapping_csv": os.path.abspath(self.mapping_csv_path)
            }
        }

if __name__ == "__main__":
    indexer = GlobalIdentityIndex()
    index_data = indexer.build_index(save=True)
    report = indexer.validate_index()

    print("=" * 70)
    print("      GLOBAL IDENTITY INDEX CREATION & VALIDATION REPORT      ")
    print("=" * 70)
    print(f"Validation Status              : {report['validation_status']}")
    print(f"Total Local Tracks Indexed     : {report['total_local_tracks']}")
    print(f"Unique Global IDs Discovered   : {report['unique_global_ids']}")
    print(f"Cross-Camera Identities        : {report['multicamera_identities_count']}")
    print(f"Unresolved Matches             : {report['unresolved_matches_count']}")
    print("\n--- Local Tracks per Camera ---")
    for cam, count in report['tracks_per_camera'].items():
        print(f"  {cam:<10}: {count} tracks")
    print("\n--- Match Status Distribution ---")
    for st, count in report['match_status_distribution'].items():
        print(f"  {st:<25}: {count}")
    print("\n--- Output Index Files ---")
    for key, path in report['output_files'].items():
        print(f"  {key:<22}: {path}")
    print("=" * 70)

    # Test sample lookup
    print("\n--- Sample Query Test: Global ID G013 ---")
    res = indexer.query_global_id("G013")
    if res:
        print(f"Global ID       : {res['global_id']}")
        print(f"Cameras Visited : {res['cameras']}")
        print(f"Total Sightings : {res['total_sightings']}")
        print(f"Timeline:")
        for s in res['traversal_timeline']:
            print(f"  [{s['camera_id']}] Track {s['local_track_id']:>2} | Frames: {s['first_frame']:>3}-{s['last_frame']:>3} | Time: {s['first_seen_seconds']:>6.3f}s - {s['last_seen_seconds']:>6.3f}s ({s['duration_seconds']:>5.2f}s) | Status: {s['match_status']:<15} | Conf: {s['match_confidence']:.4f}")
    print("=" * 70)
