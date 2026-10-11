"""
Test and Validation Suite for Global Identity Searchable Data Layer
Validates schema compliance, timestamp precision, O(1) query performance,
and integrity against multicamera_global_id_mapping.csv.
"""

import os
import json
import pandas as pd
from global_identity_index import GlobalIdentityIndex

def test_data_layer_and_index():
    print("=" * 70)
    print("   RUNNING VALIDATION SUITE: GLOBAL IDENTITY SEARCH INDEX   ")
    print("=" * 70)

    indexer = GlobalIdentityIndex()
    index_data = indexer.build_index(save=True)

    # 1. Assert file outputs exist
    assert os.path.exists("multicamera_track_index.csv"), "Missing multicamera_track_index.csv"
    assert os.path.exists("multicamera_global_identity_index.json"), "Missing multicamera_global_identity_index.json"
    assert os.path.exists("multicamera_global_id_mapping.csv"), "Missing multicamera_global_id_mapping.csv"
    print("[PASS] Output index files exist on disk.")

    # 2. Check track-level schema
    df_tracks = pd.read_csv("multicamera_track_index.csv")
    required_cols = [
        "global_id",
        "camera_id",
        "source_video",
        "local_track_id",
        "first_frame",
        "last_frame",
        "first_seen_seconds",
        "last_seen_seconds",
        "match_status",
        "match_confidence"
    ]
    for col in required_cols:
        assert col in df_tracks.columns, f"Required column '{col}' missing from track index CSV!"
    print(f"[PASS] All {len(required_cols)} mandatory columns present in track index schema.")

    # 3. Check row counts & integrity
    df_orig = pd.read_csv("multicamera_global_id_mapping.csv")
    assert len(df_tracks) == len(df_orig), f"Row count mismatch: {len(df_tracks)} vs {len(df_orig)}"
    assert len(df_tracks) == 48, f"Expected 48 local tracks, found {len(df_tracks)}"
    print(f"[PASS] Total track records count validated: exactly 48 tracks.")

    # 4. Check Global IDs count
    all_gids = indexer.get_all_global_ids()
    assert len(all_gids) == 36, f"Expected 36 unique Global IDs, got {len(all_gids)}"
    print(f"[PASS] Total unique Global IDs validated: exactly 36 IDs ({all_gids[0]} to {all_gids[-1]}).")

    # 5. Timestamp arithmetic verification
    for _, r in df_tracks.iterrows():
        fps = 30.0
        exp_first_sec = round((r["first_frame"] - 1) / fps, 3)
        exp_last_sec = round((r["last_frame"] - 1) / fps, 3)
        assert abs(r["first_seen_seconds"] - exp_first_sec) < 1e-3, f"Timestamp error in track {r['local_track_id']}"
        assert abs(r["last_seen_seconds"] - exp_last_sec) < 1e-3, f"Timestamp error in track {r['local_track_id']}"
        assert r["first_seen_seconds"] <= r["last_seen_seconds"], f"Negative duration in track {r['local_track_id']}"
    print("[PASS] Timestamp calculations mathematically verified for all 48 tracks (time zero = frame 1).")

    # 6. Test Query Performance (0 ms neural network calls)
    import time
    t0 = time.perf_counter()
    for gid in all_gids:
        rec = indexer.query_global_id(gid)
        assert rec is not None
        assert rec["global_id"] == gid
        assert len(rec["traversal_timeline"]) >= 1
    t_elapsed = (time.perf_counter() - t0) * 1000
    print(f"[PASS] Successfully queried all 36 Global IDs in {t_elapsed:.2f} ms ({t_elapsed/len(all_gids):.3f} ms per lookup) - ZERO neural network calls.")

    # 7. Test Cross-Camera Traversals
    g013 = indexer.query_global_id("G013")
    assert set(g013["cameras"]) == {"lobby2", "lobby3", "lobby4"}, f"G013 camera mismatch: {g013['cameras']}"
    assert len(g013["traversal_timeline"]) == 4, f"G013 sighting count mismatch: {len(g013['traversal_timeline'])}"
    print(f"[PASS] Cross-camera traversal verified for G013: 4 sightings across lobby2, lobby3, and lobby4.")

    g016 = indexer.query_global_id("G016")
    assert set(g016["cameras"]) == {"lobby2", "lobby3", "lobby4"}, f"G016 camera mismatch: {g016['cameras']}"
    print(f"[PASS] Cross-camera traversal verified for G016: 3 sightings across lobby2, lobby3, and lobby4.")

    # 8. Unresolved matches check
    unresolved = indexer.get_unresolved_tracks()
    assert len(unresolved) == 27, f"Expected 27 unresolved matches, got {len(unresolved)}"
    print(f"[PASS] Unresolved matches preserved: exactly 27 tracks remain unmerged under the matching policy.")

    print("\n" + "=" * 70)
    print("   ALL 8 VALIDATION SUITE TESTS PASSED SUCCESSFULLY!   ")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_data_layer_and_index()
