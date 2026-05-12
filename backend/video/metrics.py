import numpy as np
from collections import defaultdict
from typing import Dict, List

# SoccerPitchConfiguration outputs coords in cm; scale to metres
PITCH_SCALE = 0.01


def compute_speed_kmh(prev_xy: np.ndarray, curr_xy: np.ndarray, fps: float) -> float:
    """
    Given two consecutive pitch positions (in pitch config units / cm),
    compute speed in km/h.
    """
    if prev_xy is None or curr_xy is None:
        return 0.0
    dist_m = np.linalg.norm(curr_xy - prev_xy) * PITCH_SCALE
    speed_ms = dist_m * fps
    return speed_ms * 3.6


def build_analytics(frame_data: List[dict], fps: float) -> dict:
    """
    Aggregate per-frame records into the final analytics payload.

    frame_data: list of {
        frame_idx, timestamp_s,
        players: [{tracker_id, team, pitch_xy, speed_kmh}],
        ball: {pitch_xy} or None,
        possession_team: 0 | 1 | None
    }

    Returns:
        possession: {team_a_pct, team_b_pct}
        players: {tracker_id: {speeds, distances, heatmap_points, avg_speed, max_speed, distance_m, team}}
        ball_trajectory: [{timestamp_s, pitch_xy}]
    """
    possession_counts: Dict = defaultdict(int, {0: 0, 1: 0, None: 0})
    player_records: Dict[int, dict] = defaultdict(lambda: {
        "speeds": [], "pitch_xy_history": [], "team": None
    })
    ball_trajectory = []

    for fd in frame_data:
        possession_counts[fd.get("possession_team")] += 1

        ball_xy = fd.get("ball", {}).get("pitch_xy") if fd.get("ball") else None
        if ball_xy is not None:
            ball_trajectory.append({
                "timestamp_s": fd["timestamp_s"],
                "pitch_xy": ball_xy if isinstance(ball_xy, list) else ball_xy.tolist(),
            })

        for p in fd.get("players", []):
            tid = p["tracker_id"]
            player_records[tid]["speeds"].append(p["speed_kmh"])
            player_records[tid]["pitch_xy_history"].append(p["pitch_xy"])
            if p["team"] is not None:
                player_records[tid]["team"] = int(p["team"])

    # Possession %
    total_possession_frames = possession_counts[0] + possession_counts[1]
    if total_possession_frames > 0:
        possession = {
            "team_a_pct": round(possession_counts[0] / total_possession_frames * 100, 1),
            "team_b_pct": round(possession_counts[1] / total_possession_frames * 100, 1),
        }
    else:
        possession = {"team_a_pct": 50.0, "team_b_pct": 50.0}

    # Per-player stats
    players_analytics = {}
    for tid, data in player_records.items():
        speeds = data["speeds"]
        xy_history = data["pitch_xy_history"]

        distance_m = 0.0
        for i in range(1, len(xy_history)):
            if xy_history[i] is not None and xy_history[i - 1] is not None:
                distance_m += np.linalg.norm(
                    np.array(xy_history[i]) - np.array(xy_history[i - 1])
                ) * PITCH_SCALE

        valid_speeds = [s for s in speeds if s > 0]
        players_analytics[str(tid)] = {
            "team": data["team"],
            "avg_speed_kmh": round(float(np.mean(valid_speeds)), 2) if valid_speeds else 0.0,
            "max_speed_kmh": round(float(np.max(valid_speeds)), 2) if valid_speeds else 0.0,
            "distance_m": round(distance_m, 1),
            "heatmap_points": [
                (xy if isinstance(xy, list) else xy.tolist())
                for xy in xy_history if xy is not None
            ],
            "speed_timeline": [
                {"t": i / fps, "speed": s} for i, s in enumerate(speeds)
            ],
        }

    # Match momentum: a centred rolling possession window over time.
    # Output is one sample per processed frame; advantage ∈ [-1, +1]
    #   +1 → Team A held the ball through the entire window
    #   -1 → Team B held the ball through the entire window
    #    0 → balanced or no possession recorded in the window
    window_seconds = 3.0
    window_frames = max(1, int(window_seconds * fps))
    half = window_frames // 2
    momentum_timeline = []
    for i, fd in enumerate(frame_data):
        lo = max(0, i - half)
        hi = min(len(frame_data), i + half + 1)
        window = frame_data[lo:hi]
        team_a = sum(1 for w in window if w.get("possession_team") == 0)
        team_b = sum(1 for w in window if w.get("possession_team") == 1)
        total = team_a + team_b
        advantage = ((team_a - team_b) / total) if total > 0 else 0.0
        momentum_timeline.append({
            "t": round(float(fd["timestamp_s"]), 2),
            "advantage": round(float(advantage), 3),
        })

    return {
        "possession": possession,
        "players": players_analytics,
        "ball_trajectory": ball_trajectory,
        "momentum_timeline": momentum_timeline,
    }
