import bisect
import cv2
import os
import shutil
import subprocess
import time
import numpy as np
import supervision as sv
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from inference_sdk import InferenceHTTPClient
from typing import Callable

from sports.annotators.soccer import draw_pitch

from backend.config import settings
from backend.video.homography import compute_homography, get_player_feet_xy
from backend.video.team_classifier import TeamClassifier
from backend.video.metrics import build_analytics
from backend.video.annotator import (
    PITCH_CONFIG,
    build_annotators,
    draw_frame,
    draw_pitch_map,
    draw_voronoi_map,
)

PLAYER_MODEL_ID = "football-players-detection-3zvbc/11"
PITCH_MODEL_ID = "football-field-detection-f07vi/14"

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

MAX_PLAUSIBLE_SPEED_KMH = 40.0
HOMOGRAPHY_SMOOTH_WINDOW = 5
PIPELINE_DEPTH = 4
PITCH_INTERVAL = 5
INFERENCE_WORKERS = 6
FIT_AT_FRAME = 25
FIT_MIN_CROPS = 50
DETECTION_CONFIDENCE = 0.15  # aggressive: in per-frame mode we want every player caught
SPEED_WINDOW = 5  # at 25fps per-frame data, this is a 0.2s window — ideal
POSITION_SMOOTH_WINDOW = 5  # bumped from 3: per-frame YOLO has small wobble; 5-frame avg removes it
INFERENCE_MAX_RETRIES = 3  # transient DNS/network failures recover on retry
INFERENCE_RETRY_BACKOFF_S = 0.5  # 0.5 → 1.0 → 2.0 between attempts
TEAM_HISTORY_SIZE = 15           # rolling deque of predictions per tracker_id
TEAM_OVERLAP_IOU = 0.15          # boxes with IoU > this skip team prediction (mixed crops)
TEAM_PREDICT_UNTIL_CLEAN = 7     # stop predicting for a tid after this many clean predictions


def _detect_overlapping_boxes(xyxy: np.ndarray, threshold: float) -> np.ndarray:
    """Mark each box True if its IoU with any other box exceeds `threshold`.

    Used to suppress team-classifier predictions on confused/overlapping crops.
    When two players' bboxes overlap heavily, the YOLO crop contains pixels from
    both jerseys, the SigLIP classifier sees mixed colors, and predicts garbage —
    which then gets cached and breaks team assignment for the rest of the video.
    """
    n = len(xyxy)
    if n < 2:
        return np.zeros(n, dtype=bool)
    flagged = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            ax1, ay1, ax2, ay2 = xyxy[i]
            bx1, by1, bx2, by2 = xyxy[j]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            iou = inter / (area_a + area_b - inter + 1e-6)
            if iou > threshold:
                flagged[i] = True
                flagged[j] = True
    return flagged


def _robust_infer(client: InferenceHTTPClient, frame: np.ndarray, model_id: str) -> dict:
    """Inference call with exponential backoff retry on transient errors.

    Why this matters: residential DNS resolvers (router-level) drop intermittently,
    causing getaddrinfo failures that wipe out 50+ consecutive frames if a single
    call fails permanently. A short retry catches DNS hiccups and 5xx blips.
    """
    last_exc = None
    for attempt in range(INFERENCE_MAX_RETRIES):
        try:
            return client.infer(frame, model_id=model_id)
        except Exception as e:
            last_exc = e
            if attempt < INFERENCE_MAX_RETRIES - 1:
                time.sleep(INFERENCE_RETRY_BACKOFF_S * (2 ** attempt))
    raise last_exc


def _crop_bgr(frame: np.ndarray, xyxy: np.ndarray) -> list:
    h, w = frame.shape[:2]
    crops = []
    for box in xyxy:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            crops.append(frame[y1:y2, x1:x2])
        else:
            crops.append(np.zeros((4, 4, 3), dtype=np.uint8))
    return crops


def _apply_homography_matrix(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    if matrix is None or len(points) == 0:
        return np.empty((0, 2))
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = homogeneous @ matrix.T
    w = transformed[:, 2:3]
    return transformed[:, :2] / np.where(w == 0, 1, w)


def _resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team: np.ndarray,
    goalkeepers: sv.Detections,
    team_fitted: bool,
) -> np.ndarray:
    if len(goalkeepers) == 0:
        return np.array([], dtype=int)
    if len(players) == 0:
        return np.zeros(len(goalkeepers), dtype=int)

    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pl_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    if team_fitted:
        team_0 = pl_xy[players_team == 0]
        team_1 = pl_xy[players_team == 1]
        if len(team_0) > 0 and len(team_1) > 0:
            c0 = team_0.mean(axis=0)
            c1 = team_1.mean(axis=0)
            return np.array(
                [0 if np.linalg.norm(xy - c0) < np.linalg.norm(xy - c1) else 1 for xy in gk_xy],
                dtype=int,
            )

    median_x = float(np.median(pl_xy[:, 0]))
    return np.array([0 if xy[0] < median_x else 1 for xy in gk_xy], dtype=int)


def _make_video_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    print(f"[VideoWriter] {path} ({width}x{height} @ {fps:.1f} fps)")
    return writer


def _resolve_ffmpeg_binary() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _ffmpeg_to_h264(src: str, dst: str) -> bool:
    binary = _resolve_ffmpeg_binary()
    if not binary:
        return False
    try:
        subprocess.run(
            [
                binary, "-y", "-i", src,
                # CRF 18 = visually lossless H.264; preset slow trades a bit of CPU for sharper output
                "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                dst,
            ],
            check=True, capture_output=True,
        )
        return True
    except Exception as e:
        print(f"[ffmpeg] re-encode failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# State computation (Pass 1 → analytics-ready snapshots)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_state(
    raw: dict,
    *,
    team_predictions: dict,
    team_clean_count: dict,
    prev_player_xy: dict,
    speed_history: dict,
    frame_data: list,
    team_classifier: TeamClassifier,
    effective_fps: float,
) -> dict:
    """Take a raw detection bundle and produce a fully-resolved 'computed state':
    team labels, projected pitch coords, smoothed speeds, possession.

    Returns a flat dict containing everything Pass 2 needs to render this frame.
    """
    players: sv.Detections = raw["players"]
    goalkeepers: sv.Detections = raw["goalkeepers"]
    referees: sv.Detections = raw["referees"]
    ball: sv.Detections = raw["ball_detections"]
    player_crops: list = raw["player_crops"]
    transformer_matrix: np.ndarray | None = raw["transformer_matrix"]
    frame_idx: int = raw["frame_idx"]
    timestamp_s: float = raw["timestamp_s"]

    # 1. Team classification — overlap-aware rolling majority vote
    #
    # Why not a single cached label? When two players collide (challenges, scrums,
    # set-pieces), their bboxes overlap → the crop contains pixels from both
    # jerseys → SigLIP predicts garbage → that garbage was previously cached and
    # locked in for the rest of the video. Now we:
    #   (a) detect overlapping boxes via IoU
    #   (b) only call the classifier for tids that haven't accumulated enough
    #       *clean* (non-overlap) predictions yet — saves compute once stable
    #   (c) only APPEND to the prediction history if the crop was clean
    #   (d) the final label is a majority vote over the rolling deque, so a
    #       single bad prediction can't flip the player's team
    players_team = np.zeros(len(players), dtype=int)
    if team_classifier.fitted and len(players) > 0:
        player_tids = (
            players.tracker_id if players.tracker_id is not None else [None] * len(players)
        )
        overlapping = _detect_overlapping_boxes(players.xyxy, threshold=TEAM_OVERLAP_IOU)

        # Pick which players still need predictions (haven't stabilised yet)
        indices_to_predict = []
        for i, tid in enumerate(player_tids):
            tid_int = int(tid) if tid is not None else None
            if tid_int is None:
                indices_to_predict.append(i)
                continue
            if team_clean_count.get(tid_int, 0) < TEAM_PREDICT_UNTIL_CLEAN:
                indices_to_predict.append(i)

        if indices_to_predict:
            crops_subset = [player_crops[i] for i in indices_to_predict]
            new_labels = team_classifier.predict_crops(crops_subset)
            for j, i in enumerate(indices_to_predict):
                pred = int(new_labels[j])
                tid = player_tids[i]
                if tid is None:
                    # No tracker → use this single prediction; nothing to remember
                    players_team[i] = pred
                    continue
                tid_int = int(tid)
                if not overlapping[i]:
                    team_predictions[tid_int].append(pred)
                    team_clean_count[tid_int] = team_clean_count.get(tid_int, 0) + 1
                # If overlapping: deliberately drop this prediction.

        # Final assignment: majority vote of the (clean) prediction history
        for i, tid in enumerate(player_tids):
            tid_int = int(tid) if tid is not None else None
            if tid_int is not None and len(team_predictions.get(tid_int, [])) > 0:
                counts = np.bincount(list(team_predictions[tid_int]))
                players_team[i] = int(counts.argmax())
            # else: stays 0 — first frame for an untrackable detection

    # 2. Goalkeeper team assignment
    gk_team = _resolve_goalkeepers_team_id(
        players, players_team, goalkeepers, team_classifier.fitted
    )

    # 3. Merge players + GKs + referees
    merge_parts, team_parts = [], []
    if len(players) > 0:
        merge_parts.append(players)
        team_parts.append(players_team)
    if len(goalkeepers) > 0:
        merge_parts.append(goalkeepers)
        team_parts.append(gk_team)
    if len(referees) > 0:
        merge_parts.append(referees)
        team_parts.append(np.full(len(referees), 2, dtype=int))

    if merge_parts:
        all_outfield = sv.Detections.merge(merge_parts) if len(merge_parts) > 1 else merge_parts[0]
        all_team = np.concatenate(team_parts) if len(team_parts) > 1 else team_parts[0]
    else:
        all_outfield = sv.Detections.empty()
        all_team = np.array([], dtype=int)

    # 4. Pitch projection
    outfield_pitch_xy = np.empty((0, 2))
    ball_pitch_xy = None
    if transformer_matrix is not None and len(all_outfield) > 0:
        feet = get_player_feet_xy(all_outfield)
        if len(feet) > 0:
            outfield_pitch_xy = _apply_homography_matrix(transformer_matrix, feet)

    ball_xyxy_arr = ball.xyxy.copy() if len(ball) > 0 else np.empty((0, 4))
    if transformer_matrix is not None and len(ball) > 0:
        ball_feet = get_player_feet_xy(ball)
        if len(ball_feet) > 0:
            projected = _apply_homography_matrix(transformer_matrix, ball_feet)
            if len(projected) > 0:
                ball_pitch_xy = projected[0]

    # 5. Speeds (rolling-median smoothed per tracker_id)
    # When raw_speed exceeds the plausible threshold it's almost always a tracker
    # ID swap (ID jumped to a different physical player). Reject the sample
    # entirely instead of clipping — clipped values still pollute the median.
    speeds: dict = {}
    if all_outfield.tracker_id is not None:
        for i, tid in enumerate(all_outfield.tracker_id):
            tid_int = int(tid)
            curr = outfield_pitch_xy[i] if i < len(outfield_pitch_xy) else None
            prev = prev_player_xy.get(tid_int)
            if curr is not None and prev is not None:
                dist_m = float(np.linalg.norm(curr - prev) * 0.01)
                raw_speed = dist_m * effective_fps * 3.6
                if raw_speed > MAX_PLAUSIBLE_SPEED_KMH:
                    # Likely tracker swap: keep prior smoothed speed; do NOT pollute history
                    speeds[tid_int] = (
                        round(float(np.median(speed_history[tid_int])), 1)
                        if speed_history[tid_int]
                        else 0.0
                    )
                else:
                    speed_history[tid_int].append(raw_speed)
                    speeds[tid_int] = round(float(np.median(speed_history[tid_int])), 1)
            else:
                speeds[tid_int] = 0.0
            if curr is not None:
                prev_player_xy[tid_int] = curr

    # 6. Possession (refs excluded)
    possession_team = None
    if ball_pitch_xy is not None and len(outfield_pitch_xy) > 0:
        possession_mask = all_team != 2
        if possession_mask.any():
            candidates_xy = outfield_pitch_xy[possession_mask]
            candidates_team = all_team[possession_mask]
            dists = np.linalg.norm(candidates_xy - ball_pitch_xy, axis=1)
            nearest = int(np.argmin(dists))
            possession_team = int(candidates_team[nearest])

    # 7. Append per-frame analytics record (only at processed-frame cadence)
    players_record = []
    if all_outfield.tracker_id is not None:
        for i, tid in enumerate(all_outfield.tracker_id):
            players_record.append({
                "tracker_id": int(tid),
                "team": int(all_team[i]) if i < len(all_team) else None,
                "pitch_xy": outfield_pitch_xy[i].tolist() if i < len(outfield_pitch_xy) else None,
                "speed_kmh": speeds.get(int(tid), 0.0),
            })
    frame_data.append({
        "frame_idx": frame_idx,
        "timestamp_s": timestamp_s,
        "players": players_record,
        "ball": {"pitch_xy": ball_pitch_xy.tolist() if ball_pitch_xy is not None else None},
        "possession_team": possession_team,
    })

    # Flatten into a render-friendly state dict
    return {
        "src_frame_idx": frame_idx,
        "timestamp_s": timestamp_s,
        "all_xyxy": all_outfield.xyxy.copy() if len(all_outfield) > 0 else np.empty((0, 4)),
        "all_team": all_team.copy(),
        "all_tids": (
            all_outfield.tracker_id.copy()
            if all_outfield.tracker_id is not None
            else np.full(len(all_outfield), -1, dtype=int)
        ),
        "all_pitch_xy": outfield_pitch_xy.copy(),
        "ball_xyxy": ball_xyxy_arr,
        "ball_pitch_xy": ball_pitch_xy.copy() if ball_pitch_xy is not None else None,
        "transformer_matrix": transformer_matrix,
        "speeds": dict(speeds),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Position smoothing & interpolation (Pass 2 → smooth output)
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_positions_inplace(states: list[dict], window: int = POSITION_SMOOTH_WINDOW) -> None:
    """Apply a centred rolling-average to per-tracker bbox + pitch_xy across processed states.
    Reduces detection jitter before interpolation, eliminating 'wobble' on stationary players.
    """
    if window < 2 or len(states) < 2:
        return

    # Collect raw trajectory per tracker_id
    traj_xyxy: dict = defaultdict(list)
    traj_pitch: dict = defaultdict(list)
    for s_idx, s in enumerate(states):
        for i, tid in enumerate(s["all_tids"]):
            tid_int = int(tid)
            if tid_int < 0:
                continue
            traj_xyxy[tid_int].append((s_idx, s["all_xyxy"][i]))
            if i < len(s["all_pitch_xy"]):
                traj_pitch[tid_int].append((s_idx, s["all_pitch_xy"][i]))

    half = window // 2

    def _rolling(samples: list[tuple[int, np.ndarray]]) -> dict[int, np.ndarray]:
        out = {}
        n = len(samples)
        for k in range(n):
            lo = max(0, k - half)
            hi = min(n, k + half + 1)
            arr = np.array([samples[m][1] for m in range(lo, hi)])
            out[samples[k][0]] = arr.mean(axis=0)
        return out

    smoothed_xyxy = {tid: _rolling(samples) for tid, samples in traj_xyxy.items()}
    smoothed_pitch = {tid: _rolling(samples) for tid, samples in traj_pitch.items()}

    # Write smoothed values back into states
    for s_idx, s in enumerate(states):
        for i, tid in enumerate(s["all_tids"]):
            tid_int = int(tid)
            if tid_int < 0:
                continue
            if s_idx in smoothed_xyxy.get(tid_int, {}):
                s["all_xyxy"][i] = smoothed_xyxy[tid_int][s_idx]
            if s_idx in smoothed_pitch.get(tid_int, {}) and i < len(s["all_pitch_xy"]):
                s["all_pitch_xy"][i] = smoothed_pitch[tid_int][s_idx]


def _interpolate_state_at(src_idx: int, states: list[dict], state_indices: list[int]) -> dict:
    """Return an interpolated render-state for a given source-frame index.

    Uses linear interpolation per tracker_id between bracketing processed states.
    Tracker IDs only present on one side are held at that side's position.
    """
    if not states:
        return {
            "all_xyxy": np.empty((0, 4)),
            "all_team": np.array([], dtype=int),
            "all_tids": np.array([], dtype=int),
            "all_pitch_xy": np.empty((0, 2)),
            "ball_xyxy": np.empty((0, 4)),
            "ball_pitch_xy": None,
            "speeds": {},
        }

    # Find bracketing positions in state_indices
    pos = bisect.bisect_left(state_indices, src_idx)
    if pos < len(state_indices) and state_indices[pos] == src_idx:
        return _state_to_render_dict(states[pos])

    if pos == 0:
        return _state_to_render_dict(states[0])
    if pos >= len(state_indices):
        return _state_to_render_dict(states[-1])

    left = states[pos - 1]
    right = states[pos]
    left_idx = state_indices[pos - 1]
    right_idx = state_indices[pos]
    if right_idx == left_idx:
        return _state_to_render_dict(left)
    alpha = (src_idx - left_idx) / (right_idx - left_idx)
    return _interpolate_two_states(left, right, alpha)


def _state_to_render_dict(s: dict) -> dict:
    return {
        "all_xyxy": s["all_xyxy"],
        "all_team": s["all_team"],
        "all_tids": s["all_tids"],
        "all_pitch_xy": s["all_pitch_xy"],
        "ball_xyxy": s["ball_xyxy"],
        "ball_pitch_xy": s["ball_pitch_xy"],
        "speeds": s["speeds"],
    }


def _interpolate_two_states(left: dict, right: dict, alpha: float) -> dict:
    left_tid_to_idx = {int(t): i for i, t in enumerate(left["all_tids"])}
    right_tid_to_idx = {int(t): i for i, t in enumerate(right["all_tids"])}

    common_tids = set(left_tid_to_idx) & set(right_tid_to_idx)
    only_left = set(left_tid_to_idx) - common_tids

    out_xyxy, out_team, out_tids, out_pitch = [], [], [], []

    for tid in common_tids:
        li = left_tid_to_idx[tid]
        ri = right_tid_to_idx[tid]
        bbox_l = left["all_xyxy"][li]
        bbox_r = right["all_xyxy"][ri]
        out_xyxy.append(bbox_l + (bbox_r - bbox_l) * alpha)
        out_team.append(int(left["all_team"][li]))
        out_tids.append(tid)
        if li < len(left["all_pitch_xy"]) and ri < len(right["all_pitch_xy"]):
            pl = left["all_pitch_xy"][li]
            pr = right["all_pitch_xy"][ri]
            out_pitch.append(pl + (pr - pl) * alpha)
        elif li < len(left["all_pitch_xy"]):
            out_pitch.append(left["all_pitch_xy"][li])
        elif ri < len(right["all_pitch_xy"]):
            out_pitch.append(right["all_pitch_xy"][ri])

    # Players only in left frame: fade them by holding for the bracket (avoids pop)
    for tid in only_left:
        li = left_tid_to_idx[tid]
        out_xyxy.append(left["all_xyxy"][li])
        out_team.append(int(left["all_team"][li]))
        out_tids.append(tid)
        if li < len(left["all_pitch_xy"]):
            out_pitch.append(left["all_pitch_xy"][li])

    # Ball: interpolate when both sides have it
    ball_xyxy = np.empty((0, 4))
    ball_pitch_xy = None
    has_left_ball = len(left["ball_xyxy"]) > 0
    has_right_ball = len(right["ball_xyxy"]) > 0
    if has_left_ball and has_right_ball:
        ball_xyxy = (left["ball_xyxy"] + (right["ball_xyxy"] - left["ball_xyxy"]) * alpha)
        if left["ball_pitch_xy"] is not None and right["ball_pitch_xy"] is not None:
            ball_pitch_xy = (
                left["ball_pitch_xy"]
                + (right["ball_pitch_xy"] - left["ball_pitch_xy"]) * alpha
            )
        else:
            ball_pitch_xy = left["ball_pitch_xy"] or right["ball_pitch_xy"]
    elif has_left_ball:
        ball_xyxy = left["ball_xyxy"]
        ball_pitch_xy = left["ball_pitch_xy"]
    elif has_right_ball:
        ball_xyxy = right["ball_xyxy"]
        ball_pitch_xy = right["ball_pitch_xy"]

    # Inherit speeds from left (constant within the bracket — already smoothed)
    speeds = dict(left["speeds"])

    return {
        "all_xyxy": np.array(out_xyxy) if out_xyxy else np.empty((0, 4)),
        "all_team": np.array(out_team, dtype=int),
        "all_tids": np.array(out_tids, dtype=int),
        "all_pitch_xy": np.array(out_pitch) if out_pitch else np.empty((0, 2)),
        "ball_xyxy": ball_xyxy,
        "ball_pitch_xy": ball_pitch_xy,
        "speeds": speeds,
    }


def _build_detections_for_render(state: dict) -> tuple[sv.Detections, sv.Detections]:
    """Construct sv.Detections objects expected by draw_frame from a render state."""
    if len(state["all_xyxy"]) > 0:
        outfield = sv.Detections(
            xyxy=state["all_xyxy"].astype(np.float32),
            class_id=state["all_team"].astype(int),
            confidence=np.ones(len(state["all_xyxy"]), dtype=np.float32),
            tracker_id=state["all_tids"].astype(int),
        )
    else:
        outfield = sv.Detections.empty()

    if len(state["ball_xyxy"]) > 0:
        ball = sv.Detections(
            xyxy=state["ball_xyxy"].astype(np.float32),
            class_id=np.zeros(len(state["ball_xyxy"]), dtype=int),
            confidence=np.ones(len(state["ball_xyxy"]), dtype=np.float32),
        )
    else:
        ball = sv.Detections.empty()

    return outfield, ball


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    job_id: str,
    video_path: str,
    progress_callback: Callable[[str, int, str], None],
    team_classifier: TeamClassifier = None,
) -> dict:
    """Two-pass pipeline:
        Pass 1: inference on every Nth frame → list of computed states (analytics-ready).
        Pass 2: read every source frame, interpolate state, render 3 separate output videos.

    Outputs three browser-ready H.264 mp4s:
        {job_id}_main.mp4     — annotated source: ellipses, ball, speed labels
        {job_id}_pitch.mp4    — top-down 2D pitch: player dots + ball
        {job_id}_voronoi.mp4  — top-down Voronoi territory map
    """

    client = InferenceHTTPClient(
        api_url=settings.inference_server_url,
        api_key=settings.roboflow_api_key,
    )
    if team_classifier is None:
        team_classifier = TeamClassifier()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_step = max(1, int(fps / settings.frame_sample_rate))
    effective_fps = fps / frame_step

    annotators = build_annotators()
    # Looser ByteTrack tuning to reduce ID swaps:
    #   - lower activation threshold catches small/distant players earlier
    #   - longer lost_track_buffer keeps an ID alive through occlusions
    #   - lower matching threshold accepts more aggressive re-identification
    tracker = sv.ByteTrack(
        track_activation_threshold=0.20,
        lost_track_buffer=90,
        minimum_matching_threshold=0.7,
    )
    tracker.reset()

    # ── Pass 1 setup ─────────────────────────────────────────────────────
    frame_data: list = []
    prev_player_xy: dict = {}
    team_predictions: dict = defaultdict(lambda: deque(maxlen=TEAM_HISTORY_SIZE))
    team_clean_count: dict = {}
    speed_history: dict = defaultdict(lambda: deque(maxlen=SPEED_WINDOW))
    homography_buffer = deque(maxlen=HOMOGRAPHY_SMOOTH_WINDOW)
    current_transformer_matrix: np.ndarray | None = None

    fit_crops_pool: list = []
    team_fitted = False
    raw_buffer: list[dict] = []  # raw bundles awaiting team classifier fit
    processed_states: list[dict] = []

    progress_callback("detecting", 5, "Starting detection pipeline...")
    inference_pool = ThreadPoolExecutor(max_workers=INFERENCE_WORKERS)

    def sample_frames():
        f = 0
        while True:
            ret, frame_arr = cap.read()
            if not ret:
                return
            if f % frame_step == 0:
                yield f, frame_arr
            f += 1

    frame_iter = sample_frames()
    pending: deque = deque()
    next_processed_idx = 0
    last_pitch_result = None

    def submit_frame(pidx: int, src_idx: int, frm: np.ndarray):
        pf = inference_pool.submit(_robust_infer, client, frm, PLAYER_MODEL_ID)
        pitchf = None
        if pidx % PITCH_INTERVAL == 0:
            pitchf = inference_pool.submit(_robust_infer, client, frm, PITCH_MODEL_ID)
        pending.append((pidx, src_idx, frm, pf, pitchf))

    for _ in range(PIPELINE_DEPTH):
        try:
            sidx, frm = next(frame_iter)
        except StopIteration:
            break
        submit_frame(next_processed_idx, sidx, frm)
        next_processed_idx += 1

    processed_count = 0
    while pending:
        pidx, frame_idx, frame, player_future, pitch_future = pending.popleft()
        timestamp_s = frame_idx / fps

        try:
            player_result = player_future.result()
        except Exception as e:
            print(f"[Pipeline] player inference failed at frame {frame_idx}: {e}")
            try:
                sidx_next, frm_next = next(frame_iter)
                submit_frame(next_processed_idx, sidx_next, frm_next)
                next_processed_idx += 1
            except StopIteration:
                pass
            continue

        if pitch_future is not None:
            try:
                last_pitch_result = pitch_future.result()
            except Exception as e:
                print(f"[Pipeline] pitch inference failed at frame {frame_idx}: {e}")
        pitch_result = last_pitch_result

        detections = sv.Detections.from_inference(player_result)
        if detections.confidence is not None:
            detections = detections[detections.confidence >= DETECTION_CONFIDENCE]

        ball_detections = detections[detections.class_id == BALL_CLASS_ID]

        non_ball = detections[detections.class_id != BALL_CLASS_ID]
        non_ball = non_ball.with_nms(threshold=0.5, class_agnostic=True)
        non_ball = tracker.update_with_detections(detections=non_ball)

        players = non_ball[non_ball.class_id == PLAYER_CLASS_ID]
        goalkeepers = non_ball[non_ball.class_id == GOALKEEPER_CLASS_ID]
        referees = non_ball[non_ball.class_id == REFEREE_CLASS_ID]

        player_crops = _crop_bgr(frame, players.xyxy)

        if pitch_result is not None:
            keypoints = sv.KeyPoints.from_inference(pitch_result)
            homography = compute_homography(keypoints)
            if homography.valid:
                homography_buffer.append(homography.transformer.m)
                current_transformer_matrix = np.mean(np.array(homography_buffer), axis=0)
        transformer_snapshot = (
            current_transformer_matrix.copy() if current_transformer_matrix is not None else None
        )

        raw = {
            "frame_idx": frame_idx,
            "timestamp_s": timestamp_s,
            "players": players,
            "goalkeepers": goalkeepers,
            "referees": referees,
            "ball_detections": ball_detections,
            "player_crops": player_crops,
            "transformer_matrix": transformer_snapshot,
        }
        raw_buffer.append(raw)

        if not team_fitted:
            for c in player_crops:
                if c.shape[0] >= 8 and c.shape[1] >= 8:
                    fit_crops_pool.append(c)
            if pidx >= FIT_AT_FRAME and len(fit_crops_pool) >= FIT_MIN_CROPS:
                progress_callback(
                    "fitting", 30,
                    f"Fitting team classifier on {len(fit_crops_pool)} crops...",
                )
                team_classifier.fit_crops(fit_crops_pool)
                team_fitted = team_classifier.fitted
                fit_crops_pool = []

        if team_fitted:
            while raw_buffer:
                rb = raw_buffer.pop(0)
                processed_states.append(_compute_state(
                    rb,
                    team_predictions=team_predictions,
                    team_clean_count=team_clean_count,
                    prev_player_xy=prev_player_xy,
                    speed_history=speed_history,
                    frame_data=frame_data,
                    team_classifier=team_classifier,
                    effective_fps=effective_fps,
                ))

        processed_count += 1
        if processed_count % 5 == 0:
            pct = 5 + int((frame_idx / max(total_frames, 1)) * 60)
            stage = "detecting" if team_fitted else "collecting_crops"
            progress_callback(stage, pct, f"Inference frame {frame_idx}/{total_frames}...")

        try:
            sidx_next, frm_next = next(frame_iter)
            submit_frame(next_processed_idx, sidx_next, frm_next)
            next_processed_idx += 1
        except StopIteration:
            pass

    # End-of-stream: late-fit if necessary, then drain
    if raw_buffer:
        if not team_fitted and len(fit_crops_pool) >= 4:
            print(f"[Pipeline] late fit on {len(fit_crops_pool)} crops at end of video")
            team_classifier.fit_crops(fit_crops_pool)
            team_fitted = team_classifier.fitted
        for rb in raw_buffer:
            processed_states.append(_compute_state(
                rb,
                team_predictions=team_predictions,
                team_clean_count=team_clean_count,
                prev_player_xy=prev_player_xy,
                speed_history=speed_history,
                frame_data=frame_data,
                team_classifier=team_classifier,
                effective_fps=effective_fps,
            ))
        raw_buffer.clear()

    cap.release()
    inference_pool.shutdown(wait=True)

    # ── Position smoothing (eliminates per-frame detection jitter) ────────
    progress_callback("smoothing", 68, "Smoothing player trajectories...")
    _smooth_positions_inplace(processed_states, window=POSITION_SMOOTH_WINDOW)

    # ── Pass 2: render every source frame at native fps with interpolation ─
    progress_callback("rendering", 70, "Rendering smooth output videos...")

    os.makedirs(settings.processed_dir, exist_ok=True)
    main_raw = os.path.join(settings.processed_dir, f"{job_id}_main_raw.mp4")
    pitch_raw = os.path.join(settings.processed_dir, f"{job_id}_pitch_raw.mp4")
    voronoi_raw = os.path.join(settings.processed_dir, f"{job_id}_voronoi_raw.mp4")
    main_path = os.path.join(settings.processed_dir, f"{job_id}_main.mp4")
    pitch_path = os.path.join(settings.processed_dir, f"{job_id}_pitch.mp4")
    voronoi_path = os.path.join(settings.processed_dir, f"{job_id}_voronoi.mp4")

    sample_pitch = draw_pitch(PITCH_CONFIG)
    pitch_h, pitch_w = sample_pitch.shape[:2]

    main_writer = _make_video_writer(main_raw, fps, width, height)
    pitch_writer = _make_video_writer(pitch_raw, fps, pitch_w, pitch_h)
    voronoi_writer = _make_video_writer(voronoi_raw, fps, pitch_w, pitch_h)

    state_indices = [s["src_frame_idx"] for s in processed_states]

    cap2 = cv2.VideoCapture(video_path)
    src_idx = 0
    while True:
        ret, frame = cap2.read()
        if not ret:
            break

        rstate = _interpolate_state_at(src_idx, processed_states, state_indices)
        outfield, ball_dets = _build_detections_for_render(rstate)

        # Main: annotated source frame
        main_frame = draw_frame(
            frame, outfield, ball_dets,
            rstate["all_team"], rstate["speeds"], annotators,
        )
        main_writer.write(main_frame)

        # Pitch: top-down 2D dots
        pitch_frame = draw_pitch_map(
            rstate["all_pitch_xy"], rstate["all_team"], rstate["ball_pitch_xy"],
        )
        if pitch_frame.shape[:2] != (pitch_h, pitch_w):
            pitch_frame = cv2.resize(pitch_frame, (pitch_w, pitch_h))
        pitch_writer.write(pitch_frame)

        # Voronoi: territorial control
        voronoi_frame = draw_voronoi_map(rstate["all_pitch_xy"], rstate["all_team"])
        if voronoi_frame.shape[:2] != (pitch_h, pitch_w):
            voronoi_frame = cv2.resize(voronoi_frame, (pitch_w, pitch_h))
        voronoi_writer.write(voronoi_frame)

        src_idx += 1
        if src_idx % 30 == 0:
            pct = 70 + int((src_idx / max(total_frames, 1)) * 20)
            progress_callback("rendering", pct, f"Rendering frame {src_idx}/{total_frames}...")

    cap2.release()
    main_writer.release()
    pitch_writer.release()
    voronoi_writer.release()

    # ── Re-encode all 3 to web-safe H.264 ─────────────────────────────────
    progress_callback("encoding", 92, "Re-encoding videos for browser playback...")
    for raw, dst in [
        (main_raw, main_path),
        (pitch_raw, pitch_path),
        (voronoi_raw, voronoi_path),
    ]:
        if _ffmpeg_to_h264(raw, dst):
            try:
                os.remove(raw)
            except OSError:
                pass
        else:
            print(f"[Encode] WARNING: ffmpeg failed for {raw}, falling back to mp4v.")
            shutil.move(raw, dst)

    progress_callback("computing_metrics", 95, "Computing match analytics...")
    analytics = build_analytics(frame_data, effective_fps)

    progress_callback("done", 100, "Processing complete!")
    return {
        "main_video_path": main_path,
        "pitch_video_path": pitch_path,
        "voronoi_video_path": voronoi_path,
        "output_video_path": main_path,  # backward compat: legacy single-output field
        "analytics": analytics,
    }
