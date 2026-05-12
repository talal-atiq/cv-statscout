import cv2
import numpy as np
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

PITCH_CONFIG = SoccerPitchConfiguration()

# Class-id → palette index when annotating frame ellipses.
# 0 = team A (cyan), 1 = team B (pink), 2 = referee (gold)
TEAM_PALETTE = sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"])

# Voronoi blend uses these per-team colours (cv2 BGR).
TEAM_A_HEX = "#00BFFF"
TEAM_B_HEX = "#FF1493"
BALL_HEX = "#FFFFFF"

# Voronoi rendering parameters
VORONOI_PADDING = 50
VORONOI_SCALE = 0.1
VORONOI_OPACITY = 0.45
VORONOI_BLEND_STEEPNESS = 15


def build_annotators() -> dict:
    return {
        "ellipse": sv.EllipseAnnotator(color=TEAM_PALETTE, thickness=2),
        "label": sv.LabelAnnotator(
            color=TEAM_PALETTE,
            text_color=sv.Color.from_hex("#FFFFFF"),
            text_position=sv.Position.TOP_CENTER,  # above box reduces overlap with adjacent labels
            text_scale=0.35,
            text_thickness=1,
            text_padding=2,
        ),
        "ball_triangle": sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"), base=20, height=17
        ),
    }


def draw_frame(
    frame: np.ndarray,
    outfield_detections: sv.Detections,
    ball_detections: sv.Detections,
    team_labels: np.ndarray,
    speeds: dict,
    annotators: dict,
) -> np.ndarray:
    annotated = frame.copy()

    if len(outfield_detections) > 0 and len(team_labels) == len(outfield_detections):
        coloured = sv.Detections(
            xyxy=outfield_detections.xyxy.copy(),
            confidence=outfield_detections.confidence,
            class_id=team_labels.astype(int),
            tracker_id=outfield_detections.tracker_id,
        )
        annotated = annotators["ellipse"].annotate(annotated, coloured)

        tracker_ids = coloured.tracker_id if coloured.tracker_id is not None else []
        # Clean broadcast-style labels: only the speed, only when the player is moving.
        # Hides #TID clutter and avoids drawing noise on stationary players.
        labels = []
        for tid in tracker_ids:
            spd = speeds.get(int(tid), 0.0)
            labels.append(f"{spd:.0f} km/h" if spd >= 5.0 else "")
        if labels:
            annotated = annotators["label"].annotate(annotated, coloured, labels=labels)

    if len(ball_detections) > 0:
        ball_padded = sv.Detections(
            xyxy=sv.pad_boxes(xyxy=ball_detections.xyxy, px=10),
            confidence=ball_detections.confidence,
            class_id=ball_detections.class_id,
        )
        annotated = annotators["ball_triangle"].annotate(annotated, ball_padded)

    return annotated


def _draw_voronoi_overlay(
    pitch_img: np.ndarray,
    team_a_xy: np.ndarray,
    team_b_xy: np.ndarray,
) -> np.ndarray:
    """Smooth-blended Voronoi overlay showing territorial control.

    Ported from the football_ai notebook (cell 53). For each pixel of the pitch
    we compute the distance to the nearest player of each team, then blend the
    two team colours by ratio. tanh smoothing avoids hard cell edges, giving the
    "broadcast graphics" look.
    """
    if len(team_a_xy) == 0 or len(team_b_xy) == 0:
        return pitch_img

    scaled_width = int(PITCH_CONFIG.width * VORONOI_SCALE)
    scaled_length = int(PITCH_CONFIG.length * VORONOI_SCALE)

    voronoi = np.zeros_like(pitch_img, dtype=np.uint8)

    color_a = np.array(sv.Color.from_hex(TEAM_A_HEX).as_bgr(), dtype=np.uint8)
    color_b = np.array(sv.Color.from_hex(TEAM_B_HEX).as_bgr(), dtype=np.uint8)

    y_coords, x_coords = np.indices((
        scaled_width + 2 * VORONOI_PADDING,
        scaled_length + 2 * VORONOI_PADDING,
    ))
    y_coords -= VORONOI_PADDING
    x_coords -= VORONOI_PADDING

    def _min_distance(xy: np.ndarray) -> np.ndarray:
        # distances from each (player) xy to every pixel in the scaled grid
        dx = xy[:, 0][:, None, None] * VORONOI_SCALE - x_coords
        dy = xy[:, 1][:, None, None] * VORONOI_SCALE - y_coords
        return np.min(np.sqrt(dx * dx + dy * dy), axis=0)

    min_a = _min_distance(team_a_xy)
    min_b = _min_distance(team_b_xy)

    # Blend factor in [0, 1]. 1 = fully team A, 0 = fully team B.
    distance_ratio = min_b / np.clip(min_a + min_b, a_min=1e-5, a_max=None)
    blend = np.tanh((distance_ratio - 0.5) * VORONOI_BLEND_STEEPNESS) * 0.5 + 0.5

    for c in range(3):
        voronoi[:, :, c] = (blend * color_a[c] + (1 - blend) * color_b[c]).astype(np.uint8)

    # Voronoi shape may not exactly match pitch_img if the sports lib pads differently;
    # crop or pad to fit.
    h, w = pitch_img.shape[:2]
    voronoi = voronoi[:h, :w]
    if voronoi.shape != pitch_img.shape:
        # Pad with zeros if smaller
        padded = np.zeros_like(pitch_img)
        vh, vw = voronoi.shape[:2]
        padded[:vh, :vw] = voronoi
        voronoi = padded

    return cv2.addWeighted(voronoi, VORONOI_OPACITY, pitch_img, 1 - VORONOI_OPACITY, 0)


def draw_pitch_map(
    player_pitch_xy: np.ndarray,
    team_labels: np.ndarray,
    ball_pitch_xy: np.ndarray | None,
) -> np.ndarray:
    """Top-down 2D pitch view: player dots + ball, no Voronoi overlay.

    Use this for the "positions" panel. For territorial control, see draw_voronoi_map.
    """
    pitch_img = draw_pitch(PITCH_CONFIG)

    if len(player_pitch_xy) > 0 and len(team_labels) > 0 and len(player_pitch_xy) == len(team_labels):
        for team_idx, hex_colour in [(0, TEAM_A_HEX), (1, TEAM_B_HEX)]:
            mask = team_labels == team_idx
            if mask.any():
                pitch_img = draw_points_on_pitch(
                    config=PITCH_CONFIG,
                    xy=player_pitch_xy[mask],
                    face_color=sv.Color.from_hex(hex_colour),
                    edge_color=sv.Color.WHITE,
                    radius=14,
                    pitch=pitch_img,
                )

    if ball_pitch_xy is not None:
        pitch_img = draw_points_on_pitch(
            config=PITCH_CONFIG,
            xy=ball_pitch_xy[np.newaxis],
            face_color=sv.Color.from_hex(BALL_HEX),
            edge_color=sv.Color.BLACK,
            radius=8,
            pitch=pitch_img,
        )

    return pitch_img


def draw_voronoi_map(
    player_pitch_xy: np.ndarray,
    team_labels: np.ndarray,
) -> np.ndarray:
    """Top-down 2D pitch with Voronoi territorial control overlay only.

    Smooth tanh-blended team color fill showing which team controls each area.
    No player dots, no ball — pure tactical "space control" visualisation.
    Referees (class 2) are excluded — only teams 0 and 1 contribute to control.
    """
    pitch_img = draw_pitch(PITCH_CONFIG)

    if len(player_pitch_xy) > 0 and len(team_labels) > 0 and len(player_pitch_xy) == len(team_labels):
        team_a_xy = player_pitch_xy[team_labels == 0]
        team_b_xy = player_pitch_xy[team_labels == 1]
        pitch_img = _draw_voronoi_overlay(pitch_img, team_a_xy, team_b_xy)

    return pitch_img
