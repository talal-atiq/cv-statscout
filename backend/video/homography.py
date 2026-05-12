import numpy as np
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from dataclasses import dataclass
from typing import Optional

PITCH_CONFIG = SoccerPitchConfiguration()


@dataclass
class HomographyResult:
    transformer: Optional[ViewTransformer]
    valid: bool
    keypoint_count: int


def compute_homography(
    keypoints: sv.KeyPoints,
    confidence_threshold: float = 0.5,
) -> HomographyResult:
    """
    Given pitch keypoints detected in a frame, compute the ViewTransformer
    that maps frame pixel coords -> pitch coords.

    Requires at least 4 high-confidence keypoints. Returns valid=False otherwise,
    so the caller can fall back to the last valid transformer.
    """
    if keypoints.xy is None or len(keypoints.xy) == 0:
        return HomographyResult(transformer=None, valid=False, keypoint_count=0)

    confidence = keypoints.confidence[0]
    xy = keypoints.xy[0]

    pitch_vertices = np.array(PITCH_CONFIG.vertices)
    if len(confidence) != len(pitch_vertices):
        # Detector returned a partial keypoint set — we can't know which vertex each
        # one corresponds to. Skip this frame; caller falls back to cached homography.
        return HomographyResult(transformer=None, valid=False, keypoint_count=0)

    mask = confidence > confidence_threshold
    frame_points = xy[mask]
    pitch_points = pitch_vertices[mask]

    keypoint_count = len(frame_points)

    if keypoint_count < 4:
        return HomographyResult(transformer=None, valid=False, keypoint_count=keypoint_count)

    try:
        transformer = ViewTransformer(
            source=frame_points.astype(np.float32),
            target=pitch_points.astype(np.float32),
        )
        return HomographyResult(transformer=transformer, valid=True, keypoint_count=keypoint_count)
    except Exception:
        return HomographyResult(transformer=None, valid=False, keypoint_count=keypoint_count)


def get_player_feet_xy(detections: sv.Detections) -> np.ndarray:
    """
    Extract the bottom-centre of each bounding box — approximates player feet,
    which is what we want to project via homography (not the box centre).
    """
    if len(detections) == 0:
        return np.empty((0, 2))
    xyxy = detections.xyxy
    feet_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
    feet_y = xyxy[:, 3]
    return np.column_stack([feet_x, feet_y])
