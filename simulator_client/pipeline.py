"""Safe integrated pipeline used by the simulator runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator_client.kalman_tracker_bindings import KalmanTracker
from simulator_client.protocol import AimMessage, Matrix3x3
from detector import BoundingBox, Detection, detect_mnist_board
from simulator_client.target_selector import TargetSelector

@dataclass(frozen=True)
class PipelineResult:
    aim: AimMessage
    used_fallback: bool
    reason: str


class FallbackPipeline:
    def __init__(
        self,
        latency: float = 1.0,
        target_depth: float = 10.0,
        threshold: int = 100,
        board_width_meters: float = 0.4,
        board_height_meters: float = 0.2,
        latency_multiplier: float = 1.0,
    ) -> None:
        self.latency = latency
        self.target_depth = target_depth
        self.threshold = threshold
        self.board_width_meters = board_width_meters
        self.board_height_meters = board_height_meters
        self.latency_multiplier = latency_multiplier
        self._warmup_process_noise = 15.0
        self._stable_process_noise = 1.0
        self._warmup_measurement_noise = 0.3
        self._stable_measurement_noise = 0.8
        self._noise_warmup_frames = 8
        self.tracker = KalmanTracker(
            process_noise=self._warmup_process_noise,
            measurement_noise=self._warmup_measurement_noise,
        )
        self._last_time: float | None = None
        self._last_target: Detection | None = None
        self._lost_count: int = 0
        self.selector = TargetSelector(lost_threshold=5, min_confidence=0.2)
        self._tracking_frames = 0

    def process_rgb_image(self, image: np.ndarray, camera_matrix: Matrix3x3, timestamp: float = 0.0) -> PipelineResult:
        try:
            dt = timestamp - self._last_time if self._last_time is not None else 0.05
            self._last_time = timestamp
            dt = max(0.001, min(dt, 1.0))

            detections = detect_mnist_board(image, threshold=self.threshold)

            # Filter to confident detections
            good = [d for d in detections if d.class_id >= 0 and d.confidence >= 0.2]
            if not good:
                good = detections  # fallback: use any detection
            
            # Estimate 3D positions for all detections
            positions = [self._estimate_position(d, camera_matrix) for d in good]

            target = self.selector.select(good)
            if target is None:
                target =good[0] if good else None
            if target is None:
                self._lost_count += 1
                if self.tracker.is_tracking and self._lost_count <= 20:
                    px, py, pz = self.tracker.predict(self.latency * self.latency_multiplier)
                    pz = max(1.0, min(pz, 200.0))
                    return PipelineResult(
                        AimMessage(float(px), float(py), float(pz)),
                        used_fallback=False, reason="coasting",
                    )
                self.tracker.reset()
                self._last_target = None
                self._tracking_frames = 0
                return self._center_fallback(image, camera_matrix, f"no target in {len(detections)} detections")
            self._lost_count = 0
            cur_x, cur_y, cur_z = self._estimate_position(target, camera_matrix)
            x,y,z=self.tracker.get_position()
            if (self._last_target is not None and target.class_id != self._last_target.class_id) or np.sqrt((cur_x-x)**2+(cur_y-y)**2+(cur_z-z)**2)>5.0:
                self.tracker.reset()
                self._tracking_frames = 0
            self._last_target = target

            self._apply_adaptive_tracker_noise()
            self.tracker.update(cur_x, cur_y, cur_z, dt)
            self._tracking_frames += 1
            pred_x, pred_y, pred_z = self.tracker.predict(self.latency * self.latency_multiplier)
           
            # Clamp
            pred_z = max(1.0, min(pred_z, 200.0))
            max_xy = abs(pred_z) * 2.0
            pred_x = max(-max_xy, min(pred_x, max_xy))
            pred_y = max(-max_xy, min(pred_y, max_xy))

            # Build detection visualization data
            dets_payload = []
            for d, (px, py, pz) in zip(good, positions):
                dets_payload.append({
                    "classId": d.class_id,
                    "confidence": d.confidence,
                    "bbox": {
                        "x": d.bbox.x, "y": d.bbox.y,
                        "width": d.bbox.width, "height": d.bbox.height,
                    },
                    "corners": [
                        [float(d.corners[0][0]), float(d.corners[0][1])],
                        [float(d.corners[1][0]), float(d.corners[1][1])],
                        [float(d.corners[2][0]), float(d.corners[2][1])],
                        [float(d.corners[3][0]), float(d.corners[3][1])],
                    ],
                    "position": {"x": float(px), "y": float(py), "z": float(pz)},
                })

            return PipelineResult(
                AimMessage(
                    float(pred_x), float(pred_y), float(pred_z),
                    detections=dets_payload,
                ),
                used_fallback=False, reason="tracking",
            )
        except NotImplementedError as exc:
            return self._center_fallback(image, camera_matrix, f"student function not implemented: {exc}")
        except Exception as exc:
            return self._center_fallback(image, camera_matrix, f"pipeline error: {exc}")

    def _estimate_position(self, detection: Detection, camera_matrix: Matrix3x3) -> tuple[float, float, float]:
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]

        bbox_w = max(detection.bbox.width, 1.0)
        z_est = fx * self.board_width_meters / bbox_w
        z_est = max(1.0, min(z_est, 200.0))

        u, v = detection.bbox.center
        ray_x = (u - cx) / fx
        ray_y = (v - cy) / fy
        return (ray_x * z_est, ray_y * z_est, z_est)

    def _apply_adaptive_tracker_noise(self) -> None:
        progress = min(self._tracking_frames, self._noise_warmup_frames) / self._noise_warmup_frames
        process_noise = self._lerp(self._warmup_process_noise, self._stable_process_noise, progress)
        measurement_noise = self._lerp(self._warmup_measurement_noise, self._stable_measurement_noise, progress)
        self.tracker.set_noise(process_noise, measurement_noise)

    @staticmethod
    def _lerp(start: float, end: float, progress: float) -> float:
        return start + (end - start) * progress

    def _center_fallback(self, image: np.ndarray, camera_matrix: Matrix3x3, reason: str) -> PipelineResult:
        height, width = image.shape[:2]
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]
        u, v = (width - 1) * 0.5, (height - 1) * 0.5
        ray_x = (u - cx) / fx
        ray_y = (v - cy) / fy
        px, py, pz = ray_x * self.target_depth, ray_y * self.target_depth, self.target_depth
        return PipelineResult(AimMessage(float(px), float(py), float(pz)), used_fallback=True, reason=reason)
