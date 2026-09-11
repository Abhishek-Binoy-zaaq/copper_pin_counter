import cv2
import numpy as np


class MachineStabilizer:
    """
    Real-time optical flow machine tracker to neutralize camera wobble and vibration.
    
    How it works:
    1. Detects high-contrast feature corners on the rigid machine structure around the feeder bed.
    2. Uses pyramidal Lucas-Kanade optical flow (calcOpticalFlowPyrLK) to track machine displacement
       frame-by-frame with sub-pixel precision (< 1ms execution time).
    3. Employs RANSAC affine estimation (estimateAffinePartial2D) to reject moving pins or personnel
       and capture only genuine camera translation and rotation.
    4. Dynamically shifts the 4-point ROI polygon so the inspection zone physically sticks to the
       feeder channels, ensuring parallel lane alignment even under intense camera shake.
    """
    def __init__(self, max_corners=120, quality_level=0.015, min_distance=10):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

        self.ref_gray = None
        self.ref_pts = None
        self.base_poly_px = None
        self.base_poly_norm = None
        self.frame_shape = None
        self.is_initialized = False

        self.last_tracked_poly_px = None
        self.last_tracked_poly_norm = None
        self.last_transform = None
        self.inlier_count = 0
        self.is_locked = False

    def reset(self):
        """Resets the tracker reference state."""
        self.ref_gray = None
        self.ref_pts = None
        self.base_poly_px = None
        self.base_poly_norm = None
        self.frame_shape = None
        self.is_initialized = False
        self.last_tracked_poly_px = None
        self.last_tracked_poly_norm = None
        self.last_transform = None
        self.inlier_count = 0
        self.is_locked = False

    def initialize(self, frame, roi_polygon_norm):
        """
        Initializes the machine anchor points around the feeder bed.
        roi_polygon_norm: list of 4 (x, y) normalized coordinates.
        """
        if frame is None or len(roi_polygon_norm) != 4:
            return False

        h, w = frame.shape[:2]
        self.frame_shape = (h, w)
        self.base_poly_norm = [(float(p[0]), float(p[1])) for p in roi_polygon_norm]
        self.base_poly_px = np.array([
            [p[0] * w, p[1] * h] for p in self.base_poly_norm
        ], dtype=np.float32)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Build mask for rigid machine features:
        # Include an expanded margin around the feeder bed, but exclude the active sliding channel center
        # to prevent moving copper pins from being chosen as stationary anchor points.
        mask = np.zeros((h, w), dtype=np.uint8)

        # Bounding box of ROI
        xs = [p[0] for p in self.base_poly_px]
        ys = [p[1] for p in self.base_poly_px]
        min_x, max_x = max(0, int(min(xs))), min(w, int(max(xs)))
        min_y, max_y = max(0, int(min(ys))), min(h, int(max(ys)))

        # Expand anchor search zone around the feeder bed
        pad_x = int((max_x - min_x) * 0.45)
        pad_y = int((max_y - min_y) * 0.35)
        anchor_x1 = max(0, min_x - pad_x)
        anchor_x2 = min(w, max_x + pad_x)
        anchor_y1 = max(0, min_y - pad_y)
        anchor_y2 = min(h, max_y + pad_y)

        # Anchor area on surrounding rigid chassis
        mask[anchor_y1:anchor_y2, anchor_x1:anchor_x2] = 255
        
        # Exclude inner 60% of ROI where moving pins pass
        inner_pad_x = int((max_x - min_x) * 0.15)
        inner_pad_y = int((max_y - min_y) * 0.15)
        mask[min_y + inner_pad_y:max_y - inner_pad_y, min_x + inner_pad_x:max_x - inner_pad_x] = 0

        # Detect Shi-Tomasi corners on stationary machine structures
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask
        )

        if corners is None or len(corners) < 8:
            # Fallback: search anywhere in anchor bounding box without inner exclusion
            mask[anchor_y1:anchor_y2, anchor_x1:anchor_x2] = 255
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level * 0.5,
                minDistance=self.min_distance,
                mask=mask
            )

        if corners is not None and len(corners) >= 6:
            self.ref_gray = gray.copy()
            self.ref_pts = corners.reshape(-1, 1, 2).astype(np.float32)
            self.last_tracked_poly_px = self.base_poly_px.copy()
            self.last_tracked_poly_norm = list(self.base_poly_norm)
            self.inlier_count = len(corners)
            self.is_initialized = True
            self.is_locked = True
            return True

        return False

    def update(self, current_frame):
        """
        Calculates optical flow from the reference frame and returns the dynamically adjusted ROI polygon.
        Returns:
            tracked_poly_px: (4, 2) int32 coordinates of ROI in current frame
            tracked_poly_norm: list of 4 (x, y) normalized coordinates
            is_locked: True if tracking confidence is high
        """
        if not self.is_initialized or current_frame is None:
            if self.base_poly_px is not None:
                return self.base_poly_px.astype(np.int32), self.base_poly_norm, False
            return None, None, False

        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if len(current_frame.shape) == 3 else current_frame

        # Track keypoints from reference frame using pyramidal Lucas-Kanade
        tracked_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.ref_gray,
            curr_gray,
            self.ref_pts,
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if tracked_pts is not None and status is not None:
            valid_idx = (status.flatten() == 1)
            src_inliers = self.ref_pts[valid_idx]
            dst_inliers = tracked_pts[valid_idx]

            if len(src_inliers) >= 6:
                # Estimate 2D rigid transform (translation + rotation + uniform scale) with RANSAC
                M, inlier_mask = cv2.estimateAffinePartial2D(
                    src_inliers,
                    dst_inliers,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.5
                )

                if M is not None and inlier_mask is not None:
                    inliers_count = int(np.sum(inlier_mask))
                    self.inlier_count = inliers_count

                    if inliers_count >= 5:
                        # Transform base polygon to current frame coordinates
                        pts_transformed = cv2.transform(
                            np.array([self.base_poly_px], dtype=np.float32),
                            M
                        )[0]

                        h, w = self.frame_shape
                        norm_poly = [
                            (float(np.clip(p[0] / w, 0.0, 1.0)), float(np.clip(p[1] / h, 0.0, 1.0)))
                            for p in pts_transformed
                        ]

                        self.last_tracked_poly_px = pts_transformed
                        self.last_tracked_poly_norm = norm_poly
                        self.last_transform = M
                        self.is_locked = True

                        # Re-anchor if keypoint retention is decaying
                        if inliers_count < (len(self.ref_pts) * 0.4) and inliers_count < 25:
                            self.reanchor(current_frame, norm_poly)

                        return pts_transformed.astype(np.int32), norm_poly, True

        # If tracking lost temporarily, return last known good position
        self.is_locked = False
        if self.last_tracked_poly_px is not None:
            return self.last_tracked_poly_px.astype(np.int32), self.last_tracked_poly_norm, False

        return self.base_poly_px.astype(np.int32), self.base_poly_norm, False

    def reanchor(self, frame, current_poly_norm):
        """Refreshes keypoints using the current frame as new reference to prevent tracking drift."""
        self.initialize(frame, current_poly_norm)
