import cv2
import numpy as np

from src.detection.stabilizer import MachineStabilizer

# Calibrated default ROI coordinates for copper pin feeder bed (ymin, xmin, ymax, xmax)
# Isolates the 8-lane ramp located between the yellow guide rollers
DEFAULT_ROI = (0.58, 0.28, 0.70, 0.54)

# Default 4-point polygon derived from DEFAULT_ROI
DEFAULT_ROI_POLY = [
    (0.28, 0.58),  # Top-Left (xmin, ymin)
    (0.54, 0.58),  # Top-Right (xmax, ymin)
    (0.54, 0.70),  # Bottom-Right (xmax, ymax)
    (0.28, 0.70)   # Bottom-Left (xmin, ymax)
]


class ROIPreprocessor:
    """
    Applies a 4-point perspective transform to extract a top-down rectified image
    of the 8-lane copper pin feeder bed, eliminating perspective distortion so all
    8 channels become vertical, uniform-width parallel slices.
    
    Includes dynamic MachineStabilizer to lock the ROI to the feeder bed under camera wobble.
    """
    def __init__(self, roi_box=None, roi_polygon=None, warp_size=(240, 160), enable_stabilizer=True):
        """
        roi_box: tuple of (ymin, xmin, ymax, xmax) normalized coordinates
        roi_polygon: list of 4 (x, y) normalized coordinates [(x_tl, y_tl), (x_tr, y_tr), (x_br, y_br), (x_bl, y_bl)]
        warp_size: (width, height) of the rectified output image
        enable_stabilizer: whether to track machine features to counteract camera wobble
        """
        self.warp_w, self.warp_h = warp_size
        self.stabilizer_enabled = enable_stabilizer
        self.stabilizer = MachineStabilizer() if enable_stabilizer else None
        
        if roi_box is not None:
            self.set_roi_box(roi_box)
        elif roi_polygon is not None:
            self.set_roi_polygon(roi_polygon)
        else:
            self.set_roi_box(DEFAULT_ROI)

        # Precompute destination points for perspective transform
        self.pts_dst = np.array([
            [0, 0],
            [self.warp_w - 1, 0],
            [self.warp_w - 1, self.warp_h - 1],
            [0, self.warp_h - 1]
        ], dtype=np.float32)

    def set_roi_polygon(self, roi_polygon, frame=None):
        """Sets the 4-corner polygon in normalized coordinates (x, y) and re-initializes stabilizer if frame provided."""
        if len(roi_polygon) == 4:
            self.roi_polygon = [(float(p[0]), float(p[1])) for p in roi_polygon]
            if self.stabilizer:
                self.stabilizer.reset()
                if frame is not None:
                    self.stabilizer.initialize(frame, self.roi_polygon)

    def set_roi_box(self, roi_box, frame=None):
        """Converts a normalized (ymin, xmin, ymax, xmax) bounding box into 4 polygon corners."""
        ymin, xmin, ymax, xmax = roi_box
        self.roi_polygon = [
            (xmin, ymin),  # TL
            (xmax, ymin),  # TR
            (xmax, ymax),  # BR
            (xmin, ymax)   # BL
        ]
        if self.stabilizer:
            self.stabilizer.reset()
            if frame is not None:
                self.stabilizer.initialize(frame, self.roi_polygon)

    def initialize_stabilizer(self, frame):
        """Initializes the machine stabilizer anchor points on the given frame."""
        if self.stabilizer and frame is not None:
            return self.stabilizer.initialize(frame, self.roi_polygon)
        return False

    def warp(self, frame, stabilize=True):
        """
        Transforms the quadrilateral feeder bed in `frame` into a top-down rectified image.
        Dynamically adjusts for camera wobble if stabilizer is active.
        Returns:
            rectified: (warp_h, warp_w, 3) image where lanes are perfectly vertical
            pts_src_px: (4, 2) int32 pixel coordinates on the original frame
            M: (3, 3) perspective transform matrix (original -> rectified)
            M_inv: (3, 3) inverse perspective transform matrix (rectified -> original)
        """
        if frame is None:
            return None, np.zeros((4, 2), dtype=np.int32), None, None

        h, w = frame.shape[:2]

        # Auto-initialize stabilizer on the first frame if needed
        if stabilize and self.stabilizer_enabled and self.stabilizer:
            if not self.stabilizer.is_initialized:
                self.stabilizer.initialize(frame, self.roi_polygon)

        # Retrieve stabilized or static coordinates
        if stabilize and self.stabilizer_enabled and self.stabilizer and self.stabilizer.is_initialized:
            pts_src_px, _, is_locked = self.stabilizer.update(frame)
            pts_src = pts_src_px.astype(np.float32)
        else:
            pts_src = np.array([
                [p[0] * w, p[1] * h] for p in self.roi_polygon
            ], dtype=np.float32)

        # Compute forward and inverse perspective matrices
        M = cv2.getPerspectiveTransform(pts_src, self.pts_dst)
        M_inv = cv2.getPerspectiveTransform(self.pts_dst, pts_src)

        # Apply perspective warp
        rectified = cv2.warpPerspective(frame, M, (self.warp_w, self.warp_h), flags=cv2.INTER_LINEAR)
        return rectified, pts_src.astype(np.int32), M, M_inv

    def crop(self, frame):
        """
        Backwards-compatible wrapper returning (rectified, pts_src_px).
        """
        rectified, pts_src_px, _, _ = self.warp(frame)
        return rectified, pts_src_px