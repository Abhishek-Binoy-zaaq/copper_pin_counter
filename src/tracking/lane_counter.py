import cv2
import numpy as np
from collections import deque


class LanePinTracker:
    """
    Industrial Copper Pin Counter for 8-lane conveyor bed.
    
    Key Features:
    1. Preprocessing: CLAHE adaptive contrast equalization on L-channel (LAB)
       combined with restricted amber/orange HSV segmentation (rejecting yellow paint)
       and specular highlight recovery (threshold 175).
    2. Vertical Morphological Filtering: (1, 7) vertical kernel preserving vertical
       copper pin lines while stripping machinery and fastener blobs.
    3. Rolling Temporal Integration: 5-frame deque buffer per lane to confirm presence
       (requires >= 3 of 5 frames occupied) eliminating transient glare/shadow flicker.
    4. Batch-Pulse State Machine: Latches active lanes during the pulse and triggers
       the tally strictly upon batch exit/clearance or safety timeout (active_frames_count > 60).
    5. Per-Lane Verification: Tallies only lanes verified present during the cycle
       when verified_count >= min_pulse_lanes.
    6. Debounced Lockout: 120-frame (~4 seconds at 30 FPS) cooldown preventing any
       duplicate counts during feeder pauses or machine vibration.
    """
    def __init__(
        self,
        num_lanes=8,
        brightness_thresh=85,
        density_thresh=0.12,
        min_pulse_lanes=3,
        cooldown_frames=120,
        clear_frames_required=4,
        temporal_window=5,
        temporal_min_hits=3
    ):
        self.num_lanes = num_lanes
        self.brightness_thresh = brightness_thresh
        self.density_thresh = density_thresh
        self.min_pulse_lanes = min_pulse_lanes
        self.cooldown_frames = cooldown_frames
        self.clear_frames_required = clear_frames_required
        self.temporal_window = temporal_window
        self.temporal_min_hits = temporal_min_hits

        self.total_count = 0
        self.last_increment = 0
        self.state = "IDLE"  # "IDLE", "BATCH_ACTIVE", "COOLDOWN"

        self.cooldown_timer = 0
        self.clear_counter = 0
        self.active_frames_count = 0
        self.batch_lanes_latched = [False] * num_lanes

        # Rolling temporal buffer: deque of 5 booleans per lane
        self.lane_history = [deque(maxlen=self.temporal_window) for _ in range(num_lanes)]
        
        self.debug_mask = None
        self.current_densities = [0.0] * num_lanes
        self.current_occupied = [False] * num_lanes

        # CLAHE operator for adaptive local contrast equalization
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def reset(self):
        """Resets all accumulators, histories, and state machine variables."""
        self.total_count = 0
        self.last_increment = 0
        self.state = "IDLE"
        self.cooldown_timer = 0
        self.clear_counter = 0
        self.active_frames_count = 0
        self.batch_lanes_latched = [False] * self.num_lanes
        self.lane_history = [deque(maxlen=self.temporal_window) for _ in range(self.num_lanes)]
        self.debug_mask = None
        self.current_densities = [0.0] * self.num_lanes
        self.current_occupied = [False] * self.num_lanes

    def _segment_pins(self, rectified_bgr):
        """
        Applies CLAHE local contrast equalization and multi-cue segmentation:
        - CLAHE on LAB L-channel to balance factory lighting and lift copper edges
        - Restricted amber/reddish-orange HSV filter [H: 5-18, S: 45-200, V: 60-255]
          to strictly reject bright yellow machine paint
        - High-brightness specular highlight mask (threshold 175) to capture reflective glares
        - Tall vertical morphological kernel (1, 7) to preserve vertical pin lines
          while eliminating square machinery blobs
        """
        # 1. CLAHE Equalization in LAB space
        lab = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_eq = self.clahe.apply(l_chan)
        lab_eq = cv2.merge((l_eq, a_chan, b_chan))
        enhanced_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # 2. HSV Copper Chromaticity Mask (rejects bright yellow paint > 18)
        hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
        copper_mask = cv2.inRange(hsv, (5, 45, 60), (18, 200, 255))

        # 3. Specular Highlights
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        _, specular_mask = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

        # 4. Multi-cue Fusion: bitwise OR
        pin_mask = cv2.bitwise_or(copper_mask, specular_mask)

        # 5. Tall vertical kernel to preserve vertical pin lines and eliminate square blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        pin_mask = cv2.morphologyEx(pin_mask, cv2.MORPH_OPEN, kernel)

        return pin_mask

    def process_frame(self, rectified_bgr):
        """
        Processes a single rectified frame where channels are vertical and parallel.
        Returns:
            total_count: Cumulative pins counted
            new_counts: Pins incremented in this frame (if exit trigger fired)
            lane_statuses: Status metadata per lane for UI visualization
            (y_top, y_bottom): Vertical inspection zone pixel boundaries
        """
        h, w = rectified_bgr.shape[:2]
        if h == 0 or w == 0:
            return self.total_count, 0, [], (0, 0)

        # 1. Segment copper pins
        pin_mask = self._segment_pins(rectified_bgr)
        self.debug_mask = pin_mask

        # 2. Central Inspection Band (middle 60% of rectified bed)
        y_zone_top = int(h * 0.20)
        y_zone_bottom = int(h * 0.80)

        lane_w = w / self.num_lanes
        new_counts = 0
        lane_statuses = []
        confirmed_occupied_lanes = [False] * self.num_lanes

        # 3. Lane-by-Lane Evaluation & Rolling Temporal Buffer
        for i in range(self.num_lanes):
            x1 = int(i * lane_w)
            x2 = int((i + 1) * lane_w)

            lane_roi = pin_mask[y_zone_top:y_zone_bottom, x1:x2]
            total_px = max(1, lane_roi.size)
            active_px = np.count_nonzero(lane_roi)
            density = active_px / total_px
            self.current_densities[i] = density

            # Instantaneous raw occupancy
            raw_occupied = density >= self.density_thresh

            # Update rolling temporal history
            self.lane_history[i].append(raw_occupied)

            # Confirmed occupied only if sustained across >= temporal_min_hits of last 5 frames
            confirmed = sum(self.lane_history[i]) >= self.temporal_min_hits
            confirmed_occupied_lanes[i] = confirmed
            self.current_occupied[i] = confirmed

        active_count = sum(confirmed_occupied_lanes)

        # 4. Batch-Pulse State Machine with Deadlock Prevention
        if self.state == "IDLE":
            # Pulse arrives when multiple lanes sustain presence simultaneously
            if active_count >= self.min_pulse_lanes:
                self.state = "BATCH_ACTIVE"
                self.active_frames_count = 0
                self.clear_counter = 0
                self.batch_lanes_latched = list(confirmed_occupied_lanes)
            else:
                self.last_increment = 0

        elif self.state == "BATCH_ACTIVE":
            self.active_frames_count += 1

            # Latch any lane that sustains presence at any point during this pulse
            for i in range(self.num_lanes):
                if confirmed_occupied_lanes[i]:
                    self.batch_lanes_latched[i] = True

            # Exit condition: lanes drop below threshold OR safety timeout exceeded (batch pulses last ~1-2s)
            exit_condition = (active_count < 2) or (self.active_frames_count > 60)
            if exit_condition:
                self.clear_counter += 1
                if self.clear_counter >= self.clear_frames_required or self.active_frames_count > 60:
                    verified_count = sum(1 for present in self.batch_lanes_latched if present)
                    
                    # Verify a true pulse occurred before incrementing total_count
                    if verified_count >= self.min_pulse_lanes:
                        self.total_count += verified_count
                        self.last_increment = verified_count
                        new_counts = verified_count
                    else:
                        self.last_increment = 0
                        new_counts = 0

                    # Reset active frames counter when transitioning to COOLDOWN
                    self.active_frames_count = 0
                    self.state = "COOLDOWN"
                    self.cooldown_timer = self.cooldown_frames
            else:
                self.clear_counter = 0

        elif self.state == "COOLDOWN":
            self.cooldown_timer -= 1
            if self.cooldown_timer <= 0:
                # Cooldown expired: unconditionally re-arm for next batch
                self.cooldown_timer = 0
                self.state = "IDLE"
                self.batch_lanes_latched = [False] * self.num_lanes
                self.clear_counter = 0
                self.active_frames_count = 0
                for q in self.lane_history:
                    q.clear()

        # 5. Construct Status Metadata for Overlays
        for i in range(self.num_lanes):
            x1 = int(i * lane_w)
            x2 = int((i + 1) * lane_w)
            lane_statuses.append({
                "lane_id": i,
                "x_start": x1,
                "x_end": x2,
                "occupied": self.current_occupied[i],
                "latched": self.batch_lanes_latched[i],
                "density": self.current_densities[i],
                "state": self.state,
                "cooldown_remaining": self.cooldown_timer
            })

        return self.total_count, new_counts, lane_statuses, (y_zone_top, y_zone_bottom)


ProductionCounter = LanePinTracker