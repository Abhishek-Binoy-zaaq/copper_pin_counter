import customtkinter as ctk
import cv2
from PIL import Image
from tkinter import filedialog, messagebox
import numpy as np
import threading
import time

from src.detection.preprocessor import ROIPreprocessor, DEFAULT_ROI, DEFAULT_ROI_POLY
from src.tracking.lane_counter import ProductionCounter

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class PinCounterApp(ctk.CTk):
    """
    Production Counter Desktop Application for 8-lane copper pin factory CCTV.
    Features:
    - Dynamic machine-anchored optical flow stabilization to counteract camera wobble
    - 4-point perspective warp for top-down rectified pin analysis
    - Inverse perspective overlay projecting lane states back onto oblique camera view
    - CLAHE + HSV segmented mask preview
    - Thread-safe CustomTkinter UI dispatch with queue throttling
    """
    def __init__(self):
        super().__init__()
        self.title("Copper Pin Production Counter")
        self.geometry("1240x750")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.cap = None
        self.is_playing = False
        self.stop_stream = False
        self.video_thread = None
        self.current_ctk_img = None
        self.current_video_path = None
        self.show_mask_view = False
        self.ui_update_pending = False
        self.stabilize_enabled = True
        
        # Feeder bed ROI preprocessor with camera wobble stabilization
        self.preprocessor = ROIPreprocessor(warp_size=(240, 160), enable_stabilizer=True)
        self.counter = ProductionCounter(num_lanes=8)

        self._setup_layout()

    def _setup_layout(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Column: Video Playback Frame
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        self.video_display = ctk.CTkLabel(
            self.left_frame,
            text="No Video Loaded\n\nClick 'Browse Video' to load Trial.mp4",
            font=("Arial", 16)
        )
        self.video_display.pack(expand=True, fill="both", padx=10, pady=10)

        # Right Column: Dashboard Controls & Counters
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        ctk.CTkLabel(self.right_frame, text="Production Dashboard", font=("Arial", 18, "bold")).pack(pady=(15, 8))

        self.btn_select = ctk.CTkButton(self.right_frame, text="Browse Video", command=self.browse_video)
        self.btn_select.pack(pady=5, padx=20, fill="x")

        self.btn_toggle = ctk.CTkButton(self.right_frame, text="Start Playback", state="disabled", command=self.toggle_playback)
        self.btn_toggle.pack(pady=5, padx=20, fill="x")

        self.btn_calibrate = ctk.CTkButton(
            self.right_frame,
            text="Calibrate Custom ROI",
            state="disabled",
            fg_color="#4A5568",
            hover_color="#2D3748",
            command=self.calibrate_roi
        )
        self.btn_calibrate.pack(pady=5, padx=20, fill="x")

        self.btn_reset_roi = ctk.CTkButton(
            self.right_frame,
            text="Reset Default ROI",
            fg_color="#4A5568",
            hover_color="#2D3748",
            command=self.reset_roi
        )
        self.btn_reset_roi.pack(pady=5, padx=20, fill="x")

        self.btn_reset = ctk.CTkButton(
            self.right_frame,
            text="Reset Count to 0",
            fg_color="#C53030",
            hover_color="#9B2C2C",
            command=self.reset_counter
        )
        self.btn_reset.pack(pady=5, padx=20, fill="x")

        self.stabilizer_switch = ctk.CTkSwitch(
            self.right_frame,
            text="Auto-Stabilize (Lock to Bed)",
            command=self.toggle_stabilizer
        )
        self.stabilizer_switch.select()
        self.stabilizer_switch.pack(pady=(8, 4), padx=20)

        self.mask_switch = ctk.CTkSwitch(self.right_frame, text="Show Pin Occupancy Mask", command=self.toggle_mask_view)
        self.mask_switch.pack(pady=(4, 8), padx=20)

        self.lbl_timestamp = ctk.CTkLabel(self.right_frame, text="Timestamp: 00:00:00", font=("Arial", 14))
        self.lbl_timestamp.pack(pady=(8, 4))

        # State Card
        self.card_state = ctk.CTkFrame(self.right_frame, corner_radius=8)
        self.card_state.pack(pady=8, padx=20, fill="x")
        ctk.CTkLabel(self.card_state, text="Machine State", font=("Arial", 12)).pack(pady=(6, 2))
        self.val_state = ctk.CTkLabel(self.card_state, text="IDLE", font=("Arial", 18, "bold"), text_color="#A0AEC0")
        self.val_state.pack(pady=(0, 6))

        # Increment Card
        self.card_current = ctk.CTkFrame(self.right_frame, corner_radius=8)
        self.card_current.pack(pady=8, padx=20, fill="x")
        ctk.CTkLabel(self.card_current, text="Current Batch Increment", font=("Arial", 12)).pack(pady=(6, 2))
        self.val_current = ctk.CTkLabel(self.card_current, text="0", font=("Arial", 28, "bold"), text_color="#3B8ED0")
        self.val_current.pack(pady=(0, 6))

        # Total Count Card
        self.card_total = ctk.CTkFrame(self.right_frame, corner_radius=8)
        self.card_total.pack(pady=8, padx=20, fill="x")
        ctk.CTkLabel(self.card_total, text="Total Pins Counted", font=("Arial", 14, "bold")).pack(pady=(8, 2))
        self.val_total = ctk.CTkLabel(self.card_total, text="0", font=("Arial", 40, "bold"), text_color="#2ECC71")
        self.val_total.pack(pady=(0, 8))

    def _stop_current_stream(self):
        """Cleanly stops the background video worker thread before releasing video resources."""
        self.is_playing = False
        self.stop_stream = True
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=0.6)
        self.video_thread = None
        self.stop_stream = False

    def toggle_stabilizer(self):
        self.stabilize_enabled = (self.stabilizer_switch.get() == 1)
        self.preprocessor.stabilizer_enabled = self.stabilize_enabled

    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if file_path:
            self.current_video_path = file_path
            self._stop_current_stream()

            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(file_path)
            self.counter.reset()
            self.val_total.configure(text="0")
            self.val_current.configure(text="0")
            self.val_state.configure(text="IDLE", text_color="#A0AEC0")
            self.btn_toggle.configure(state="normal", text="Start Playback")
            self.btn_calibrate.configure(state="normal")
            
            ret, frame = self.cap.read()
            if ret:
                self.preprocessor.initialize_stabilizer(frame)
                rectified, pts_src, _, _ = self.preprocessor.warp(frame, stabilize=self.stabilize_enabled)
                preview = frame.copy()
                cv2.polylines(preview, [pts_src], isClosed=True, color=(0, 255, 0), thickness=2)
                self._render_frame(preview)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def calibrate_roi(self):
        if not self.current_video_path:
            return
        
        self._stop_current_stream()
        self.btn_toggle.configure(text="Start Playback")

        temp_cap = cv2.VideoCapture(self.current_video_path)
        ret, frame = temp_cap.read()
        temp_cap.release()

        if not ret or frame is None:
            messagebox.showerror("Error", "Could not read frame for calibration.")
            return

        h, w = frame.shape[:2]
        roi = cv2.selectROI("Select Feeder Bed ROI (Press ENTER to confirm, ESC to cancel)", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        rx, ry, rw, rh = roi
        if rw > 10 and rh > 10:
            norm_box = (ry / h, rx / w, (ry + rh) / h, (rx + rw) / w)
            self.preprocessor.set_roi_box(norm_box, frame=frame)
            self.counter.reset()
            rectified, pts_src, _, _ = self.preprocessor.warp(frame, stabilize=self.stabilize_enabled)
            preview = frame.copy()
            cv2.polylines(preview, [pts_src], isClosed=True, color=(0, 255, 0), thickness=2)
            self._render_frame(preview)
            messagebox.showinfo("ROI Updated", f"Feeder ROI calibrated!\nx=[{rx}:{rx+rw}], y=[{ry}:{ry+rh}]")

    def reset_roi(self):
        self.counter.reset()
        if self.cap and self.cap.isOpened():
            curr_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            if ret:
                self.preprocessor.set_roi_box(DEFAULT_ROI, frame=frame)
                rectified, pts_src, _, _ = self.preprocessor.warp(frame, stabilize=self.stabilize_enabled)
                preview = frame.copy()
                cv2.polylines(preview, [pts_src], isClosed=True, color=(0, 255, 0), thickness=2)
                self._render_frame(preview)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, curr_pos)
        else:
            self.preprocessor.set_roi_box(DEFAULT_ROI)

    def reset_counter(self):
        self.counter.reset()
        self.val_total.configure(text="0")
        self.val_current.configure(text="0")
        self.val_state.configure(text="IDLE", text_color="#A0AEC0")

    def toggle_mask_view(self):
        self.show_mask_view = self.mask_switch.get() == 1

    def toggle_playback(self):
        if not self.is_playing:
            self.is_playing = True
            self.btn_toggle.configure(text="Pause")
            if self.video_thread is None or not self.video_thread.is_alive():
                self.video_thread = threading.Thread(target=self._stream_loop, daemon=True)
                self.video_thread.start()
        else:
            self.is_playing = False
            self.btn_toggle.configure(text="Resume")

    def _stream_loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30.0
        if not fps or fps <= 0 or fps > 120:
            fps = 30.0
        target_frame_time = 1.0 / fps

        while self.cap and self.cap.isOpened() and not self.stop_stream:
            if not self.is_playing:
                time.sleep(0.04)
                continue

            loop_start = time.time()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.is_playing = False
                self.after(0, self._on_playback_finished)
                break

            fh, fw = frame.shape[:2]

            # 1. Warp perspective to top-down rectified image (with wobble stabilization)
            rectified, pts_src, M, M_inv = self.preprocessor.warp(frame, stabilize=self.stabilize_enabled)
            
            # 2. Track pin batch pulse on rectified channels
            total_count, delta_count, statuses, (y_top, y_bottom) = self.counter.process_frame(rectified)

            # 3. Draw overlays
            # Dynamic ROI polygon boundary (Green if locked by stabilizer, Red if static/lost)
            is_locked = (self.stabilize_enabled and self.preprocessor.stabilizer and self.preprocessor.stabilizer.is_locked)
            roi_color = (0, 255, 0) if is_locked else (0, 0, 255)
            cv2.polylines(frame, [pts_src], isClosed=True, color=roi_color, thickness=2)

            # Project rectified inspection zones and lane dividers back onto original perspective view
            rw, rh = self.preprocessor.warp_w, self.preprocessor.warp_h
            
            # Boundary lines (yellow top, cyan bottom)
            boundary_pts_rect = np.array([
                [[0, y_top]], [[rw - 1, y_top]],
                [[0, y_bottom]], [[rw - 1, y_bottom]]
            ], dtype=np.float32)
            boundary_pts_orig = cv2.perspectiveTransform(boundary_pts_rect, M_inv).astype(np.int32)
            
            cv2.line(frame, tuple(boundary_pts_orig[0][0]), tuple(boundary_pts_orig[1][0]), (255, 255, 0), 1)
            cv2.line(frame, tuple(boundary_pts_orig[2][0]), tuple(boundary_pts_orig[3][0]), (0, 255, 255), 1)

            # Lane dividers and occupancy indicators
            lane_pts_rect = []
            for st in statuses:
                x_mid = int((st["x_start"] + st["x_end"]) / 2)
                lane_pts_rect.extend([
                    [[st["x_start"], 0]], [[st["x_start"], rh - 1]],
                    [[x_mid, y_top]], [[x_mid, y_bottom]]
                ])

            lane_pts_orig = cv2.perspectiveTransform(np.array(lane_pts_rect, dtype=np.float32), M_inv).astype(np.int32)

            for i, st in enumerate(statuses):
                # Divider line
                div_top = tuple(lane_pts_orig[i * 4][0])
                div_bot = tuple(lane_pts_orig[i * 4 + 1][0])
                cv2.line(frame, div_top, div_bot, (80, 80, 80), 1)

                # Active pin occupancy indicator
                if st["occupied"]:
                    occ_top = tuple(lane_pts_orig[i * 4 + 2][0])
                    occ_bot = tuple(lane_pts_orig[i * 4 + 3][0])
                    occ_mid = ((occ_top[0] + occ_bot[0]) // 2, (occ_top[1] + occ_bot[1]) // 2)
                    cv2.line(frame, occ_top, occ_bot, (0, 255, 0), 2)
                    cv2.circle(frame, occ_mid, 3, (0, 255, 0), -1)

                if st["latched"]:
                    occ_bot = tuple(lane_pts_orig[i * 4 + 3][0])
                    cv2.putText(frame, "OK", (occ_bot[0] - 8, occ_bot[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            # Live Machine State Banner on Video
            current_state = self.counter.state
            lock_label = " [STABILIZED]" if is_locked else ""
            if current_state == "BATCH_ACTIVE":
                banner_text = f"STATUS: BATCH PULSE ACTIVE{lock_label}"
                banner_color = (0, 255, 0)
            elif current_state == "COOLDOWN":
                banner_text = f"STATUS: COOLDOWN ({self.counter.cooldown_timer}){lock_label}"
                banner_color = (0, 165, 255)
            else:
                banner_text = f"STATUS: IDLE (WAITING FOR BATCH){lock_label}"
                banner_color = (200, 200, 200)

            # Draw banner above top-left corner of ROI
            tl_x, tl_y = pts_src[0]
            cv2.putText(frame, banner_text, (max(10, tl_x - 30), max(25, tl_y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, banner_color, 1)

            # If user enabled mask view, warp the binary pin occupancy mask directly into ROI
            if self.show_mask_view and self.counter.debug_mask is not None:
                mask_bgr = cv2.cvtColor(self.counter.debug_mask, cv2.COLOR_GRAY2BGR)
                warped_mask = cv2.warpPerspective(mask_bgr, M_inv, (fw, fh))
                poly_mask = cv2.fillPoly(np.zeros((fh, fw), dtype=np.uint8), [pts_src], 255)
                frame[poly_mask > 0] = warped_mask[poly_mask > 0]

            # 4. Push updates to UI (strictly dispatched to main thread)
            msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            formatted_time = time.strftime('%H:%M:%S', time.gmtime(msec / 1000.0))
            
            if not self.ui_update_pending:
                self.ui_update_pending = True
                self.after(0, self._update_ui_state, frame, formatted_time, delta_count, total_count, current_state)

            # Smooth playback pacing
            elapsed = time.time() - loop_start
            time.sleep(max(0.001, target_frame_time - elapsed))

    def _on_playback_finished(self):
        self.btn_toggle.configure(text="Finished", state="disabled")
        self.val_state.configure(text="COMPLETED", text_color="#A0AEC0")

    def _update_ui_state(self, frame, timestamp, delta_count, total_count, state_str):
        try:
            self._render_frame(frame)
            self.lbl_timestamp.configure(text=f"Timestamp: {timestamp}")
            
            if delta_count > 0:
                self.val_current.configure(text=f"+{delta_count}")
            elif state_str == "IDLE":
                self.val_current.configure(text=str(self.counter.last_increment))

            self.val_total.configure(text=str(total_count))

            # Update State Badge
            if state_str == "BATCH_ACTIVE":
                self.val_state.configure(text="BATCH PULSE", text_color="#2ECC71")
            elif state_str == "COOLDOWN":
                self.val_state.configure(text=f"COOLDOWN ({self.counter.cooldown_timer})", text_color="#E67E22")
            else:
                self.val_state.configure(text="IDLE", text_color="#A0AEC0")
        finally:
            self.ui_update_pending = False

    def _render_frame(self, bgr_frame):
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        
        display_w = 820
        display_h = int(h * (display_w / w))
        
        img = Image.fromarray(cv2.resize(rgb_frame, (display_w, display_h)))
        self.current_ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(display_w, display_h))
        self.video_display.configure(image=self.current_ctk_img, text="")

    def on_closing(self):
        self._stop_current_stream()
        if self.cap:
            self.cap.release()
        self.destroy()

