import cv2
import os
from pathlib import Path

# Anchor paths to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_PATH = PROJECT_ROOT / "src" / "test" / "Trial.mp4"
OUTPUT_DIR = PROJECT_ROOT / "src" / "train" / "dataset" / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not VIDEO_PATH.exists():
    # If the video is inside an assets or input folder, adjust here
    print(f"Error: Video not found at {VIDEO_PATH}")
    exit(1)

cap = cv2.VideoCapture(str(VIDEO_PATH))
roi_norm = (0.58, 0.28, 0.70, 0.54)

frame_idx = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract every 6th frame to get varied movement states
    if frame_idx % 6 == 0:
        h, w = frame.shape[:2]
        ymin, xmin, ymax, xmax = (
            int(roi_norm[0] * h),
            int(roi_norm[1] * w),
            int(roi_norm[2] * h),
            int(roi_norm[3] * w)
        )
        crop = frame[ymin:ymax, xmin:xmax]

        save_path = OUTPUT_DIR / f"pin_crop_{saved_count:04d}.jpg"
        cv2.imwrite(str(save_path), crop)
        saved_count += 1

    frame_idx += 1

cap.release()
print(f"Done! Saved {saved_count} cropped images to: {OUTPUT_DIR}")