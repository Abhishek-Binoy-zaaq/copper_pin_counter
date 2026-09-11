# Copper Pin Production Counter

An OpenCV and CustomTkinter desktop app that automatically counts copper pins produced on an industrial hairpin / pin-making machine using factory CCTV footage.

---

## What Does This Project Do?

In factories making copper pins (like those used in electric motor stators or heat exchangers), pins travel down an 8-lane guide bed in periodic pulses. Every ~10 to 15 seconds, a batch of 8 pins shoots forward for about 1–2 seconds, gets cut or formed, and clears out.

Counting them accurately from a distant, angled CCTV camera is tricky because:
- **Angle & Perspective:** The ceiling camera looks at the machine from an oblique angle, making parallel channels look slanted and narrower at the top.
- **Glare & Reflections:** Shiny cylindrical copper reflects overhead factory lights intensely.
- **Machine Structure:** Parts of the machine are painted yellow or have shiny metal brackets that can easily confuse simple color or edge detectors.

This application fixes those issues using practical computer vision techniques (no heavy deep learning needed) and displays a live count in a clean desktop dashboard.

---

## How It Works

Here is what happens behind the scenes for every video frame:

1. **Straightening the View (Perspective Warp):**
   Instead of a simple rectangular box, the app takes the 4 corners of the feeder bed and flattens it into a straight, top-down view. This turns the converging angled lanes into 8 perfectly vertical, equal-width strips.

2. **Finding the Copper Pins:**
   - Enhances local contrast with CLAHE so dark channels and bright pins stand out clearly regardless of changing room light.
   - Filters for reddish-orange / amber copper hues while ignoring the bright yellow paint on the machine clamps.
   - Picks up specular glares on the copper pins so reflections don't leave hollow gaps.
   - Uses a vertical line filter ($1 \times 7$ pixels) to keep straight pin shapes while wiping out square bolts and horizontal machine beams.

3. **Checking Lane Occupancy (With Memory):**
   - Slices the straightened view into 8 vertical lanes.
   - Each lane keeps a short 5-frame rolling memory. A lane is only marked active if a pin is visible for at least 3 of those 5 frames. This completely filters out random 1-frame light flashes or camera noise.

4. **Batch State Machine (Zero Double-Counts):**
   - **IDLE:** Waits until at least 3 lanes detect pins at the same time.
   - **BATCH ACTIVE:** Tracks exactly which of the 8 lanes have pins during the 1–2 second feed stroke.
   - **CLEAR & COUNT:** When the pins slide out of the bed (or after a 2-second safety timeout), it counts only the lanes that actually had pins (e.g. +8 if all lanes were full, +7 if one lane was empty).
   - **COOLDOWN:** Locks out counting for ~4 seconds (120 frames) while the bed is empty so machine vibrations or paused pins can never trigger a double-count.

---

## Project Structure

```text
copper_pin_production_counter/
│
├── main.py                     # Entry point to launch the application
├── README.md                   # Project documentation
│
└── src/
    ├── detection/
    │   └── preprocessor.py     # 4-point perspective warp and ROI geometry
    │
    ├── tracking/
    │   └── lane_counter.py     # Pin segmentation, temporal deques, and state machine
    │
    └── interface/
        └── app.py              # CustomTkinter GUI, video loop, and dashboard cards
```

---

## Getting Started

### 1. Requirements

Make sure you have Python 3.9+ installed. You will need the following packages:

```bash
pip install opencv-python customtkinter pillow numpy
```

### 2. Run the App

Open your terminal in the project directory and run:

```bash
python main.py
```

---

## How to Use the App

1. **Load a Video:** Click **Browse Video** and select your video file (for example, `src/test/Trial.mp4`).
2. **Check the Red Box:** A red box will highlight the flat black feeder bed directly above the yellow clamp mechanism.
3. **Start Counting:** Click **Start Playback**.
   - As pins feed down the channels, green lines appear on the active lanes.
   - Once the pins exit, the **Total Pins Counted** tally increments (e.g. `+8`), and the status switches to `COOLDOWN` before resetting for the next batch.
4. **Extra Controls:**
   - **Calibrate Custom ROI:** If your camera angle changes, click this to drag a new box around the feeder bed.
   - **Reset Default ROI:** Reverts back to the calibrated coordinates.
   - **Show Pin Occupancy Mask:** Turn this switch on to see the live black-and-white computer vision mask directly inside the video.
   - **Reset Count to 0:** Resets the running total back to zero anytime.
