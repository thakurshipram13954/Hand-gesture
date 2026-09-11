# Hand Gesture Camera Controller 📷🖐️

A real-time Python Computer Vision application using **OpenCV** and **MediaPipe Hands** to control camera zoom (Smooth Zoom IN / Zoom OUT) and capture photographs with a **3-Second Delay Timer & 3-Second Cooldown** when holding a closed-fist gesture.

---

## 🌟 Key Features

- **Real-Time Hand Tracking**: Powered by MediaPipe Hands (21 landmark keypoints) at 30+ FPS.
- **Gesture-Based Digital Zoom**:
  - **Zoom IN**: Spread thumb and index finger tip apart.
  - **Zoom OUT**: Bring thumb and index finger tip close together.
- **Clean Saved Photographs**: Saved photos in `captured_photos/` are 100% clean (no landmark lines, gesture lines, or HUD overlays), while the live screen preview displays interactive gesture tracking feedback.
- **Closed Wrist / Fist Photo Capture with Persistent 3-Second Delay Timer**:
  - Closing your wrist/fist triggers a **3-Second Visual Countdown** (`3... 2... 1... CAPTURE!`).
  - **Persistent Countdown**: Once triggered, the 3-second timer runs continuously regardless of whether your wrist opens, changes gesture, or leaves the camera view.
  - Gives you time to prepare and pose naturally before the clean photograph is saved.
  - Includes a **3-Second Cooldown** after clicking a photo to prevent accidental repeated shooting.
- **Scale Normalization**: Euclidean landmark distance is scaled relative to palm size ($D_{\text{ref}}$: Wrist `0` to Knuckle `9`), rendering tracking depth-independent whether your hand is close to or far from the webcam.
- **Smooth Zoom Transitions**: Exponential Moving Average (EMA) filtering prevents jumpy zoom behavior.
- **Heads-Up Display (HUD)**: Modern overlay displaying gesture status, zoom percentage bar, FPS, 3s countdown ring, and photo popup notifications.
- **Keyboard Fallback Controls**: Shortcuts (`Q` to Quit, `R` to Reset Zoom, `S` for Instant Save).

---

## 📁 Project Structure

```
hand_gesture_camera_controller/
├── main.py                      # Modular Application Entry Point (with 3s Timer)
├── hand_detector.py             # MediaPipe Hands Wrapper & Visualizer
├── gesture_detector.py          # Scale Normalization & Gesture Classifier
├── camera_controller.py         # Digital Zoom Cropping, 3s Timer & Photo Saver
├── utils.py                     # HUD Visual Overlay & 3s Countdown Renderer
├── hand_gesture_controller.py   # Starter Single-File Script (Beginner Edition)
├── test_gesture_logic.py        # Non-Interactive Verification Test Suite
├── captured_photos/             # Saved Photographs (photo_001.jpg, etc.)
└── README.md                    # Documentation & Portfolio Guide
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+ (Tested on Python 3.14)
- A working webcam

### Step 1: Install Dependencies
Open your terminal / command prompt and run:

```bash
pip install opencv-python mediapipe numpy
```

---

## 🚀 How to Run

### Option 1: Starter Single-File Edition (Beginners)
```bash
python hand_gesture_controller.py
```

### Option 2: Modular Architecture (Recommended for Portfolio)
```bash
python main.py
```

### Option 3: Run Verification Test Suite (No Webcam Required)
```bash
python test_gesture_logic.py
```

---

## 🧠 Gesture Logic & Mathematics

### 1. Distance Normalization Math
$$\text{Palm Scale } D_{\text{ref}} = \sqrt{(x_9 - x_0)^2 + (y_9 - y_0)^2}$$

$$\text{Normalized Pinch Distance } d_{\text{norm}} = \frac{\sqrt{(x_8 - x_4)^2 + (y_8 - y_4)^2}}{D_{\text{ref}}}$$

### 2. Gesture Priority & 3-Second Timer Workflow
1. **Closed Fist (Priority 1 - Photo Capture)**:
   - When fingertips are folded toward palm ($< 0.85 \times D_{\text{ref}}$), a fist is recognized.
   - Initiates **3-Second Delay Timer** (`GET READY! TAKING PHOTO IN: 3.. 2.. 1`).
   - At $t = 3.0\text{s}$, frame is captured and saved to `captured_photos/photo_XXX.jpg`.
   - Activates **3-Second Cooldown** before the next photo sequence can be triggered.
2. **Zoom IN (Priority 2)**: $d_{\text{norm}} > 0.55$ (Fingers spreading apart).
3. **Zoom OUT (Priority 3)**: $d_{\text{norm}} < 0.35$ (Fingers pinching together).
4. **Neutral / Dead-Zone**: $0.35 \le d_{\text{norm}} \le 0.55$ (Holds constant zoom).

---

## 🎮 Controls

| Action | Hand Gesture | Keyboard Key |
| :--- | :--- | :--- |
| **Zoom IN** | Thumb & Index tip apart | — |
| **Zoom OUT** | Thumb & Index tip close | — |
| **3-Sec Photo Capture** | Hold Closed Fist | — |
| **Instant Save Photo** | — | `S` |
| **Reset Zoom (100%)**| — | `R` |
| **Quit Application** | — | `Q` or `ESC` |

---

## 📄 License
MIT License. Free for educational and personal portfolio projects.
