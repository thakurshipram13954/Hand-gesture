"""
===================================================================
HAND GESTURE CAMERA CONTROLLER - STARTER SINGLE-FILE EDITION
===================================================================
A beginner-friendly complete Python script combining OpenCV and MediaPipe
to control webcam zoom (IN/OUT) and capture photos via hand gestures
with a 3-Second Delay Timer & 3-Second Cooldown.

Compatible with both legacy MediaPipe (mp.solutions) and MediaPipe Tasks API.

Dependencies:
    pip install opencv-python mediapipe numpy

Run Command:
    python hand_gesture_controller.py
"""

import os
import sys
import math
import time
import urllib.request
from collections import deque
import cv2
import numpy as np
import mediapipe as mp

# =================================================================
# 1. SETUP & CONFIGURATION
# =================================================================
OUTPUT_DIR = "captured_photos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Zoom Parameters
MIN_ZOOM = 1.0       # 100% Zoom (Original image size)
MAX_ZOOM = 3.0       # 300% Zoom
ALPHA = 0.15          # Smooth zoom exponential filter weight
ZOOM_IN_THRESH = 0.55   # Normalized pinch distance threshold for Zoom IN
ZOOM_OUT_THRESH = 0.35  # Normalized pinch distance threshold for Zoom OUT

# Timer Settings
COOLDOWN_SEC = 3.0       # 3-Second time gap between photos
COUNTDOWN_DURATION = 3.0 # 3-Second countdown before clicking photo

# MediaPipe Initialization (Dual API Compatibility)
USE_LEGACY = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')

if USE_LEGACY:
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
else:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    task_filename = "hand_landmarker.task"
    task_path = os.path.join(os.path.dirname(__file__), task_filename)
    if not os.path.exists(task_path):
        print("[INFO] Downloading MediaPipe Hand Landmarker model file...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, task_path)

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    tasks_detector = vision.HandLandmarker.create_from_options(options)

# State Variables
current_zoom = 1.0
target_zoom = 1.0
last_photo_time = 0.0
photo_banner_time = 0.0
last_saved_file = ""

# Countdown State
is_counting_down = False
countdown_start_time = None

# Multi-frame gesture consensus filter
gesture_history = deque(maxlen=3)

# =================================================================
# 2. HELPER ALGORITHMS
# =================================================================
def euclidean_distance(pt1, pt2):
    """Calculates 2D Euclidean distance between two points."""
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

def get_palm_scale(lm_list):
    """Calculates reference palm scale distance (Wrist 0 to Middle MCP 9)."""
    if len(lm_list) <= 9:
        return 1.0
    w_x, w_y = lm_list[0][1], lm_list[0][2]
    m_x, m_y = lm_list[9][1], lm_list[9][2]
    scale = euclidean_distance((w_x, w_y), (m_x, m_y))
    return scale if scale > 0 else 1.0

def is_closed_fist(lm_list, palm_scale):
    """Checks if all fingertips are folded toward the palm base."""
    if len(lm_list) < 21:
        return False
    wrist = (lm_list[0][1], lm_list[0][2])
    fingertip_ids = [8, 12, 16, 20]
    folded_count = 0

    for tip_id in fingertip_ids:
        tip = (lm_list[tip_id][1], lm_list[tip_id][2])
        dist_to_wrist = euclidean_distance(tip, wrist)
        if (dist_to_wrist / palm_scale) < 0.85:
            folded_count += 1

    thumb_tip = (lm_list[4][1], lm_list[4][2])
    index_mcp = (lm_list[5][1], lm_list[5][2])
    thumb_folded = (euclidean_distance(thumb_tip, index_mcp) / palm_scale) < 0.70

    return folded_count >= 3 and thumb_folded

def detect_gesture(lm_list):
    """Classifies hand gesture with strict priority."""
    if not lm_list or len(lm_list) < 21:
        gesture_history.clear()
        return "NONE", 0.0

    palm_scale = get_palm_scale(lm_list)
    thumb_tip = (lm_list[4][1], lm_list[4][2])
    index_tip = (lm_list[8][1], lm_list[8][2])

    norm_pinch_dist = euclidean_distance(thumb_tip, index_tip) / palm_scale

    if is_closed_fist(lm_list, palm_scale):
        raw_gesture = "PHOTO_CAPTURE"
    elif norm_pinch_dist > ZOOM_IN_THRESH:
        raw_gesture = "ZOOM_IN"
    elif norm_pinch_dist < ZOOM_OUT_THRESH:
        raw_gesture = "ZOOM_OUT"
    else:
        raw_gesture = "NONE"

    gesture_history.append(raw_gesture)
    if len(gesture_history) == 3 and len(set(gesture_history)) == 1:
        return gesture_history[0], norm_pinch_dist
    return "NONE", norm_pinch_dist

def apply_digital_zoom(frame, zoom):
    """Crops frame center and resizes to original width & height."""
    if zoom <= 1.001:
        return frame.copy()
    h, w, _ = frame.shape
    crop_h = int(h / zoom)
    crop_w = int(w / zoom)

    start_y = max(0, (h - crop_h) // 2)
    start_x = max(0, (w - crop_w) // 2)

    cropped = frame[start_y:start_y + crop_h, start_x:start_x + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def save_photo(frame):
    """Saves photo with auto-incrementing filename."""
    global last_photo_time, photo_banner_time, last_saved_file
    now = time.time()

    idx = 1
    while True:
        fn = f"photo_{idx:03d}.jpg"
        fp = os.path.join(OUTPUT_DIR, fn)
        if not os.path.exists(fp):
            break
        idx += 1

    cv2.imwrite(fp, frame)
    last_photo_time = now
    photo_banner_time = now
    last_saved_file = fn
    return True

def draw_custom_landmarks(img, landmarks):
    h, w, _ = img.shape
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (0, 17)                                # Palm Base
    ]
    for start_idx, end_idx in connections:
        p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(img, p1, p2, (255, 200, 0), 2, cv2.LINE_AA)

    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 4, (0, 255, 128), -1, cv2.LINE_AA)

# =================================================================
# 3. MAIN EXECUTION LOOP
# =================================================================
def open_webcam():
    """Attempts to initialize webcam across camera indices and Windows backends."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith('win') else [cv2.CAP_ANY]
    for cam_idx in range(4):
        for backend in backends:
            cap = cv2.VideoCapture(cam_idx, backend)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    return cap
                cap.release()
    return None

def main():
    global current_zoom, target_zoom, is_counting_down, countdown_start_time

    cap = open_webcam()
    if cap is None:
        print("[ERROR] Could not open webcam.")
        print("\n[DIAGNOSTICS] Windows Camera Driver Access Checklist:")
        print("  1. Windows Privacy Settings: Ensure 'Let desktop apps access your camera' is ON.")
        print("  2. Exclusive App Lock: Close Zoom, Teams, Discord, Chrome, or Windows Camera app.")
        print("  3. Device Manager: Check if Camera driver is enabled in Device Manager.")
        print("  4. Physical Privacy Switch: Verify laptop webcam slider or Fn key isn't blocking.")
        return

    print("Hand Gesture Controller Started (3s Delay Timer). Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Clean frame for saving photos without gesture lines or HUD overlays
        clean_frame = frame.copy()
        display_frame = frame.copy()

        lm_list = []
        if USE_LEGACY:
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(frame_rgb)
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(display_frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                for lm_id, lm in enumerate(hand_lms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([lm_id, cx, cy])
        else:
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = tasks_detector.detect(mp_image)
            if results.hand_landmarks:
                hand_lms = results.hand_landmarks[0]
                draw_custom_landmarks(display_frame, hand_lms)
                for lm_id, lm in enumerate(hand_lms):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([lm_id, cx, cy])

        # Detect Gesture
        gesture, norm_pinch_dist = detect_gesture(lm_list)

        # Update Zoom Level
        if gesture == "ZOOM_IN":
            target_zoom = min(MAX_ZOOM, target_zoom + 0.04)
        elif gesture == "ZOOM_OUT":
            target_zoom = max(MIN_ZOOM, target_zoom - 0.04)

        # Apply Exponential Smoothing to Zoom
        current_zoom = ALPHA * target_zoom + (1.0 - ALPHA) * current_zoom
        current_zoom = max(MIN_ZOOM, min(MAX_ZOOM, current_zoom))

        # Crop and Scale Frame
        clean_zoomed_frame = apply_digital_zoom(clean_frame, current_zoom)
        zoomed_display_frame = apply_digital_zoom(display_frame, current_zoom)

        # Handle 3-Second Closed Wrist/Fist Countdown (Persistent once started)
        now = time.time()
        countdown_left = 0.0

        if gesture == "PHOTO_CAPTURE" and not is_counting_down and (now - last_photo_time) >= COOLDOWN_SEC:
            is_counting_down = True
            countdown_start_time = now

        if is_counting_down:
            elapsed = now - countdown_start_time
            countdown_left = COUNTDOWN_DURATION - elapsed
            if countdown_left <= 0.0:
                is_counting_down = False
                countdown_start_time = None
                countdown_left = 0.0
                save_photo(clean_zoomed_frame)
                print(f"[ACTION] Clean Photo Saved after 3s countdown: {last_saved_file}")

        # Draw UI Overlay on preview frame
        cv2.rectangle(zoomed_display_frame, (15, 15), (w - 15, 80), (30, 30, 30), -1)
        cv2.rectangle(zoomed_display_frame, (15, 15), (w - 15, 80), (0, 220, 255), 2)
        cv2.putText(zoomed_display_frame, f"Gesture: {gesture}", (30, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
        cv2.putText(zoomed_display_frame, f"Zoom: {int(current_zoom * 100)}%", (370, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Render 3-Second Countdown Display
        if countdown_left > 0.0:
            sec_num = int(math.ceil(countdown_left))
            cv2.circle(zoomed_display_frame, (w // 2, h // 2), 80, (0, 0, 0), -1)
            cv2.circle(zoomed_display_frame, (w // 2, h // 2), 80, (0, 220, 255), 3)
            cv2.putText(zoomed_display_frame, "CLOSED WRIST - TAKING PHOTO IN", (w // 2 - 180, h // 2 - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(zoomed_display_frame, str(sec_num), (w // 2 - 25, h // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 255, 128), 6)

        elif (time.time() - photo_banner_time) < 1.5:
            cv2.rectangle(zoomed_display_frame, (w//2 - 200, h//2 - 30), (w//2 + 200, h//2 + 30), (0, 150, 0), -1)
            cv2.putText(zoomed_display_frame, f"PHOTO CAPTURED! ({last_saved_file})", (w//2 - 180, h//2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Hand Gesture Camera Controller", zoomed_display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            target_zoom = 1.0
            current_zoom = 1.0
        elif key in (ord('s'), ord('S')):
            save_photo(clean_zoomed_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
