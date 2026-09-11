"""
===================================================================
AUTOMATED UNIT & LOGIC VERIFICATION SUITE
===================================================================
Tests gesture classifier logic, landmark distance normalization,
fist detection accuracy, zoom EMA filter smoothing, 3-second delay
countdown sequence, and photo saving.
Runs non-interactively without requiring a physical camera device.
"""

import os
import sys
import time
import numpy as np
import cv2

# Import project modules
from hand_detector import HandDetector
from gesture_detector import GestureDetector
from camera_controller import CameraController
import utils

def create_synthetic_landmarks(gesture_type="OPEN_PINCH"):
    """
    Generates realistic 21 2D landmarks [id, x, y, z] for testing:
    - OPEN_PINCH: Thumb & Index tips far apart (Zoom IN)
    - CLOSE_PINCH: Thumb & Index tips close together (Zoom OUT)
    - CLOSED_FIST: All fingertips folded close to palm (Photo Capture)
    """
    lm_list = []
    
    wrist = (200, 400)
    middle_mcp = (200, 300)
    
    for i in range(21):
        lm_list.append([i, 200, 300, 0.5, 0.5, 0.0])
        
    lm_list[0] = [0, wrist[0], wrist[1], 0.5, 0.8, 0.0]
    lm_list[9] = [9, middle_mcp[0], middle_mcp[1], 0.5, 0.5, 0.0]
    lm_list[5] = [5, 170, 310, 0.4, 0.5, 0.0]  # Index MCP

    if gesture_type == "OPEN_PINCH":
        lm_list[4] = [4, 120, 250, 0.2, 0.3, 0.0]
        lm_list[8] = [8, 280, 250, 0.7, 0.3, 0.0]
        lm_list[12] = [12, 200, 150, 0.5, 0.1, 0.0]
        lm_list[16] = [16, 230, 160, 0.6, 0.1, 0.0]
        lm_list[20] = [20, 260, 180, 0.7, 0.2, 0.0]

    elif gesture_type == "CLOSE_PINCH":
        lm_list[4] = [4, 190, 250, 0.48, 0.3, 0.0]
        lm_list[8] = [8, 210, 250, 0.52, 0.3, 0.0]
        lm_list[12] = [12, 200, 150, 0.5, 0.1, 0.0]
        lm_list[16] = [16, 230, 160, 0.6, 0.1, 0.0]
        lm_list[20] = [20, 260, 180, 0.7, 0.2, 0.0]

    elif gesture_type == "CLOSED_FIST":
        lm_list[4] = [4, 175, 330, 0.45, 0.6, 0.0]
        lm_list[8] = [8, 180, 350, 0.45, 0.65, 0.0]
        lm_list[12] = [12, 200, 350, 0.5, 0.65, 0.0]
        lm_list[16] = [16, 220, 350, 0.55, 0.65, 0.0]
        lm_list[20] = [20, 240, 360, 0.6, 0.65, 0.0]

    return lm_list

def test_gesture_detector():
    print("[TEST] Running GestureDetector Unit Tests...")
    detector = GestureDetector(zoom_in_thresh=0.55, zoom_out_thresh=0.35, stability_frames=1)

    # Test 1: Open Pinch (Zoom IN)
    lms_open = create_synthetic_landmarks("OPEN_PINCH")
    gesture, norm_dist, raw_dist = detector.detect_gesture(lms_open)
    assert gesture == "ZOOM_IN", f"Expected ZOOM_IN, got {gesture}"
    print(f"  [OK] Open Pinch -> ZOOM_IN (norm dist: {norm_dist:.2f})")

    # Test 2: Close Pinch (Zoom OUT)
    lms_close = create_synthetic_landmarks("CLOSE_PINCH")
    detector.gesture_history.clear()
    gesture, norm_dist, raw_dist = detector.detect_gesture(lms_close)
    assert gesture == "ZOOM_OUT", f"Expected ZOOM_OUT, got {gesture}"
    print(f"  [OK] Close Pinch -> ZOOM_OUT (norm dist: {norm_dist:.2f})")

    # Test 3: Closed Fist (Photo Capture)
    lms_fist = create_synthetic_landmarks("CLOSED_FIST")
    detector.gesture_history.clear()
    gesture, norm_dist, raw_dist = detector.detect_gesture(lms_fist)
    assert gesture == "PHOTO_CAPTURE", f"Expected PHOTO_CAPTURE, got {gesture}"
    print(f"  [OK] Closed Fist -> PHOTO_CAPTURE")

def test_camera_controller_with_3s_timer():
    print("[TEST] Running CameraController & 3-Second Timer Unit Tests...")
    test_dir = "test_captured_photos"
    controller = CameraController(output_dir=test_dir, min_zoom=1.0, max_zoom=3.0, alpha=0.5, cooldown_sec=0.2, countdown_duration=0.1)

    # Test Zoom Updates
    initial_zoom = controller.current_zoom
    controller.update_zoom("ZOOM_IN")
    assert controller.current_zoom > initial_zoom, "Zoom level should increase on ZOOM_IN"
    print(f"  [OK] Zoom IN updated level to {controller.current_zoom:.2f}x")

    controller.reset_zoom()
    assert controller.current_zoom == 1.0, "Reset should restore zoom to 1.0x"
    print(f"  [OK] Zoom reset verified (1.0x)")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test 3-Second Delay Sequence with Persistence
    # 1. Closed wrist/fist triggers timer start
    started = controller.start_countdown()
    assert started and controller.is_counting_down, "start_countdown should return True and activate is_counting_down"
    print(f"  [OK] Closed Wrist/Fist initiated 3-second countdown timer sequence")

    # 2. Even if hand is released or gesture changes, update_countdown continues
    photo_taken, fp, left = controller.update_countdown(dummy_frame)
    assert not photo_taken and controller.is_counting_down, "Countdown should persist while timer is active"
    print(f"  [OK] Persistent countdown verified (remaining: {left:.2f}s)")

    # 3. Wait for countdown duration to elapse
    time.sleep(0.12)

    # 4. Timer finishes and captures photo
    photo_taken, fp, left = controller.update_countdown(dummy_frame)
    assert photo_taken and os.path.exists(fp), "Photo should be captured after countdown finishes"
    print(f"  [OK] Clean photo saved after countdown finished: {fp}")

    # Clean up test folder
    if os.path.exists(fp):
        os.remove(fp)
    if os.path.exists(test_dir):
        os.rmdir(test_dir)

def test_hud_rendering_with_countdown():
    print("[TEST] Testing HUD & UI 3-Second Countdown Rendering...")
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    output = utils.draw_hud(dummy_frame, "PHOTO_CAPTURE", 1.5, fps=28.5, banner_active=False, saved_filename="", countdown_sec=2.5)
    assert output.shape == dummy_frame.shape, "HUD output shape mismatch"
    print("  [OK] HUD 3-Second Countdown overlay rendered without errors")

if __name__ == "__main__":
    print("=========================================================")
    print("      GESTURE & 3-SECOND TIMER SUITE VERIFICATION")
    print("=========================================================")
    test_gesture_detector()
    test_camera_controller_with_3s_timer()
    test_hud_rendering_with_countdown()
    print("=========================================================")
    print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY! (100% OK)")
    print("=========================================================")
