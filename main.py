import sys
import time
import cv2

from hand_detector import HandDetector
from gesture_detector import GestureDetector
from camera_controller import CameraController
import utils

def open_webcam():
    """
    Attempts to initialize webcam using indices 0 to 3 with multiple OpenCV backends.
    Tries DirectShow (CAP_DSHOW), Media Foundation (CAP_MSMF), and default API.
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith('win') else [cv2.CAP_ANY]
    backend_names = {cv2.CAP_DSHOW: "DirectShow (CAP_DSHOW)", cv2.CAP_MSMF: "Media Foundation (CAP_MSMF)", cv2.CAP_ANY: "Default (CAP_ANY)"}

    for cam_idx in range(4):
        for backend in backends:
            cap = cv2.VideoCapture(cam_idx, backend)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    b_name = backend_names.get(backend, "Default")
                    print(f"[INFO] Webcam successfully opened on camera index {cam_idx} using {b_name}.")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    return cap
                cap.release()
    return None

def main():
    print("=" * 60)
    print("  HAND GESTURE CAMERA CONTROLLER - 3-SECOND TIMER EDITION")
    print("=" * 60)
    print("Controls:")
    print("  - Pinch fingers apart: Zoom IN")
    print("  - Pinch fingers close: Zoom OUT")
    print("  - Closed Fist: 3-Second Timer -> Capture Photo")
    print("  - [Q]: Quit")
    print("  - [R]: Reset Zoom (100%)")
    print("  - [S]: Instant Save Photo")
    print("-" * 60)

    # Initialize video capture
    cap = open_webcam()
    if cap is None:
        print("[ERROR] Could not access any webcam.")
        print("\n[DIAGNOSTICS] Windows Camera Driver Access Checklist:")
        print("  1. Windows Privacy Settings: Go to Settings > Privacy & security > Camera.")
        print("     Ensure 'Camera access' and 'Let desktop apps access your camera' are enabled.")
        print("  2. App Locks: Close other apps that may be holding exclusive lock on the camera driver")
        print("     (e.g., Zoom, Microsoft Teams, Discord, Chrome, or Windows Camera App).")
        print("  3. Device Manager: Open Device Manager > Cameras / Imaging devices.")
        print("     Verify driver is enabled and functioning.")
        print("  4. Hardware Privacy Switch: Check if your laptop has a physical shutter or Fn toggle key.")
        print("  5. Antivirus Protection: Verify your antivirus/firewall isn't blocking python.exe.")
        sys.exit(1)

    # Initialize project modules (cooldown = 3.0s, countdown = 3.0s)
    hand_detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.7)
    gesture_detector = GestureDetector(zoom_in_thresh=0.55, zoom_out_thresh=0.35, stability_frames=3)
    camera_controller = CameraController(output_dir="captured_photos", min_zoom=1.0, max_zoom=3.0, alpha=0.15, cooldown_sec=3.0, countdown_duration=3.0)

    prev_time = time.time()
    fps = 0.0

    window_name = "Hand Gesture Camera Controller"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Mirror webcam frame for intuitive interaction
            frame = cv2.flip(frame, 1)

            # Keep a pristine clean frame copy for photo capture (NO gesture lines/HUD)
            clean_frame = frame.copy()
            display_frame = frame.copy()

            # 1. Hand Detection & Landmark Extraction on display frame
            processed_display_frame = hand_detector.find_hands(display_frame, draw=True)
            lm_list = hand_detector.get_landmark_positions(processed_display_frame)

            # 2. Gesture Recognition
            gesture, norm_pinch_dist, raw_dist = gesture_detector.detect_gesture(lm_list)

            # Draw visual pinch line on display frame if in pinch mode
            if lm_list and gesture != "PHOTO_CAPTURE":
                utils.draw_pinch_line(processed_display_frame, lm_list)

            # 3. Update Zoom Level
            current_zoom = camera_controller.update_zoom(gesture, norm_pinch_dist if lm_list else None)

            # 4. Apply Digital Camera Zoom (Crop & Scale)
            clean_zoomed_frame = camera_controller.apply_zoom(clean_frame)
            zoomed_display_frame = camera_controller.apply_zoom(processed_display_frame)

            # 5. Handle 3-Second Closed Wrist/Fist Photo Capture Sequence
            if gesture == "PHOTO_CAPTURE":
                camera_controller.start_countdown()

            photo_captured, filepath, countdown_left = camera_controller.update_countdown(clean_zoomed_frame)
            if photo_captured:
                print(f"[ACTION] 3-Second Timer Complete! Clean photo saved: {filepath}")

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
            prev_time = curr_time

            # 6. Render Modern HUD Overlay with 3s Countdown Display on preview frame
            banner_active = camera_controller.is_banner_active()
            saved_filename = camera_controller.last_saved_filename
            output_frame = utils.draw_hud(zoomed_display_frame, gesture, current_zoom, fps, banner_active, saved_filename, countdown_sec=countdown_left)

            # Render frame to OpenCV window
            cv2.imshow(window_name, output_frame)

            # 7. Keyboard Shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 'Q' or ESC
                print("[INFO] Exiting application...")
                break
            elif key in (ord('r'), ord('R')):
                camera_controller.reset_zoom()
                print("[ACTION] Zoom level reset to 100%.")
            elif key in (ord('s'), ord('S')):
                success, filepath = camera_controller.capture_photo(clean_zoomed_frame)
                if success:
                    print(f"[ACTION] Instant Photo Saved: {filepath}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam and resources released successfully.")

if __name__ == "__main__":
    main()
