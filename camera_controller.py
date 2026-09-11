import os
import time
import cv2
import numpy as np

class CameraController:
    """
    Manages camera digital zooming, frame cropping, image saving,
    3-second capture countdown timer, and photo capture cooldown logic.
    """
    def __init__(self, output_dir="captured_photos", min_zoom=1.0, max_zoom=3.0, alpha=0.15, cooldown_sec=3.0, countdown_duration=3.0):
        self.output_dir = output_dir
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.alpha = alpha  # Exponential smoothing factor
        self.cooldown_sec = cooldown_sec
        self.countdown_duration = countdown_duration  # 3-second delay timer

        self.current_zoom = 1.0
        self.target_zoom = 1.0
        self.last_photo_time = 0.0

        # Countdown State
        self.countdown_start_time = None
        self.is_counting_down = False

        # UI Banner state
        self.photo_captured_banner_time = 0.0
        self.last_saved_filename = ""

        # Ensure photo directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def update_zoom(self, gesture, norm_pinch_dist=None, step=0.03):
        """
        Updates target zoom level based on detected gesture or pinch distance,
        and applies exponential smoothing for natural, fluid motion.
        """
        if gesture == "ZOOM_IN":
            if norm_pinch_dist is not None:
                desired = 1.0 + (norm_pinch_dist - 0.50) * 3.5
                self.target_zoom = max(self.target_zoom, desired)
            else:
                self.target_zoom += step

        elif gesture == "ZOOM_OUT":
            if norm_pinch_dist is not None:
                desired = 1.0 + (norm_pinch_dist - 0.30) * 3.0
                self.target_zoom = min(self.target_zoom, desired)
            else:
                self.target_zoom -= step

        # Clamp target zoom within valid boundaries [1.0, 3.0]
        self.target_zoom = float(np.clip(self.target_zoom, self.min_zoom, self.max_zoom))

        # Exponential Moving Average (EMA) smoothing for stable transitions
        self.current_zoom = self.alpha * self.target_zoom + (1.0 - self.alpha) * self.current_zoom
        self.current_zoom = float(np.clip(self.current_zoom, self.min_zoom, self.max_zoom))

        return self.current_zoom

    def reset_zoom(self):
        """Resets zoom back to 100% (1.0x)."""
        self.target_zoom = 1.0
        self.current_zoom = 1.0

    def apply_zoom(self, frame):
        """
        Crops center of frame based on current zoom level
        and resizes it back to original resolution.
        """
        if self.current_zoom <= 1.001:
            return frame.copy()

        h, w, c = frame.shape
        crop_h = int(h / self.current_zoom)
        crop_w = int(w / self.current_zoom)

        start_y = max(0, (h - crop_h) // 2)
        start_x = max(0, (w - crop_w) // 2)
        end_y = min(h, start_y + crop_h)
        end_x = min(w, start_x + crop_w)

        cropped_region = frame[start_y:end_y, start_x:end_x]
        zoomed_frame = cv2.resize(cropped_region, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed_frame

    def can_start_photo_sequence(self):
        """Checks if cooldown period (3.0s) has elapsed since last photo."""
        now = time.time()
        return (now - self.last_photo_time) >= self.cooldown_sec and not self.is_counting_down

    def start_countdown(self):
        """
        Initiates 3-second countdown sequence when closed wrist/fist gesture is detected,
        provided cooldown has elapsed and countdown is not already active.
        """
        now = time.time()
        if not self.is_counting_down and (now - self.last_photo_time) >= self.cooldown_sec:
            self.is_counting_down = True
            self.countdown_start_time = now
            return True
        return False

    def update_countdown(self, clean_zoomed_frame):
        """
        Unconditionally updates active 3-second countdown timer.
        When countdown completes, saves clean_zoomed_frame as photo.
        Returns: (photo_captured: bool, filepath: str, countdown_sec_left: float)
        """
        if not self.is_counting_down:
            return False, "", 0.0

        now = time.time()
        elapsed = now - self.countdown_start_time
        remaining = self.countdown_duration - elapsed

        if remaining <= 0.0:
            # Countdown finished! Take clean photo now!
            self.is_counting_down = False
            self.countdown_start_time = None
            success, filepath = self.capture_photo(clean_zoomed_frame)
            return True, filepath, 0.0
        else:
            return False, "", remaining

    def trigger_fist_gesture(self, clean_zoomed_frame):
        """
        Convenience function called when closed fist gesture is detected.
        Initiates countdown if ready and updates timer state.
        Returns: (photo_captured: bool, filepath: str, countdown_sec_left: float)
        """
        self.start_countdown()
        return self.update_countdown(clean_zoomed_frame)

    def cancel_countdown_if_broken(self):
        """
        No-op: Timer runs continuously for 3 seconds once triggered,
        regardless of whether wrist opens or hand leaves frame.
        """
        pass

    def capture_photo(self, zoomed_frame):
        """
        Saves current zoomed frame as a JPG photo with auto-incrementing filename.
        """
        idx = 1
        while True:
            filename = f"photo_{idx:03d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            if not os.path.exists(filepath):
                break
            idx += 1

        cv2.imwrite(filepath, zoomed_frame)
        now = time.time()
        self.last_photo_time = now
        self.photo_captured_banner_time = now
        self.last_saved_filename = filename

        return True, filepath

    def is_banner_active(self, banner_duration=1.5):
        """Returns True if 'Photo Captured!' UI notification banner should be displayed."""
        return (time.time() - self.photo_captured_banner_time) < banner_duration
