import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    """
    Dual-compatible MediaPipe Hand Detector:
    Supports legacy `mp.solutions.hands` (MediaPipe <= 0.10.14) AND
    modern `mediapipe.tasks.python.vision.HandLandmarker` (MediaPipe >= 0.10.15 / 1.0.0+ / Python 3.14).
    """
    def __init__(self, mode=False, max_hands=1, model_complexity=1, detection_con=0.7, track_con=0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_con = detection_con
        self.track_con = track_con
        self.results = None

        # Check if legacy mp.solutions interface exists
        self.use_legacy_solutions = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')

        if self.use_legacy_solutions:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.mode,
                max_num_hands=self.max_hands,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.detection_con,
                min_tracking_confidence=self.track_con
            )
            self.mp_draw = mp.solutions.drawing_utils
        else:
            # Modern MediaPipe Tasks API for Python 3.14 / MediaPipe 1.0+
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            task_filename = "hand_landmarker.task"
            task_path = os.path.join(os.path.dirname(__file__), task_filename)

            if not os.path.exists(task_path):
                print("[INFO] Downloading MediaPipe Hand Landmarker model file...")
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, task_path)
                print(f"[INFO] Task file saved to {task_path}")

            base_options = python.BaseOptions(model_asset_path=task_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.detection_con,
                min_hand_presence_confidence=self.track_con,
                min_tracking_confidence=self.track_con
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

    def find_hands(self, img, draw=True):
        """
        Processes image frame to detect hands and optionally draws landmarks.
        """
        if self.use_legacy_solutions:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(img_rgb)
            if self.results.multi_hand_landmarks and draw:
                for hand_lms in self.results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=2)
                    )
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            self.results = self.detector.detect(mp_image)
            if self.results.hand_landmarks and draw:
                for hand_lms in self.results.hand_landmarks:
                    self.draw_custom_landmarks(img, hand_lms)
        return img

    def draw_custom_landmarks(self, img, landmarks):
        """
        Draws landmarks and connecting skeletal lines for MediaPipe Tasks API.
        """
        h, w, _ = img.shape
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (0, 17)                                # Palm Base
        ]
        # Draw skeletal bones
        for start_idx, end_idx in connections:
            p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(img, p1, p2, (255, 200, 0), 2, cv2.LINE_AA)

        # Draw joint nodes
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 4, (0, 255, 128), -1, cv2.LINE_AA)

    def get_landmark_positions(self, img, hand_no=0):
        """
        Extracts landmark coordinates for specified hand.
        Returns a list of [id, pixel_x, pixel_y, norm_x, norm_y, norm_z]
        """
        lm_list = []
        if self.use_legacy_solutions:
            if self.results and self.results.multi_hand_landmarks:
                if hand_no < len(self.results.multi_hand_landmarks):
                    hand = self.results.multi_hand_landmarks[hand_no]
                    h, w, _ = img.shape
                    for lm_id, lm in enumerate(hand.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([lm_id, cx, cy, lm.x, lm.y, lm.z])
        else:
            if self.results and self.results.hand_landmarks:
                if hand_no < len(self.results.hand_landmarks):
                    hand = self.results.hand_landmarks[hand_no]
                    h, w, _ = img.shape
                    for lm_id, lm in enumerate(hand):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([lm_id, cx, cy, lm.x, lm.y, lm.z])
        return lm_list

    def get_bounding_box(self, img, lm_list):
        """
        Computes bounding box around detected hand for visual highlighting.
        """
        if not lm_list:
            return None
        x_coords = [lm[1] for lm in lm_list]
        y_coords = [lm[2] for lm in lm_list]
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        padding = 20
        h, w, _ = img.shape
        xmin = max(0, xmin - padding)
        ymin = max(0, ymin - padding)
        xmax = min(w, xmax + padding)
        ymax = min(h, ymax + padding)
        return (xmin, ymin, xmax, ymax)
