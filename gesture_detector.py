import math
from collections import deque

class GestureDetector:
    """
    Analyzes MediaPipe hand landmarks to classify gestures:
    1. CLOSED_FIST (Photo Capture - Priority 1)
    2. ZOOM_IN (Pinch open - Priority 2)
    3. ZOOM_OUT (Pinch close - Priority 3)
    4. NONE (Default state)
    
    Includes scale normalization and multi-frame stability filtering.
    """
    def __init__(self, zoom_in_thresh=0.55, zoom_out_thresh=0.35, stability_frames=3):
        self.zoom_in_thresh = zoom_in_thresh
        self.zoom_out_thresh = zoom_out_thresh
        self.stability_frames = stability_frames
        
        # History queue for multi-frame gesture verification
        self.gesture_history = deque(maxlen=stability_frames)
        
        # Dead-zone parameters
        self.last_pinch_distance = None
        self.deadzone_margin = 0.03

    @staticmethod
    def euclidean_distance(pt1, pt2):
        """Calculates 2D Euclidean distance between two (x, y) landmark points."""
        return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

    def get_palm_scale(self, lm_list):
        """
        Calculates reference palm scale distance (Wrist 0 to Middle MCP 9).
        This scale normalizes hand size regardless of camera distance.
        """
        if len(lm_list) <= 9:
            return 1.0
        wrist = (lm_list[0][1], lm_list[0][2])
        middle_mcp = (lm_list[9][1], lm_list[9][2])
        scale = self.euclidean_distance(wrist, middle_mcp)
        return scale if scale > 0 else 1.0

    def is_closed_fist(self, lm_list, palm_scale):
        """
        Detects if the hand forms a closed fist.
        Conditions:
        - Fingertips (8, 12, 16, 20) folded close to Wrist (0) / Palm MCPs.
        - Distance from tip to wrist < 0.85 * palm_scale for main fingers.
        """
        if len(lm_list) < 21:
            return False

        wrist = (lm_list[0][1], lm_list[0][2])
        
        # Fingertips: Index (8), Middle (12), Ring (16), Pinky (20)
        fingertip_ids = [8, 12, 16, 20]
        folded_count = 0

        for tip_id in fingertip_ids:
            tip = (lm_list[tip_id][1], lm_list[tip_id][2])
            dist_to_wrist = self.euclidean_distance(tip, wrist)
            # Normalized tip-to-wrist ratio
            ratio = dist_to_wrist / palm_scale
            if ratio < 0.85:
                folded_count += 1

        # Thumb tip (4) distance to Index MCP (5)
        thumb_tip = (lm_list[4][1], lm_list[4][2])
        index_mcp = (lm_list[5][1], lm_list[5][2])
        thumb_folded = (self.euclidean_distance(thumb_tip, index_mcp) / palm_scale) < 0.70

        # At least 3 fingers folded plus thumb folded = closed fist
        return folded_count >= 3 and thumb_folded

    def detect_gesture(self, lm_list):
        """
        Main gesture evaluation with strict priority:
        1. Closed Fist -> PHOTO_CAPTURE
        2. Pinch Separation -> ZOOM_IN / ZOOM_OUT
        3. Stable output across consecutive frames
        """
        if not lm_list or len(lm_list) < 21:
            self.gesture_history.clear()
            return "NONE", 0.0, 0.0

        palm_scale = self.get_palm_scale(lm_list)
        
        # Landmark coordinates
        thumb_tip = (lm_list[4][1], lm_list[4][2])
        index_tip = (lm_list[8][1], lm_list[8][2])

        # Raw & normalized pinch distance (Thumb tip to Index tip)
        raw_pinch_dist = self.euclidean_distance(thumb_tip, index_tip)
        norm_pinch_dist = raw_pinch_dist / palm_scale

        # Priority 1: Closed Fist Check
        if self.is_closed_fist(lm_list, palm_scale):
            raw_gesture = "PHOTO_CAPTURE"
        # Priority 2 & 3: Zoom In / Zoom Out with Hysteresis/Dead-zone
        else:
            if norm_pinch_dist > self.zoom_in_thresh:
                raw_gesture = "ZOOM_IN"
            elif norm_pinch_dist < self.zoom_out_thresh:
                raw_gesture = "ZOOM_OUT"
            else:
                raw_gesture = "NONE"

        # Apply multi-frame consensus filtering for rock-solid stability
        self.gesture_history.append(raw_gesture)
        
        # If all frames in history agree, adopt the gesture
        if len(self.gesture_history) == self.stability_frames:
            if len(set(self.gesture_history)) == 1:
                confirmed_gesture = self.gesture_history[0]
            else:
                confirmed_gesture = "NONE"
        else:
            confirmed_gesture = "NONE"

        return confirmed_gesture, norm_pinch_dist, raw_pinch_dist
