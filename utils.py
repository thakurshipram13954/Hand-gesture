import math
import cv2
import numpy as np

def draw_hud(frame, gesture, zoom_level, fps=0, banner_active=False, saved_filename="", countdown_sec=0.0):
    """
    Renders HUD overlay:
    - Top Glassmorphism Status Bar (Gesture badge, Zoom %, FPS counter)
    - 3-Second Capture Countdown Overlay (Big prominent countdown ring: 3..2..1)
    - Photo Capture Banner Notification
    - Keyboard Controls Guide Footer
    """
    h, w, _ = frame.shape

    # 1. Top HUD Glassmorphism Background Panel
    panel_height = 85
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (w - 15, panel_height), (20, 20, 25), -1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (15, 15), (w - 15, panel_height), (0, 220, 255), 2)

    # 2. Gesture Badge & Styling
    gesture_colors = {
        "ZOOM_IN": (0, 255, 128),     # Vibrant Green
        "ZOOM_OUT": (0, 165, 255),    # Amber
        "PHOTO_CAPTURE": (0, 0, 255), # Red Accent
        "NONE": (180, 180, 180)       # Gray
    }
    color = gesture_colors.get(gesture, (180, 180, 180))

    cv2.putText(frame, "GESTURE:", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    display_gesture = "CLOSED FIST (PHOTO)" if gesture == "PHOTO_CAPTURE" else gesture.replace("_", " ")
    cv2.putText(frame, display_gesture, (120, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    # 3. Zoom Level Meter
    zoom_pct = int(zoom_level * 100)
    zoom_str = f"ZOOM: {zoom_pct}%"
    cv2.putText(frame, zoom_str, (390, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw Zoom Progress Bar
    bar_x1, bar_y1 = 390, 58
    bar_w, bar_h = 160, 12
    fill_ratio = max(0.0, min(1.0, (zoom_level - 1.0) / 2.0))
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + int(bar_w * fill_ratio), bar_y1 + bar_h), (0, 220, 255), -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (255, 255, 255), 1)

    # 4. FPS Counter
    if fps > 0:
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # 5. Prominent 3-Second Photo Countdown Overlay (Center Screen)
    if countdown_sec > 0.0:
        sec_num = int(math.ceil(countdown_sec))
        center_x, center_y = w // 2, h // 2

        # Outer semi-transparent dark circle box
        cd_overlay = frame.copy()
        cv2.circle(cd_overlay, (center_x, center_y), 90, (0, 0, 0), -1)
        cv2.addWeighted(cd_overlay, 0.6, frame, 0.4, 0, frame)

        # Pulsing circle border
        cv2.circle(frame, (center_x, center_y), 90, (0, 220, 255), 4, cv2.LINE_AA)

        # "GET READY!" text above number
        cv2.putText(frame, "HOLD FIST - TAKING PHOTO IN", (center_x - 175, center_y - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        # Big Countdown Number (3, 2, 1)
        num_str = str(sec_num)
        text_size, _ = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_SIMPLEX, 3.2, 7)
        tx = center_x - text_size[0] // 2
        ty = center_y + text_size[1] // 2
        cv2.putText(frame, num_str, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0, 255, 128), 7, cv2.LINE_AA)

    # 6. Photo Capture Banner Popup (Center Screen Notification)
    elif banner_active:
        banner_w, banner_h = 440, 65
        bx1 = (w - banner_w) // 2
        by1 = (h - banner_h) // 2
        
        banner_overlay = frame.copy()
        cv2.rectangle(banner_overlay, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 128, 0), -1)
        cv2.addWeighted(banner_overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 255, 0), 3)

        cv2.putText(frame, "PHOTO CAPTURED!", (bx1 + 90, by1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        if saved_filename:
            cv2.putText(frame, f"Saved: {saved_filename}", (bx1 + 120, by1 + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 255, 220), 1, cv2.LINE_AA)

    # 7. Bottom Key Map Footer
    footer_text = "[Q] Quit  |  [R] Reset Zoom (100%)  |  [S] Save Photo (Instant)"
    cv2.putText(frame, footer_text, (25, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return frame

def draw_pinch_line(frame, lm_list):
    """
    Draws visual connecting line between Thumb Tip (4) and Index Tip (8)
    with distance indicator dot.
    """
    if len(lm_list) >= 9:
        t_x, t_y = lm_list[4][1], lm_list[4][2]
        i_x, i_y = lm_list[8][1], lm_list[8][2]
        m_x, m_y = (t_x + i_x) // 2, (t_y + i_y) // 2

        cv2.line(frame, (t_x, t_y), (i_x, i_y), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (t_x, t_y), 6, (0, 255, 255), -1)
        cv2.circle(frame, (i_x, i_y), 6, (0, 255, 255), -1)
        cv2.circle(frame, (m_x, m_y), 5, (255, 255, 255), -1)
