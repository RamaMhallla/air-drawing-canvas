import cv2
import mediapipe as mp
import numpy as np
import time

# =========================
# Enhanced Design Settings
# =========================
WHITE_BG = False                # Start with black background
INIT_THICKNESS = 5              # Initial brush thickness
ERASER_MULTIPLIER = 3           # Eraser is thicker by this multiplier
PALM_CLEAR_FRAMES = 15          # Frames with open palm to clear canvas
MASK_MOVE_CHECK = True          # Require motion (BG subtraction) to draw

# Thickness bounds + smoothing
THICKNESS_MIN = 2
THICKNESS_MAX = 50
THICKNESS_ALPHA = 0.4           # smoothing factor (0..1), higher = faster response

# UI Colors
PRIMARY_COLOR = (50, 50, 200)   # Top bar color (BGR)
SECONDARY_COLOR = (30, 30, 30)  # Text color on light areas
BTN_TEXT_COLOR = (240, 240, 240)
BTN_ROUNDNESS = 0.3             # Button corner roundness (0-1)

# Slider geometry (left side)
SLIDER_X = 24
SLIDER_W = 40
SLIDER_TOP_MARGIN = 80
SLIDER_BOTTOM_MARGIN = 100

# Grid (guidelines)
GRID_ENABLED = True
GRID_SPACING = 80
GRID_MIN = 20
GRID_MAX = 200
GRID_COLOR = (80, 80, 80)
GRID_THICKNESS = 1

# Try higher preview resolution
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Info panel placement (moved below top bar)
INFO_PANEL_TOP = 70   # start Y to avoid overlapping the top bar

# =========================
# MediaPipe Initialization
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# =========================
# Helper Functions
# =========================
def make_canvas(h, w, white=WHITE_BG):
    return np.full((h, w, 3), 255, dtype=np.uint8) if white else np.zeros((h, w, 3), dtype=np.uint8)

def rounded_rectangle(img, x, y, w, h, color, radius=10, thickness=-1):
    cv2.circle(img, (x + radius, y + radius), radius, color, thickness)
    cv2.circle(img, (x + w - radius, y + radius), radius, color, thickness)
    cv2.circle(img, (x + radius, y + h - radius), radius, color, thickness)
    cv2.circle(img, (x + w - radius, y + h - radius), radius, color, thickness)
    cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, thickness)
    cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, thickness)
    cv2.rectangle(img, (x + radius, y + radius), (x + w - radius, y + h - radius), color, thickness)

def draw_button(img, x, y, w, h, label, color, active=False, radius=10):
    btn_color = (color[0]//2, color[1]//2, color[2]//2) if active else color
    rounded_rectangle(img, x, y, w, h, btn_color, radius)
    cv2.rectangle(img, (x, y + h - 2), (x + w, y + h), (0, 0, 0), 1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(img, label, (x + (w - text_size[0]) // 2, y + (h + text_size[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BTN_TEXT_COLOR, 2)

def point_in_rect(px, py, x, y, w, h):
    return x <= px <= x + w and y <= py <= y + h

def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    up = []
    for tip, pip in zip(tips, pips):
        up.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)
    return up

def map_thickness_to_knob_y(thick, y1, y2):
    return int(np.interp(thick, [THICKNESS_MIN, THICKNESS_MAX], [y2, y1]))

def map_y_to_thickness(y, y1, y2):
    return int(np.interp(y, [y2, y1], [THICKNESS_MIN, THICKNESS_MAX]))

def draw_slider(ui, thickness, x, y1, y2, w):
    track_color = (230, 230, 230)
    border_color = (180, 180, 180)
    cv2.rectangle(ui, (x, y1), (x + w, y2), track_color, -1)
    cv2.rectangle(ui, (x, y1), (x + w, y2), border_color, 2)
    knob_y = map_thickness_to_knob_y(thickness, y1, y2)
    cv2.rectangle(ui, (x+2, knob_y), (x + w - 2, y2 - 2), (210, 210, 210), -1)
    cv2.circle(ui, (x + w // 2, knob_y), 12, (120, 120, 120), -1)
    cv2.circle(ui, (x + w // 2, knob_y), 12, (70, 70, 70), 2)
    cv2.putText(ui, "Thickness", (x - 5, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.putText(ui, str(thickness), (x + w//2 - 14, y1 - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

def draw_grid(ui, x1, y1, x2, y2, spacing, color=GRID_COLOR, thickness=GRID_THICKNESS):
    if not GRID_ENABLED or spacing < 5:
        return
    for x in range(x1 + spacing, x2, spacing):
        cv2.line(ui, (x, y1), (x, y2), color, thickness)
    for y in range(y1 + spacing, y2, spacing):
        cv2.line(ui, (x1, y), (x2, y), color, thickness)

def bgr_to_hex(bgr):
    return f"#{bgr[2]:02X}{bgr[1]:02X}{bgr[0]:02X}"

# =========================
# Camera and Window Setup
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera.")

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read the first frame from the camera.")
frame = cv2.flip(frame, 1)
H, W, _ = frame.shape

SLIDER_Y1 = SLIDER_TOP_MARGIN
SLIDER_Y2 = H - SLIDER_BOTTOM_MARGIN

canvas = make_canvas(H, W, WHITE_BG)
background_color = (255, 255, 255) if WHITE_BG else (0, 0, 0)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
kernel = np.ones((3, 3), np.uint8)

# Buttons
BTN_W, BTN_H = 90, 50
BTN_Y = H - 70
BTN_SPACING = 10
buttons = [
    {"label": "Red",       "x": BTN_SPACING + 0*(BTN_W+BTN_SPACING), "y": BTN_Y, "w": BTN_W, "h": BTN_H, "color": (50, 50, 200), "action": "color"},
    {"label": "Green",     "x": BTN_SPACING + 1*(BTN_W+BTN_SPACING), "y": BTN_Y, "w": BTN_W, "h": BTN_H, "color": (50, 200, 50), "action": "color"},
    {"label": "Blue",      "x": BTN_SPACING + 2*(BTN_W+BTN_SPACING), "y": BTN_Y, "w": BTN_W, "h": BTN_H, "color": (200, 50, 50), "action": "color"},
    {"label": "Smaller",   "x": BTN_SPACING + 3*(BTN_W+BTN_SPACING), "y": BTN_Y, "w": BTN_W, "h": BTN_H, "color": (100, 100, 100), "action": "thinner"},
    {"label": "Bigger",    "x": BTN_SPACING + 4*(BTN_W+BTN_SPACING), "y": BTN_Y, "w": BTN_W, "h": BTN_H, "color": (100, 100, 100), "action": "thicker"},
    {"label": "Eraser",    "x": BTN_SPACING + 5*(BTN_W+BTN_SPACING), "y": BTN_Y, "w": BTN_W+10, "h": BTN_H, "color": (200, 200, 200), "action": "eraser"},
    {"label": "Clear All", "x": BTN_SPACING + 6*(BTN_W+BTN_SPACING)+10, "y": BTN_Y, "w": BTN_W+20, "h": BTN_H, "color": (200, 50, 50), "action": "clear"},
    {"label": "BG",        "x": BTN_SPACING + 7*(BTN_W+BTN_SPACING)+30, "y": BTN_Y, "w": BTN_W-20, "h": BTN_H, "color": (120, 120, 120), "action": "toggle_bg"},
    {"label": "G-",        "x": BTN_SPACING + 8*(BTN_W+BTN_SPACING)+50, "y": BTN_Y, "w": 60, "h": BTN_H, "color": (140, 140, 140), "action": "grid_smaller"},
    {"label": "G+",        "x": BTN_SPACING + 8*(BTN_W+BTN_SPACING)+50 + 70, "y": BTN_Y, "w": 60, "h": BTN_H, "color": (140, 140, 140), "action": "grid_bigger"},
]

draw_color = (50, 50, 200)  # default (BGR)
thickness = INIT_THICKNESS
mode = "pen"  # "pen" or "eraser"
prev_x, prev_y = 0, 0
palm_frames = 0
last_clear_time = 0
clear_cooldown = 1.0  # seconds

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        fgmask = bg_subtractor.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)

        ui = frame.copy()

        # ===== Top bar =====
        cv2.rectangle(ui, (0, 0), (W, 60), PRIMARY_COLOR, -1)
        cv2.putText(ui, "Air Drawing Canvas", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Drawing area & grid
        DRAW_Y1 = 60
        DRAW_Y2 = BTN_Y - 10
        DRAW_X1 = 0
        DRAW_X2 = W
        draw_grid(ui, DRAW_X1, DRAW_Y1, DRAW_X2, DRAW_Y2, GRID_SPACING, GRID_COLOR, GRID_THICKNESS)

        # Bottom toolbar
        cv2.rectangle(ui, (0, BTN_Y-10), (W, H), (240, 240, 240), -1)
        cv2.line(ui, (0, BTN_Y-10), (W, BTN_Y-10), (200, 200, 200), 2)
        for btn in buttons:
            active = (btn["action"] == "eraser" and mode == "eraser")
            draw_button(ui, btn["x"], btn["y"], btn["w"], btn["h"],
                        btn["label"], btn["color"], active=active, radius=int(BTN_H*BTN_ROUNDNESS))

        # ===== Info panel (moved down to avoid overlap with top bar) =====
        info_panel = np.zeros((260, 220, 3), dtype=np.uint8) + 240
        cv2.rectangle(info_panel, (0, 0), (219, 259), (200, 200, 200), 1)
        bg_name = "White" if (background_color == (255, 255, 255)) else "Black"
        cv2.putText(info_panel, "Status:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SECONDARY_COLOR, 1)
        cv2.putText(info_panel, f"Mode: {'Eraser' if mode == 'eraser' else 'Pen'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SECONDARY_COLOR, 1)
        cv2.putText(info_panel, f"Thickness: {thickness}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SECONDARY_COLOR, 1)
        cv2.putText(info_panel, f"BG: {bg_name}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SECONDARY_COLOR, 1)
        cv2.putText(info_panel, f"Grid: {GRID_SPACING}px", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SECONDARY_COLOR, 1)
        cv2.putText(info_panel, "Controls:", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SECONDARY_COLOR, 1)
        cv2.putText(info_panel, "- Index to draw", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, SECONDARY_COLOR, 1)
        cv2.putText(info_panel, "- Hover left slider to set thickness", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, SECONDARY_COLOR, 1)

        ph, pw = info_panel.shape[:2]
        inf_x1 = W - 20 - pw
        inf_y1 = INFO_PANEL_TOP
        inf_y2 = min(inf_y1 + ph, H)
        ui[inf_y1:inf_y2, inf_x1:inf_x1 + pw] = info_panel[0:(inf_y2 - inf_y1), :]

        # ===== Color/Mode indicator (draw AFTER panel so it's always visible) =====
        sw = 36
        sw_x = W - 180
        sw_y = 12  # stays in the top bar zone (0..60)
        color_to_show = background_color if mode == "eraser" else draw_color
        cv2.rectangle(ui, (sw_x, sw_y), (sw_x + sw, sw_y + sw), color_to_show, -1)
        cv2.rectangle(ui, (sw_x, sw_y), (sw_x + sw, sw_y + sw), (255, 255, 255), 2)
        label_text = f"Mode: {('Eraser' if mode=='eraser' else 'Pen')}"
        cv2.putText(ui, label_text, (sw_x + sw + 10, sw_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Slider (left)
        draw_slider(ui, thickness, SLIDER_X, SLIDER_Y1, SLIDER_Y2, SLIDER_W)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(ui, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Index fingertip
                x = int(hand_landmarks.landmark[8].x * W)
                y = int(hand_landmarks.landmark[8].y * H)
                x = np.clip(x, 0, W-1)
                y = np.clip(y, 0, H-1)

                # Fingers state
                fup = fingers_up(hand_landmarks)
                index_only = (fup == [1, 0, 0, 0])
                open_palm = (fup == [1, 1, 1, 1])

                # Regions
                inside_slider = point_in_rect(x, y, SLIDER_X, SLIDER_Y1, SLIDER_W, SLIDER_Y2 - SLIDER_Y1)
                inside_button = any(point_in_rect(x, y, b["x"], b["y"], b["w"], b["h"]) for b in buttons)
                over_ui_area = inside_slider or inside_button or (y < 60) or (y > BTN_Y - 10)

                # Adjust thickness using slider
                if inside_slider:
                    y_clamped = int(np.clip(y, SLIDER_Y1, SLIDER_Y2))
                    target = map_y_to_thickness(y_clamped, SLIDER_Y1, SLIDER_Y2)
                    thickness = int((1 - THICKNESS_ALPHA) * thickness + THICKNESS_ALPHA * target)
                    cv2.putText(ui, "Adjusting thickness...", (SLIDER_X + 50, SLIDER_Y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

                # Buttons
                if inside_button:
                    for btn in buttons:
                        if point_in_rect(x, y, btn["x"], btn["y"], btn["w"], btn["h"]):
                            if btn["action"] == "color":
                                draw_color = btn["color"]; mode = "pen"
                            elif btn["action"] == "thinner":
                                thickness = max(THICKNESS_MIN, thickness - 2)
                            elif btn["action"] == "thicker":
                                thickness = min(THICKNESS_MAX, thickness + 2)
                            elif btn["action"] == "eraser":
                                mode = "eraser"
                            elif btn["action"] == "clear":
                                now = time.time()
                                if now - last_clear_time > clear_cooldown:
                                    canvas[:] = background_color
                                    last_clear_time = now
                            elif btn["action"] == "toggle_bg":
                                old_bg = background_color
                                background_color = (255, 255, 255) if old_bg == (0, 0, 0) else (0, 0, 0)
                                mask = np.all(canvas == np.array(old_bg, dtype=np.uint8), axis=2)
                                canvas[mask] = background_color
                            elif btn["action"] == "grid_smaller":
                                GRID_SPACING = max(GRID_MIN, GRID_SPACING - 10)
                            elif btn["action"] == "grid_bigger":
                                GRID_SPACING = min(GRID_MAX, GRID_SPACING + 10)

                # Clear with open palm (hold)
                if open_palm:
                    palm_frames += 1
                    cv2.putText(ui, "Clearing...", (W//2 - 100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    if palm_frames >= PALM_CLEAR_FRAMES:
                        canvas[:] = background_color
                        palm_frames = 0
                        prev_x, prev_y = 0, 0
                        last_clear_time = time.time()
                else:
                    palm_frames = 0

                # Drawing (index only, not over UI)
                movement_ok = (fgmask[y, x] > 0) if MASK_MOVE_CHECK else True
                if index_only and movement_ok and not over_ui_area:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y
                    color_to_use = background_color if mode == "eraser" else draw_color
                    thick_to_use = max(1, thickness * ERASER_MULTIPLIER) if mode == "eraser" else thickness
                    cv2.line(canvas, (prev_x, prev_y), (x, y), color_to_use, thick_to_use, lineType=cv2.LINE_AA)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = 0, 0

        # Combine layers
        combined = cv2.addWeighted(ui, 0.6, canvas, 0.4, 0)

        # Bottom hints
        cv2.putText(combined,
                    "Index to draw |Open palm to clear | BG toggles background | G-/G+ grid size",
                    (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SECONDARY_COLOR, 1)

        cv2.imshow("Air Drawing Canvas - Enhanced", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"drawing_{timestamp}.png", canvas)
            cv2.imwrite(f"preview_{timestamp}.png", combined)
            print(f"Saved: drawing_{timestamp}.png and preview_{timestamp}.png")

cap.release()
cv2.destroyAllWindows()
