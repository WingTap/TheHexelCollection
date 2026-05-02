"""
🖐️  Hand-Tracking Mouse Controller v2
----------------------------------------
Move palm        → moves the mouse
Pinch (👌)       → click on release
Hold pinch still → double-click every second
Hold + move 100px in 1s → DRAG (mouseDown → follow → mouseUp on release)

Keyboard gesture (✌️ peace sign) → toggles virtual QWERTY keyboard
Type by moving cursor over keys and pinching.

Requirements:
    pip install opencv-python mediapipe pyautogui numpy

Run:
    python3 hand_mouse.py   |   press Q to quit
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import tkinter as tk
import threading

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0

# ─── CONFIG ────────────────────────────────────────────────────────────────────

TRACKING_ZONE        = 0.35
SMOOTHING            = 0.80

PINCH_THRESHOLD      = 0.13
PINCH_CONSEC_FRAMES  = 2
CLICK_COOLDOWN       = 0.4

HOLD_THRESHOLD       = 0.2       # seconds before hold-mode activates
DRAG_PX_THRESHOLD    = 100        # screen pixels moved within 1s to enter drag
DRAG_TIME_WINDOW     = 1.0        # seconds window to measure drag movement
HOLD_DCLICK_INTERVAL = 1.0

# Peace sign hold duration to toggle keyboard
PEACE_HOLD_FRAMES    = 20         # consecutive frames of peace sign

# ─── MEDIAPIPE ─────────────────────────────────────────────────────────────────

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)

# Landmark indices
WRIST      = 0
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_TIP  = 20
THUMB_TIP  = 4;  THUMB_IP   = 3


# ─── HAND HELPERS ──────────────────────────────────────────────────────────────

def lm_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])

def finger_up(landmarks, tip, pip, w, h):
    """Returns True if finger is extended (tip above pip, i.e. lower y)."""
    return lm_xy(landmarks, tip, w, h)[1] < lm_xy(landmarks, pip, w, h)[1]

def hand_size(landmarks, w, h):
    return float(np.linalg.norm(
        lm_xy(landmarks, WRIST, w, h) - lm_xy(landmarks, INDEX_MCP, w, h)
    )) + 1e-6

def pinch_ratio(landmarks, w, h):
    return float(np.linalg.norm(
        lm_xy(landmarks, THUMB_TIP, w, h) - lm_xy(landmarks, INDEX_TIP, w, h)
    )) / hand_size(landmarks, w, h)

def cursor_anchor(landmarks, w, h):
    a = lm_xy(landmarks, INDEX_MCP, w, h)
    b = lm_xy(landmarks, PINKY_MCP, w, h)
    return (a + b) / 2
def zone_to_screen(pt, frame_w, frame_h, screen_w, screen_h, zone):
    margin_x = frame_w * (1 - zone) / 2
    margin_y = frame_h * (1 - zone) / 2
    cx = np.clip(pt[0], margin_x, frame_w - margin_x)
    cy = np.clip(pt[1], margin_y, frame_h - margin_y)
    nx = (cx - margin_x) / (frame_w * zone)
    ny = (cy - margin_y) / (frame_h * zone)
    return (float(np.clip(nx * screen_w, 0, screen_w - 1)),
            float(np.clip(ny * screen_h, 0, screen_h - 1)))


# ─── VIRTUAL KEYBOARD ──────────────────────────────────────────────────────────

ROWS = [
    ['`','1','2','3','4','5','6','7','8','9','0','-','=','⌫'],
    ['q','w','e','r','t','y','u','i','o','p','[',']','\\'],
    ['a','s','d','f','g','h','j','k','l',';',"'",'↵'],
    ['⇧','z','x','c','v','b','n','m',',','.','/',  '⇧'],
    ['SPACE', '×'],
]

KEY_MAP = {
    '⌫': 'backspace', '↵': 'return', '⇧': 'shift',
    'SPACE': 'space',  '×': None,   # × closes keyboard
}

class VirtualKeyboard:
    def __init__(self, on_close):
        self.on_close   = on_close
        self.shift_on   = False
        self.root       = None
        self.thread     = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.root = tk.Tk()
        self.root.title("⌨️  Hand Keyboard")
        self.root.attributes('-topmost', True)
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(False, False)

        # Position at bottom centre of screen
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.update_idletasks()

        pad = 6
        btn_w, btn_h = 5, 2   # in tk character units

        for row_i, row in enumerate(ROWS):
            frame = tk.Frame(self.root, bg='#1a1a2e')
            frame.pack(padx=pad, pady=(pad if row_i == 0 else 2))
            for key in row:
                w = 24 if key == 'SPACE' else (btn_w + 1 if len(key) > 1 else btn_w)
                b = tk.Button(
                    frame,
                    text=key,
                    width=w,
                    height=btn_h,
                    font=('Consolas', 13, 'bold'),
                    bg='#16213e',
                    fg='#e0e0e0',
                    activebackground='#0f3460',
                    activeforeground='white',
                    relief='flat',
                    bd=0,
                    cursor='hand2',
                    command=lambda k=key: self._press(k),
                )
                b.pack(side='left', padx=2)

        self.root.update_idletasks()
        kw = self.root.winfo_width()
        kh = self.root.winfo_height()
        x  = (sw - kw) // 2
        y  = sh - kh - 40
        self.root.geometry(f'+{x}+{y}')
        self.root.protocol('WM_DELETE_WINDOW', self._close)
        self.root.mainloop()

    def _press(self, key):
        if key == '×':
            self._close()
            return
        mapped = KEY_MAP.get(key)
        if mapped:
            pyautogui.press(mapped)
        else:
            char = key.upper() if self.shift_on else key
            pyautogui.typewrite(char, interval=0)
            if self.shift_on:
                self.shift_on = False

    def _close(self):
        self.on_close()
        if self.root:
            self.root.destroy()
            self.root = None

    def destroy(self):
        if self.root:
            self.root.after(0, self.root.destroy)
            self.root = None


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    screen_w, screen_h = pyautogui.size()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌  Cannot open webcam.")
        return

    print("🖐️  Hand Mouse v2 started.")
    print("    Move palm         → mouse")
    print("    Pinch 👌          → click")
    print("    Hold pinch still  → double-click")
    print("    Hold + move 100px → drag")
    print("    Q                 → quit\n")

    smooth_x, smooth_y = screen_w / 2.0, screen_h / 2.0

    # Pinch state machine
    #   idle → pinching → (click | hold_still | dragging)
    pinch_counter    = 0
    pinching         = False
    pinch_start_time = 0.0
    pinch_start_pos  = (0.0, 0.0)   # screen pos when pinch began
    drag_active      = False
    drag_decided     = False         # have we resolved hold → drag or dclick?
    last_click_time  = 0.0
    last_dclick_time = 0.0



    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        # Tracking zone guide
        mx = int(w * (1 - TRACKING_ZONE) / 2)
        my = int(h * (1 - TRACKING_ZONE) / 2)
        cv2.rectangle(frame, (mx, my), (w - mx, h - my), (55, 55, 55), 1)

        now = time.time()

        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0].landmark

            mp_drawing.draw_landmarks(
                frame, res.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            # ── Cursor movement ─────────────────────────────────────────────
            anchor   = cursor_anchor(lms, w, h)
            tx, ty   = zone_to_screen(anchor, w, h, screen_w, screen_h, TRACKING_ZONE)
            smooth_x = SMOOTHING * smooth_x + (1 - SMOOTHING) * tx
            smooth_y = SMOOTHING * smooth_y + (1 - SMOOTHING) * ty
            sx, sy   = int(smooth_x), int(smooth_y)

            # Always use moveTo — we manage mouseDown/mouseUp manually,
            # so dragTo must NOT be used (it presses the button again internally).
            pyautogui.moveTo(sx, sy)

            cv2.circle(frame, tuple(anchor.astype(int)), 8, (0, 220, 255), -1)


            # ── Pinch state machine ─────────────────────────────────────────
            pr          = pinch_ratio(lms, w, h)
            is_pinching = pr < PINCH_THRESHOLD

            thumb_pt = lm_xy(lms, THUMB_TIP, w, h).astype(int)
            index_pt = lm_xy(lms, INDEX_TIP, w, h).astype(int)
            p_color  = (0, 0, 255) if is_pinching else (0, 255, 120)
            cv2.line(frame, tuple(thumb_pt), tuple(index_pt), p_color, 2)
            cv2.circle(frame, tuple(thumb_pt), 6, p_color, -1)
            cv2.circle(frame, tuple(index_pt), 6, p_color, -1)

            if is_pinching:
                pinch_counter += 1

                # First frame of sustained pinch
                if pinch_counter >= PINCH_CONSEC_FRAMES and not pinching:
                    pinching         = True
                    drag_decided     = False
                    pinch_start_time = now
                    pinch_start_pos  = (smooth_x, smooth_y)

                if pinching and not drag_decided:
                    held      = now - pinch_start_time
                    moved     = np.linalg.norm(
                        np.array([smooth_x, smooth_y]) - np.array(pinch_start_pos)
                    )

                    if held >= HOLD_THRESHOLD:
                        if moved >= DRAG_PX_THRESHOLD:
                            # ── Enter DRAG ──────────────────────────────────
                            drag_active  = True
                            drag_decided = True
                            # Jump back to where the pinch originally happened,
                            # press down there, then let the loop dragTo current pos
                            pyautogui.moveTo(int(pinch_start_pos[0]), int(pinch_start_pos[1]))
                            pyautogui.mouseDown(button='left')
                            print(f"🔒  Drag started from ({int(pinch_start_pos[0])}, {int(pinch_start_pos[1])})")
                        elif held >= DRAG_TIME_WINDOW:
                            # Held long enough, didn't move → DOUBLE-CLICK mode
                            drag_decided = True

                if pinching and drag_decided and not drag_active:
                    # Double-click mode: fire every second
                    if now - last_dclick_time >= HOLD_DCLICK_INTERVAL:
                        pyautogui.doubleClick()
                        last_dclick_time = now
                        print(f"🖱️🖱️  Double-click!")

            else:
                if pinching:
                    held = now - pinch_start_time

                    if drag_active:
                        # Release drag
                        pyautogui.mouseUp(button='left')
                        drag_active  = False
                        drag_decided = False
                        print(f"🔓  Drag released")
                    elif not drag_decided:
                        # Quick pinch → single click
                        if now - last_click_time >= CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click_time = now
                            print(f"🖱️   Click! ({held:.2f}s)")

                pinching      = False
                pinch_counter = 0

            # ── HUD ─────────────────────────────────────────────────────────
            if drag_active:
                state_str   = "DRAGGING"
                state_color = (0, 120, 255)
            elif pinching and drag_decided:
                state_str   = "DOUBLE-CLICK MODE"
                state_color = (0, 0, 255)
            elif pinching:
                held    = now - pinch_start_time
                moved   = np.linalg.norm(np.array([smooth_x, smooth_y]) - np.array(pinch_start_pos))
                state_str   = f"PINCH {held:.1f}s  {moved:.0f}px"
                state_color = (0, 180, 255)
            else:
                state_str   = f"pinch: {pr:.3f}"
                state_color = (0, 255, 120)

            cv2.putText(frame, state_str,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)
            cv2.putText(frame, f"Mouse ({sx}, {sy})",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        else:
            if drag_active:
                pyautogui.mouseUp(button='left')
                drag_active = False
            pinching      = False
            pinch_counter = 0
            cv2.putText(frame, "No hand detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)

        cv2.putText(frame, "Q = quit", (w - 80, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        cv2.imshow("🖐  Hand Mouse v2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if drag_active:
        pyautogui.mouseUp(button='left')
    if keyboard:
        keyboard.destroy()
    cam.release()
    cv2.destroyAllWindows()
    print("👋  Done.")


if __name__ == "__main__":
    main()

