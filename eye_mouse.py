"""
👁️  Eye-Tracking Mouse Controller v4
---------------------------------------
Uses a CALIBRATION-BASED approach — the only reliable way to do this.

Instead of trying to math out head movement, we:
1. Show dots at known screen positions
2. Record where your iris is when you look at each dot
3. Fit a linear model: iris_position → screen_position
4. Use that model in real-time

This absorbs head tilt, distance, eye shape, everything.

Blink:  quick blink = click | hold blink = double-click every 1s

Requirements:
    pip install opencv-python mediapipe pyautogui numpy

Run:
    python3 eye_mouse.py

During calibration:
    - A red dot will appear on your screen
    - Look at it, then press SPACE
    - Repeat for all 9 dots
    - Done — mouse follows your eyes

Press R to recalibrate, Q to quit (in the camera window).
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import tkinter as tk
import threading

# ─── CONFIG ────────────────────────────────────────────────────────────────────

SMOOTHING            = 0.85   # cursor smoothing 0–1 (higher = smoother but laggier)
SAMPLES_PER_DOT      = 30     # iris frames averaged per calibration point
BLINK_THRESHOLD      = 0.21
BLINK_CONSEC_FRAMES  = 2
CLICK_COOLDOWN       = 0.5
HOLD_DCLICK_INTERVAL = 1.0
HOLD_THRESHOLD       = 0.4

# ─── MEDIAPIPE ─────────────────────────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

L_TOP, L_BOT, L_TOP2, L_BOT2 = 159, 145, 158, 153
L_LEFT, L_RIGHT = 33, 133
R_TOP, R_BOT, R_TOP2, R_BOT2 = 386, 374, 385, 380
R_LEFT, R_RIGHT = 362, 263


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def get_iris(frame, flip=True):
    """Returns averaged (x, y) iris position in frame coords, or None."""
    if flip:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None, None, None
    lms = res.multi_face_landmarks[0].landmark

    def lm_xy(idx):
        return np.array([lms[idx].x * w, lms[idx].y * h])

    l_iris = np.mean([lm_xy(i) for i in LEFT_IRIS],  axis=0)
    r_iris = np.mean([lm_xy(i) for i in RIGHT_IRIS], axis=0)
    iris   = (l_iris + r_iris) / 2

    # EAR
    def dist(a, b): return np.linalg.norm(lm_xy(a) - lm_xy(b))
    l_ear = (dist(L_TOP, L_BOT) + dist(L_TOP2, L_BOT2)) / (2 * dist(L_LEFT, L_RIGHT) + 1e-6)
    r_ear = (dist(R_TOP, R_BOT) + dist(R_TOP2, R_BOT2)) / (2 * dist(R_LEFT, R_RIGHT) + 1e-6)
    avg_ear = (l_ear + r_ear) / 2

    return iris, avg_ear, (frame if not flip else cv2.flip(frame, 1))


# ─── CALIBRATION OVERLAY (tkinter fullscreen dot) ──────────────────────────────

class CalibDot:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg='black')
        self.root.attributes('-alpha', 0.85)

        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        self.sw = self.root.winfo_screenwidth()
        self.sh = self.root.winfo_screenheight()

        self.label = self.canvas.create_text(
            self.sw // 2, self.sh // 2 - 60,
            text="", fill="white",
            font=("Arial", 22)
        )
        self.dot = None
        self.ready = threading.Event()
        self.root.bind('<space>', self._on_space)
        self.root.bind('<q>', lambda e: self.root.quit())

    def _on_space(self, event):
        self.ready.set()

    def show_dot(self, x, y, msg="Look at the dot, then press SPACE"):
        self.canvas.delete('all')
        self.canvas.create_text(
            self.sw // 2, 40,
            text=msg, fill='white', font=('Arial', 20)
        )
        # Outer ring
        self.canvas.create_oval(x-18, y-18, x+18, y+18, outline='white', width=2)
        # Inner dot
        self.canvas.create_oval(x-6, y-6, x+6, y+6, fill='red', outline='red')
        self.ready.clear()
        self.root.update()

    def show_message(self, msg):
        self.canvas.delete('all')
        self.canvas.create_text(
            self.sw // 2, self.sh // 2,
            text=msg, fill='white', font=('Arial', 26), justify='center'
        )
        self.root.update()

    def wait_for_space(self, timeout=30):
        """Block until space pressed (polls tkinter so window stays responsive)."""
        deadline = time.time() + timeout
        while not self.ready.is_set():
            self.root.update()
            time.sleep(0.016)
            if time.time() > deadline:
                return False
        return True

    def destroy(self):
        self.root.destroy()


# ─── CALIBRATION LOGIC ─────────────────────────────────────────────────────────

def run_calibration(cam):
    """
    Show 9 dots across the screen, collect iris samples at each,
    return (model_x, model_y) — two linear regressors iris→screen.
    """
    sw, sh = pyautogui.size()
    margin = 100

    # 9-point grid
    xs = [margin, sw // 2, sw - margin]
    ys = [margin, sh // 2, sh - margin]
    points = [(x, y) for y in ys for x in xs]

    dot = CalibDot()
    dot.show_message("EYE TRACKER CALIBRATION\n\nLook at each red dot\nand press SPACE\n\nPress SPACE to begin")
    dot.wait_for_space()

    iris_pts   = []
    screen_pts = []

    for i, (sx, sy) in enumerate(points):
        msg = f"Point {i+1} of {len(points)} — look at the dot and press SPACE"
        dot.show_dot(sx, sy, msg)
        dot.wait_for_space()

        # Collect SAMPLES_PER_DOT iris readings after space pressed
        dot.show_dot(sx, sy, f"Collecting... ({SAMPLES_PER_DOT} frames)")
        samples = []
        attempts = 0
        while len(samples) < SAMPLES_PER_DOT and attempts < SAMPLES_PER_DOT * 5:
            ret, frame = cam.read()
            if not ret:
                attempts += 1
                continue
            iris, _, _ = get_iris(frame)
            if iris is not None:
                samples.append(iris)
            attempts += 1
            dot.root.update()

        if len(samples) < 5:
            dot.show_message(f"Couldn't detect eyes at point {i+1}.\nMake sure your face is visible.\nPress SPACE to retry.")
            dot.wait_for_space()
            # retry this point
            points.insert(i+1, (sx, sy))
            continue

        avg_iris = np.mean(samples, axis=0)
        iris_pts.append(avg_iris)
        screen_pts.append([sx, sy])

    dot.show_message("✅  Calibration complete!\n\nMove your eyes to control the mouse.\nQuick blink = click | Hold blink = double-click\n\nPress SPACE to start")
    dot.wait_for_space()
    dot.destroy()

    # Fit linear model: [iris_x, iris_y, 1] → screen_x / screen_y
    iris_arr   = np.array(iris_pts)
    screen_arr = np.array(screen_pts)
    A = np.hstack([iris_arr, np.ones((len(iris_arr), 1))])  # Nx3

    # Least-squares fit
    mx, _, _, _ = np.linalg.lstsq(A, screen_arr[:, 0], rcond=None)
    my, _, _, _ = np.linalg.lstsq(A, screen_arr[:, 1], rcond=None)

    print(f"✅  Calibration done with {len(iris_pts)} points.")
    print(f"    Model X: {mx}")
    print(f"    Model Y: {my}")
    return mx, my


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE    = 0

    screen_w, screen_h = pyautogui.size()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌  Cannot open webcam.")
        return

    print("👁️  Eye Tracker v4 — calibration-based")
    print("    Close the camera window at any time with Q.")
    print("    A calibration overlay will appear on your screen.\n")

    # ── CALIBRATION ────────────────────────────────────────────────────────────
    model_x, model_y = run_calibration(cam)

    # ── TRACKING LOOP ──────────────────────────────────────────────────────────
    smooth_x, smooth_y = screen_w / 2.0, screen_h / 2.0

    blink_counter    = 0
    eye_closed       = False
    eye_closed_start = 0.0
    last_click_time  = 0.0
    last_dclick_time = 0.0

    print("🟢  Tracking. Look around to move the mouse.")
    print("    Press R in camera window to recalibrate, Q to quit.\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        iris, avg_ear, _ = get_iris(frame)

        if iris is not None:
            # ── Gaze → screen via linear model ─────────────────────────────
            feat = np.array([iris[0], iris[1], 1.0])
            tx   = float(np.clip(feat @ model_x, 0, screen_w  - 1))
            ty   = float(np.clip(feat @ model_y, 0, screen_h - 1))

            smooth_x = SMOOTHING * smooth_x + (1 - SMOOTHING) * tx
            smooth_y = SMOOTHING * smooth_y + (1 - SMOOTHING) * ty

            pyautogui.moveTo(int(smooth_x), int(smooth_y))

            # ── Blink detection ─────────────────────────────────────────────
            now = time.time()
            if avg_ear < BLINK_THRESHOLD:
                blink_counter += 1
                if blink_counter >= BLINK_CONSEC_FRAMES and not eye_closed:
                    eye_closed       = True
                    eye_closed_start = now

                if eye_closed:
                    held = now - eye_closed_start
                    if held >= HOLD_THRESHOLD:
                        if now - last_dclick_time >= HOLD_DCLICK_INTERVAL:
                            pyautogui.doubleClick()
                            last_dclick_time = now
                            print(f"🖱️🖱️  Double-click! ({held:.1f}s)")
            else:
                if eye_closed:
                    held = now - eye_closed_start
                    if held < HOLD_THRESHOLD:
                        if now - last_click_time >= CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click_time = now
                            print(f"🖱️   Click! ({held:.2f}s blink)")
                eye_closed    = False
                blink_counter = 0

            # ── Debug overlay ───────────────────────────────────────────────
            ear_color = (0, 0, 255) if avg_ear < BLINK_THRESHOLD else (0, 255, 100)
            hold_str  = f" HOLD {now - eye_closed_start:.1f}s" if eye_closed else ""
            cv2.putText(display, f"EAR {avg_ear:.3f}{hold_str}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_color, 2)
            cv2.putText(display, f"Mouse ({int(smooth_x)}, {int(smooth_y)})",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(display, "R=recalibrate  Q=quit",
                        (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # Draw iris dot
            h, w = display.shape[:2]
            iris_disp = iris.copy()
            iris_disp[0] = w - iris_disp[0]   # mirror to match flip
            cv2.circle(display, tuple(iris_disp.astype(int)), 5, (0, 255, 255), -1)

        else:
            cv2.putText(display, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("👁  Eye Mouse v4", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\n🔄  Recalibrating...")
            cv2.destroyAllWindows()
            model_x, model_y = run_calibration(cam)
            print("🟢  Back to tracking.\n")

    cam.release()
    cv2.destroyAllWindows()
    print("👋  Done.")


if __name__ == "__main__":
    main()
