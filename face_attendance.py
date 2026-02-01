import cv2
import face_recognition
import numpy as np
import pickle
import os
import argparse
import pandas as pd
import time
from collections import deque
from datetime import datetime
from math import hypot

# =====================================================
# CONFIG
# =====================================================
ENC_FILE = "encodings_v2.pickle"
ATT_FILE = "attendance_v2.csv"

CAPTURE_IMAGES = 35
FRAME_WIDTH = 640

LIVENESS_TIMEOUT = 2.0
MOVEMENT_THRESHOLD_RATIO = 0.08
BLINK_EAR_THRESHOLD = 0.20

attendance_state = {}        # name -> "IN" or "OUT"
last_seen_frame = {}         # name -> frame index
FRAME_GAP = 30               # ~1 second if camera ~30fps
frame_idx = 0


# =====================================================
# STORAGE
# =====================================================
def save_encodings(encs, names, ids):
    with open(ENC_FILE, "wb") as f:
        pickle.dump({"encodings": encs, "names": names, "ids": ids}, f)

def load_encodings():
    if not os.path.exists(ENC_FILE):
        return [], [], []
    with open(ENC_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"], data["ids"]

def ensure_attendance_file():
    """
    Ensure attendance file exists. New canonical schema:
      columns = ["name", "id", "date", "timestamp", "state"]
    where state is 1 for Punch-In and 0 for Punch-Out.

    If an old file exists (punch_in / punch_out columns), we keep it as-is
    and the code will handle that format as backward-compatible.
    """
    if not os.path.exists(ATT_FILE):
        df = pd.DataFrame(columns=["name", "id", "date", "timestamp", "state"])
        df.to_csv(ATT_FILE, index=False)

def _detect_schema_and_read():
    """
    Read attendance CSV and return DataFrame plus schema info.
    Returns: df, schema_type
      schema_type == "new" => has 'state' column
      schema_type == "old" => has 'punch_in'/'punch_out' columns (legacy)
      schema_type == "empty" => file missing/empty (df created)
    """
    if not os.path.exists(ATT_FILE):
        # create new canonical file
        ensure_attendance_file()
        return pd.read_csv(ATT_FILE), "new"

    df = pd.read_csv(ATT_FILE)
    cols = set(df.columns.str.lower())
    if "state" in cols and "timestamp" in cols:
        return df, "new"
    elif "punch_in" in cols or "punch_out" in cols:
        return df, "old"
    else:
        # Unknown/empty header - treat as new canonical
        ensure_attendance_file()
        df = pd.read_csv(ATT_FILE)
        return df, "new"

def log_attendance(name, sid):
    ensure_attendance_file()

    # Normalize ID to string (CRITICAL)
    sid = str(sid)

    df, schema = _detect_schema_and_read()

    # Work on a copy
    df_new = df.copy()

    # Normalize ID column if present
    if "id" in df_new.columns:
        df_new["id"] = df_new["id"].astype(str)

    now = datetime.now()
    date = now.date().isoformat()
    ts = now.isoformat(timespec="seconds")

    # -----------------------------
    # Determine last state
    # -----------------------------
    if df_new.empty:
        last_state = None
    else:
        if "state" in df_new.columns:
            mask = (
                (df_new["name"] == name) &
                (df_new["id"] == sid) &
                (df_new["date"] == date)
            )
            user_rows = df_new[mask]
            if user_rows.empty:
                last_state = None
            else:
                last_state = int(user_rows.iloc[-1]["state"])
        else:
            # Legacy fallback (punch_in / punch_out)
            mask = (
                (df_new["name"] == name) &
                (df_new["id"] == sid) &
                (df_new["date"] == date)
            )
            user_rows = df_new[mask]
            if user_rows.empty:
                last_state = None
            else:
                last = user_rows.iloc[-1]
                if pd.isna(last.get("punch_out", None)) or last.get("punch_out", "") == "":
                    last_state = 1
                else:
                    last_state = 0

    # -----------------------------
    # Toggle state
    # -----------------------------
    if last_state == 1:
        new_state = 0
        action = "Punch-Out"
    else:
        new_state = 1
        action = "Punch-In"

    # Ensure canonical columns exist
    for col in ["name", "id", "date", "timestamp", "state"]:
        if col not in df_new.columns:
            df_new[col] = ""

    # Append new row (Pandas 2.x safe)
    new_row = {
        "name": name,
        "id": sid,
        "date": date,
        "timestamp": ts,
        "state": new_state,
    }

    df_new = pd.concat(
        [df_new, pd.DataFrame([new_row])],
        ignore_index=True
    )

    df_new.to_csv(ATT_FILE, index=False)
    print(f"[ATTENDANCE] {name} ({sid}) → {action}")

# =====================================================
# LIGHTING-AWARE PREPROCESS (NOVELTY #1)
# =====================================================
def lighting_preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness < 70:
        gamma = 1.8
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        corrected = cv2.LUT(frame, table)

        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR), "DARK"

    if mean_brightness > 180:
        return frame, "BRIGHT"

    return frame, "OK"


# =====================================================
# CONFIDENCE VALIDATOR
# =====================================================
class ConfidenceValidator:
    def __init__(self, window=8, min_hits=6, thresh=0.5, max_var=0.02):
        self.buffer = deque(maxlen=window)
        self.window = window
        self.min_hits = min_hits
        self.thresh = thresh
        self.max_var = max_var

    def update(self, name, dist):
        self.buffer.append((name, dist))

    def is_confident(self):
        if len(self.buffer) < self.window:
            return False, None

        names = [n for n, _ in self.buffer]
        dists = [d for _, d in self.buffer]

        candidate = max(set(names), key=names.count)
        if candidate == "Unknown":
            return False, None

        if (
            names.count(candidate) >= self.min_hits and
            np.mean(dists) < self.thresh and
            np.var(dists) < self.max_var
        ):
            self.buffer.clear()
            return True, candidate

        return False, None


# =====================================================
# LIVENESS (NOVELTY #2 – SPOOF PROOF)
# =====================================================
def eye_aspect_ratio(eye):
    if len(eye) < 6:
        return None
    def d(a, b): return hypot(a[0]-b[0], a[1]-b[1])
    return (d(eye[1], eye[5]) + d(eye[2], eye[4])) / (2.0 * d(eye[0], eye[3]))

def perform_liveness(cam):
    start = time.time()
    centroids = []

    while time.time() - start < LIVENESS_TIMEOUT:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)

        if boxes:
            t, r, b, l = boxes[0]
            centroids.append((l + r) / 2)

            if max(centroids) - min(centroids) > FRAME_WIDTH * MOVEMENT_THRESHOLD_RATIO:
                return True

            landmarks = face_recognition.face_landmarks(rgb)
            if landmarks:
                lm = landmarks[0]
                for eye in ["left_eye", "right_eye"]:
                    ear = eye_aspect_ratio(lm.get(eye, []))
                    if ear and ear < BLINK_EAR_THRESHOLD:
                        return True
    return False


# =====================================================
# REGISTRATION
# =====================================================
def register(name, sid):
    encs, names, ids = load_encodings()
    if sid in ids:
        print("[ERROR] ID already registered.")
        return

    cam = cv2.VideoCapture(0)
    samples = []

    print(f"[INFO] Registering {name} ({sid})")

    while len(samples) < CAPTURE_IMAGES:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])))
        frame, lighting = lighting_preprocess(frame)

        if lighting == "BRIGHT":
            cv2.putText(frame, "Backlight detected", (10,60), 0, 0.7, (0,0,255), 2)
            cv2.imshow("Register", frame)
            cv2.waitKey(1)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)

        if len(boxes) == 1:
            enc = face_recognition.face_encodings(rgb, boxes)[0]
            samples.append(enc)
            cv2.putText(frame, f"{len(samples)}/{CAPTURE_IMAGES}", (10,30),
                        0, 0.8, (0,255,0), 2)

        cv2.imshow("Register", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    encs.extend(samples)
    names.extend([name]*len(samples))
    ids.extend([sid]*len(samples))
    save_encodings(encs, names, ids)
    print("[DONE] Registration complete")


# =====================================================
# RECOGNITION (WITH BBOX + CONFIDENCE)
# =====================================================
def recognize():
    encs, names, ids = load_encodings()
    if not encs:
        print("[INFO] No registered users.")
        return

    name_to_id = dict(zip(names, ids))
    validator = ConfidenceValidator()

    # Latch: prevents multiple triggers while face stays visible
    seen_latch = set()

    cam = cv2.VideoCapture(0)
    print("[RUN] Recognition started. Press 'q' to quit.")

    def get_current_state(name, sid):
        """
        Returns:
            1 -> IN  (last recorded state is Punch-In)
            0 -> OUT (last recorded state is Punch-Out or no record)
        Handles both new canonical schema (state column) and legacy schema.
        """
        if not os.path.exists(ATT_FILE):
            return 0

        df = pd.read_csv(ATT_FILE)
        if df.empty:
            return 0

        # Prefer canonical 'state' column if available
        if "state" in df.columns and "timestamp" in df.columns:
            sid = str(sid)
            df["id"] = df["id"].astype(str)

            df_user = df[
                (df["name"] == name) &
                (df["id"] == sid)
            ]

            if df_user.empty:
                return 0
            try:
                last_state = int(df_user.iloc[-1]["state"])
                return 1 if last_state == 1 else 0
            except Exception:
                return 0
        else:
            # Legacy handling: check last row's punch_out presence
            df_user = df[(df["name"] == name) & (df["id"] == sid)]
            if df_user.empty:
                return 0
            last = df_user.iloc[-1]
            if ("punch_out" in df.columns) and (pd.isna(last.get("punch_out", None)) or last.get("punch_out", "") == ""):
                return 1
            else:
                return 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            # Resize
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))

            # Lighting novelty
            frame_proc, lighting = lighting_preprocess(frame)
            if lighting == "BRIGHT":
                
                status_text = ""

                if name != "Unknown":
                    current_state = get_current_state(name, name_to_id[name])
                    status_text = "IN" if current_state == 1 else "OUT"

                cv2.putText(
                    frame,
                    f"{name} | {confidence:.1f}% | {status_text}",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                cv2.imshow("Face Attendance", frame)
                cv2.waitKey(1)
                continue

            rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)

            if encodings:
                # Single-face assumption
                dists = face_recognition.face_distance(encs, encodings[0])
                idx = np.argmin(dists)
                dist = float(dists[idx])
                confidence = max(0.0, (1.0 - dist) * 100.0)

                if dist < 0.6:
                    name = names[idx]
                    sid = name_to_id[name]
                else:
                    name = "Unknown"

                # Draw bounding box + confidence
                top, right, bottom, left = boxes[0]
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame,
                    f"{name} | {confidence:.1f}%",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                # Temporal confidence
                validator.update(name, dist)
                valid, confirmed = validator.is_confident()

                if valid and confirmed != "Unknown":
                    # Fire ONLY once per appearance
                    if confirmed not in seen_latch:
                        if perform_liveness(cam):
                            state = get_current_state(confirmed, name_to_id[confirmed])

                            # State-based action
                            # If state == 0 => currently OUT => next is Punch-In (1)
                            # If state == 1 => currently IN  => next is Punch-Out (0)
                            log_attendance(confirmed, name_to_id[confirmed])

                            seen_latch.add(confirmed)

            else:
                # No face → reset latch (allows next re-entry)
                seen_latch.clear()

            cv2.imshow("Face Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["register", "recognize"])
    parser.add_argument("--name")
    parser.add_argument("--id")
    args = parser.parse_args()

    if args.mode == "register":
        if not args.name or not args.id:
            print("Usage: --mode register --name NAME --id ID")
        else:
            register(args.name, args.id)
    else:
        recognize()
