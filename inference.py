import numpy as np
import keras
import mediapipe as mp
from collections import deque, Counter
import warnings
import cv2
import base64

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────
MAX_FRAMES = 96
CONFIDENCE_THRESHOLD = 0.60

SELECTED_LANDMARKS = (
    list(range(0, 468, 2)) +
    list(range(468, 489)) +
    [500, 501, 502, 503, 504, 505] +
    list(range(522, 543))
)
N_SELECTED = len(SELECTED_LANDMARKS)
NOSE_IDX = 0
L_SHOULDER_IDX = 255
R_SHOULDER_IDX = 256

# ── Load model once at startup ─────────────────────────────
def load_model(model_path: str, classes_path: str):
    original_dense_init = keras.layers.Dense.__init__
    def patched_dense_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        original_dense_init(self, *args, **kwargs)
    keras.layers.Dense.__init__ = patched_dense_init

    model = keras.models.load_model(model_path, compile=False)
    classes = np.load(classes_path, allow_pickle=True)
    return model, classes

# ── Feature extraction (unchanged from your script) ────────
def extract_kaggle_format(results):
    frame_data = np.zeros((543, 3), dtype=np.float32)
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            frame_data[i] = [lm.x, lm.y, lm.z]
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            frame_data[468 + i] = [lm.x, lm.y, lm.z]
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            frame_data[489 + i] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            frame_data[522 + i] = [lm.x, lm.y, lm.z]
    return frame_data[SELECTED_LANDMARKS]

def engineer_live_window(features):
    centered_scaled = np.zeros((MAX_FRAMES, N_SELECTED, 3), dtype=np.float32)
    deltas = np.zeros((MAX_FRAMES, N_SELECTED, 3), dtype=np.float32)
    frame_is_valid = np.zeros(MAX_FRAMES, dtype=bool)
    last_valid_anchor = np.zeros(3, dtype=np.float32)
    anchor_initialized = False

    for t in range(MAX_FRAMES):
        frame = features[t]
        valid_mask = np.any(frame != 0, axis=1)
        if not np.any(valid_mask): continue
        if valid_mask[NOSE_IDX]:
            anchor = frame[NOSE_IDX].copy()
            last_valid_anchor = anchor
            anchor_initialized = True
        elif anchor_initialized:
            anchor = last_valid_anchor
        else:
            continue
        anchored = np.zeros_like(frame)
        anchored[valid_mask] = frame[valid_mask] - anchor
        if valid_mask[L_SHOULDER_IDX] and valid_mask[R_SHOULDER_IDX]:
            scale = np.linalg.norm(anchored[L_SHOULDER_IDX] - anchored[R_SHOULDER_IDX])
        else:
            scale = np.sqrt(np.mean(anchored[valid_mask] ** 2))
        scale = max(float(scale), 1e-3)
        centered_scaled[t, valid_mask] = anchored[valid_mask] / scale
        frame_is_valid[t] = True

    for t in range(1, MAX_FRAMES):
        if frame_is_valid[t] and frame_is_valid[t - 1]:
            deltas[t] = centered_scaled[t] - centered_scaled[t - 1]

    engineered = np.hstack([
        centered_scaled.reshape(MAX_FRAMES, -1),
        deltas.reshape(MAX_FRAMES, -1)
    ])
    return engineered.astype(np.float32)

# ── Per-connection session state ───────────────────────────
class InferenceSession:
    def __init__(self):
        self.frames_buffer = deque(maxlen=MAX_FRAMES)
        self.prediction_history = deque(maxlen=15)
        self.no_hands_counter = 0
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def close(self):
        self.holistic.close()

    def process_frame(self, frame_bytes: bytes, model, classes) -> dict:
        # Decode base64 frame from browser
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"sign": "No Sign", "confidence": 0.0, "status": "error"}

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        hands_detected = bool(
            results.left_hand_landmarks or results.right_hand_landmarks
        )
        self.no_hands_counter = 0 if hands_detected else self.no_hands_counter + 1

        keypoints = extract_kaggle_format(results)
        self.frames_buffer.append(keypoints)

        if len(self.frames_buffer) < MAX_FRAMES:
            pct = int(len(self.frames_buffer) / MAX_FRAMES * 100)
            return {"sign": "buffering", "confidence": 0.0, "buffer_pct": pct}

        if self.no_hands_counter > 10:
            self.prediction_history.clear()
            return {"sign": "No Sign", "confidence": 0.0, "buffer_pct": 100}

        raw_seq = np.array(self.frames_buffer)
        engineered = engineer_live_window(raw_seq)
        model_input = np.expand_dims(engineered, axis=0)

        preds = model.predict(model_input, verbose=0)[0]
        best_idx = int(np.argmax(preds))
        confidence = float(preds[best_idx])

        if confidence > CONFIDENCE_THRESHOLD:
            self.prediction_history.append(classes[best_idx])
            sign = Counter(self.prediction_history).most_common(1)[0][0]
            return {"sign": sign, "confidence": confidence, "buffer_pct": 100}

        return {"sign": "No Sign", "confidence": 0.0, "buffer_pct": 100}