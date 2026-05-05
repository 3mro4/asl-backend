import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import cv2
from collections import deque, Counter
import warnings

warnings.filterwarnings("ignore")

MAX_FRAMES = 30
CONFIDENCE_THRESHOLD = 0.5


def load_model(model_path: str, labels_path: str):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    prediction_fn = interpreter.get_signature_runner("serving_default")

    with open(labels_path, 'r') as f:
        sign_to_idx = json.load(f)
    idx_to_sign = {v: k for k, v in sign_to_idx.items()}

    return prediction_fn, idx_to_sign


def extract_keypoints(results):
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.full(468 * 3, np.nan)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.full(21 * 3, np.nan)
    pose = np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.full(33 * 3, np.nan)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.full(21 * 3, np.nan)

    all_kp = np.concatenate([face, lh, pose, rh])
    return np.reshape(all_kp, (543, 3))


class InferenceSession:
    def __init__(self):
        self.frames_buffer = deque(maxlen=MAX_FRAMES)
        self.prediction_history = deque(maxlen=10)
        self.no_hands_counter = 0
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def close(self):
        self.holistic.close()

    def process_frame(self, frame_bytes: bytes, model, classes) -> dict:
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"sign": "No Sign", "confidence": 0.0, "buffer_pct": 0}

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        hands_detected = bool(
            results.left_hand_landmarks or results.right_hand_landmarks
        )
        self.no_hands_counter = 0 if hands_detected else self.no_hands_counter + 1

        keypoints = extract_keypoints(results)
        self.frames_buffer.append(keypoints)

        pct = int(len(self.frames_buffer) / MAX_FRAMES * 100)

        if len(self.frames_buffer) < MAX_FRAMES:
            return {"sign": "buffering", "confidence": 0.0, "buffer_pct": pct}

        if self.no_hands_counter > 10:
            self.prediction_history.clear()
            return {"sign": "No Sign", "confidence": 0.0, "buffer_pct": 100}

        input_data = np.array(list(self.frames_buffer), dtype=np.float32)
        input_data = np.nan_to_num(input_data, nan=0.0)  # replace NaN with 0

        prediction = model(inputs=input_data)
        probabilities = prediction['outputs'][0]
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])

        if confidence > CONFIDENCE_THRESHOLD:
            sign = classes.get(best_idx, 'Unknown')
            self.prediction_history.append(sign)
            smoothed = Counter(self.prediction_history).most_common(1)[0][0]
            return {"sign": smoothed, "confidence": confidence, "buffer_pct": 100}

        return {"sign": "No Sign", "confidence": 0.0, "buffer_pct": 100}
