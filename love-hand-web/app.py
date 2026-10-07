from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, model_complexity=0,
    min_detection_confidence=0.65, min_tracking_confidence=0.55
)

PARTNER_NAME = "SHEILA"
BIRTHDAY_MESSAGE = "HAPPY BIRTHDAY SAYANG"

def landmark_dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def count_fingers(landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    count = sum(landmarks[tip].y < landmarks[pip].y - 0.02
                for tip, pip in zip(tips, pips))
    if landmark_dist(landmarks[4], landmarks[17]) > landmark_dist(landmarks[3], landmarks[17]) + 0.02:
        count += 1
    return count

@app.route("/")
def index():
    return render_template("index.html",
        partner_name=PARTNER_NAME, birthday_message=BIRTHDAY_MESSAGE)

@app.post("/process_frame")
def process_frame():
    if "frame" not in request.files:
        return jsonify(ok=False, error="frame tidak ditemukan"), 400
    data = request.files["frame"].read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(ok=False, error="frame tidak valid"), 400

    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    mode, finger_count, hand_x, hand_y = "scatter", None, None, None
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        finger_count = count_fingers(lm)
        mode = {0:"heart", 1:"zodiac", 2:"love", 3:"bday"}.get(finger_count, "scatter")
        ids = [0, 5, 9, 13, 17]
        hand_x = sum(lm[i].x for i in ids) / len(ids)
        hand_y = sum(lm[i].y for i in ids) / len(ids)

    return jsonify(ok=True, mode=mode, finger_count=finger_count,
                   hand_x=hand_x, hand_y=hand_y)

@app.get("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
