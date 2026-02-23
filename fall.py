import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import math
import time

# ---------------------------
# Load Models
# ---------------------------
model = YOLO("yolov8n.pt")   # single model for tracking

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

CONFIRM_FRAMES = 5
SNAPSHOT_COOLDOWN = 10  # seconds

person_states = {}

# ---------------------------
# Angle Function
# ---------------------------
def calculate_angle(a, b):
    return abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))

# ---------------------------
# Video Input
# ---------------------------
cap = cv2.VideoCapture("test_video.mp4")

print("Industrial Safety Monitoring Started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    results = model.track(frame, persist=True)

    machinery_boxes = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else None
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ---------------------------
            # MACHINERY DETECTION
            # ---------------------------
            if cls in [2, 5, 7]:  # car, bus, truck
                machinery_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (255, 0, 255), 2)
                cv2.putText(frame, "MACHINERY",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 255), 2)

            # ---------------------------
            # PERSON DETECTION
            # ---------------------------
            if cls == 0 and track_id is not None:

                if track_id not in person_states:
                    person_states[track_id] = {
                        "fall_counter": 0,
                        "is_fallen": False,
                        "snapshot_taken": False,
                        "last_snapshot_time": 0
                    }

                state = person_states[track_id]

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                width = x2 - x1
                height = y2 - y1
                ratio = width / height if height != 0 else 0

                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_crop)

                fall_detected = False
                injured = False
                danger_zone = False

                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    h, w, _ = person_crop.shape

                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    left_hip = landmarks[23]
                    right_hip = landmarks[24]
                    nose = landmarks[0]

                    shoulder_center = (
                        int((left_shoulder.x + right_shoulder.x) / 2 * w),
                        int((left_shoulder.y + right_shoulder.y) / 2 * h)
                    )

                    hip_center = (
                        int((left_hip.x + right_hip.x) / 2 * w),
                        int((left_hip.y + right_hip.y) / 2 * h)
                    )

                    body_angle = calculate_angle(shoulder_center, hip_center)

                    head_y = int(nose.y * h)
                    head_drop = head_y > hip_center[1]

                    score = 0
                    if ratio > 1.1:
                        score += 1
                    if body_angle < 55:
                        score += 1
                    if head_drop:
                        score += 1

                    if score >= 2:
                        state["fall_counter"] += 1
                    else:
                        state["fall_counter"] = 0

                    if state["fall_counter"] >= CONFIRM_FRAMES:
                        fall_detected = True
                        state["is_fallen"] = True
                    else:
                        state["is_fallen"] = False

                # ---------------------------
                # CHECK MACHINERY OVERLAP
                # ---------------------------
                for mx1, my1, mx2, my2 in machinery_boxes:
                    if x1 < mx2 and x2 > mx1 and y1 < my2 and y2 > my1:
                        danger_zone = True

                # ---------------------------
                # SNAPSHOT LOGIC (ONCE PER FALL)
                # ---------------------------
                if fall_detected and not state["snapshot_taken"]:
                    filename = f"fall_person_{track_id}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Fall captured for Person {track_id}")
                    state["snapshot_taken"] = True
                    state["last_snapshot_time"] = current_time

                # If person stands up → reset
                if not fall_detected:
                    state["snapshot_taken"] = False

                # ---------------------------
                # INJURY DETECTION
                # ---------------------------
                if fall_detected and danger_zone:
                    injured = True
                    cv2.putText(frame,
                                f"PERSON {track_id} INJURED / STUCK",
                                (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 3)

                # Display status
                if fall_detected:
                    cv2.putText(frame,
                                f"FALL DETECTED ID:{track_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

    cv2.imshow("S3 Industrial Safety Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Monitoring Stopped.")