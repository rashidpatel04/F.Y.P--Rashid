import cv2
import random
import numpy as np
import pyttsx3
import threading
import time
from ultralytics import YOLO

# ==============================
# 🎯 YOLOv8 Setup
# ==============================
with open("utils/coco.txt", "r") as f:
    class_list = f.read().split("\n")

model = YOLO("weights/yolov8n.pt")

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
          for _ in range(len(class_list))]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera.")
    exit()

cap.set(3, 640)
cap.set(4, 480)
print("✅ Camera opened successfully.")

# ==============================
# ⚙️ Helper Functions
# ==============================
def get_position_and_distance(bb, frame_wid, frame_hyt):
    """Estimate direction and distance of an object."""
    x_center = (bb[0] + bb[2]) / 2
    area = (bb[2] - bb[0]) * (bb[3] - bb[1])

    if x_center < frame_wid / 3:
        pos = "left"
    elif x_center > 2 * frame_wid / 3:
        pos = "right"
    else:
        pos = "center"

    if area > 130000:
        dist = "very close"
    elif area > 75000:
        dist = "near"
    else:
        dist = "far"

    return pos, dist

# ==============================
# 🔊 Reliable Async Voice Engine
# ==============================
def speak(text):
    """Initialize pyttsx3 engine fresh each time for reliable speech."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 165)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)  # choose voice here
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"❌ Voice error: {e}")

def speak_async(text):
    """Run speech in a separate thread to not block detection."""
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# ==============================
# 🧩 Main Loop
# ==============================
last_announcements = {}
ANNOUNCE_GAP = 4  # seconds between same object alerts
DETECTION_CONF = 0.55

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    frame_hyt, frame_wid = frame.shape[:2]
    results = model.predict(source=[frame], conf=DETECTION_CONF, save=False, verbose=False)
    boxes = results[0].boxes
    current_time = time.time()

    active_objects = set()

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            clsID = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < DETECTION_CONF:
                continue

            bb = box.xyxy[0].cpu().numpy().astype(int)
            obj = class_list[clsID]
            pos, dist = get_position_and_distance(bb, frame_wid, frame_hyt)
            label = f"{obj} ({conf:.2f})"

            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), colors[clsID], 2)
            cv2.putText(frame, label, (bb[0], bb[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            active_objects.add((obj, pos, dist))

    # Trigger voice alert only for new or changed detections
    for obj, pos, dist in active_objects:
        message = f"{obj} on the {pos}, {dist}"
        last_time = last_announcements.get(obj, 0)
        if current_time - last_time > ANNOUNCE_GAP:
            print(f"🔊 {message}")
            speak_async(message)
            last_announcements[obj] = current_time

    cv2.imshow("Blind Assistance - Real Time", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting safely...")
        speak_async("System stopped successfully.")
        break

# ==============================
# 🧹 Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()
