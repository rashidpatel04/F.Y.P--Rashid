import cv2
import random
import numpy as np
import threading
import time
import os
from gtts import gTTS
import pygame
from ultralytics import YOLO

# ==============================
# 🎯 YOLOv10 Setup
# ==============================
model = YOLO("yolov10n.pt")  # Lightweight YOLOv10 model
class_list = list(model.names.values())

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
          for _ in range(len(class_list))]

# ==============================
# 📸 Camera Setup
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera.")
    exit()
cap.set(3, 640)
cap.set(4, 480)
print("✅ Camera opened successfully.")

# ==============================
# 🧠 Helper: Position & Distance
# ==============================
def get_position_and_distance(bb, frame_wid, frame_hyt):
    x_center = (bb[0] + bb[2]) / 2
    area = (bb[2] - bb[0]) * (bb[3] - bb[1])
    pos = "left" if x_center < frame_wid / 3 else "right" if x_center > 2 * frame_wid / 3 else "center"
    if area > 130000:
        dist = "very close"
    elif area > 75000:
        dist = "near"
    else:
        dist = "far"
    return pos, dist

# ==============================
# 🔊 Speech System (Real-time)
# ==============================
pygame.mixer.init()
speech_lock = threading.Lock()

def speak_now(text):
    try:
        with speech_lock:
            filename = os.path.join(os.getenv("TEMP", "."), f"temp_{int(time.time()*1000)}.mp3")
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
            os.remove(filename)
    except Exception as e:
        print(f"❌ Voice error: {e}")

def speak_async(text):
    if threading.active_count() < 6:
        threading.Thread(target=speak_now, args=(text,), daemon=True).start()

# ==============================
# 🧩 Main Loop (Debounced)
# ==============================
last_announcements = {}
ANNOUNCE_GAP = 10  # seconds per object
DETECTION_CONF = 0.55

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame grab failed.")
        break

    frame_hyt, frame_wid = frame.shape[:2]
    results = model.predict(source=frame, conf=DETECTION_CONF, verbose=False)
    boxes = results[0].boxes
    current_time = time.time()

    current_objects = set()

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            clsID = int(box.cls[0])
            bb = box.xyxy[0].cpu().numpy().astype(int)
            obj = list(class_list)[clsID]
            pos, dist = get_position_and_distance(bb, frame_wid, frame_hyt)

            label = f"{obj} ({dist})"
            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), colors[clsID], 2)
            cv2.putText(frame, label, (bb[0], bb[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            current_objects.add((obj, pos, dist))

    # 🔊 Announce immediately but only once per 10 seconds per object name
    for obj, pos, dist in current_objects:
        last_time = last_announcements.get(obj, 0)
        if current_time - last_time > ANNOUNCE_GAP:
            message = f"{obj} on the {pos}, {dist}"
            print(f"🔊 {message}")
            speak_async(message)
            last_announcements[obj] = current_time

    cv2.imshow("Blind Assistance - Real-time YOLOv10", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting safely...")
        speak_async("System stopped successfully.")
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
