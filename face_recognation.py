import cv2
import os
import json
import requests
import time
from datetime import datetime
import locale

# ================== LOCALE INDONESIA ==================
try:
    locale.setlocale(locale.LC_TIME, "id_ID.utf8")
except:
    locale.setlocale(locale.LC_TIME, "Indonesian_indonesia")

# ================== TELEGRAM ==================
TOKEN = "8325775090:AAH5pCRxOCcoX5rLuqNEJGj6womPjkfuZZg"
CHAT_ID = "7033858750"
URL_SEND_MESSAGE = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
URL_SEND_PHOTO   = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

# ================== PATHS =====================
dataset_path   = "data_set_model"
trainer_yml    = os.path.join("trainer", "trainer.yml")
labels_path    = os.path.join("trainer", "labels.json")
haar_file      = "haarcascade_frontalface_default.xml"

# Pastikan trainer.yml ada
if not os.path.exists(trainer_yml):
    raise FileNotFoundError("trainer/trainer.yml tidak ditemukan. Jalankan training dulu.")

# Buat labels.json jika belum ada
if not os.path.exists(labels_path):
    os.makedirs("trainer", exist_ok=True)
    names = sorted(
        [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    )
    id2name = {str(i): name for i, name in enumerate(names)}
    with open(labels_path, "w") as f:
        json.dump(id2name, f, ensure_ascii=False, indent=2)

# Load mapping id -> nama
with open(labels_path, "r") as f:
    id2name = json.load(f)

# ================== MODEL =====================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_yml)

if os.path.exists(haar_file):
    face_cascade = cv2.CascadeClassifier(haar_file)
else:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ================== PARAMETER =================
CONF_THRESH = 50
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 5
EMPTY_DELAY = 5  # detik

# ================== CAMERA ====================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

last_status = None
last_seen_time = time.time()

# ================== FUNGSI NOTIF ==============
def kirim_notif(nama, frame):
    ts = datetime.now().strftime("%A, %d-%m-%Y %H:%M:%S")
    text = f"🚪 Ruangan D II .01 telah digunakan oleh {nama} pada {ts}"
    try:
        requests.post(URL_SEND_MESSAGE, data={"chat_id": CHAT_ID, "text": text}, timeout=5)
        img_path = "detected_face.jpg"
        cv2.imwrite(img_path, frame)
        with open(img_path, "rb") as img:
            requests.post(URL_SEND_PHOTO, data={"chat_id": CHAT_ID}, files={"photo": img}, timeout=10)
    except Exception as e:
        print("[WARN] Gagal kirim Telegram:", e)

def kirim_notif_kosong(frame):
    ts = datetime.now().strftime("%A, %d-%m-%Y %H:%M:%S")
    text = f"📭 Ruangan D II .01 kosong pada {ts}"
    try:
        requests.post(URL_SEND_MESSAGE, data={"chat_id": CHAT_ID, "text": text}, timeout=5)
        img_path = "ruangan_kosong.jpg"
        cv2.imwrite(img_path, frame)
        with open(img_path, "rb") as img:
            requests.post(URL_SEND_PHOTO, data={"chat_id": CHAT_ID}, files={"photo": img}, timeout=10)
    except Exception as e:
        print("[WARN] Gagal kirim Telegram:", e)

# ================== LOOP ======================
while True:
    ok, frame = cap.read()
    if not ok:
        print("Gagal membuka kamera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS)

    if len(faces) > 0:
        last_seen_time = time.time()
        if last_status != "occupied":
            last_status = "occupied"  # Status berubah dari kosong ke terisi

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label_id, confidence = recognizer.predict(face_roi)

            if confidence <= CONF_THRESH:
                nama = id2name.get(str(label_id), "Tidak dikenal")
                color = (0, 255, 0)
                cv2.putText(frame, f"{nama}", (x+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Hanya kirim notif saat status berubah
                if last_status == "occupied":
                    kirim_notif(nama, frame)

            else:
                color = (0, 0, 255)
                cv2.putText(frame, "Tidak dikenal", (x+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    else:
        if time.time() - last_seen_time > EMPTY_DELAY and last_status != "empty":
            kirim_notif_kosong(frame)
            last_status = "empty"  # Status berubah ke kosong

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC keluar
        break

cap.release()
cv2.destroyAllWindows()
