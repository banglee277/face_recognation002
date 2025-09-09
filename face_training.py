import cv2
import os
import numpy as np
from PIL import Image
import json

# ====== CONFIG ======
dataset_path = "data_set_model"
trainer_path = "trainer"

if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Simpan mapping nama ↔ ID
labels = {}
current_id = 0
faces = []
ids = []

print("🔧 Mulai proses training...")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue
    
    labels[current_id] = person_name
    print(f"👤 Training {person_name} dengan ID {current_id}")

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        try:
            gray_img = Image.open(img_path).convert("L")  # Grayscale
        except:
            continue
        img_numpy = np.array(gray_img, "uint8")

        detected_faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in detected_faces:
            faces.append(img_numpy[y:y+h, x:x+w])
            ids.append(current_id)

    current_id += 1

if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
    trainer_file = os.path.join(trainer_path, "trainer.yml")
    recognizer.save(trainer_file)

    with open(os.path.join(trainer_path, "labels.json"), "w") as f:
        json.dump(labels, f)

    print(f"✅ Training selesai! Model disimpan di: {trainer_file}")
    print(f"📊 Total orang: {len(labels)} | Total wajah: {len(faces)}")
else:
    print("⚠️ Tidak ada wajah yang terdeteksi. Training gagal.")
