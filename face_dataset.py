import cv2
import os

# Path untuk simpan dataset
dataset_path = "data_set_model"

# Minta nama orang untuk folder
person_name = input("Masukkan nama orang: ").strip()
person_folder = os.path.join(dataset_path, person_name)
os.makedirs(person_folder, exist_ok=True)

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # lebar
cap.set(4, 480)  # tinggi

count = 0
print("[INFO] Mulai pengambilan gambar, tekan 'ESC' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1
        # Simpan foto wajah
        img_path = os.path.join(person_folder, f"{count}.jpg")
        cv2.imwrite(img_path, gray[y:y+h, x:x+w])

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(count), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Dataset', frame)
    k = cv2.waitKey(100) & 0xff
    if k == 27 or count >= 30:  # ESC atau 30 foto
        break

print("[INFO] Dataset berhasil dibuat.")
cap.release()
cv2.destroyAllWindows()
