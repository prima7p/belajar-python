
import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections

# 1. Pastikan model hand_landmarker.task tersedia
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Mengunduh model hand_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Selesai mengunduh model.")

# 2. Inisialisasi HandLandmarker (MediaPipe Tasks API terbaru)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)
landmarker = vision.HandLandmarker.create_from_options(options)

# 3. Buka Webcam (Gunakan index 1 untuk kamera internal laptop)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses. Pastikan kamera terhubung dan tidak digunakan oleh aplikasi lain.")
    exit()

print("Program berjalan! Arahkan tangan Anda ke kamera.")
print("Tekan huruf 'q' pada keyboard untuk keluar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Gagal membaca webcam.")
        break

    # Balik gambar secara horizontal (mirror) agar gerakan terasa natural
    frame = cv2.flip(frame, 1)

    # Konversi frame BGR ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Deteksi landmark tangan
    result = landmarker.detect(mp_image)

    # Gambar titik-titik dan garis penghubung tangan jika terdeteksi
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS
            )

    # Tampilkan jendela video
    cv2.imshow("Hand Tracking - Python", frame)

    # Tekan 'q' untuk berhenti
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
