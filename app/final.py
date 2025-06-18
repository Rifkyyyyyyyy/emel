import cv2
import numpy as np
import mediapipe as mp
from detector import GestureDetector

# Inisialisasi detektor gesture
detector = GestureDetector(
    "models/cnn_model_mnist.h5",
    "models/label_map.json",
    image_size=(28, 28),
    threshold=0.8
)

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Tidak dapat mengakses webcam.")
    exit()

input_text = ""
last_prediction = None
cooldown = 20
cooldown_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame tidak terbaca.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    black_img = np.zeros((150, 600, 3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_coords) * w) - 20
            ymin = int(min(y_coords) * h) - 20
            xmax = int(max(x_coords) * w) + 20
            ymax = int(max(y_coords) * h) + 20

            # Pastikan ROI masih dalam frame
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            # Gambar kotak tangan
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

            roi = frame[ymin:ymax, xmin:xmax]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            if cooldown_counter == 0:
                try:
                    prediction = detector.predict(roi_gray)
                    if prediction and prediction != last_prediction:
                        input_text += prediction
                        last_prediction = prediction
                        cooldown_counter = cooldown
                except Exception as e:
                    print("🚨 Error prediksi:", e)

            if cooldown_counter > 0:
                cooldown_counter -= 1
            break

    # Tampilkan prediksi
    cv2.putText(
        black_img,
        input_text[-50:],
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Live Webcam", frame)
    cv2.imshow("Prediksi", black_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        input_text = ""
        last_prediction = None
        cooldown_counter = 0

cap.release()
cv2.destroyAllWindows()
