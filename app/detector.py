import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

class GestureDetector:
    def __init__(self, model_path, label_path, image_size=(28, 28), threshold=0.8):
        self.model = load_model(model_path)
        self.label_map = self._load_label_map(label_path)
        self.image_size = image_size
        self.threshold = threshold

    def _load_label_map(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def predict(self, image):
        # Pastikan gambar grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(image, self.image_size)
        img = img.astype("float32") / 255.0
        img = img.reshape(1, self.image_size[0], self.image_size[1], 1)

        preds = self.model.predict(img, verbose=0)[0]
        idx = np.argmax(preds)
        confidence = preds[idx]
        
        # --- TAMBAH BARIS INI UNTUK DEBUGGING ---
        print(f"All predictions: {preds}")
        print(f"Predicted index: {idx}, Label: {self.label_map.get(str(idx), '')}, Confidence: {confidence:.4f}")

        if confidence < self.threshold:
            return None

        label = self.label_map.get(str(idx), "")
        return label
