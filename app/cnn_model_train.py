import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# Setup path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '../data')
model_dir = os.path.join(base_dir, '../models')
output_dir = os.path.join(base_dir, '../outputs')

# Buat folder jika belum ada
os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load dataset
train_df = pd.read_csv(os.path.join(data_dir, "sign_mnist_train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "sign_mnist_test.csv"))

# Buang label 25 (Z) karena perlu gerakan, tidak bisa direpresentasikan statis
train_df = train_df[train_df['label'] != 25]
test_df = test_df[test_df['label'] != 25]

# Pisahkan fitur dan label
train_y = train_df['label'].values
train_x = train_df.drop('label', axis=1).values
test_y = test_df['label'].values
test_x = test_df.drop('label', axis=1).values

# Normalisasi dan reshape
train_x = train_x.reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_x = test_x.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encoding
num_classes = train_y.max() + 1  # aman walaupun ada label hilang
train_y = to_categorical(train_y, num_classes)
test_y = to_categorical(test_y, num_classes)

# Split validasi
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)

# Bangun model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Simpan model terbaik
model_path = os.path.join(model_dir, "cnn_model_mnist.h5")
checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1, mode="max")

# Training
history = model.fit(
    train_x, train_y,
    validation_data=(val_x, val_y),
    batch_size=128,
    epochs=15,
    callbacks=[checkpoint],
    verbose=1
)

# Evaluasi
score = model.evaluate(test_x, test_y, verbose=0)
print(f"Test Accuracy: {score[1]*100:.2f}%")

# Plot akurasi dan simpan
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(output_dir, "acc_loss_plot.png"))
plt.close()
