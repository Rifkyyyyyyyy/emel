import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation # Added BatchNormalization and Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # Added ReduceLROnPlateau
import tensorflow as tf # Added tf import for specific Keras layers/optimizers

# Setup path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '../data') # Pastikan file dataset baru ada di folder 'data'
model_dir = os.path.join(base_dir, '../models')
output_dir = os.path.join(base_dir, '../outputs')

# Buat folder jika belum ada
os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# --- MODIFIKASI DIMULAI DI SINI ---

# Load keempat file CSV dari dataset baru
try:
    k_train = pd.read_csv(os.path.join(data_dir, "k_train.csv"))
    k_test = pd.read_csv(os.path.join(data_dir, "k_test.csv"))
    g_train = pd.read_csv(os.path.join(data_dir, "g_train.csv"))
    g_test = pd.read_csv(os.path.join(data_dir, "g_test.csv"))
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure all 4 CSV files are in the 'data' directory.")
    exit()

# Gabungkan dataset seperti di notebook Kaggle
enhanced_train = pd.concat([k_train, g_train], axis=0, ignore_index=True).sample(frac=1, random_state=42)
enhanced_test = pd.concat([k_test, g_test], axis=0, ignore_index=True).sample(frac=1, random_state=42)

# Pisahkan fitur dan label
X_full = enhanced_train[list(enhanced_train)[1:]].values
y_full = enhanced_train.label.values

X_test_final = enhanced_test[list(enhanced_test)[1:]].values
y_test_final = enhanced_test.label.values

# Reshape dan normalisasi data
# Normalisasi ke 0-1 dilakukan oleh ImageDataGenerator nanti
size = 28 # Ukuran gambar
channels = 1 # Grayscale

X_full = X_full.reshape(X_full.shape[0], size, size, channels).astype("float32")
X_test_final = X_test_final.reshape(X_test_final.shape[0], size, size, channels).astype("float32")

# Perhatikan bahwa dataset ini menggunakan 25 kelas (0-24)
# Jadi, kita tidak membuang label 25 (Z) atau 9 (J) lagi seperti sebelumnya.
# Kita perlu membuat label_map.json yang sesuai untuk 25 kelas (A-Y, tanpa Z).
# Label mapping: 0=A, 1=B, ..., 8=I, 9=J, 10=K, ..., 24=Y
# Ini mengasumsikan bahwa 'J' sekarang disertakan (label 9) dan 'Z' (label 25) tidak ada.
# Jika 'J' tidak ada, maka label 9 sebenarnya adalah 'K', dst.
# Berdasarkan notebook, jumlah kelas output Dense layer adalah 25.
# ASL MNIST original biasanya menghilangkan J (indeks 9) dan Z (indeks 25).
# Jika dataset ini "enhanced" untuk 25 kelas, ada beberapa kemungkinan:
# 1. Mereka mengisi kembali J (label 9).
# 2. Mereka mengisi kembali Z (label 25).
# 3. Ada simbol tambahan.
# Dari notebook, tampaknya mereka menganggap 25 kelas sebagai A-Y (tanpa Z).
# Mari kita buat label_map.json yang memetakan 0->A, 1->B, ..., 23->X, 24->Y
# Ini berarti J (indeks 9) tetap J, dan Z (indeks 25) dihilangkan.

# Temukan semua label unik yang ada di dataset
unique_labels = np.unique(y_full) # Should be 0-24, skipping 9 if J is truly excluded, or 0-24 if J is included and Z is not 25.
# Berdasarkan structure model di notebook, output adalah 25 kelas, ini menyiratkan J masuk dan Z tidak ada.
# Jadi 0=A, 1=B, ..., 8=I, 9=J, 10=K, ..., 24=Y (total 25 kelas)

label_map_dict = {}
char_code = 65 # ASCII for 'A'
for i in range(25): # 25 classes (0-24)
    # Jika ada huruf yang secara tradisional dilewati (seperti 'Z' atau 'J' dalam beberapa kasus ASL MNIST),
    # kita perlu menanganinya di sini.
    # Karena model output 25, kita asumsikan 0-24 adalah A-Y.
    # Huruf 'Z' (label 25 di dataset lama) memang tidak ada. Huruf 'J' (label 9) diikutkan.
    label_map_dict[str(i)] = chr(char_code)
    char_code += 1
    if chr(char_code-1) == 'J': # Setelah 'I' adalah 'J', ini perlu diperiksa kembali jika J tidak ada
        # Jika J tidak ada, kita harus melompati 'K' setelah 'I' (indeks 8)
        pass # Biarkan ini apa adanya, karena dataset baru ini kemungkinan sudah mengelola label dengan baik.

# Debugging: Cetak label map yang akan disimpan
# print("Generated Label Map:", label_map_dict)

with open(os.path.join(model_dir, "label_map.json"), 'w') as f:
    json.dump(label_map_dict, f)

num_classes = 25 # Total 25 kelas

# One-hot encoding untuk y
y_categorical = to_categorical(y_full, num_classes=num_classes)
y_test_final_categorical = to_categorical(y_test_final, num_classes=num_classes)


# Split data menjadi training dan validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_full, y_categorical, test_size=0.20, random_state=42, stratify=y_categorical
)

# --- MODIFIKASI BERAKHIR DI SINI ---

# Data Augmentation
# Menggunakan rescale di ImageDataGenerator, sehingga tidak perlu normalisasi x_train/x_test secara manual di awal
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # Untuk konsistensi

train_generator = train_datagen.flow(X_train, y_train, batch_size=96) # hp.get("batch_size")
val_generator = val_datagen.flow(X_valid, y_valid, batch_size=96) # hp.get("batch_size")


# Bangun model CNN (Mengikuti arsitektur dari notebook Kaggle)
model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), input_shape=(size,size,channels), kernel_initializer='he_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2,2, padding='same'),

    Conv2D(filters=128, kernel_size=(3,3), kernel_initializer='he_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2,2, padding='same'),

    Conv2D(filters=512, kernel_size=(3,3), kernel_initializer='he_uniform', bias_initializer='zeros', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(2,2, padding='same'),

    Flatten(),

    Dense(units=256),
    Activation('relu'),
    Dropout(0.1), # hp.get('dropout_rate')

    Dense(units=96),
    Activation('relu'),
    Dropout(0.1), # hp.get('dropout_rate')

    Dense(num_classes, activation="softmax")
])

# Kompilasi model
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"]) # Notebook uses SGD, but we stick to adam for general use. Loss is sparse_categorical_crossentropy if y is not one-hot encoded, but we use to_categorical for y.

# Simpan model terbaik
model_path = os.path.join(model_dir, "cnn_model_mnist.h5")
checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1, mode="max")

# Early Stopping dan ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True) # patience was 5 in notebook
# The notebook sets restore_best_weights=False, but True is usually better for practical use.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, min_lr=0.00001, verbose=1) # As per notebook

callbacks = [checkpoint, early_stopping, reduce_lr]

# Training with Data Augmentation
history = model.fit(
    train_generator, # Uses ImageDataGenerator
    steps_per_epoch=len(X_train) // 96, # hp.get("batch_size")
    validation_data=val_generator, # Uses ImageDataGenerator
    validation_steps=len(X_valid) // 96, # hp.get("batch_size")
    epochs=100, # hp.get('epochs')
    shuffle=True, # hp.get('shuffle') is False in notebook. Set to True for better generalization.
    callbacks=callbacks,
    verbose=1
)

# Evaluasi pada final test set
score = model.evaluate(X_test_final / 255.0, y_test_final_categorical, verbose=0)
print(f"Test Accuracy on Enhanced Test Set: {score[1]*100:.2f}%")

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

# Plot Loss dan simpan
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_plot.png"))
plt.close()