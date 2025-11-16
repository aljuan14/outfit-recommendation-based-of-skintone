import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout  # Impor Dropout
import os

# --- TAHAP 1: PERSIAPAN DATA (MENGGUNAKAN SEMUA DATA) ---

# Langsung arahkan ke folder dataset LENGKAP Anda
DATASET_PATH = 'data_skintone'

# Cek apakah folder data asli ada
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Folder dataset lengkap tidak ditemukan di {DATASET_PATH}")

IMAGE_SIZE = (224, 224)
# Batch size yang lebih standar untuk dataset besar
BATCH_SIZE = 32

print(f"Memuat dataset lengkap dari: {DATASET_PATH}")

# Augmentasi dan Normalisasi Data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% validasi, 80% training
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Memuat Data Training & Validasi dari folder LENGKAP
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

class_labels = list(train_generator.class_indices.keys())
num_classes = len(class_labels)
print(f"Label Kelas (ditemukan {num_classes} kelas): {class_labels}")

# --- TAHAP 2: PEMBANGUNAN MODEL CNN (DENGAN DROPOUT) ---

base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # TAMBAHKAN LAPISAN DROPOUT UNTUK ANTI-OVERFITTING
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- TAHAP 3: PELATIHAN DAN EVALUASI ---

# Latih lebih lama karena datanya jauh lebih banyak
EPOCHS = 20

print(f"Memulai pelatihan model FINAL dengan {EPOCHS} epoch...")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

print("Evaluasi model final pada data validasi...")
loss, accuracy = model.evaluate(validation_generator)
print(f"Akurasi Model FINAL pada data validasi: {accuracy*100:.2f}%")

# Simpan model final
model_filename = 'skin_tone_classifier_final.h5'
# (Ganti .h5 menjadi .keras jika Anda ingin menggunakan format baru)
model.save(model_filename)
print(f"Model FINAL berhasil disimpan sebagai {model_filename}")
