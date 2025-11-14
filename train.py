from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Tentukan lokasi folder dataset Anda
# Asumsi: dataset Anda memiliki struktur folder seperti ini:
# /dataset_root/
# ├── Light/
# ├── Medium/
# └── Dark/
DATASET_PATH = 'path/to/your/kaggle/dataset'

# 2. Tentukan Hyperparameter
IMAGE_SIZE = (224, 224)  # Ukuran standar untuk Transfer Learning
BATCH_SIZE = 32         # Jumlah gambar yang diproses dalam satu waktu

# 3. Augmentasi dan Normalisasi Data
# Normalisasi: Mengubah nilai piksel dari 0-255 menjadi 0-1 (rescale=1./255)
# Augmentasi: Membuat variasi data (misalnya memutar, membalik) agar model lebih kuat (hanya untuk Training)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Menggunakan 20% data untuk Validasi
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# 4. Memuat dan Membagi Data (Training & Validation)
# Class_mode='categorical' karena ini adalah klasifikasi multi-kelas (Light, Medium, Dark)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # Data untuk pelatihan
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  # Data untuk validasi
)

# Mengetahui urutan label kelas:
class_labels = list(train_generator.class_indices.keys())
print(f"Label Kelas: {class_labels}")
# Output: ['Dark', 'Light', 'Medium'] (tergantung urutan abjad folder)


# 1. Muat Model Pre-trained (MobileNetV2)
# weights='imagenet': Menggunakan bobot yang sudah dilatih
# include_top=False: Menghilangkan lapisan klasifikasi terakhir, karena kita akan buat yang baru
base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),  # Ukuran input (224, 224, 3)
    include_top=False,
    weights='imagenet'
)

# 2. Freeze Lapisan Dasar
# Mencegah bobot (weights) dari MobileNetV2 berubah selama pelatihan
base_model.trainable = False

# 3. Buat Arsitektur Model Akhir (Sequential Model)
num_classes = len(class_labels)  # Jumlah kelas: 3 (Dark, Light, Medium)

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Mengubah output MobileNetV2 menjadi vektor
    Dense(128, activation='relu'),  # Lapisan tersembunyi
    Dense(num_classes, activation='softmax')  # Lapisan Output
])

# 4. Kompilasi Model
model.compile(
    optimizer='adam',
    # Fungsi loss standar untuk klasifikasi multi-kelas
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Tampilkan ringkasan model
model.summary()

# 1. Pelatihan Model
# Training akan memakan waktu tergantung spesifikasi komputer/GPU Anda
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=10  # Mulai dengan 10 epoch, bisa disesuaikan
)

# 2. Evaluasi Final (Opsional, tapi Direkomendasikan)
# Gunakan data test terpisah jika Anda memilikinya, atau generator validasi
loss, accuracy = model.evaluate(validation_generator)
print(f"Akurasi Model pada data validasi: {accuracy*100:.2f}%")

# 3. Simpan Model
model.save('skin_tone_classifier.h5')
print("Model berhasil disimpan sebagai skin_tone_classifier.h5")
