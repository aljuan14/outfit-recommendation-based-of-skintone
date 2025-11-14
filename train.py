import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os
import shutil
import glob

# --- TAHAP 0: PERSIAPAN DATASET 100 GAMBAR/KELAS ---

ORIGINAL_DATA_PATH = 'data_skintone'
# Mengganti nama folder subset baru
SUBSET_DATA_PATH = 'data_skintone_100'
# Mengubah jumlah gambar per kelas
NUM_IMAGES_PER_CLASS = 100

print(f"Mempersiapkan dataset 100 gambar/kelas di: {SUBSET_DATA_PATH}")

# Cek apakah folder data asli ada
if not os.path.exists(ORIGINAL_DATA_PATH):
    raise FileNotFoundError(
        f"Folder dataset asli tidak ditemukan di {ORIGINAL_DATA_PATH}")

# Dapatkan nama-nama kelas
class_names = [d for d in os.listdir(ORIGINAL_DATA_PATH) if os.path.isdir(
    os.path.join(ORIGINAL_DATA_PATH, d))]

for class_name in class_names:
    original_class_dir = os.path.join(ORIGINAL_DATA_PATH, class_name)
    subset_class_dir = os.path.join(SUBSET_DATA_PATH, class_name)

    # Buat folder kelas di dataset subset
    os.makedirs(subset_class_dir, exist_ok=True)

    # Temukan semua gambar (jpg, png, jpeg)
    image_files = glob.glob(os.path.join(original_class_dir, '*.jpg')) + \
        glob.glob(os.path.join(original_class_dir, '*.png')) + \
        glob.glob(os.path.join(original_class_dir, '*.jpeg'))

    # Ambil 100 gambar pertama
    images_to_copy = image_files[:NUM_IMAGES_PER_CLASS]

    # Salin gambar-gambar tersebut ke folder subset
    print(f"Menyalin {len(images_to_copy)} gambar untuk kelas: {class_name}")
    for img_path in images_to_copy:
        shutil.copy(img_path, subset_class_dir)

print("Dataset 100 gambar/kelas berhasil dibuat.")
print("-" * 30)

# --- TAHAP 1: PERSIAPAN DATA (MENGGUNAKAN DATASET 100) ---

DATASET_PATH = SUBSET_DATA_PATH
IMAGE_SIZE = (224, 224)
# Batch size bisa sedikit lebih besar sekarang
BATCH_SIZE = 16

# Augmentasi dan Normalisasi Data
datagen = ImageDataGenerator(
    rescale=1./255,
    # 20% validasi (20 gambar/kelas), 80% training (80 gambar/kelas)
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Memuat Data Training & Validasi dari folder SUBSET
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

# --- TAHAP 2: PEMBANGUNAN MODEL CNN (TRANSFER LEARNING) ---

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
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- TAHAP 3: PELATIHAN DAN EVALUASI ---

# Kita bisa tambah epoch sedikit karena data lebih banyak
EPOCHS = 10

print(f"Memulai pelatihan model dengan {EPOCHS} epoch...")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

print("Evaluasi model pada data validasi...")
loss, accuracy = model.evaluate(validation_generator)
print(f"Akurasi Model pada data validasi: {accuracy*100:.2f}%")

# Simpan model dengan nama baru
model_filename = 'skin_tone_classifier_100.h5'
model.save(model_filename)
print(f"Model uji coba berhasil disimpan sebagai {model_filename}")
