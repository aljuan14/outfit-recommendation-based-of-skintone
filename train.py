import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collections
import os

# ============================================================
# KONFIGURASI
# ============================================================
DATASET_PATH = 'data_skintone'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10

# ============================================================
# TAHAP 1: VALIDASI DAN PERSIAPAN DATA
# ============================================================
print("="*60)
print("TAHAP 1: PERSIAPAN DATA")
print("="*60)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Folder dataset tidak ditemukan di {DATASET_PATH}")

print(f"✓ Dataset ditemukan: {DATASET_PATH}")

# Augmentasi dan Normalisasi Data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

# Generator untuk Training
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Generator untuk Validasi
validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_labels = list(train_generator.class_indices.keys())
num_classes = len(class_labels)

print(f"\n✓ Jumlah kelas: {num_classes}")
print(f"✓ Label kelas: {class_labels}")
print(f"✓ Total gambar training: {train_generator.samples}")
print(f"✓ Total gambar validasi: {validation_generator.samples}")

# Cek distribusi kelas
print("\n" + "="*60)
print("DISTRIBUSI KELAS")
print("="*60)

class_counts = collections.Counter(train_generator.classes)
for class_idx, count in sorted(class_counts.items()):
    percentage = (count / train_generator.samples) * 100
    print(
        f"{class_labels[class_idx]:20s}: {count:5d} gambar ({percentage:.1f}%)")

# Hitung class weights untuk data yang tidak seimbang
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\n✓ Class weights dihitung untuk mengatasi data imbalance")

# ============================================================
# TAHAP 2: PEMBANGUNAN MODEL
# ============================================================
print("\n" + "="*60)
print("TAHAP 2: PEMBANGUNAN MODEL")
print("="*60)

# Base model MobileNetV2
base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Arsitektur model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Model berhasil dibuat")
model.summary()

# ============================================================
# TAHAP 3: TRAINING AWAL (TRANSFER LEARNING)
# ============================================================
print("\n" + "="*60)
print("TAHAP 3: TRAINING AWAL (TRANSFER LEARNING)")
print("="*60)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model_initial.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print(f"Memulai training dengan {INITIAL_EPOCHS} epochs maksimal...")
print("(Training akan berhenti lebih awal jika tidak ada peningkatan)")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# ============================================================
# TAHAP 4: FINE-TUNING
# ============================================================
print("\n" + "="*60)
print("TAHAP 4: FINE-TUNING (UNFREEZE BEBERAPA LAYER)")
print("="*60)

# Unfreeze layer terakhir dari base model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(
    f"✓ Layer yang di-unfreeze: {sum([1 for layer in base_model.layers if layer.trainable])}")

# Compile ulang dengan learning rate lebih kecil
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks untuk fine-tuning
callbacks_fine = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model_finetuned.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )
]

print(f"\nMemulai fine-tuning dengan {FINE_TUNE_EPOCHS} epochs maksimal...")

history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks_fine,
    class_weight=class_weight_dict
)

# ============================================================
# TAHAP 5: EVALUASI MODEL
# ============================================================
print("\n" + "="*60)
print("TAHAP 5: EVALUASI MODEL")
print("="*60)

# Evaluasi pada data validasi
loss, accuracy = model.evaluate(validation_generator)
print(f"\n✓ Loss: {loss:.4f}")
print(f"✓ Accuracy: {accuracy*100:.2f}%")

# Prediksi untuk confusion matrix dan classification report
print("\nMembuat prediksi untuk evaluasi detail...")
validation_generator.reset()
y_pred = model.predict(validation_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# ============================================================
# TAHAP 6: VISUALISASI
# ============================================================
print("\n" + "="*60)
print("TAHAP 6: MEMBUAT VISUALISASI")
print("="*60)

# Gabungkan history dari initial training dan fine-tuning
total_acc = history.history['accuracy'] + history_fine.history['accuracy']
total_val_acc = history.history['val_accuracy'] + \
    history_fine.history['val_accuracy']
total_loss = history.history['loss'] + history_fine.history['loss']
total_val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# Plot 1: Training History
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(total_acc, label='Train Accuracy', linewidth=2)
plt.plot(total_val_acc, label='Val Accuracy', linewidth=2)
plt.axvline(x=len(history.history['accuracy']),
            color='r', linestyle='--', label='Fine-tuning Start')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(total_loss, label='Train Loss', linewidth=2)
plt.plot(total_val_loss, label='Val Loss', linewidth=2)
plt.axvline(x=len(history.history['loss']), color='r',
            linestyle='--', label='Fine-tuning Start')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.bar(class_labels, [class_counts[i] for i in range(num_classes)])
plt.xlabel('Kelas')
plt.ylabel('Jumlah Gambar')
plt.title('Distribusi Data Training')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✓ Training history disimpan: training_history.png")

# Plot 2: Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels,
            cbar_kws={'label': 'Jumlah Prediksi'})
plt.title('Confusion Matrix - Skin Tone Classifier', fontsize=16, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix disimpan: confusion_matrix.png")

# Plot 3: Per-class Accuracy
plt.figure(figsize=(12, 6))
per_class_acc = cm.diagonal() / cm.sum(axis=1)
bars = plt.bar(class_labels, per_class_acc * 100,
               color='skyblue', edgecolor='navy')
plt.xlabel('Kelas', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Accuracy per Kelas', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.ylim([0, 100])
plt.grid(axis='y', alpha=0.3)

# Tambahkan nilai di atas setiap bar
for bar, acc in zip(bars, per_class_acc):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Per-class accuracy disimpan: per_class_accuracy.png")

plt.show()

# ============================================================
# TAHAP 7: MENYIMPAN MODEL FINAL
# ============================================================
print("\n" + "="*60)
print("TAHAP 7: MENYIMPAN MODEL FINAL")
print("="*60)

model_filename = 'skin_tone_classifier_final.keras'
model.save(model_filename)
print(f"✓ Model final disimpan: {model_filename}")

# Simpan juga class labels
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)
print("✓ Class labels disimpan: class_labels.json")

# Simpan model summary ke file teks
with open('model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print("✓ Model summary disimpan: model_summary.txt")

# ============================================================
# SELESAI
# ============================================================
print("\n" + "="*60)
print("✓✓✓ TRAINING SELESAI! ✓✓✓")
print("="*60)
print(f"\nHasil Akhir:")
print(f"- Accuracy: {accuracy*100:.2f}%")
print(f"- Loss: {loss:.4f}")
print(f"\nFile yang dihasilkan:")
print(f"1. {model_filename} (Model final)")
print(f"2. best_model_initial.keras (Model terbaik sebelum fine-tuning)")
print(f"3. best_model_finetuned.keras (Model terbaik setelah fine-tuning)")
print(f"4. training_history.png (Grafik training)")
print(f"5. confusion_matrix.png (Confusion matrix)")
print(f"6. per_class_accuracy.png (Accuracy per kelas)")
print(f"7. class_labels.json (Label kelas)")
print(f"8. model_summary.txt (Arsitektur model)")
print("\n" + "="*60)
