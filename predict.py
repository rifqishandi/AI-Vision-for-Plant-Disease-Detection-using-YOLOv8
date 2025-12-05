from ultralytics import YOLO
import os
from PIL import Image

# --- KONFIGURASI PATH ---
MODEL_PATH = 'models/PlantVillagev1.pt'      # Model YOLO Anda
DATA_FOLDER = 'data/'                        # Folder berisi banyak gambar

# 1. Cek apakah model ada
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model tidak ditemukan di: {MODEL_PATH}")
    exit()

# 2. Cek folder data
if not os.path.exists(DATA_FOLDER):
    print(f"ERROR: Folder data tidak ditemukan: {DATA_FOLDER}")
    exit()

# 3. Load model
print(f"Memuat model dari: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# 4. Ambil semua file gambar dalam folder
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(SUPPORTED_EXT)]

if not image_files:
    print("ERROR: Tidak ada file gambar (.jpg/.png) di folder data/")
    exit()

print(f"\nDitemukan {len(image_files)} gambar. Memulai proses prediksi...\n")

# 5. Loop prediksi tiap gambar
for img_name in image_files:
    img_path = os.path.join(DATA_FOLDER, img_name)

    print("\n" + "="*60)
    print(f"MENGANALISIS GAMBAR: {img_name}")
    print("="*60)

    try:
        # Jalankan prediksi YOLO
        results = model(img_path, conf=0.5, verbose=False)
        result = results[0]
        probs = result.probs

        # Top-1 kelas
        top1_index = probs.top1
        top1_label = result.names[top1_index]
        top1_conf = probs.top1conf.item() * 100

        print(f"Status Tanaman (Kelas Terdeteksi): {top1_label}")
        print(f"Tingkat Keyakinan               : {top1_conf:.2f}%")

        # Top-5 probabilities
        print("\nTop 5 Probabilitas Kelas:")
        top5_indices = probs.top5
        for i, index in enumerate(top5_indices):
            label = result.names[index]
            confidence = probs.data[index].item() * 100
            print(f"  {i+1}. {label:<25} | {confidence:.2f}%")

        print("="*60)

    except Exception as e:
        print(f"ERROR saat memproses {img_name}: {e}")

print("\n\n*** Semua gambar selesai dianalisis. ***")
