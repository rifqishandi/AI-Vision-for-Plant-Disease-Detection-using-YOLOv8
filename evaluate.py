from ultralytics import YOLO
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    # -------------------------------------------------------
    # 1. KONFIGURASI
    # -------------------------------------------------------
    # Path ke model terbaik hasil training Anda
    model_path = 'runs/classify/train/weights/best.pt' 
    
    # Path ke folder dataset VALIDASI atau TEST Anda
    # Pastikan strukturnya: dataset/val/nama_kelas/gambar.jpg
    val_dir = 'PlantVillage_Split/val' 

    # Load Model
    print("🔄 Memuat model...")
    model = YOLO(model_path)

    # -------------------------------------------------------
    # 2. JALANKAN PREDIKSI (INFERENCE)
    # -------------------------------------------------------
    y_true = []  # Label Asli
    y_pred = []  # Label Prediksi
    class_names = [] # Nama Kelas (Penyakit)

    # Mendapatkan daftar nama kelas dari folder val
    class_names = sorted(os.listdir(val_dir))
    
    print("🚀 Sedang mengevaluasi data validation... (Mohon tunggu)")

    # Loop melalui setiap folder kelas
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        # Loop setiap gambar dalam folder kelas
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            # Prediksi menggunakan model YOLO
            # verbose=False agar terminal tidak penuh log
            results = model(img_path, verbose=False) 
            
            # Ambil index kelas dengan probabilitas tertinggi (Top-1)
            pred_idx = results[0].probs.top1
            
            # Simpan ke list
            y_true.append(class_idx)
            y_pred.append(pred_idx)

    # -------------------------------------------------------
    # 3. HITUNG METRIKS
    # -------------------------------------------------------
    print("\n" + "="*50)
    print("📊 HASIL EVALUASI MODEL")
    print("="*50)

    # A. Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Overall Accuracy: {acc:.2%}")
    print("-" * 50)

    # B. Classification Report (Precision, Recall, F1-Score)
    print("\n📋 Laporan Klasifikasi Per Kelas:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # -------------------------------------------------------
    # 4. VISUALISASI CONFUSION MATRIX
    # -------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediksi Model')
    plt.ylabel('Label Asli (Aktual)')
    plt.title('Confusion Matrix - Plant Disease Detection')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_model()
