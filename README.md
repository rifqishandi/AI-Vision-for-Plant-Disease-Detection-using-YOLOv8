# 🌱 AI Vision for Plant Disease Detection using YOLOv8
> **Penerapan IoT dan AI pada Precision Farming: Studi Kasus Deteksi Kesehatan Tanaman**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8_Classification-green?style=flat&logo=ultralytics)
![Status](https://img.shields.io/badge/Status-Active-success)
![Platform](https://img.shields.io/badge/Platform-IoT%20Edge-orange)

## 📖 Latar Belakang (Background)

Proyek ini bertujuan untuk mengembangkan sistem **Deteksi Kesehatan Tanaman** cerdas sebagai bagian dari solusi *Precision Farming*. Dalam pertanian konvensional, keterlambatan identifikasi penyakit sering kali menyebabkan kerugian panen yang signifikan.

Sistem ini mengintegrasikan **Kecerdasan Buatan (AI)** menggunakan algoritma **YOLOv8 (You Only Look Once)** dalam mode **Klasifikasi** untuk mendiagnosis penyakit tanaman secara cepat. Model ini dirancang agar ringan dan efisien, sehingga siap diimplementasikan pada arsitektur **Internet of Things (IoT)** berbasis *Edge Computing* (seperti Raspberry Pi) di lahan pertanian.

## 🚀 Fitur Utama

* **High Accuracy:** Mencapai akurasi **99.73%** pada dataset validasi PlantVillage.
* **AI Model:** Menggunakan arsitektur `yolov8-cls` yang dioptimalkan untuk klasifikasi cepat.
* **Real-time Ready:** Dirancang untuk inferensi cepat pada perangkat dengan sumber daya terbatas (*resource-constrained devices*).
* **IoT Integration:** Struktur output data siap dikirim melalui protokol ringan seperti MQTT.

## 🏗️ Arsitektur Sistem IoT

Sistem ini dirancang dengan arsitektur *Edge-Cloud Hybrid* untuk efisiensi data:

![Desain Arsitektur IoT](https://github.com/rifqishandi/AI-Vision-for-Plant-Disease-Detection-using-YOLOv8/raw/main/assets/Desain%20Arsitektur%20IoTT.PNG)

1.  **Device Layer:** Sensor kamera mengambil citra tanaman di lahan.
2.  **Edge Layer:** Model YOLOv8 berjalan lokal untuk mendeteksi penyakit secara instan.
3.  **Connectivity:** Hasil diagnosis dikirim via protokol komunikasi (WiFi/LoRa).
4.  **Cloud & User Layer:** Data disimpan di database dan ditampilkan pada Dashboard Petani.

## 📂 Struktur Direktori

```text
├── data/              # Folder penyimpanan dataset (Train/Val)
├── models/            # Menyimpan bobot model hasil training (best.pt)
├── predict.py         # Script Python untuk menjalankan prediksi/inferensi
├── requirements.txt   # Daftar library yang dibutuhkan (YOLO, Scikit-learn, dll)
└── README.md          # Dokumentasi proyek ini
