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

* **AI Model:** Menggunakan `yolov8n-cls` (Nano) yang sangat ringan dan cepat.
* **Real-time Ready:** Dioptimalkan untuk inferensi cepat pada perangkat dengan sumber daya terbatas (*resource-constrained devices*).

## 📂 Struktur Direktori

```text
├── data/              # Sampel data atau folder dataset (split train/val)
├── models/            # Menyimpan bobot model hasil training (best.pt, .onnx)
├── predict.py         # Script utama untuk menjalankan prediksi pada gambar
├── requirements.txt   # Daftar dependensi library Python
└── README.md          # Dokumentasi proyek
