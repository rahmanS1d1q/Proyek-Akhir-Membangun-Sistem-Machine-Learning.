# Submission Sistem Machine Learning
**Nama:** Muhammad Rahman Shiddiq  
**Topik:** Sistem Machine Learning (Training, Monitoring, dan Workflow CI)

Repository ini berisi implementasi lengkap sistem machine learning yang mencakup:
- Proses training dan evaluasi model
- Logging dan monitoring model
- Workflow CI untuk retraining model menggunakan MLflow Project

---

## 📂 Struktur Proyek

Struktur folder pada proyek ini adalah sebagai berikut:

SMSML_Muhammad-Rahman-Shiddiq
├── Eksperimen_SML_Muhammad-Rahman-Shiddiq.txt
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   ├── DagsHub.txt
│   ├── namadataset_preprocessing/
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   ├── artifacts/
│   │   ├── classification_report.json
│   │   ├── confusion_matrix.png
│   │   ├── pr_curve.png
│   │   └── roc_curve.png
│   ├── screenshoot_artifak.jpg
│   └── screenshoot_dashboard.jpg
├── Monitoring dan Logging/
│   ├── inference.py
│   ├── prometheus.yml
│   ├── prometheus_exporter.py
│   ├── bukti_serving.jpg
│   ├── bukti monitoring Prometheus/
│   │   └── monitoring.jpg
│   └── bukti alerting Grafana/
│       ├── notifikasi.png
│       └── rules_.png
└── Workflow-CI.txt

---

## 🧠 Membangun Model

Folder **`Membangun_model/`** berisi proses utama training model machine learning.

### File penting:
- `modelling.py`  
  Script utama untuk melatih model machine learning menggunakan dataset hasil preprocessing serta melakukan logging menggunakan MLflow.

- `modelling_tuning.py`  
  Digunakan untuk proses tuning model.

- `requirements.txt`  
  Berisi dependency yang digunakan, antara lain:
  - pandas
  - numpy
  - scikit-learn
  - mlflow

### Dataset
Dataset yang digunakan merupakan hasil preprocessing dan disimpan pada folder:
Membangun_model/namadataset_preprocessing/

Dataset terdiri dari:
- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

---

## 📊 Artefak dan Evaluasi Model

Hasil evaluasi model disimpan pada folder **`artifacts/`**, meliputi:
- Classification report (`classification_report.json`)
- Confusion matrix
- ROC Curve
- Precision-Recall Curve

Artefak ini dihasilkan selama proses training dan evaluasi model menggunakan MLflow.

---

## 📈 Monitoring dan Logging

Folder **`Monitoring dan Logging/`** berisi implementasi monitoring sistem machine learning.

### Komponen utama:
- `inference.py`  
  Script untuk melakukan serving/inference model.

- `prometheus_exporter.py`  
  Digunakan untuk mengekspor metrik ke Prometheus.

- `prometheus.yml`  
  Konfigurasi Prometheus untuk monitoring metrik aplikasi.

### Bukti Monitoring:
- Monitoring menggunakan **Prometheus**
- Alerting dan visualisasi menggunakan **Grafana**
- Screenshot bukti monitoring dan alerting disertakan di dalam folder

---

## 🔁 Workflow CI

File **`Workflow-CI.txt`** menjelaskan konsep workflow CI yang digunakan untuk retraining model.

Workflow CI dirancang untuk:
- Menjalankan ulang training model secara otomatis
- Menggunakan pendekatan MLflow Project
- Mendukung integrasi dengan GitHub Actions

Workflow ini memastikan proses retraining dapat dilakukan secara konsisten dan reproducible.

---

## 🛠 Teknologi yang Digunakan

- Python
- Scikit-learn
- MLflow
- Prometheus
- Grafana
- GitHub Actions (Workflow CI)

---

## ✅ Kesimpulan

Proyek ini mengimplementasikan sistem machine learning end-to-end yang mencakup:
- Training dan evaluasi model
- Logging dan monitoring performa model
- Workflow CI untuk otomatisasi retraining

Seluruh komponen disusun untuk memenuhi kebutuhan submission akademik pada modul Sistem Machine Learning.

---

