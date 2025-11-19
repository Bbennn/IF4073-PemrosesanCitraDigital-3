# Tugas Besar 3 IF4073 - Pemrosesan Citra Digital
## Image Segmentation

### A. Conventional Image Segmentation
### B. DeepLearning-based Image Segmentation

#### Extract Dataset
##### 1\. Ekstrak archive.zip
##### 2\. Masuk ke folder hasil ekstraksi
##### 3\. Pindahkan semua isi file ke folder dataset\deeplearning

#### How to run
Ikuti langkah-langkah berikut
##### 1\. Buat Virtual Environemnt

Gunakan virtual environemnt untuk mengisolasi dependecies projek

```bash
python -m venv venv
```

##### 2\. Aktifkan Virtual Environment

Jalankan command yang sesuai dengan OS:

| Operating System | Command |
| :--- | :--- |
| **Windows (Command Prompt)** | `venv\Scripts\activate.bat` |
| **Windows (PowerShell)** | `venv\Scripts\Activate.ps1` |
| **macOS / Linux** | `source venv/bin/activate` |

##### 3\. Install Requirements
```bash
pip install -r requirement.txt --no-cache-dir
```

##### 4\. Jalankan Program
```bash
python main_gui.py
```