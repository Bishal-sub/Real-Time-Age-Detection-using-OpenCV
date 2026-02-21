# 🎂 Real-Time Age Detection using OpenCV

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time computer vision application that detects faces from a webcam feed and predicts the age range of each detected person using deep learning models.

This project uses OpenCV’s DNN module with pretrained Caffe and TensorFlow models to perform fast and efficient inference.

---

## 📌 Features

- 🎥 Real-time face detection using webcam
- 🧠 Age range prediction using deep learning
- ⚡ Fast inference using OpenCV DNN
- 🖼️ Bounding box with predicted age label
- 🧩 Lightweight and easy to run

---

## 🧠 Age Categories Predicted

The model predicts one of the following age ranges:

- (0-2)
- (4-6)
- (8-12)
- (15-20)
- (25-32)
- (38-43)
- (48-53)
- (60-100)

---

## 🛠️ Technologies Used

- Python 3.x
- OpenCV
- NumPy
- Caffe model (Age Detection)
- TensorFlow model (Face Detection)

---

## 📂 Project Structure

```
age-detection/
│
├── age_detection.py
├── age_deploy.prototxt
├── age_net.caffemodel
├── opencv_face_detector.pbtxt
├── opencv_face_detector_uint8.pb
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/age-detection.git
cd age-detection
```

### 2️⃣ (Optional but Recommended) Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have requirements.txt:

```bash
pip install opencv-python numpy
```

---

## ▶️ Usage

Run the script:

```bash
python age_detection.py
```

- Your webcam will open
- Press **Q** to exit the application

---

## 🔍 How It Works

1. Webcam captures live video frames.
2. Frames are converted into blobs.
3. Face detection model detects faces.
4. Each face is passed into the age prediction model.
5. Predicted age range is displayed above the bounding box.

---


## 🚀 Future Improvements

- Add gender detection
- Add emotion detection
- Save detection results
- Deploy as web app (Flask / FastAPI)
- Convert into mobile app
- Optimize model for better accuracy

---

## ⚠️ Notes

- Ensure all model files are in the same directory.
- Works best in good lighting conditions.
- Webcam permission must be enabled.
- Model files should not exceed GitHub size limits (100MB per file).



Your Name  
GitHub: https://github.com/bishal-sub


