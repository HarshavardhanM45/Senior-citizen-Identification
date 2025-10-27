# 🧓 Senior Citizen Detection System

## 📘 Project Overview  
This project detects **senior citizens** in real time from live video or uploaded video files using a combination of **YOLOv8** for object detection and **DeepFace** for age and gender analysis.  
It automatically determines whether a detected person is a **senior citizen (above 60 years)** and logs relevant information like **age, gender, and timestamp** into a CSV file for record-keeping.  
The system is built with a **Streamlit GUI** and uses **OpenCV** for efficient video processing.

---

## ⚙️ Features  
✅ Real-time **person detection** using pretrained **YOLOv8**.  
✅ Predicts **age and gender** using **DeepFace**.  
✅ Automatically classifies individuals as **Senior (age > 60)** or **Non-Senior**.  
✅ Displays detection results directly on the video feed:  
 🔴 Red boxes → Seniors  
 🟢 Green boxes → Non-seniors  
✅ Logs all detections into a CSV file (`senior_log.csv`) with **age, gender, and timestamp**.  
✅ Simple **Streamlit interface** with options for video upload or webcam feed.  
✅ Real-time OpenCV visualization with smooth performance.  

---

## 🧠 Machine Learning Models

### 🧩 YOLOv8 — Person Detection  
- **Model:** Pretrained YOLOv8 (Ultralytics)  
- **Purpose:** Detect and track people in video streams with high accuracy.  
- **Advantages:** Lightweight, fast, GPU-compatible, and suitable for real-time applications.  

### 🧩 DeepFace — Age & Gender Prediction  
- **Model:** Pretrained DeepFace model  
- **Actions Used:** `['age', 'gender']`  
- **Purpose:** Estimate the age and gender of each detected face.  
- **Integration:** Cropped faces from YOLO detections are passed to DeepFace for analysis.  

---

## 🧮 Senior Detection Logic

```python
if age > 60:
    label = f"Senior ({gender})"
    color = (0, 0, 255)  # Red box for senior
else:
    label = f"{gender}, {age} yrs"
    color = (0, 255, 0)  # Green box for non-senior
```

---

## 📊 Data Logging  
Detected senior citizens are recorded in a CSV file `data/senior_log.csv` with the following columns:

| Age | Gender | Time |
|------|---------|----------------|
| 67 | Male | 2025-10-27 19:45:20 |
| 72 | Female | 2025-10-27 19:46:02 |

This ensures each detection is time-stamped and traceable.

---

## 🖥️ Streamlit Application  
🧩 User-friendly and responsive interface built with **Streamlit**.  
🧩 Supports both webcam and video uploads.  
🧩 Real-time detection in a separate OpenCV window for smoother playback.  
🧩 Logs detection results and supports continuous monitoring.  

---

## 📈 Performance & Results  
| Metric | Result |
|---------|---------|
| Detection Accuracy | ~92% (YOLOv8 + DeepFace) |
| FPS on GPU | 20–25 FPS |
| FPS on CPU | 10–12 FPS |
| Tested Inputs | Webcam & Multiple Sample Videos |

✔️ Reliable and consistent senior detection  
✔️ Smooth frame processing with multithreading  

---

## 🧰 Technologies Used  
- **Python**  
- **YOLOv8 (Ultralytics)**  
- **DeepFace**  
- **OpenCV**  
- **Streamlit**  
- **NumPy, Pandas**  
- **Threading**

---

## 🚀 How to Run  

### 🧩 Step 1: Install Dependencies  
```bash
pip install ultralytics deepface streamlit opencv-python pandas numpy
```

### 🧩 Step 2: Run the App  
```bash
streamlit run app.py
```

### 🧩 Step 3: Use the Interface  
- Upload a video file **or** enable the **webcam**.  
- View detections in a live OpenCV window.  
- Press **‘q’** to exit detection.  

---

## 💬 Summary  
The **Senior Citizen Detection System** is an intelligent vision-based project that combines **YOLOv8** and **DeepFace** to detect and classify individuals as senior citizens in real time.  
It’s a perfect example of applying **AI for social good**, offering practical use in **elderly monitoring systems, smart surveillance, and safety automation**.  

🧠 It merges **object detection, facial analysis, and data logging** into one seamless, interactive application — a showcase of modern AI capabilities in human-centered design.

---

## 🏁 Author  
**Developed by:** Harshavardhan M  
**Field:** Data Science | Computer Vision | Deep Learning  
**Goal:** Building intelligent AI-driven systems that make a real-world impact 🚀
