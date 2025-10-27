import streamlit as st
import cv2
from ultralytics import YOLO
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import tempfile
import os
import threading

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Senior Citizen Detection",
    page_icon="🧓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Title & Info
# -------------------------------
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🧓 Senior Citizen Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Upload a video or use webcam. Detection will be shown in a separate OpenCV window.</p>", unsafe_allow_html=True)
st.write("---")

# -------------------------------
# Sidebar Options
# -------------------------------
st.sidebar.header("Detection Settings")
use_webcam = st.sidebar.checkbox("Use Webcam")
video_file = st.sidebar.file_uploader("Upload Video", type=["mp4","avi","mov"])
skip_frames = st.sidebar.slider("Skip Frames (for faster detection)", min_value=1, max_value=10, value=3)

# -------------------------------
# Load YOLO model
# -------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("yolov8.pt")  # make sure the file is in the same folder

model = load_yolo()

# -------------------------------
# CSV Logging
# -------------------------------
if not os.path.exists("data"):
    os.makedirs("data")
LOG_FILE = "data/senior_log.csv"

def log_data(age, gender):
    df = pd.DataFrame({
        "Age":[age],
        "Gender":[gender],
        "Time":[datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    })
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)

# -------------------------------
# Video Processing Function
# -------------------------------
def process_video(video_path=None, use_webcam=False, skip_frames=3):
    cap = cv2.VideoCapture(0 if use_webcam else video_path)
    frame_count = 0
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        frame = cv2.resize(frame, (640,480))

        # Only process every N-th frame
        if frame_count % skip_frames == 0:
            results = model(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

            for (x1, y1, x2, y2) in boxes:
                x1, y1 = max(0,x1), max(0,y1)
                x2, y2 = min(frame.shape[1]-1,x2), min(frame.shape[0]-1,y2)
                cropped_face = frame[y1:y2, x1:x2]
                try:
                    analysis = DeepFace.analyze(cropped_face,
                                                actions=['age','gender'],
                                                enforce_detection=False)
                    age = int(analysis[0]['age'])
                    gender = analysis[0]['dominant_gender']

                    if age > 60:
                        label = f"Senior ({gender})"
                        log_data(age, gender)
                        color = (0,0,255)
                    else:
                        label = f"{gender}, {age} yrs"
                        color = (0,255,0)

                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except:
                    continue
            last_frame = frame.copy()
        else:
            if last_frame is not None:
                frame = last_frame.copy()

        cv2.imshow("Senior Citizen Detection", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Detection finished. CSV log saved in {LOG_FILE}")

# -------------------------------
# Start Button
# -------------------------------
st.write("---")
st.markdown("<h3 style='color: #FF4B4B;'>Start Detection</h3>", unsafe_allow_html=True)
start_btn = st.button("▶️ Start Detection", key="start_btn")

if start_btn:
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        st.info("Processing video... OpenCV window will pop up.")
        threading.Thread(target=process_video, args=(tfile.name, False, skip_frames)).start()
    elif use_webcam:
        st.info("Starting webcam... OpenCV window will pop up.")
        threading.Thread(target=process_video, args=(None, True, skip_frames)).start()
    else:
        st.warning("Please upload a video or select webcam.")
