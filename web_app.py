import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as numpy

st.set_page_config(page_title="The AI Engineer", page_icon="🤖")
st.title("🤖 The AI Engineer - Live Image Recognition")
st.write("Let's detect objects in real-time using your laptop camera.")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()
run_cam = st.checkbox('Start Webcam / Turn on the camera.')
FRAME_WINDOW = st.image([])
if run_cam:
    cap = cv2.VideoCapture(0)
    while run_cam:
        success, frame = cap.read()
        if not success:
            st.error("The camera is not working!")
            break
        results = model(frame, stream=True)
        for r in results:
            frame = r.plot()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
    cap.release()
else:
    st.write('The camera has been stopped.')
