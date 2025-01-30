import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from utils import preprocess_frame, load_model

# Konfigurasi Halaman
st.set_page_config(
    page_title="Students Engagement Detection",
    page_icon="🎓",
    layout="centered"
)

# CSS untuk membuat tata letak lebih terpusat
st.markdown(
    """
    <style>
    .stApp {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .block-container {
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul Halaman
st.title("📊 Student Engagement Detection")

# Pilihan Model
st.subheader("🔍 Choose Your AI Model")
model_option = st.selectbox(
    "Model:",
    ["MobileNetV2", "ResNet50", "EfficientNet"]
)

# Load Model berdasarkan pilihan user
MODEL_PATHS = {
    "MobileNetV2": "models/student_engagement_mobilenetv2_final.keras",
    "ResNet50": "models/student_engagement_resnet50_final.keras",
    "EfficientNet": "models/student_engagement_efficientnet_final.keras"
}

model = load_model(MODEL_PATHS[model_option])

# Opsi Input: Webcam atau Upload Video
st.subheader("📹 Input Data")
input_option = st.radio(
    "Choose Input:",
    ["Webcam", "Upload Video"]
)

if input_option == "Upload Video":
    uploaded_file = st.file_uploader("Unggah Video", type=["mp4", "avi"])
    if uploaded_file:
        st.video(uploaded_file)
        cap = cv2.VideoCapture(uploaded_file.name)
elif input_option == "Webcam":
    st.write("📹 Activating Webcam...")
    cap = cv2.VideoCapture(0)

# Tombol untuk memulai deteksi
st.subheader("🚀 Start Detection")
if st.button("Start Detection"):
    if not cap.isOpened():
        st.error("❌ Input video is not available.")
    else:
        st.write("🚦 Detection process is started...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("❌ Finished. Video is not available")
                break
            
            # Preprocessing Frame
            preprocessed_frame = preprocess_frame(frame)
            prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))
            engagement_status = "Engaged" if prediction[0][0] > 0.5 else "Disengaged"
            
            # Tampilkan Frame dengan Status
            cv2.putText(frame, f"Status: {engagement_status}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            st.image(frame, channels="BGR")
        
        cap.release()
        cv2.destroyAllWindows()

# Footer
st.write("---")
st.write("© 2024 Students Engagement Detection | Developed using Streamlit")