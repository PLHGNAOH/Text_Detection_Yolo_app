import json
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# =========================
# Config
# =========================
MODEL_PATH = "runs/detect/train/weights/best.pt"

st.set_page_config(
    page_title="OCR Text Detection Demo",
    layout="wide"  # đổi sang wide để hiển thị ngang đẹp hơn
)

st.title("Text Detection (YOLO)")
st.write("Upload an image and click Run to detect text regions.")

# =========================
# Load model (cache)
# =========================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# =========================
# Visualization function
# =========================
def visualize_bbox(img, predictions, font=cv2.FONT_HERSHEY_SIMPLEX):

    for prediction in predictions:
        conf_score = prediction["confidence"]

        bbox = prediction["box"]
        xmin = int(bbox["x1"])
        ymin = int(bbox["y1"])
        xmax = int(bbox["x2"])
        ymax = int(bbox["y2"])

        # Draw rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Draw confidence text
        text = f"{conf_score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)

        cv2.rectangle(
            img,
            (xmin, ymin - th - 6),
            (xmin + tw, ymin),
            (0, 255, 0),
            -1,
        )
        cv2.putText(img, text, (xmin, ymin - 4), font, 0.7, (0, 0, 0), 2)

    return img


# =========================
# UI Controls
# =========================
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

run_button = st.button("Run Detection")

# =========================
# Inference
# =========================
if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img, channels="BGR", use_container_width=True)

    if run_button:
        with st.spinner("Running inference..."):
            results = model(img, verbose=False)
            predictions = json.loads(results[0].to_json())

            visualized_img = visualize_bbox(
                img.copy(),
                predictions
            )

        with col2:
            st.subheader("Detection Result")
            st.image(visualized_img, channels="BGR", use_container_width=True)

            st.subheader("Raw Predictions (JSON)")
            st.json(predictions)