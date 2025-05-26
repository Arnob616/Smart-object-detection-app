import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"

# Async fix must come FIRST before any imports
import asyncio
import nest_asyncio
nest_asyncio.apply()

# Torch path workaround BEFORE torch import
import torch
try:
    # Prevent Streamlit from inspecting torch.classes
    torch.classes.__path__._path = []  # type: ignore 
except AttributeError:
    pass

# Now import other packages
import streamlit as st
import cv2
import numpy as np
import json
import io
import shutil
import warnings
import time
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# Rest of your code...

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Objectify-Smart Object & Edge Detection App", layout="wide")

# ======================
# CONFIGURATION
# ======================
MODEL_PATH = "yolov8n.pt"
CACHE_DIR = os.path.expanduser("~/.cache/ultralytics")
MAX_IMAGE_SIZE = 800
EDGE_METHODS = ["Canny", "Sobel", "Laplacian"]  # Removed Scharr
DEFAULT_CLASSES_TO_SHOW = 3

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ======================
# CUSTOM CSS
# ======================
def apply_theme(high_contrast):
    if high_contrast:
        st.markdown(
            """
            <style>
            .main {background-color: #FFFFFF; padding: 20px;}
            .stImage > img {max-width: 100%; height: auto; border: 2px solid #000000; border-radius: 5px;}
            .stSidebar {background-color: #FFFFFF; border-right: 2px solid #000000;}
            .stExpander {background-color: #FFFFFF; border: 2px solid #000000; border-radius: 5px;}
            h1, h2, h3, p, div, label, span {color: #000000 !important;}
            .stButton>button {background-color: #000000; color: #FFFFFF; border-radius: 5px; border: 2px solid #000000;}
            .stButton>button:hover {background-color: #333333;}
            .stDataFrame {background-color: #FFFFFF; border: 2px solid #000000; border-radius: 5px;}
            .stSelectbox > div, .stMultiSelect > div, .stCheckbox > label, .stSlider > label {color: #000000 !important;}
            .stDataFrame table, .stDataFrame th, .stDataFrame td {color: #000000 !important; border-color: #000000;}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .main {background-color: #000000; padding: 20px;}
            .stImage > img {max-width: 100%; height: auto; border: 1px solid #ffffff; border-radius: 5px;}
            .stSidebar {background-color: #000000;}
            .stExpander {background-color: #000000; border: 1px solid #ffffff; border-radius: 5px;}
            h1, h2, h3, p, div, label, span {color: #ffffff !important;}
            .stButton>button {background-color: #333333; color: #ffffff; border-radius: 5px; border: none;}
            .stButton>button:hover {background-color: #555555;}
            .stDataFrame {background-color: #000000; border: 1px solid #ffffff; border-radius: 5px;}
            .stSelectbox > div, .stMultiSelect > div, .stCheckbox > label, .stSlider > label {color: #ffffff !important;}
            .stDataFrame table, .stDataFrame th, .stDataFrame td {color: #ffffff !important; border-color: #ffffff;}
            </style>
            """,
            unsafe_allow_html=True
        )

# ======================
# HELPER FUNCTIONS
# ======================
@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 model with cache management and error handling"""
    try:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            st.info("Cleared Ultralytics cache for fresh yolov8n.pt download.")
            
        if not os.path.exists(MODEL_PATH):
            st.info("Downloading yolov8n.pt... This may take a moment.")
            
        return YOLO(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}. Ensure internet connection and update ultralytics/torch.")

def adjust_image_properties(image, brightness=100, contrast=100, saturation=100):
    """Adjust image brightness, contrast, and saturation"""
    img = np.array(image)
    if brightness != 100:
        img = cv2.convertScaleAbs(img, alpha=brightness/100)
    if contrast != 100:
        img = cv2.convertScaleAbs(img, alpha=contrast/100, beta=0)
    if saturation != 100 and len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * (saturation/100), 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(img)

def preprocess_image(image: Image.Image) -> tuple:
    """Process uploaded image and return grayscale and RGB versions"""
    try:
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_rgb = img_array.copy()
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        # Resize maintaining aspect ratio
        scale = MAX_IMAGE_SIZE / max(img_gray.shape)
        if scale < 1:
            img_gray = cv2.resize(img_gray, (0, 0), fx=scale, fy=scale)
            img_rgb = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale)
            
        return img_gray, img_rgb
    except Exception as e:
        raise RuntimeError(f"Image processing failed: {str(e)}")

def apply_preprocessing(image: np.ndarray, params: dict) -> np.ndarray:
    """Apply preprocessing steps to the image"""
    try:
        processed = image.copy()
        if params['gaussian']:
            processed = cv2.GaussianBlur(processed, 
                                        (params['gaussian_kernel'], params['gaussian_kernel']), 0)
        if params['threshold']:
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
        if params['hist_eq']:
            processed = cv2.equalizeHist(processed)
        if params['morph']:
            kernel = np.ones((params['morph_kernel'], params['morph_kernel']), np.uint8)
            processed = cv2.dilate(processed, kernel, iterations=1)
        return processed
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")

def apply_edge_detection(method: str, image: np.ndarray, params: dict) -> np.ndarray:
    """Apply selected edge detection algorithm"""
    try:
        if method == "Canny":
            return cv2.Canny(image, params['threshold1'], params['threshold2'])
        elif method == "Sobel":
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=params['kernel_size'])
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=params['kernel_size'])
            edges = cv2.magnitude(sobelx, sobely)
        elif method == "Laplacian":
            edges = cv2.Laplacian(image, cv2.CV_64F, ksize=params['laplacian_kernel'])
        elif method == "Scharr":
            scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
            edges = cv2.magnitude(scharrx, scharry)
            
        return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    except Exception as e:
        raise RuntimeError(f"Edge detection failed: {str(e)}")

def detect_objects(model, image: np.ndarray, selected_classes: list) -> tuple:
    """Run YOLO object detection and return results with visualization"""
    try:
        start_time = time.time()
        results = model(image)
        detections = []
        counts = {}
        viz_image = image.copy()
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                if selected_classes and class_name not in selected_classes:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Draw bounding boxes
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(viz_image, f"{class_name} {conf:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                counts[class_name] = counts.get(class_name, 0) + 1
                
        return viz_image, detections, counts, time.time()-start_time
    except Exception as e:
        raise RuntimeError(f"Object detection failed: {str(e)}")

def generate_heatmap(edges: np.ndarray) -> np.ndarray:
    """Generate edge intensity heatmap"""
    try:
        return cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    except Exception as e:
        raise RuntimeError(f"Heatmap generation failed: {str(e)}")

# ======================
# STREAMLIT UI
# ======================
def main():
    st.title("Objectify-Smart object and edge detection application")
    
    # High Contrast Mode Toggle
    high_contrast = st.sidebar.checkbox("High Contrast Mode", value=False)
    apply_theme(high_contrast)
    
    # Model loading with error handling
    try:
        model = load_yolo_model()
    except Exception as e:
        st.error(f"""Model Error: {str(e)}
                    Try these steps:
                    1. Update dependencies: `pip install --upgrade ultralytics torch`
                    2. Ensure internet connection for downloading yolov8n.pt
                    3. Verify PyTorch version: `pip show torch`
                    4. Manually download yolov8n.pt from https://github.com/ultralytics/assets/releases and place it in the project directory""")
        st.stop()

    # Input Selection
    input_method = st.sidebar.radio("Input Method", ["File Upload", "Webcam Capture"])
    
    # File upload/webcam capture section
    with st.container():
        st.subheader("Image Input")
        img = None
        
        if input_method == "File Upload":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")
            if uploaded_file:
                img = Image.open(uploaded_file)
        else:
            webcam_img = st.camera_input("Take a picture...")
            if webcam_img:
                img = Image.open(webcam_img)

        if not img:
            st.markdown("*Built by Arnob Bokshi. Upload an image or use webcam to begin processing.*")
            return

    # Image Adjustments
    with st.sidebar.expander("Image Adjustments", expanded=True):
        brightness = st.slider("Brightness", 0, 200, 100)
        contrast = st.slider("Contrast", 0, 200, 100)
        saturation = st.slider("Saturation", 0, 200, 100)
        adjusted_img = adjust_image_properties(img, brightness, contrast, saturation)

    # Image preprocessing
    try:
        img_gray, img_rgb = preprocess_image(adjusted_img)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        st.stop()

    # ======================
    # CONTROL PANEL
    # ======================
    with st.sidebar:
        st.header("Processing Parameters")
        
        # Edge detection settings
        with st.expander("Edge Detection Settings", expanded=True):
            edge_method = st.selectbox("Edge Detection Method", EDGE_METHODS, key="edge_method")
            edge_params = {}
            
            if edge_method == "Canny":
                edge_params['threshold1'] = st.slider("Canny Threshold 1", 0, 500, 100, key="canny_t1")
                edge_params['threshold2'] = st.slider("Canny Threshold 2", 0, 500, 200, key="canny_t2")
            elif edge_method == "Sobel":
                edge_params['kernel_size'] = st.slider("Kernel Size", 3, 15, 3, step=2, key="kernel_size")
            elif edge_method == "Laplacian":
                edge_params['laplacian_kernel'] = st.slider("Laplacian Kernel", 3, 15, 3, step=2, key="laplacian_kernel")

        # Preprocessing options
        with st.expander("Preprocessing Options", expanded=True):
            preprocess_params = {
                'gaussian': st.checkbox("Gaussian Blur", True, key="gaussian"),
                'gaussian_kernel': st.slider("Gaussian Kernel", 3, 15, 5, step=2, key="gaussian_kernel"),
                'threshold': st.checkbox("Adaptive Thresholding", key="threshold"),
                'hist_eq': st.checkbox("Histogram Equalization", key="hist_eq"),
                'morph': st.checkbox("Morphological Operations", key="morph"),
                'morph_kernel': st.slider("Morph Kernel", 3, 15, 3, step=2, key="morph_kernel")
            }

        # Object detection options
        with st.expander("Object Detection Settings", expanded=True):
            yolo_enabled = st.checkbox("Enable YOLO Detection", True, key="yolo_enabled")
            draw_contours = st.checkbox("Draw Edge Contours", True, key="draw_contours")
            class_options = list(model.names.values()) if yolo_enabled else []
            selected_classes = st.multiselect("Filter Classes", 
                                             class_options, 
                                             default=class_options[:DEFAULT_CLASSES_TO_SHOW] if class_options else [],
                                             key="class_filter")

    # ======================
    # IMAGE PROCESSING
    # ======================
    try:
        # Apply preprocessing
        start_preprocess = time.time()
        processed_img = apply_preprocessing(img_gray, preprocess_params)
        preprocess_time = time.time() - start_preprocess

        # Edge detection
        start_edge = time.time()
        edges = apply_edge_detection(edge_method, processed_img, edge_params)
        edge_time = time.time() - start_edge
        
        contour_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        if draw_contours:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

        # Object detection
        yolo_image = img_rgb.copy()
        detections = []
        counts = {}
        yolo_time = 0
        if yolo_enabled:
            yolo_image, detections, counts, yolo_time = detect_objects(model, img_rgb, selected_classes)

        # Generate heatmap
        start_heatmap = time.time()
        heatmap = generate_heatmap(edges)
        heatmap_time = time.time() - start_heatmap

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        st.stop()

    # ======================
    # RESULTS DISPLAY
    # ======================
    with st.container():
        st.subheader("Processing Results")
        
        # Performance Metrics
        with st.expander("Performance Metrics", expanded=True):
            metrics = {
                "Preprocessing": f"{preprocess_time*1000:.2f} ms",
                "Edge Detection": f"{edge_time*1000:.2f} ms",
                "Object Detection": f"{yolo_time*1000:.2f} ms" if yolo_enabled else "N/A",
                "Heatmap Generation": f"{heatmap_time*1000:.2f} ms"
            }
            st.table(pd.DataFrame(list(metrics.items()), columns=["Process", "Time"]))

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(adjusted_img, caption="Adjusted Input Image", use_container_width=True)
            st.image(Image.fromarray(processed_img if len(processed_img.shape) == 2 else cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)), 
                     caption="Preprocessed Image", use_container_width=True)
        
        with col2:
            st.image(Image.fromarray(contour_img), caption=f"{edge_method} Edges with Contours", use_container_width=True)
            if yolo_enabled:
                st.image(Image.fromarray(yolo_image), caption="YOLO Object Detection", use_container_width=True)
            st.image(Image.fromarray(heatmap), caption="Edge Intensity Heatmap", use_container_width=True)

        # Detection analytics
        if yolo_enabled and detections:
            with st.expander("Detection Analytics", expanded=True):
                st.subheader("Detected Objects")
                st.dataframe(pd.DataFrame(detections), use_container_width=True)
                
                st.subheader("Object Counts")
                count_df = pd.DataFrame(list(counts.items()), columns=["Object Name", "Count"])
                st.dataframe(count_df, use_container_width=True)

        # Download options
        with st.expander("Download Options", expanded=False):
            buf = io.BytesIO()
            if yolo_enabled:
                Image.fromarray(yolo_image).save(buf, format="PNG")
            else:
                Image.fromarray(contour_img).save(buf, format="PNG")
            st.download_button(
                label="Download Processed Image",
                data=buf.getvalue(),
                file_name="processed_image.png",
                mime="image/png",
                key="download_image"
            )
            
            if yolo_enabled and detections:
                json_data = json.dumps(detections, indent=2)
                st.download_button(
                    label="Download JSON Results",
                    data=json_data,
                    file_name="detections.json",
                    mime="application/json",
                    key="download_json"
                )

    st.markdown("---")
    st.markdown("*Built by Arnob Bokshi using Streamlit, OpenCV, and YOLOv8*")

if __name__ == "__main__":
    main()