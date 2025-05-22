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
EDGE_METHODS = ["Canny", "Sobel", "Laplacian"]  # Removed Prewitt
DEFAULT_CLASSES_TO_SHOW = 3

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Custom CSS for full black theme with white text
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
            
        return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    except Exception as e:
        raise RuntimeError(f"Edge detection failed: {str(e)}")

def detect_objects(model, image: np.ndarray, selected_classes: list) -> tuple:
    """Run YOLO object detection and return results with visualization"""
    try:
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
                
        return viz_image, detections, counts
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
    st.markdown("Upload an image to apply edge detection and object detection with customizable settings.")

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

    # File upload section
    with st.container():
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")
        if not uploaded_file:
            st.markdown("*Built by Arnob Bokshi. Upload an image to begin processing.*")
            return

    # Image preprocessing
    try:
        img_gray, img_rgb = preprocess_image(Image.open(uploaded_file))
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
            else:  # Laplacian
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
        processed_img = apply_preprocessing(img_gray, preprocess_params)

        # Edge detection
        edges = apply_edge_detection(edge_method, processed_img, edge_params)
        contour_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        if draw_contours:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

        # Object detection
        yolo_image = img_rgb.copy()
        detections = []
        counts = {}
        if yolo_enabled:
            yolo_image, detections, counts = detect_objects(model, img_rgb, selected_classes)

        # Generate heatmap
        heatmap = generate_heatmap(edges)

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        st.stop()

    # ======================
    # RESULTS DISPLAY
    # ======================
    with st.container():
        st.subheader("Processing Results")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(Image.fromarray(img_rgb), caption="Original Image", use_container_width=True)
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