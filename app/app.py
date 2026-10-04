
import os
import sys
from pathlib import Path

# ================================================================
# CPU SETTINGS
# Must be set before importing deep-learning libraries
# ================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

# ================================================================
# LIGHTWEIGHT IMPORTS
# ================================================================

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Car Colour Detection AI",
    page_icon="🚘",
    layout="wide"
)

# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_DIR / "src"

COLOR_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "car_color_best.keras"
)

YOLO_MODEL_PATH = PROJECT_DIR / "yolo26s.pt"

# ================================================================
# ADD SRC TO PYTHON PATH
# ================================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ================================================================
# COLOUR CLASS MAPPING
# ================================================================

ID_TO_LABEL = {
    0: "beige",
    1: "black",
    2: "blue",
    3: "brown",
    4: "gold",
    5: "green",
    6: "grey",
    7: "orange",
    8: "pink",
    9: "purple",
    10: "red",
    11: "silver",
    12: "tan",
    13: "white",
    14: "yellow"
}

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def get_count(value):
    """
    Convert a detection list or integer into a safe count.
    """

    if value is None:
        return 0

    if isinstance(value, (list, tuple)):
        return len(value)

    if isinstance(value, np.ndarray):
        return len(value)

    try:
        return int(value)

    except (TypeError, ValueError):
        return 0


def get_output_image(result):
    """
    Safely extract an output image from the pipeline result.
    """

    if isinstance(result, np.ndarray):
        return result

    if not isinstance(result, dict):
        return None

    possible_keys = [
        "image",
        "output_image",
        "annotated_image",
        "result_image",
        "frame"
    ]

    for key in possible_keys:
        image = result.get(key)

        if isinstance(image, np.ndarray):
            return image

    return None


def get_detection_items(result, key):
    """
    Return a detection collection safely.
    """

    if not isinstance(result, dict):
        return []

    value = result.get(key, [])

    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    if isinstance(value, np.ndarray):
        return list(value)

    return []


def get_metric_count(result, key, fallback_key=None):
    """
    Extract a count from the result.

    Supports integer counts and lists of detected objects.
    """

    if not isinstance(result, dict):
        return 0

    value = result.get(key)

    if value is not None:
        return get_count(value)

    if fallback_key:
        return get_count(result.get(fallback_key, []))

    return 0


def prepare_display_image(image):
    """
    Convert a valid BGR image into RGB for Streamlit.
    """

    if image is None:
        return None

    if not isinstance(image, np.ndarray):
        return None

    if image.size == 0:
        return None

    if len(image.shape) == 2:
        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2RGBA
        )

    return None


# ================================================================
# CHECK MODEL FILES
# ================================================================

if not COLOR_MODEL_PATH.exists():
    st.error(
        f"Colour model not found:\n\n{COLOR_MODEL_PATH}"
    )
    st.stop()


if not YOLO_MODEL_PATH.exists():
    st.error(
        f"YOLO model not found:\n\n{YOLO_MODEL_PATH}"
    )
    st.stop()


# ================================================================
# LOAD MODELS ON DEMAND
# ================================================================

@st.cache_resource
def load_models():
    """
    Load TensorFlow, PyTorch, YOLO and the colour model
    only when the user requests detection.

    The cache prevents repeated model loading during
    normal Streamlit reruns.
    """

    # ------------------------------------------------------------
    # IMPORT DEEP-LEARNING LIBRARIES
    # ------------------------------------------------------------

    import tensorflow as tf
    import torch

    # ------------------------------------------------------------
    # CONFIGURE CPU
    # ------------------------------------------------------------

    try:
        tf.config.set_visible_devices([], "GPU")

    except Exception:
        pass

    try:
        torch.set_num_threads(2)

    except Exception:
        pass

    # ------------------------------------------------------------
    # IMPORT PROJECT MODULES
    # ------------------------------------------------------------

    from detector import VehicleDetector
    from color_classifier import CarColorClassifier
    from pipeline import CarColorPipeline

    # ------------------------------------------------------------
    # LOAD YOLO DETECTOR
    # ------------------------------------------------------------

    detector = VehicleDetector(
        model_path=YOLO_MODEL_PATH,
        conf=0.25,
        iou=0.45,
        max_det=300,
        device="cpu",
        tile_size=960,
        
    )

    # ------------------------------------------------------------
    # LOAD COLOUR CLASSIFICATION MODEL
    # ------------------------------------------------------------

    color_model = tf.keras.models.load_model(
        COLOR_MODEL_PATH,
        compile=False
    )

    classifier = CarColorClassifier(
        model=color_model,
        img_size=(224, 224),
        id_to_label=ID_TO_LABEL
    )

    # ------------------------------------------------------------
    # CREATE COMPLETE PIPELINE
    # ------------------------------------------------------------

    pipeline = CarColorPipeline(
        detector=detector,
        classifier=classifier
    )

    return detector, classifier, pipeline


# ================================================================
# HEADER
# ================================================================

st.title("🚘 Car Colour Detection AI")

st.write(
    """
    Upload a traffic image to detect cars, classify their colours,
    count people and identify traffic lights.
    """
)

# ================================================================
# MODEL INFORMATION
# ================================================================

with st.expander("🔧 Model Information"):

    st.write(
        f"**YOLO Model:** `{YOLO_MODEL_PATH.name}`"
    )

    st.write(
        f"**Colour Model:** `{COLOR_MODEL_PATH.name}`"
    )

    st.write("**Colour Classes:**")

    st.write(
        ", ".join(ID_TO_LABEL.values())
    )

    st.write("**Inference Device:** CPU")

    st.info(
        "Models are loaded when you click the Detect button."
    )

# ================================================================
# IMAGE UPLOAD
# ================================================================

uploaded_file = st.file_uploader(
    "📤 Upload a traffic image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)

# ================================================================
# PROCESS IMAGE
# ================================================================

if uploaded_file is not None:

    # ------------------------------------------------------------
    # READ IMAGE
    # ------------------------------------------------------------

    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8
    )

    image_array = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image_array is None:
        st.error(
            "Unable to read the uploaded image. "
            "Please try another image."
        )
        st.stop()

    # ------------------------------------------------------------
    # DISPLAY ORIGINAL IMAGE
    # ------------------------------------------------------------

    st.subheader("📷 Input Image")

    original_rgb = prepare_display_image(
        image_array
    )

    if original_rgb is not None:
        st.image(
            original_rgb,
            width="stretch"
        )

    # ------------------------------------------------------------
    # PROCESS BUTTON
    # ------------------------------------------------------------

    if st.button(
        "🔍 Detect Cars & Colours",
        type="primary"
    ):

        # --------------------------------------------------------
        # LOAD MODELS
        # --------------------------------------------------------

        try:
            with st.spinner("Loading AI models..."):
                detector, classifier, pipeline = load_models()

        except Exception as e:
            st.error("Model loading failed.")
            st.exception(e)
            st.stop()

        # --------------------------------------------------------
        # RUN DETECTION
        # --------------------------------------------------------

        try:
            with st.spinner(
                "Detecting cars, people, traffic lights and colours..."
            ):
                result = pipeline.process(image_array)

        except Exception as e:
            st.error("Processing failed.")
            st.exception(e)
            st.stop()

        # ========================================================
        # VALIDATE PIPELINE RESULT
        # ========================================================

        if result is None:
            st.error(
                "The pipeline returned no result."
            )
            st.stop()

        if not isinstance(result, (dict, np.ndarray)):
            st.error(
                "Unexpected pipeline result format."
            )

            st.write(
                "Returned type:",
                type(result).__name__
            )

            st.stop()

        st.success(
            "Detection completed successfully."
        )

        # ========================================================
        # EXTRACT DETECTIONS
        # ========================================================

        cars = get_detection_items(
            result,
            "cars"
        )

        people = get_detection_items(
            result,
            "people"
        )

        traffic_lights = get_detection_items(
            result,
            "traffic_lights"
        )

        # --------------------------------------------------------
        # METRIC COUNTS
        # --------------------------------------------------------

        car_count = get_metric_count(
            result,
            "car_count",
            "cars"
        )

        people_count = get_metric_count(
            result,
            "people_count",
            "people"
        )

        traffic_light_count = get_metric_count(
            result,
            "traffic_light_count",
            "traffic_lights"
        )

        blue_cars = get_metric_count(
            result,
            "blue_cars"
        )

        other_cars = get_metric_count(
            result,
            "other_cars"
        )

        # --------------------------------------------------------
        # CHECK COLOUR COUNTS
        # --------------------------------------------------------

        colour_counts_available = (
            isinstance(result, dict)
            and (
                "blue_cars" in result
                or "other_cars" in result
            )
        )

        # ========================================================
        # DETECTION SUMMARY
        # ========================================================

        st.subheader(
            "📊 Detection Summary"
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "🚗 Cars",
                car_count
            )

        with col2:
            st.metric(
                "🔵 Blue Cars",
                blue_cars if colour_counts_available else "—"
            )

        with col3:
            st.metric(
                "🚙 Other Cars",
                other_cars if colour_counts_available else "—"
            )

        with col4:
            st.metric(
                "👤 People",
                people_count
            )

        with col5:
            st.metric(
                "🚦 Traffic Lights",
                traffic_light_count
            )

        # ========================================================
        # OUTPUT IMAGE
        # ========================================================

        st.subheader(
            "🖼️ Detection Result"
        )

        output_image = get_output_image(
            result
        )

        if output_image is not None:

            output_rgb = prepare_display_image(
                output_image
            )

            if output_rgb is not None:
                st.image(
                    output_rgb,
                    width="stretch",
                    caption=(
                        "Detected cars, people and traffic lights"
                    )
                )

            else:
                st.warning(
                    "The output image format is not supported."
                )

        else:
            st.warning(
                "The pipeline did not return an annotated image. "
                "Detection counts are shown above."
            )

            if isinstance(result, dict):
                st.write(
                    "Available result keys:",
                    list(result.keys())
                )

        # ========================================================
        # BOUNDING BOX LEGEND
        # ========================================================

        st.subheader(
            "🎨 Bounding Box Legend"
        )

        legend_col1, legend_col2, legend_col3 = st.columns(3)

        with legend_col1:
            st.markdown(
                "🔴 **Red box** → Blue car"
            )

        with legend_col2:
            st.markdown(
                "🔵 **Blue box** → Other colour car"
            )

        with legend_col3:
            st.markdown(
                "🟢 **Green box** → Person"
            )

        st.markdown(
            "🟡 **Yellow box** → Traffic light"
        )

        # ========================================================
        # CAR DETAILS
        # ========================================================

        car_details = []

        if isinstance(result, dict):
            car_details = result.get(
                "car_details",
                []
            )

        if car_details is None:
            car_details = []

        if (
            isinstance(car_details, (list, tuple))
            and len(car_details) > 0
        ):

            st.subheader(
                "🚗 Car Details"
            )

            try:
                dataframe = pd.DataFrame(
                    car_details
                )

                st.dataframe(
                    dataframe,
                    width="stretch",
                    hide_index=True
                )

            except Exception as e:
                st.warning(
                    "Car details could not be displayed."
                )
                st.exception(e)

        elif len(cars) > 0:
            st.subheader(
                "🚗 Detected Car Information"
            )

            st.write(
                f"Number of detected cars: **{car_count}**"
            )

        else:
            st.info(
                "No cars were detected in this image."
            )

        # ========================================================
        # PEOPLE DETAILS
        # ========================================================

        if people_count > 0:
            st.subheader(
                "👥 People Detection"
            )

            st.write(
                f"People detected: **{people_count}**"
            )

        # ========================================================
        # TRAFFIC LIGHT DETAILS
        # ========================================================

        if traffic_light_count > 0:
            st.subheader(
                "🚦 Traffic Light Detection"
            )

            st.write(
                f"Traffic lights detected: "
                f"**{traffic_light_count}**"
            )

# ================================================================
# FOOTER
# ================================================================

st.divider()

st.caption(
    "Car Colour Detection AI | "
    "YOLO Object Detection + Custom CNN Colour Classifier"
)
