import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import tensorflow as tf

from ultralytics import YOLO


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
    PROJECT_DIR /
    "models" /
    "car_color_best.keras"
)

YOLO_MODEL_PATH = (
    PROJECT_DIR /
    "yolo26s.pt"
)


# ================================================================
# ADD SRC TO PYTHON PATH
# ================================================================

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )


# ================================================================
# IMPORT PROJECT MODULES
# ================================================================

from detector import VehicleDetector
from color_classifier import CarColorClassifier
from pipeline import CarColorPipeline


# ================================================================
# EXACT COLOR CLASS MAPPING USED DURING TRAINING
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
# CHECK MODEL FILES
# ================================================================

if not COLOR_MODEL_PATH.exists():

    st.error(
        f"Color model not found:\n\n"
        f"{COLOR_MODEL_PATH}"
    )

    st.stop()


if not YOLO_MODEL_PATH.exists():

    st.error(
        f"YOLO model not found:\n\n"
        f"{YOLO_MODEL_PATH}"
    )

    st.stop()


# ================================================================
# LOAD MODELS
# ================================================================

@st.cache_resource
def load_models():

    # ------------------------------------------------------------
    # YOLO OBJECT DETECTOR
    # ------------------------------------------------------------

    detector = VehicleDetector(
        model_path=str(
            YOLO_MODEL_PATH
        ),
        confidence=0.20
    )

    # ------------------------------------------------------------
    # CAR COLOR CNN
    # ------------------------------------------------------------

    color_model = tf.keras.models.load_model(
        COLOR_MODEL_PATH
    )

    classifier = CarColorClassifier(
        model=color_model,
        img_size=(224, 224),
        id_to_label=ID_TO_LABEL
    )

    # ------------------------------------------------------------
    # COMPLETE PIPELINE
    #
    # IMPORTANT:
    # classifier= NOT color_model=
    # ------------------------------------------------------------

    pipeline = CarColorPipeline(
        detector=detector,
        classifier=classifier
    )

    return (
        detector,
        classifier,
        pipeline
    )


# ================================================================
# LOAD
# ================================================================

try:

    detector, classifier, pipeline = load_models()

except Exception as e:

    st.error(
        "Model loading failed."
    )

    st.exception(e)

    st.stop()


# ================================================================
# HEADER
# ================================================================

st.title(
    "🚘 Car Colour Detection AI"
)

st.write(
    "Detect cars, identify their colours, "
    "count cars and people, and detect traffic lights."
)


# ================================================================
# MODEL INFORMATION
# ================================================================

with st.expander(
    "🔧 Model Information"
):

    st.write(
        f"**YOLO Model:** "
        f"`{YOLO_MODEL_PATH.name}`"
    )

    st.write(
        f"**Colour Model:** "
        f"`{COLOR_MODEL_PATH.name}`"
    )

    st.write(
        "**Colour Classes:**"
    )

    st.write(
        ", ".join(
            ID_TO_LABEL.values()
        )
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
    # READ UPLOADED IMAGE
    # ------------------------------------------------------------

    file_bytes = uploaded_file.read()

    image_array = cv2.imdecode(
        __import__("numpy").frombuffer(
            file_bytes,
            dtype=__import__("numpy").uint8
        ),
        cv2.IMREAD_COLOR
    )

    if image_array is None:

        st.error(
            "Unable to read the uploaded image."
        )

        st.stop()


    # ------------------------------------------------------------
    # ORIGINAL IMAGE
    # ------------------------------------------------------------

    st.subheader(
        "📷 Input Image"
    )

    original_rgb = cv2.cvtColor(
        image_array,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        original_rgb,
        use_container_width=True
    )


    # ------------------------------------------------------------
    # PROCESS BUTTON
    # ------------------------------------------------------------

    if st.button(
        "🔍 Detect Cars & Colours",
        type="primary"
    ):

        with st.spinner(
            "Detecting cars, people, traffic lights and colours..."
        ):

            try:

                result = pipeline.process(
                    image_array
                )

            except Exception as e:

                st.error(
                    "Processing failed."
                )

                st.exception(e)

                st.stop()


        # ========================================================
        # RESULTS
        # ========================================================

        st.success(
            "Detection completed successfully."
        )


        # --------------------------------------------------------
        # METRICS
        # --------------------------------------------------------

        cars = result.get(
            "cars",
            0
        )

        blue_cars = result.get(
            "blue_cars",
            0
        )

        other_cars = result.get(
            "other_cars",
            0
        )

        people = result.get(
            "people",
            0
        )

        traffic_lights = result.get(
            "traffic_lights",
            0
        )


        st.subheader(
            "📊 Detection Summary"
        )


        col1, col2, col3, col4, col5 = st.columns(5)


        with col1:

            st.metric(
                "🚗 Cars",
                cars
            )


        with col2:

            st.metric(
                "🔵 Blue Cars",
                blue_cars
            )


        with col3:

            st.metric(
                "🚙 Other Cars",
                other_cars
            )


        with col4:

            st.metric(
                "👤 People",
                people
            )


        with col5:

            st.metric(
                "🚦 Traffic Lights",
                traffic_lights
            )


        # --------------------------------------------------------
        # OUTPUT IMAGE
        # --------------------------------------------------------

        st.subheader(
            "🖼️ Detection Result"
        )


        output_image = result.get(
            "image"
        )


        if output_image is not None:

            output_rgb = cv2.cvtColor(
                output_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                output_rgb,
                use_container_width=True
            )


        # --------------------------------------------------------
        # LEGEND
        # --------------------------------------------------------

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


        # --------------------------------------------------------
        # CAR DETAILS
        # --------------------------------------------------------

        car_details = result.get(
            "car_details",
            []
        )


        if len(car_details) > 0:

            st.subheader(
                "🚗 Car Details"
            )


            dataframe = pd.DataFrame(
                car_details
            )


            st.dataframe(
                dataframe,
                use_container_width=True,
                hide_index=True
            )


        else:

            st.info(
                "No cars were detected."
            )


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.caption(
    "Car Colour Detection AI • "
    "YOLO object detection + custom car colour classifier"
)