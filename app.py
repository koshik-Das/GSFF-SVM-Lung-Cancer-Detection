# ============================================================
# app.py
# PulmoVision
# AI-Powered Lung Cancer Detection System
#
# PIPELINE
#
# Uploaded Image
#       ↓
# Basic Image Validation
#       ↓
# Colour Image Detection
#       ↓
# CT / Chest X-ray / MRI Modality Classification
#       ↓
# CT Verification
#       ↓
# EfficientNetB0
#       ↓
# Block5c
#       ↓
# GAP + GSDP
#       ↓
# GSFF Feature Fusion
#       ↓
# RobustScaler
#       ↓
# RBF-SVM
#       ↓
# Normal / Benign / Malignant
#
# NO GRAD-CAM
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
import keras
import joblib

from PIL import Image

from tensorflow.keras.applications.efficientnet import (
    preprocess_input
)

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage
)


# ============================================================
# 2. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PulmoVision",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# 3. BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

LUNG_ICON_PATH = (
    BASE_DIR / "Lung_Icon.png"
)


# ============================================================
# 4. MODEL FILE PATHS
# ============================================================

CT_VERIFIER_PATH = (
    BASE_DIR / "CT_Verifier.keras"
)

FEATURE_EXTRACTOR_PATH = (
    BASE_DIR / "GSFF_Feature_Extractor.keras"
)

SCALER_PATH = (
    BASE_DIR / "RobustScaler.pkl"
)

SVM_PATH = (
    BASE_DIR / "SVM_Classifier.pkl"
)


# ============================================================
# 5. IMAGE SETTINGS
# ============================================================

CT_VERIFIER_IMAGE_SIZE = (
    128,
    128
)

FEATURE_EXTRACTOR_IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# 6. THRESHOLDS
# ============================================================

# Minimum confidence required for CT verification
MODALITY_THRESHOLD = 0.60


# Average channel difference above this value
# indicates a genuine colour image.
COLOR_TOLERANCE = 8.0


# ============================================================
# 7. CLASS MAPPINGS
# ============================================================

# Lung cancer classification
class_names = [
    "Normal",
    "Benign",
    "Malignant"
]


# Modality classifier mapping
#
# This exactly matches your training code:
#
# 0 = CHEST_XRAY
# 1 = CT
# 2 = MRI
#
modality_names = [
    "CHEST_XRAY",
    "CT",
    "MRI"
]


# ============================================================
# 8. CUSTOM GSDP LAYER
# ============================================================
#
# Your saved GSFF feature extractor contains GSDP.
#
# GSDP:
#
# Standard deviation over spatial dimensions
# axis = (1, 2)
#
# Input:
#     (batch, 14, 14, 112)
#
# Output:
#     (batch, 112)
#
# GAP  = 112 features
# GSDP = 112 features
#
# GSFF = 224 features
#
# IMPORTANT:
# There is NO GMP here.
#
# ============================================================


@keras.saving.register_keras_serializable(
    package="GSFF"
)
class GSDP(keras.layers.Layer):

    def __init__(
        self,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )


    def call(
        self,
        inputs
    ):

        return keras.ops.std(
            inputs,
            axis=(1, 2)
        )


    def get_config(self):

        config = super().get_config()

        return config


# ============================================================
# 9. MODEL FILE VALIDATION
# ============================================================

def validate_model_file(
    path,
    model_name
):

    if not path.exists():

        st.error(
            f"{model_name} was not found."
        )

        st.code(
            str(path)
        )

        st.stop()


    if path.stat().st_size == 0:

        st.error(
            f"{model_name} file is empty."
        )

        st.stop()


# ============================================================
# 10. CHECK ALL REQUIRED FILES
# ============================================================

required_files = {

    "CT Verifier":
        CT_VERIFIER_PATH,

    "GSFF Feature Extractor":
        FEATURE_EXTRACTOR_PATH,

    "RobustScaler":
        SCALER_PATH,

    "SVM Classifier":
        SVM_PATH

}


for model_name, model_path in (
    required_files.items()
):

    validate_model_file(
        model_path,
        model_name
    )


# ============================================================
# 11. LOAD CT VERIFIER
# ============================================================

@st.cache_resource
def load_ct_verifier():

    model = keras.models.load_model(

        CT_VERIFIER_PATH,

        compile=False

    )

    return model


# ============================================================
# 12. LOAD GSFF FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def load_feature_extractor():

    model = keras.models.load_model(

        FEATURE_EXTRACTOR_PATH,

        custom_objects={

            "GSDP":
                GSDP,

            "GSFF>GSDP":
                GSDP

        },

        compile=False,

        safe_mode=False

    )

    return model


# ============================================================
# 13. LOAD ROBUST SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    scaler = joblib.load(
        SCALER_PATH
    )

    return scaler


# ============================================================
# 14. LOAD SVM
# ============================================================

@st.cache_resource
def load_svm():

    model = joblib.load(
        SVM_PATH
    )

    return model


# ============================================================
# 15. LOAD ALL MODELS
# ============================================================

try:

    ct_verifier = (
        load_ct_verifier()
    )

except Exception as e:

    st.error(
        "CT verifier loading failed."
    )

    st.exception(e)

    st.stop()


try:

    feature_extractor = (
        load_feature_extractor()
    )

except Exception as e:

    st.error(
        "GSFF feature extractor loading failed."
    )

    st.error(
        "Make sure this is the same "
        "GSFF_Feature_Extractor.keras "
        "created during training."
    )

    st.exception(e)

    st.stop()


try:

    scaler = load_scaler()

except Exception as e:

    st.error(
        "RobustScaler loading failed."
    )

    st.exception(e)

    st.stop()


try:

    svm = load_svm()

except Exception as e:

    st.error(
        "SVM classifier loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 16. VERIFY CT VERIFIER OUTPUT
# ============================================================

try:

    ct_output_shape = (
        ct_verifier.output_shape
    )


    if ct_output_shape[-1] != 3:

        st.error(
            "CT verifier output mismatch."
        )

        st.write(
            "Expected: 3 classes"
        )

        st.write(
            f"Actual: {ct_output_shape}"
        )

        st.stop()


except Exception as e:

    st.error(
        "Unable to verify CT verifier output."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 17. VERIFY GSFF FEATURE DIMENSION
# ============================================================

try:

    feature_output_shape = (
        feature_extractor.output_shape
    )


    expected_feature_dimension = 224


    if (
        feature_output_shape[-1]
        != expected_feature_dimension
    ):

        st.error(
            "GSFF feature dimension mismatch."
        )

        st.write(
            f"Expected: "
            f"{expected_feature_dimension}"
        )

        st.write(
            f"Actual: "
            f"{feature_output_shape}"
        )

        st.stop()


except Exception as e:

    st.error(
        "Unable to verify GSFF feature dimension."
    )

    st.exception(e)

    st.stop()


# ============================================================
# 18. SESSION STATE
# ============================================================

if (
    "pdf_report"
    not in st.session_state
):

    st.session_state.pdf_report = None


# ============================================================
# 19. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 750;
        line-height: 1.15;
        margin-top: 5px;
        margin-bottom: 8px;
    }


    .brand-name {
        text-align: center;
        font-size: 17px;
        font-weight: 600;
        margin-top: 4px;
        margin-bottom: 8px;
    }


    .subtitle {
        text-align: center;
        font-size: 16px;
        line-height: 1.6;
        margin-top: 8px;
        margin-bottom: 25px;
    }


    .result-box {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #d8d8d8;
        margin-top: 15px;
        margin-bottom: 15px;
    }


    .modality-result {
        background-color: #eef5ff;
    }


    .normal-result {
        background-color: #eaf8ed;
    }


    .benign-result {
        background-color: #fff7df;
    }


    .malignant-result {
        background-color: #fdeaea;
    }


    .section-title {
        text-align: center;
        font-size: 26px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 12px;
    }


    .info-text {
        text-align: center;
        font-size: 15px;
        line-height: 1.5;
    }


    .pipeline-box {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: rgba(128,128,128,0.05);
        line-height: 1.8;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 20. HEADER
# ============================================================

title_col1, title_col2 = (
    st.columns(
        [0.8, 7.2],
        vertical_alignment="center"
    )
)


# ============================================================
# HEADER ICON
# ============================================================

with title_col1:

    if LUNG_ICON_PATH.exists():

        st.image(
            str(LUNG_ICON_PATH),
            width=95
        )

    else:

        st.markdown(
            """
            <div style="
                font-size:70px;
                text-align:center;
                line-height:1;
            ">
                🫁
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# HEADER TEXT
# ============================================================

with title_col2:

    st.markdown(
        """
        <div class="main-title">
            PulmoVision
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="brand-name">
            AI-Powered Lung Cancer Detection System
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SUBTITLE
# ============================================================

st.markdown(
    """
    <div class="subtitle">
        Intelligent analysis of lung CT images using
        deep feature extraction and machine learning.
    </div>
    """,
    unsafe_allow_html=True
)


st.info(
    "Upload a lung CT image to begin the analysis."
)


# ============================================================
# 21. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🫁 PulmoVision"
    )


    st.write(
        "AI-Powered Lung Cancer "
        "Detection System"
    )


    st.divider()


    st.write(
        "**AI Pipeline**"
    )


    st.markdown(
        """
        <div class="pipeline-box">

        🖼️ Image<br>
        ↓<br>
        🔍 Image Validation<br>
        ↓<br>
        🩻 Modality Verification<br>
        ↓<br>
        🫁 CT<br>
        ↓<br>
        🧠 EfficientNetB0<br>
        ↓<br>
        📐 Block5c<br>
        ↓<br>
        📊 GAP + GSDP<br>
        ↓<br>
        🔗 GSFF Fusion<br>
        ↓<br>
        ⚖️ RobustScaler<br>
        ↓<br>
        🎯 RBF-SVM<br>
        ↓<br>
        📋 Classification

        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()


    st.write(
        "**Classification Classes**"
    )


    st.write(
        "🟢 Normal"
    )

    st.write(
        "🟡 Benign"
    )

    st.write(
        "🔴 Malignant"
    )


    st.divider()


    st.caption(
        "Research prototype. "
        "Not intended for clinical diagnosis."
    )


# ============================================================
# 22. COLOUR IMAGE CHECK
# ============================================================

def check_color_image(
    image
):

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32
    )

    red = rgb[:, :, 0]

    green = rgb[:, :, 1]

    blue = rgb[:, :, 2]

    rg_difference = np.mean(
        np.abs(
            red - green
        )
    )

    gb_difference = np.mean(
        np.abs(
            green - blue
        )
    )

    rb_difference = np.mean(
        np.abs(
            red - blue
        )
    )

    average_difference = (
        rg_difference
        +
        gb_difference
        +
        rb_difference
    ) / 3.0

    is_color = (
        average_difference
        >
        COLOR_TOLERANCE
    )

    return (
        is_color,
        float(average_difference)
    )


# ============================================================
# 23. BASIC IMAGE VALIDATION
# ============================================================

def validate_image(
    image
):

    width, height = image.size

    if width < 64 or height < 64:

        return (
            False,
            "Image resolution is too small."
        )

    array = np.asarray(
        image
    )

    if array.size == 0:

        return (
            False,
            "Image is empty."
        )

    is_color, difference = (
        check_color_image(
            image
        )
    )

    if is_color:

        return (
            False,
            "Color image detected. Please input a lung CT image."
        )

    gray = np.asarray(
        image.convert("L"),
        dtype=np.float32
    )

    if np.std(gray) < 8:

        return (
            False,
            "Image appears blank or invalid."
        )

    dark_ratio = np.mean(
        gray < 10
    )

    if dark_ratio > 0.98:

        return (
            False,
            "Image is almost completely black."
        )

    bright_ratio = np.mean(
        gray > 245
    )

    if bright_ratio > 0.98:

        return (
            False,
            "Image is almost completely white."
        )

    return (
        True,
        "Image passed validation."
    )


# ============================================================
# 24. PREPROCESS CT VERIFIER
# ============================================================

def preprocess_for_ct_verifier(
    image
):

    image = image.convert(
        "RGB"
    )


    image = image.resize(
        CT_VERIFIER_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )


    image_array = np.asarray(
        image,
        dtype=np.float32
    )


    # Same preprocessing as training:
    # ImageDataGenerator(rescale=1/255)

    image_array /= 255.0


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# ============================================================
# 25. PREPROCESS FOR GSFF
# ============================================================

def preprocess_for_gsff(
    image
):

    image = image.convert(
        "RGB"
    )


    image = image.resize(
        FEATURE_EXTRACTOR_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )


    image_array = np.asarray(
        image,
        dtype=np.float32
    )


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # Matches training:
    #
    # X = preprocess_input(...)

    image_array = (
        preprocess_input(
            image_array
        )
    )


    return image_array


# ============================================================
# 26. MODALITY VERIFICATION
# ============================================================

def verify_modality(
    image
):

    image_array = (
        preprocess_for_ct_verifier(
            image
        )
    )


    predictions = (
        ct_verifier.predict(
            image_array,
            verbose=0
        )
    )


    probabilities = np.asarray(
        predictions[0],
        dtype=np.float32
    )


    # Safety normalization

    probability_sum = (
        np.sum(
            probabilities
        )
    )


    if probability_sum > 0:

        probabilities = (
            probabilities
            /
            probability_sum
        )


    predicted_class = int(
        np.argmax(
            probabilities
        )
    )


    confidence = float(
        probabilities[
            predicted_class
        ]
    )


    modality = (
        modality_names[
            predicted_class
        ]
    )


    return (
        modality,
        confidence,
        probabilities
    )


# ============================================================
# 27. GSFF FEATURE EXTRACTION
# ============================================================

def extract_gsff_features(
    image
):

    image_array = (
        preprocess_for_gsff(
            image
        )
    )


    features = (
        feature_extractor.predict(
            image_array,
            verbose=0
        )
    )


    features = np.asarray(
        features,
        dtype=np.float32
    )


    return features


# ============================================================
# 28. LUNG CANCER PREDICTION
# ============================================================

def predict_lung_cancer(
    image
):

    # --------------------------------------------------------
    # GSFF feature extraction
    # --------------------------------------------------------

    features = (
        extract_gsff_features(
            image
        )
    )


    # --------------------------------------------------------
    # RobustScaler
    # --------------------------------------------------------

    scaled_features = (
        scaler.transform(
            features
        )
    )


    # --------------------------------------------------------
    # RBF-SVM
    # --------------------------------------------------------

    prediction = (
        svm.predict(
            scaled_features
        )
    )


    predicted_index = int(
        prediction[0]
    )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = (
        svm.predict_proba(
            scaled_features
        )[0]
    )


    predicted_class = (
        class_names[
            predicted_index
        ]
    )


    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# 29. PDF REPORT
# ============================================================

def create_pdf_report(
    image,
    modality,
    modality_confidence,
    modality_probabilities,
    predicted_class,
    prediction_confidence,
    class_probabilities
):

    buffer = io.BytesIO()


    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=18 * mm,

        leftMargin=18 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm

    )


    styles = (
        getSampleStyleSheet()
    )


    title_style = ParagraphStyle(

        "ReportTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=18,

        leading=22,

        spaceAfter=12

    )


    subtitle_style = ParagraphStyle(

        "ReportSubtitle",

        parent=styles["Normal"],

        alignment=TA_CENTER,

        fontSize=11,

        leading=14,

        spaceAfter=15

    )


    heading_style = ParagraphStyle(

        "ReportHeading",

        parent=styles["Heading2"],

        fontSize=13,

        leading=16,

        spaceBefore=10,

        spaceAfter=8

    )


    normal_style = ParagraphStyle(

        "ReportNormal",

        parent=styles["Normal"],

        fontSize=10,

        leading=14

    )


    story = []


    # ========================================================
    # REPORT TITLE
    # ========================================================

    story.append(

        Paragraph(

            "PulmoVision<br/>"
            "AI-Powered Lung Cancer Detection System",

            title_style

        )

    )


    story.append(

        Paragraph(

            "Lung CT Image Analysis Report",

            subtitle_style

        )

    )


    report_time = (
        datetime.now().strftime(
            "%d %B %Y, %I:%M:%S %p"
        )
    )


    story.append(

        Paragraph(

            f"<b>Analysis Date:</b> "
            f"{report_time}",

            normal_style

        )

    )


    story.append(
        Spacer(1, 10)
    )


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    image_buffer = io.BytesIO()


    image.save(
        image_buffer,
        format="PNG"
    )


    image_buffer.seek(0)


    report_image = RLImage(

        image_buffer,

        width=100 * mm,

        height=100 * mm

    )


    story.append(
        report_image
    )


    story.append(
        Spacer(1, 12)
    )


    # ========================================================
    # MODALITY SECTION
    # ========================================================

    story.append(

        Paragraph(

            "1. Medical Image Modality",

            heading_style

        )

    )


    modality_data = [

        ["Parameter", "Result"],

        [
            "Detected Modality",
            modality
        ],

        [
            "Modality Confidence",
            f"{modality_confidence * 100:.2f}%"
        ],

        [
            "Chest X-ray Probability",
            f"{modality_probabilities[0] * 100:.2f}%"
        ],

        [
            "CT Probability",
            f"{modality_probabilities[1] * 100:.2f}%"
        ],

        [
            "MRI Probability",
            f"{modality_probabilities[2] * 100:.2f}%"
        ]

    ]


    modality_table = Table(

        modality_data,

        colWidths=[
            75 * mm,
            75 * mm
        ]

    )


    modality_table.setStyle(

        TableStyle(

            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                )

            ]

        )

    )


    story.append(
        modality_table
    )


    story.append(
        Spacer(1, 12)
    )


    # ========================================================
    # LUNG CANCER RESULT
    # ========================================================

    story.append(

        Paragraph(

            "2. Lung Cancer Classification",

            heading_style

        )

    )


    result_data = [

        ["Parameter", "Result"],

        [
            "Prediction",
            predicted_class
        ],

        [
            "Prediction Confidence",
            f"{prediction_confidence * 100:.2f}%"
        ],

        [
            "Normal Probability",
            f"{class_probabilities[0] * 100:.2f}%"
        ],

        [
            "Benign Probability",
            f"{class_probabilities[1] * 100:.2f}%"
        ],

        [
            "Malignant Probability",
            f"{class_probabilities[2] * 100:.2f}%"
        ]

    ]


    result_table = Table(

        result_data,

        colWidths=[
            75 * mm,
            75 * mm
        ]

    )


    result_table.setStyle(

        TableStyle(

            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                )

            ]

        )

    )


    story.append(
        result_table
    )


    story.append(
        Spacer(1, 12)
    )


    # ========================================================
    # MODEL PIPELINE
    # ========================================================

    story.append(

        Paragraph(

            "3. Model Pipeline",

            heading_style

        )

    )


    pipeline_text = (

        "CT Modality Verification → "
        "EfficientNetB0 Block5c → "
        "GAP + GSDP → "
        "GSFF Feature Fusion → "
        "RobustScaler → "
        "RBF-SVM"

    )


    story.append(

        Paragraph(

            pipeline_text,

            normal_style

        )

    )


    story.append(
        Spacer(1, 12)
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    story.append(

        Paragraph(

            "<b>Disclaimer:</b> "
            "This system is a research prototype "
            "developed for educational and research "
            "purposes. It is not intended to provide "
            "clinical diagnosis or replace professional "
            "medical evaluation.",

            normal_style

        )

    )


    document.build(
        story
    )


    buffer.seek(0)


    return buffer.getvalue()


# ============================================================
# 30. FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(

    "📤 Upload Lung CT Image",

    type=[

        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"

    ]

)


# ============================================================
# 31. PROCESS IMAGE
# ============================================================

if uploaded_file is not None:


    # ========================================================
    # READ IMAGE
    # ========================================================

    try:

        image = Image.open(
            uploaded_file
        )

        image.load()

    except Exception as e:

        st.error(
            "Unable to read the uploaded image."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🖼️ Uploaded Image'
        '</div>',
        unsafe_allow_html=True
    )


    st.image(

        image,

        caption="Uploaded Image",

        use_container_width=True

    )


    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    is_valid, validation_message = (
        validate_image(
            image
        )
    )


    if not is_valid:

        st.error(
            validation_message
        )

        st.warning(
            "Please upload a valid grayscale "
            "lung CT image."
        )

        st.stop()


    st.success(
        validation_message
    )


    # ========================================================
    # ANALYSIS BUTTON
    # ========================================================

    if st.button(

        "🔍 Analyze CT Image",

        use_container_width=True

    ):


        # ====================================================
        # STEP 1
        # MODALITY VERIFICATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Step 1 — Image Modality Verification'
            '</div>',
            unsafe_allow_html=True
        )


        with st.spinner(
            "Analyzing image modality..."
        ):

            try:

                (
                    modality,
                    modality_confidence,
                    modality_probabilities

                ) = verify_modality(
                    image
                )

            except Exception as e:

                st.error(
                    "Modality verification failed."
                )

                st.exception(e)

                st.stop()


        # ====================================================
        # MODALITY RESULT BOX
        # ====================================================

        st.markdown(
            '<div class="result-box modality-result">',
            unsafe_allow_html=True
        )


        st.write(
            f"### Detected Modality: {modality}"
        )


        st.write(
            f"Confidence: "
            f"{modality_confidence * 100:.2f}%"
        )


        st.progress(
            float(
                modality_confidence
            )
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # ====================================================
        # MODALITY PROBABILITIES
        # ====================================================

        st.subheader(
            "📊 Modality Probability Estimates"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(

                "Chest X-ray",

                f"{modality_probabilities[0] * 100:.2f}%"

            )

            st.progress(
                float(
                    modality_probabilities[0]
                )
            )


        with col2:

            st.metric(

                "CT",

                f"{modality_probabilities[1] * 100:.2f}%"

            )

            st.progress(
                float(
                    modality_probabilities[1]
                )
            )


        with col3:

            st.metric(

                "MRI",

                f"{modality_probabilities[2] * 100:.2f}%"

            )

            st.progress(
                float(
                    modality_probabilities[2]
                )
            )


        # ====================================================
        # CT ACCEPTANCE
        # ====================================================

        if (

            modality == "CT"

            and

            modality_confidence
            >= MODALITY_THRESHOLD

        ):

            st.success(

                f"✅ CT scan verified "
                f"({modality_confidence * 100:.2f}% confidence)"

            )


        # ====================================================
        # X-RAY REJECTION
        # ====================================================

        elif modality == "CHEST_XRAY":

            st.error(
                "Chest X-ray detected."
            )


            st.warning(

                "Lung cancer classification "
                "was not performed because "
                "the analysis pipeline is designed "
                "for CT images."

            )


            st.stop()


        # ====================================================
        # MRI REJECTION
        # ====================================================

        elif modality == "MRI":

            st.error(
                "MRI image detected."
            )


            st.warning(

                "Lung cancer classification "
                "was not performed because "
                "the analysis pipeline is designed "
                "for CT images."

            )


            st.stop()


        # ====================================================
        # LOW CONFIDENCE
        # ====================================================

        else:

            st.error(

                "The image could not be verified "
                "as a CT scan with sufficient confidence."

            )


            st.write(

                f"Highest predicted modality: "
                f"{modality}"

            )


            st.write(

                f"Confidence: "
                f"{modality_confidence * 100:.2f}%"

            )


            st.warning(
                "Please upload a clear lung CT image."
            )


            st.stop()


        # ====================================================
        # STEP 2
        # LUNG CANCER CLASSIFICATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Step 2 — Lung Cancer Classification'
            '</div>',
            unsafe_allow_html=True
        )


        with st.spinner(

            "Extracting GSFF features "
            "and classifying the CT image..."

        ):

            try:

                (
                    predicted_class,
                    prediction_confidence,
                    class_probabilities

                ) = predict_lung_cancer(
                    image
                )

            except Exception as e:

                st.error(
                    "Lung cancer prediction failed."
                )

                st.exception(e)

                st.stop()


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🎯 Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )


        if predicted_class == "Normal":

            st.markdown(

                f"""
                <div class="result-box normal-result">
                    <h2>🟢 Normal</h2>
                    <p>
                    No abnormal lung cancer class
                    was predicted by the model.
                    </p>
                </div>
                """,

                unsafe_allow_html=True

            )


        elif predicted_class == "Benign":

            st.markdown(

                f"""
                <div class="result-box benign-result">
                    <h2>🟡 Benign</h2>
                    <p>
                    The model classified the CT image
                    as Benign.
                    </p>
                </div>
                """,

                unsafe_allow_html=True

            )


        else:

            st.markdown(

                f"""
                <div class="result-box malignant-result">
                    <h2>🔴 Malignant</h2>
                    <p>
                    The model classified the CT image
                    as Malignant.
                    </p>
                </div>
                """,

                unsafe_allow_html=True

            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.subheader(
            "Prediction Confidence"
        )


        st.metric(

            "Confidence",

            f"{prediction_confidence * 100:.2f}%"

        )


        st.progress(

            float(
                prediction_confidence
            )

        )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.subheader(
            "📊 Class Probability Estimates"
        )


        probability_col1, probability_col2, probability_col3 = (
            st.columns(3)
        )


        with probability_col1:

            st.metric(

                "Normal",

                f"{class_probabilities[0] * 100:.2f}%"

            )

            st.progress(

                float(
                    class_probabilities[0]
                )

            )


        with probability_col2:

            st.metric(

                "Benign",

                f"{class_probabilities[1] * 100:.2f}%"

            )

            st.progress(

                float(
                    class_probabilities[1]
                )

            )


        with probability_col3:

            st.metric(

                "Malignant",

                f"{class_probabilities[2] * 100:.2f}%"

            )

            st.progress(

                float(
                    class_probabilities[2]
                )

            )


        # ====================================================
        # PDF REPORT
        # ====================================================

        st.subheader(
            "📄 Analysis Report"
        )


        try:

            pdf_bytes = create_pdf_report(

                image=image,

                modality=modality,

                modality_confidence=(
                    modality_confidence
                ),

                modality_probabilities=(
                    modality_probabilities
                ),

                predicted_class=(
                    predicted_class
                ),

                prediction_confidence=(
                    prediction_confidence
                ),

                class_probabilities=(
                    class_probabilities
                )

            )


            st.download_button(

                label="📥 Download PDF Report",

                data=pdf_bytes,

                file_name=(
                    "PulmoVision_Lung_Cancer_Report.pdf"
                ),

                mime="application/pdf",

                use_container_width=True

            )


        except Exception as e:

            st.warning(
                "PDF report could not be generated."
            )

            st.exception(e)


        # ====================================================
        # MEDICAL DISCLAIMER
        # ====================================================

        st.info(

            "⚠️ This system is a research prototype "
            "developed for educational and research "
            "purposes. It is not intended to provide "
            "clinical diagnosis or replace professional "
            "medical evaluation."

        )


# ============================================================
# 32. FOOTER
# ============================================================

st.divider()


st.markdown(

    """
    <div style="
        text-align:center;
        font-size:14px;
        line-height:1.7;
    ">

    🫁 <b>PulmoVision</b><br>

    AI-Powered Lung Cancer Detection System<br>

    <small>
    EfficientNetB0 • GAP + GSDP • GSFF •
    RobustScaler • RBF-SVM
    </small>

    </div>
    """,

    unsafe_allow_html=True

)
