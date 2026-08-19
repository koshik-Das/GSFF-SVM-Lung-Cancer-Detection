# ============================================================
# app.py
# PulmoVision - AI-Powered Lung Cancer Detection System
#
# LUNG-CANCER PIPELINE KEPT UNCHANGED
#
# Uploaded Image
#       ↓
# Basic Image Validation / Colour Image Detection
#       ↓
# CT / Chest X-ray / MRI Modality Classification
#       ↓
# CT Verification
#       ↓
# EfficientNetB0 -> Block5c
#       ↓
# GAP + GSDP -> GSFF Feature Fusion
#       ↓
# RobustScaler
#       ↓
# RBF-SVM
#       ↓
# Normal / Benign / Malignant
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import io
import hashlib

from pathlib import Path
from datetime import datetime

import joblib
import keras
import numpy as np
import streamlit as st

from PIL import Image

from tensorflow.keras.applications.efficientnet import (
    preprocess_input
)

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)

from reportlab.lib.units import mm

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PulmoVision",
    page_icon="🫁",
    layout="wide",
)


# ============================================================
# 2. BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 3. MODEL FILE PATHS
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
# HEADER IMAGE
# ============================================================

LUNG_ICON_PATH = (
    BASE_DIR / "Lung_Icon.png"
)


# ============================================================
# 4. IMAGE SETTINGS
# ============================================================

CT_VERIFIER_IMAGE_SIZE = (
    128,
    128,
)

FEATURE_EXTRACTOR_IMAGE_SIZE = (
    224,
    224,
)


# ============================================================
# 5. THRESHOLDS
# ============================================================

MODALITY_THRESHOLD = 0.60

COLOR_TOLERANCE = 8.0


# ============================================================
# 6. CLASS MAPPINGS
# ============================================================

class_names = [
    "Normal",
    "Benign",
    "Malignant",
]


# ============================================================
# MODALITY MAPPING
#
# 0 = CHEST_XRAY
# 1 = CT
# 2 = MRI
# ============================================================

modality_names = [
    "CHEST_XRAY",
    "CT",
    "MRI",
]


# ============================================================
# 7. CUSTOM GSDP LAYER
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

    def get_config(
        self
    ):

        return super().get_config()


# ============================================================
# 8. VALIDATE MODEL FILE
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
# 9. REQUIRED FILES
# ============================================================

required_files = {

    "CT Verifier":
        CT_VERIFIER_PATH,

    "GSFF Feature Extractor":
        FEATURE_EXTRACTOR_PATH,

    "RobustScaler":
        SCALER_PATH,

    "SVM Classifier":
        SVM_PATH,
}


for model_name, model_path in (
    required_files.items()
):

    validate_model_file(
        model_path,
        model_name
    )


# ============================================================
# 10. LOAD CT VERIFIER
# ============================================================

@st.cache_resource
def load_ct_verifier():

    model = keras.models.load_model(

        CT_VERIFIER_PATH,

        compile=False,
    )

    return model


# ============================================================
# 11. LOAD GSFF FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def load_feature_extractor():

    model = keras.models.load_model(

        FEATURE_EXTRACTOR_PATH,

        custom_objects={

            "GSDP":
                GSDP,

            "GSFF>GSDP":
                GSDP,
        },

        compile=False,

        safe_mode=False,
    )

    return model


# ============================================================
# 12. LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    return joblib.load(
        SCALER_PATH
    )


# ============================================================
# 13. LOAD SVM
# ============================================================

@st.cache_resource
def load_svm():

    return joblib.load(
        SVM_PATH
    )


# ============================================================
# 14. LOAD ALL MODELS
# ============================================================

try:

    ct_verifier = (
        load_ct_verifier()
    )

except Exception as e:

    st.error(
        "CT verifier loading failed."
    )

    st.exception(
        e
    )

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

    st.exception(
        e
    )

    st.stop()


try:

    scaler = (
        load_scaler()
    )

except Exception as e:

    st.error(
        "RobustScaler loading failed."
    )

    st.exception(
        e
    )

    st.stop()


try:

    svm = (
        load_svm()
    )

except Exception as e:

    st.error(
        "SVM classifier loading failed."
    )

    st.exception(
        e
    )

    st.stop()


# ============================================================
# 15. VERIFY CT VERIFIER OUTPUT
# ============================================================

try:

    ct_output_shape = (
        ct_verifier.output_shape
    )

    if (
        ct_output_shape[-1]
        !=
        3
    ):

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

    st.exception(
        e
    )

    st.stop()


# ============================================================
# 16. VERIFY FEATURE EXTRACTOR OUTPUT
# ============================================================

try:

    feature_output_shape = (
        feature_extractor.output_shape
    )

    expected_feature_dimension = (
        224
    )

    if (
        feature_output_shape[-1]
        !=
        expected_feature_dimension
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
        "Unable to verify GSFF "
        "feature dimension."
    )

    st.exception(
        e
    )

    st.stop()


# ============================================================
# 17. SESSION STATE
# ============================================================

if (
    "analysis_result"
    not in st.session_state
):

    st.session_state.analysis_result = (
        None
    )


if (
    "pdf_report"
    not in st.session_state
):

    st.session_state.pdf_report = (
        None
    )


if (
    "uploaded_signature"
    not in st.session_state
):

    st.session_state.uploaded_signature = (
        None
    )


# ============================================================
# 18. INTERFACE CSS
# ============================================================

st.markdown(
    """
    <style>


    /* =======================================================
       MAIN PAGE
       ======================================================= */

    .block-container {

        padding-top: 1.25rem;

        padding-bottom: 2.2rem;

        max-width: 100%;
    }


    h1,
    h2,
    h3 {

        color: #1f2937;
    }


    /* =======================================================
       PULMOVISION HEADER
       ======================================================= */

    .main-title {

        text-align: center;

        font-size: 42px;

        font-weight: 750;

        line-height: 1.15;

        margin-top: 4px;

        margin-bottom: 8px;

        color: #1f2937;
    }


    .brand-name {

        text-align: center;

        font-size: 17px;

        font-weight: 650;

        line-height: 1.4;

        margin-top: 4px;

        margin-bottom: 10px;

        color: #1f2937;
    }


    .subtitle {

        text-align: center;

        font-size: 16px;

        line-height: 1.6;

        margin-top: 8px;

        margin-bottom: 10px;

        color: #1f2937;
    }


    .header-divider {

        border: none;

        border-top: 1px solid #d1d5db;

        margin: 18px 0 28px 0;
    }


    /* =======================================================
       ANALYZE BUTTON
       ======================================================= */

    div[data-testid="stButton"] > button {

        width: 100%;

        min-height: 42px;

        border: 0;

        border-radius: 8px;

        background: #ff4b4b;

        color: white;

        font-weight: 600;
    }


    div[data-testid="stButton"] > button:hover {

        background: #f13f3f;

        color: white;

        border: 0;
    }


    div[data-testid="stButton"] > button:focus {

        color: white;

        border: 0;

        box-shadow: none;
    }


    /* =======================================================
       FINAL RESULT
       ======================================================= */

    .final-result {

        width: 100%;

        box-sizing: border-box;

        border-radius: 8px;

        padding: 24px 16px;

        margin-top: 8px;

        margin-bottom: 24px;

        font-size: 26px;

        font-weight: 700;
    }


    .final-normal {

        background: #e4f6e9;

        color: #008b3d;
    }


    .final-benign {

        background: #fff4d6;

        color: #9a6700;
    }


    .final-malignant {

        background: #fde8e8;

        color: #c62828;
    }


    /* =======================================================
       MODALITY RESULT
       ======================================================= */

    .modality-text {

        font-size: 26px;

        font-weight: 650;

        margin-top: 4px;

        margin-bottom: 22px;

        color: #1f2937;
    }


    /* =======================================================
       REPORT
       ======================================================= */

    .report-divider {

        border: none;

        border-top: 1px solid #d1d5db;

        margin: 30px 0 36px 0;
    }


    .research-note {

        margin-top: 24px;

        color: #6b7280;

        font-size: 13px;
    }


    div[data-testid="stDownloadButton"] > button {

        width: 100%;

        min-height: 42px;

        border-radius: 8px;
    }


    </style>
    """,

    unsafe_allow_html=True,
)


# ============================================================
# 19. COLOR IMAGE CHECK
# ============================================================

def check_color_image(
    image
):

    rgb = np.asarray(

        image.convert(
            "RGB"
        ),

        dtype=np.float32
    )


    red = (
        rgb[:, :, 0]
    )

    green = (
        rgb[:, :, 1]
    )

    blue = (
        rgb[:, :, 2]
    )


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

        float(
            average_difference
        )
    )


# ============================================================
# 20. BASIC IMAGE VALIDATION
# ============================================================

def validate_image(
    image
):

    width, height = (
        image.size
    )


    # --------------------------------------------------------
    # IMAGE RESOLUTION
    # --------------------------------------------------------

    if (
        width < 64
        or
        height < 64
    ):

        return (

            False,

            "Image resolution is too small."
        )


    # --------------------------------------------------------
    # EMPTY IMAGE
    # --------------------------------------------------------

    array = np.asarray(
        image
    )


    if (
        array.size
        ==
        0
    ):

        return (

            False,

            "Image is empty."
        )


    # --------------------------------------------------------
    # COLOR IMAGE CHECK
    # --------------------------------------------------------

    is_color, _ = (
        check_color_image(
            image
        )
    )


    if is_color:

        return (

            False,

            "Color image detected. "
            "Please input a lung CT image."
        )


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = np.asarray(

        image.convert(
            "L"
        ),

        dtype=np.float32
    )


    # --------------------------------------------------------
    # BLANK / LOW CONTRAST IMAGE
    # --------------------------------------------------------

    if (
        np.std(
            gray
        )
        <
        8
    ):

        return (

            False,

            "Image appears blank or invalid."
        )


    # --------------------------------------------------------
    # BLACK IMAGE
    # --------------------------------------------------------

    dark_ratio = np.mean(
        gray < 10
    )


    if (
        dark_ratio
        >
        0.98
    ):

        return (

            False,

            "Image is almost completely black."
        )


    # --------------------------------------------------------
    # WHITE IMAGE
    # --------------------------------------------------------

    bright_ratio = np.mean(
        gray > 245
    )


    if (
        bright_ratio
        >
        0.98
    ):

        return (

            False,

            "Image is almost completely white."
        )


    return (

        True,

        "Image passed basic validation."
    )


# ============================================================
# 21. PREPROCESS FOR CT VERIFIER
# ============================================================

def preprocess_for_ct_verifier(
    image
):

    image = image.convert(
        "RGB"
    )


    image = image.resize(

        CT_VERIFIER_IMAGE_SIZE,

        Image.Resampling.LANCZOS,
    )


    image_array = np.asarray(

        image,

        dtype=np.float32
    )


    # --------------------------------------------------------
    # SAME AS TRAINING
    # ImageDataGenerator(rescale=1/255)
    # --------------------------------------------------------

    image_array /= (
        255.0
    )


    image_array = np.expand_dims(

        image_array,

        axis=0
    )


    return image_array


# ============================================================
# 22. PREPROCESS FOR GSFF
# ============================================================

def preprocess_for_gsff(
    image
):

    image = image.convert(
        "RGB"
    )


    image = image.resize(

        FEATURE_EXTRACTOR_IMAGE_SIZE,

        Image.Resampling.LANCZOS,
    )


    image_array = np.asarray(

        image,

        dtype=np.float32
    )


    image_array = np.expand_dims(

        image_array,

        axis=0
    )


    image_array = (
        preprocess_input(
            image_array
        )
    )


    return image_array


# ============================================================
# 23. MODALITY VERIFICATION
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

            verbose=0,
        )
    )


    probabilities = np.asarray(

        predictions[0],

        dtype=np.float32
    )


    # --------------------------------------------------------
    # SAFETY NORMALIZATION
    # --------------------------------------------------------

    probability_sum = (
        np.sum(
            probabilities
        )
    )


    if (
        probability_sum
        >
        0
    ):

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
# 24. GSFF FEATURE EXTRACTION
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

            verbose=0,
        )
    )


    features = np.asarray(

        features,

        dtype=np.float32
    )


    return features


# ============================================================
# 25. LUNG CANCER PREDICTION
# ============================================================

def predict_lung_cancer(
    image
):

    # ========================================================
    # GSFF FEATURE EXTRACTION
    # ========================================================

    features = (
        extract_gsff_features(
            image
        )
    )


    # ========================================================
    # ROBUST SCALER
    # ========================================================

    scaled_features = (
        scaler.transform(
            features
        )
    )


    # ========================================================
    # RBF SVM
    # ========================================================

    prediction = (
        svm.predict(
            scaled_features
        )
    )


    predicted_index = int(
        prediction[0]
    )


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
# 26. PDF REPORT
# ============================================================

def create_pdf_report(

    image,

    modality,

    modality_confidence,

    modality_probabilities,

    predicted_class,

    prediction_confidence,

    class_probabilities,
):

    buffer = (
        io.BytesIO()
    )


    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=18 * mm,

        leftMargin=18 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm,
    )


    styles = (
        getSampleStyleSheet()
    )


    # --------------------------------------------------------
    # TITLE STYLE
    # --------------------------------------------------------

    title_style = ParagraphStyle(

        "ReportTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=18,

        leading=22,

        spaceAfter=12,
    )


    # --------------------------------------------------------
    # SUBTITLE STYLE
    # --------------------------------------------------------

    subtitle_style = ParagraphStyle(

        "ReportSubtitle",

        parent=styles["Normal"],

        alignment=TA_CENTER,

        fontSize=11,

        leading=14,

        spaceAfter=15,
    )


    # --------------------------------------------------------
    # HEADING STYLE
    # --------------------------------------------------------

    heading_style = ParagraphStyle(

        "ReportHeading",

        parent=styles["Heading2"],

        fontSize=13,

        leading=16,

        spaceBefore=10,

        spaceAfter=8,
    )


    # --------------------------------------------------------
    # NORMAL STYLE
    # --------------------------------------------------------

    normal_style = ParagraphStyle(

        "ReportNormal",

        parent=styles["Normal"],

        fontSize=10,

        leading=14,
    )


    story = []


    # ========================================================
    # TITLE
    # ========================================================

    story.append(

        Paragraph(

            "PulmoVision<br/>"
            "AI-Powered Lung Cancer Detection System",

            title_style,
        )
    )


    # ========================================================
    # REPORT SUBTITLE
    # ========================================================

    story.append(

        Paragraph(

            "Lung CT Image Analysis Report",

            subtitle_style,
        )
    )


    # ========================================================
    # DATE
    # ========================================================

    report_time = (
        datetime.now().strftime(
            "%d %B %Y, %I:%M:%S %p"
        )
    )


    story.append(

        Paragraph(

            f"<b>Analysis Date:</b> "
            f"{report_time}",

            normal_style,
        )
    )


    story.append(
        Spacer(
            1,
            10
        )
    )


    # ========================================================
    # IMAGE
    # ========================================================

    image_buffer = (
        io.BytesIO()
    )


    image.convert(
        "RGB"
    ).save(

        image_buffer,

        format="PNG"
    )


    image_buffer.seek(
        0
    )


    report_image = RLImage(

        image_buffer,

        width=100 * mm,

        height=100 * mm,
    )


    story.append(
        report_image
    )


    story.append(
        Spacer(
            1,
            12
        )
    )


    # ========================================================
    # MODALITY SECTION
    # ========================================================

    story.append(

        Paragraph(

            "1. Medical Image Modality",

            heading_style,
        )
    )


    modality_data = [

        [
            "Parameter",
            "Result"
        ],

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
        ],
    ]


    modality_table = Table(

        modality_data,

        colWidths=[
            75 * mm,
            75 * mm
        ],
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
                ),
            ]
        )
    )


    story.append(
        modality_table
    )


    story.append(
        Spacer(
            1,
            12
        )
    )


    # ========================================================
    # LUNG CANCER RESULT
    # ========================================================

    story.append(

        Paragraph(

            "2. Lung Cancer Classification",

            heading_style,
        )
    )


    result_data = [

        [
            "Parameter",
            "Result"
        ],

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
        ],
    ]


    result_table = Table(

        result_data,

        colWidths=[
            75 * mm,
            75 * mm
        ],
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
                ),
            ]
        )
    )


    story.append(
        result_table
    )


    story.append(
        Spacer(
            1,
            12
        )
    )


    # ========================================================
    # PIPELINE SECTION
    # ========================================================

    story.append(

        Paragraph(

            "3. Model Pipeline",

            heading_style,
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

            normal_style,
        )
    )


    story.append(
        Spacer(
            1,
            12
        )
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

            normal_style,
        )
    )


    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story
    )


    buffer.seek(
        0
    )


    return (
        buffer.getvalue()
    )


# ============================================================
# 27. FRIENDLY MODALITY NAME
# ============================================================

def friendly_modality_name(
    modality
):

    if (
        modality
        ==
        "CHEST_XRAY"
    ):

        return "X-ray"


    if (
        modality
        ==
        "CT"
    ):

        return "CT"


    if (
        modality
        ==
        "MRI"
    ):

        return "MRI"


    return modality


# ============================================================
# 28. FINAL RESULT
# ============================================================

def render_final_result(
    predicted_class
):

    if (
        predicted_class
        ==
        "Normal"
    ):

        css_class = (
            "final-normal"
        )


    elif (
        predicted_class
        ==
        "Benign"
    ):

        css_class = (
            "final-benign"
        )


    else:

        css_class = (
            "final-malignant"
        )


    st.markdown(

        f"""
        <div class="final-result {css_class}">
            Final Result: {predicted_class}
        </div>
        """,

        unsafe_allow_html=True,
    )


# ============================================================
# 29. RESET SESSION RESULTS
# ============================================================

def reset_analysis_state():

    st.session_state.analysis_result = (
        None
    )

    st.session_state.pdf_report = (
        None
    )


# ============================================================
# 30. HEADER
# ============================================================

header_left, header_center, header_right = (

    st.columns(

        [
            1.25,
            8.5,
            1.25
        ],

        vertical_alignment="center",
    )
)


# ============================================================
# HEADER IMAGE
# ============================================================

with header_left:

    if (
        LUNG_ICON_PATH.exists()
    ):

        st.image(

            str(
                LUNG_ICON_PATH
            ),

            width=105,
        )


    else:

        st.markdown(

            """
            <div style="
                font-size:74px;
                text-align:center;
                line-height:1;
            ">
                🫁
            </div>
            """,

            unsafe_allow_html=True,
        )


# ============================================================
# HEADER TITLE
# ============================================================

with header_center:

    st.markdown(

        """
        <div class="main-title">
            PulmoVision
        </div>

        <div class="brand-name">
            AI-Powered Lung Cancer Detection System
        </div>

        <div class="subtitle">
            Intelligent analysis of lung CT images using
            deep feature extraction and machine learning.
        </div>
        """,

        unsafe_allow_html=True,
    )


# ============================================================
# EMPTY RIGHT COLUMN
#
# This balances the lung image on the left so that
# PulmoVision remains visually centered on the page.
# ============================================================

with header_right:

    st.write(
        ""
    )


# ============================================================
# HEADER DIVIDER
# ============================================================

st.markdown(

    '<hr class="header-divider">',

    unsafe_allow_html=True,
)


# ============================================================
# 31. UPLOAD MEDICAL IMAGE
# ============================================================

st.header(
    "Upload Medical Image"
)


uploaded_file = st.file_uploader(

    "Choose an image",

    type=[

        "jpg",

        "jpeg",

        "png",

        "bmp",

        "webp",
    ],
)


# ============================================================
# 32. PROCESS IMAGE
# ============================================================

if (
    uploaded_file
    is not None
):


    # ========================================================
    # READ UPLOADED BYTES
    # ========================================================

    uploaded_bytes = (
        uploaded_file.getvalue()
    )


    # ========================================================
    # IMAGE SIGNATURE
    # ========================================================

    file_signature = (

        hashlib.sha256(
            uploaded_bytes
        ).hexdigest()
    )


    # ========================================================
    # RESET IF DIFFERENT IMAGE IS UPLOADED
    # ========================================================

    if (
        st.session_state.uploaded_signature
        !=
        file_signature
    ):

        st.session_state.uploaded_signature = (
            file_signature
        )

        reset_analysis_state()


    # ========================================================
    # OPEN IMAGE
    # ========================================================

    try:

        image = Image.open(

            io.BytesIO(
                uploaded_bytes
            )
        )

        image.load()


    except Exception as e:

        st.error(
            "Unable to read the uploaded image."
        )

        st.exception(
            e
        )

        st.stop()


    # ========================================================
    # UPLOADED MEDICAL IMAGE
    # ========================================================

    st.header(
        "Uploaded Medical Image"
    )


    st.image(

        image,

        caption="Uploaded Medical Image",

        width=500,
    )


    # ========================================================
    # ANALYSIS BUTTON
    # ========================================================

    analyze_clicked = st.button(

        "Check Your Image By Initiating AI Analysis",

        use_container_width=True,
    )


    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    if (
        analyze_clicked
    ):

        reset_analysis_state()


        # ====================================================
        # BASIC VALIDATION
        # ====================================================

        is_valid, validation_message = (

            validate_image(
                image
            )
        )


        if (
            not is_valid
        ):

            st.error(
                validation_message
            )


            st.warning(

                "Please upload a valid grayscale "
                "lung CT image."
            )


            st.stop()


        # ====================================================
        # VALIDATION SUCCESS
        # ====================================================

        st.success(
            validation_message
        )


        # ====================================================
        # MEDICAL IMAGE TYPE CLASSIFICATION
        # ====================================================

        try:

            with st.spinner(
                "Detecting medical image type..."
            ):

                (
                    modality,

                    modality_confidence,

                    modality_probabilities,

                ) = verify_modality(
                    image
                )


        except Exception as e:

            st.error(
                "Medical image type detection failed."
            )

            st.exception(
                e
            )

            st.stop()


        # ====================================================
        # DETECTED MEDICAL IMAGE TYPE
        # ====================================================

        st.header(
            "Detected Medical Image Type"
        )


        st.markdown(

            f"""
            <div class="modality-text">
                {friendly_modality_name(modality)}
            </div>
            """,

            unsafe_allow_html=True,
        )


        # ====================================================
        # CT VERIFICATION
        # ====================================================

        st.header(
            "CT Verification"
        )


        # ----------------------------------------------------
        # CT ACCEPTED
        # ----------------------------------------------------

        if (

            modality
            ==
            "CT"

            and

            modality_confidence
            >=
            MODALITY_THRESHOLD
        ):

            st.success(
                "Chest CT verified successfully."
            )


        # ----------------------------------------------------
        # X-RAY REJECTED
        # ----------------------------------------------------

        elif (
            modality
            ==
            "CHEST_XRAY"
        ):

            st.error(
                "Chest X-ray detected."
            )


            st.warning(

                "Lung cancer classification "
                "was not performed because "
                "this pipeline is designed "
                "for CT images."
            )


            st.stop()


        # ----------------------------------------------------
        # MRI REJECTED
        # ----------------------------------------------------

        elif (
            modality
            ==
            "MRI"
        ):

            st.error(
                "MRI image detected."
            )


            st.warning(

                "Lung cancer classification "
                "was not performed because "
                "this pipeline is designed "
                "for CT images."
            )


            st.stop()


        # ----------------------------------------------------
        # LOW CONFIDENCE
        # ----------------------------------------------------

        else:

            st.error(

                "The image could not be verified "
                "as a CT scan with sufficient confidence."
            )


            st.warning(
                "Please upload a clear lung CT image."
            )


            st.stop()


        # ====================================================
        # LUNG CANCER ANALYSIS
        #
        # EfficientNetB0
        #       ↓
        # Block5c
        #       ↓
        # GAP + GSDP
        #       ↓
        # GSFF
        #       ↓
        # RobustScaler
        #       ↓
        # RBF-SVM
        # ====================================================

        try:

            with st.spinner(
                "Running lung cancer analysis..."
            ):

                (
                    predicted_class,

                    prediction_confidence,

                    class_probabilities,

                ) = predict_lung_cancer(
                    image
                )


        except Exception as e:

            st.error(
                "Lung cancer prediction failed."
            )

            st.exception(
                e
            )

            st.stop()


        # ====================================================
        # CREATE PDF
        # ====================================================

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
                ),
            )


        except Exception as e:

            pdf_bytes = (
                None
            )

            st.warning(
                "PDF report could not be generated."
            )

            st.exception(
                e
            )


        # ====================================================
        # SAVE RESULTS
        # ====================================================

        st.session_state.analysis_result = {

            "modality":
                modality,

            "modality_confidence":
                modality_confidence,

            "modality_probabilities":
                modality_probabilities,

            "predicted_class":
                predicted_class,

            "prediction_confidence":
                prediction_confidence,

            "class_probabilities":
                class_probabilities,
        }


        st.session_state.pdf_report = (
            pdf_bytes
        )


    # ========================================================
    # KEEP RESULTS AFTER STREAMLIT RERUN
    # ========================================================

    result = (
        st.session_state.analysis_result
    )


    if (
        result is not None
    ):


        # ====================================================
        # SHOW VALIDATION AGAIN AFTER DOWNLOAD RERUN
        # ====================================================

        if (
            not analyze_clicked
        ):

            st.success(
                "Image passed basic validation."
            )


            # =================================================
            # DETECTED MEDICAL IMAGE TYPE
            # =================================================

            st.header(
                "Detected Medical Image Type"
            )


            st.markdown(

                f"""
                <div class="modality-text">
                    {
                        friendly_modality_name(
                            result["modality"]
                        )
                    }
                </div>
                """,

                unsafe_allow_html=True,
            )


            # =================================================
            # CT VERIFICATION
            # =================================================

            st.header(
                "CT Verification"
            )


            st.success(
                "Chest CT verified successfully."
            )


        # ====================================================
        # LUNG CANCER DETECTION
        # ====================================================

        st.header(
            "Lung Cancer Detection"
        )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        render_final_result(

            result[
                "predicted_class"
            ]
        )


        # ====================================================
        # REPORT DIVIDER
        # ====================================================

        st.markdown(

            '<hr class="report-divider">',

            unsafe_allow_html=True,
        )


        # ====================================================
        # ANALYSIS REPORT
        # ====================================================

        st.header(
            "Analysis Report"
        )


        # ====================================================
        # PDF BUTTON
        # ====================================================

        if (
            st.session_state.pdf_report
            is not None
        ):

            st.download_button(

                label=(
                    "Download Final Report (PDF)"
                ),

                data=(
                    st.session_state.pdf_report
                ),

                file_name=(
                    "PulmoVision_"
                    "Lung_Cancer_Report.pdf"
                ),

                mime=(
                    "application/pdf"
                ),

                use_container_width=True,
            )


        else:

            st.warning(
                "The PDF report is currently unavailable."
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown(

            """
            <div class="research-note">

                Research prototype for educational/research
                use only. It is not a clinical diagnosis and
                does not replace professional medical evaluation.

            </div>
            """,

            unsafe_allow_html=True,
        )
