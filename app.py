# ============================================================
# app.py
# PulmoVision
# AI-Powered Lung Cancer Detection System
#
# UI-enhanced version
#
# MODEL LOGIC — UNCHANGED:
#
# Uploaded Image
#       ↓
# Colour Image Check
#       ↓
# CT / X-ray / MRI Verification
#       ↓
# CT only
#       ↓
# EfficientNetB0 Block5c
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
# ============================================================

import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import keras
import joblib

from PIL import Image

from tensorflow.keras.applications.efficientnet import (
    preprocess_input
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PulmoVision | AI-Powered Lung Cancer Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(30, 144, 255, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(0, 200, 180, 0.07),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #f7fbff 0%,
                #eef6fb 50%,
                #f8fcff 100%
            );
    }


    /* -------------------------------------------------------
       MAIN CONTAINER
    ------------------------------------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }


    /* -------------------------------------------------------
       HERO
    ------------------------------------------------------- */

    .hero {
        padding: 35px 35px 30px 35px;
        border-radius: 25px;
        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.96),
                rgba(235,247,255,0.96)
            );
        border: 1px solid rgba(30,144,255,0.14);
        box-shadow:
            0 12px 40px rgba(0, 70, 120, 0.10);
        margin-bottom: 25px;
        animation: fadeIn 0.8s ease-out;
    }


    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -1px;
        background:
            linear-gradient(
                90deg,
                #087f8c,
                #1677c8,
                #3b4cca
            );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    .hero-subtitle-main {
        display: block;
        font-size: 20px;
        font-weight: 600;
        margin-top: 4px;
        letter-spacing: 0;
        background: none;
        -webkit-text-fill-color: #526777;
        color: #526777;
    }


    .hero-subtitle {
        font-size: 18px;
        color: #526777;
        line-height: 1.6;
    }


    /* -------------------------------------------------------
       FEATURE CARDS
    ------------------------------------------------------- */

    .feature-card {
        background: rgba(255,255,255,0.90);
        border: 1px solid rgba(30,144,255,0.12);
        border-radius: 18px;
        padding: 20px;
        min-height: 120px;
        box-shadow:
            0 8px 25px rgba(0, 70, 120, 0.07);
        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease;
        animation: slideUp 0.7s ease-out;
    }


    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow:
            0 15px 35px rgba(0, 70, 120, 0.13);
    }


    .feature-icon {
        font-size: 30px;
        margin-bottom: 8px;
    }


    .feature-title {
        font-size: 16px;
        font-weight: 700;
        color: #183b56;
    }


    .feature-text {
        font-size: 13px;
        color: #647b8c;
        margin-top: 5px;
    }


    /* -------------------------------------------------------
       SECTION TITLES
    ------------------------------------------------------- */

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #173b57;
        margin-top: 25px;
        margin-bottom: 15px;
    }


    /* -------------------------------------------------------
       RESULT CARDS
    ------------------------------------------------------- */

    .result-card {
        padding: 28px;
        border-radius: 22px;
        text-align: center;
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(30,144,255,0.12);
        box-shadow:
            0 10px 35px rgba(0, 70, 120, 0.10);
        animation: resultAppear 0.6s ease-out;
    }


    .result-label {
        font-size: 15px;
        color: #718494;
        text-transform: uppercase;
        letter-spacing: 1px;
    }


    .result-value {
        font-size: 36px;
        font-weight: 800;
        margin-top: 7px;
    }


    /* -------------------------------------------------------
       UPLOAD AREA
    ------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.75);
        border-radius: 20px;
        padding: 10px;
        border: 1px dashed rgba(22,119,200,0.35);
    }


    /* -------------------------------------------------------
       BUTTON
    ------------------------------------------------------- */

    .stButton > button {
        border-radius: 14px;
        height: 55px;
        font-size: 17px;
        font-weight: 700;
        border: none;
        background:
            linear-gradient(
                90deg,
                #087f8c,
                #1677c8
            );
        color: white;
        box-shadow:
            0 8px 20px rgba(22,119,200,0.20);
        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }


    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 12px 28px rgba(22,119,200,0.30);
    }


    /* -------------------------------------------------------
       SIDEBAR
    ------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #f4fbff 0%,
                #eaf5fa 100%
            );
        border-right: 1px solid rgba(30,144,255,0.10);
    }


    .sidebar-title {
        font-size: 22px;
        font-weight: 800;
        color: #173b57;
    }


    /* -------------------------------------------------------
       ANIMATIONS
    ------------------------------------------------------- */

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(-15px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }


    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(15px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }


    @keyframes resultAppear {
        from {
            opacity: 0;
            transform: scale(0.96);
        }

        to {
            opacity: 1;
            transform: scale(1);
        }
    }


    /* -------------------------------------------------------
       FOOTER
    ------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #7890a0;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(30,144,255,0.10);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL FILE PATHS
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
# SETTINGS
# ============================================================

CT_VERIFIER_IMAGE_SIZE = (128, 128)

FEATURE_EXTRACTOR_IMAGE_SIZE = (224, 224)

MODALITY_THRESHOLD = 0.60


class_names = [
    "Normal",
    "Benign",
    "Malignant"
]


modality_names = [
    "CHEST_XRAY",
    "CT",
    "MRI"
]


# ============================================================
# CUSTOM GSDP LAYER
# ============================================================

@keras.saving.register_keras_serializable(
    package="GSFF"
)
class GSDP(keras.layers.Layer):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)


    def call(self, inputs):

        return keras.ops.std(
            inputs,
            axis=(1, 2)
        )


    def get_config(self):

        config = super().get_config()

        return config


# ============================================================
# CHECK REQUIRED MODEL FILES
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


for model_name, model_path in required_files.items():

    if not model_path.exists():

        st.error(
            f"❌ {model_name} was not found."
        )

        st.code(
            str(model_path)
        )

        st.stop()


    if model_path.stat().st_size == 0:

        st.error(
            f"❌ {model_name} file is empty."
        )

        st.stop()


# ============================================================
# LOAD CT VERIFIER
# ============================================================

@st.cache_resource
def load_ct_verifier():

    model = keras.models.load_model(
        CT_VERIFIER_PATH,
        compile=False
    )

    return model


# ============================================================
# LOAD GSFF FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def load_feature_extractor():

    model = keras.models.load_model(

        FEATURE_EXTRACTOR_PATH,

        custom_objects={
            "GSDP": GSDP,
            "GSFF>GSDP": GSDP
        },

        compile=False,

        safe_mode=False
    )

    return model


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    return joblib.load(
        SCALER_PATH
    )


# ============================================================
# LOAD SVM
# ============================================================

@st.cache_resource
def load_svm():

    return joblib.load(
        SVM_PATH
    )


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    ct_verifier = load_ct_verifier()

except Exception as e:

    st.error(
        "❌ CT verifier loading failed."
    )

    st.exception(e)

    st.stop()


try:

    feature_extractor = load_feature_extractor()

except Exception as e:

    st.error(
        "❌ GSFF feature extractor loading failed."
    )

    st.exception(e)

    st.stop()


try:

    scaler = load_scaler()

except Exception as e:

    st.error(
        "❌ RobustScaler loading failed."
    )

    st.exception(e)

    st.stop()


try:

    svm = load_svm()

except Exception as e:

    st.error(
        "❌ SVM classifier loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY CT VERIFIER OUTPUT
# ============================================================

try:

    ct_verifier_output_shape = (
        ct_verifier.output_shape
    )

    if ct_verifier_output_shape[-1] != 3:

        st.error(
            "❌ CT verifier output mismatch."
        )

        st.write(
            "Expected output: 3 classes"
        )

        st.write(
            f"Actual output: "
            f"{ct_verifier_output_shape}"
        )

        st.stop()

except Exception as e:

    st.error(
        "❌ Unable to verify CT verifier output."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY GSFF FEATURE DIMENSION
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
            "❌ GSFF feature dimension mismatch."
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
        "❌ Unable to verify GSFF feature dimension."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE COLOUR VALIDATION
# ============================================================

def check_grayscale_image(image):

    try:

        rgb_image = image.convert("RGB")

        rgb_array = np.asarray(
            rgb_image,
            dtype=np.float32
        )

        r = rgb_array[:, :, 0]

        g = rgb_array[:, :, 1]

        b = rgb_array[:, :, 2]


        rg_difference = np.mean(
            np.abs(r - g)
        )

        gb_difference = np.mean(
            np.abs(g - b)
        )

        rb_difference = np.mean(
            np.abs(r - b)
        )


        average_channel_difference = (

            rg_difference
            + gb_difference
            + rb_difference

        ) / 3.0


        if average_channel_difference > 3.0:

            return (
                False,
                "Colour image detected."
            )


        return (
            True,
            "Grayscale image detected."
        )


    except Exception as e:

        return (
            False,
            f"Unable to validate image: {e}"
        )


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_image(image):

    width, height = image.size


    if width < 64 or height < 64:

        return (
            False,
            "Image resolution is too small."
        )


    is_grayscale, message = (
        check_grayscale_image(image)
    )


    if not is_grayscale:

        return (
            False,
            message
        )


    gray = image.convert("L")

    gray_array = np.asarray(
        gray,
        dtype=np.float32
    )


    standard_deviation = np.std(
        gray_array
    )


    if standard_deviation < 8:

        return (
            False,
            "Image appears blank or invalid."
        )


    dark_ratio = np.mean(
        gray_array < 10
    )


    if dark_ratio > 0.98:

        return (
            False,
            "Image is almost completely black."
        )


    bright_ratio = np.mean(
        gray_array > 245
    )


    if bright_ratio > 0.98:

        return (
            False,
            "Image is almost completely white."
        )


    return (
        True,
        "Image passed basic validation."
    )


# ============================================================
# PREPROCESS FOR CT VERIFIER
# ============================================================

def preprocess_for_ct_verifier(image):

    image = image.convert("RGB")

    image = image.resize(
        CT_VERIFIER_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREPROCESS FOR GSFF
# ============================================================

def preprocess_for_gsff(image):

    image = image.convert("RGB")

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

    image_array = preprocess_input(
        image_array
    )

    return image_array


# ============================================================
# MODALITY VERIFICATION
# ============================================================

def verify_modality(image):

    image_array = (
        preprocess_for_ct_verifier(
            image
        )
    )


    predictions = ct_verifier.predict(
        image_array,
        verbose=0
    )


    probabilities = np.asarray(
        predictions[0],
        dtype=np.float32
    )


    probability_sum = np.sum(
        probabilities
    )


    if probability_sum > 0:

        probabilities = (
            probabilities
            / probability_sum
        )


    predicted_class = int(
        np.argmax(probabilities)
    )


    confidence = float(
        probabilities[predicted_class]
    )


    modality = modality_names[
        predicted_class
    ]


    return (
        modality,
        confidence,
        probabilities
    )


# ============================================================
# GSFF FEATURE EXTRACTION
# ============================================================

def extract_gsff_features(image):

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
# LUNG CANCER PREDICTION
# ============================================================

def predict_lung_cancer(image):

    features = (
        extract_gsff_features(
            image
        )
    )


    scaled_features = (
        scaler.transform(
            features
        )
    )


    prediction = svm.predict(
        scaled_features
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
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🫁 PulmoVision
            <span class="hero-subtitle-main">
                AI-Powered Lung Cancer Detection System
            </span>
        </div>

        <div class="hero-subtitle">
            An AI-powered research prototype for
            lung CT image classification using deep
            feature extraction and machine learning.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">
                🧠
            </div>

            <div class="feature-title">
                Deep Feature Extraction
            </div>

            <div class="feature-text">
                EfficientNetB0 with Block5c
                feature representation.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">
                🔬
            </div>

            <div class="feature-title">
                Statistical Fusion
            </div>

            <div class="feature-text">
                GAP and GSDP features are
                combined using GSFF.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">
                🎯
            </div>

            <div class="feature-title">
                RBF-SVM Classification
            </div>

            <div class="feature-text">
                Classifies CT images into
                Normal, Benign, or Malignant.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🫁 PulmoVision</div>',
        unsafe_allow_html=True
    )


    st.write("")


    st.markdown(
        """
        **AI Pipeline**

        Image
        ↓

        Modality Verification
        ↓

        EfficientNetB0
        ↓

        Block5c
        ↓

        GAP + GSDP
        ↓

        GSFF
        ↓

        RobustScaler
        ↓

        RBF-SVM
        ↓

        Classification
        """
    )


    st.divider()


    st.markdown(
        "### Classification Classes"
    )

    st.write("🟢 Normal")

    st.write("🟡 Benign")

    st.write("🔴 Malignant")


    st.divider()


    st.caption(
        "Research prototype"
    )

    st.caption(
        "Not intended for clinical diagnosis."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "⚙️ Model Information"
):

    info_col1, info_col2 = st.columns(2)


    with info_col1:

        st.write(
            "**Feature Extractor:** "
            "EfficientNetB0"
        )

        st.write(
            "**Truncation:** Block5c"
        )

        st.write(
            "**Pooling:** GAP + GSDP"
        )

        st.write(
            "**Fusion:** GSFF"
        )


    with info_col2:

        st.write(
            "**Feature Dimension:** 224"
        )

        st.write(
            "**Scaler:** RobustScaler"
        )

        st.write(
            "**Classifier:** RBF-SVM"
        )

        st.write(
            f"**Modality Threshold:** "
            f"{MODALITY_THRESHOLD:.2f}"
        )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload CT Image</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(

    "Choose an image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],

    label_visibility="collapsed"
)


# ============================================================
# PROCESS UPLOADED IMAGE
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
        '<div class="section-title">🖼️ Uploaded Image</div>',
        unsafe_allow_html=True
    )


    image_col1, image_col2 = st.columns(
        [1.2, 1]
    )


    with image_col1:

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )


    with image_col2:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    🔍
                </div>

                <div class="feature-title">
                    Ready for Analysis
                </div>

                <div class="feature-text">
                    The uploaded image will be
                    processed through the trained
                    detection pipeline.
                </div>

            </div>
            """,
            unsafe_allow_html=True
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

        st.stop()


    st.success(
        validation_message
    )


    st.write("")


    # ========================================================
    # DETECTION BUTTON
    # ========================================================

    detect_clicked = st.button(
        "🔍  Analyze CT Image",
        use_container_width=True
    )


    if detect_clicked:


        # ====================================================
        # STEP 1 — MODALITY VERIFICATION
        # ====================================================

        st.markdown(
            '<div class="section-title">🔬 Image Analysis</div>',
            unsafe_allow_html=True
        )


        progress_bar = st.progress(
            0
        )


        status_text = st.empty()


        status_text.write(
            "Initializing image analysis..."
        )


        progress_bar.progress(
            20
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

                progress_bar.empty()

                status_text.empty()

                st.error(
                    "CT modality verification failed."
                )

                st.exception(e)

                st.stop()


        progress_bar.progress(
            45
        )


        # ====================================================
        # MODALITY PROBABILITIES
        # ====================================================

        st.markdown(
            "### Modality Analysis"
        )


        prob_col1, prob_col2, prob_col3 = (
            st.columns(3)
        )


        with prob_col1:

            st.metric(
                "Chest X-ray",
                f"{modality_probabilities[0] * 100:.1f}%"
            )

            st.progress(
                float(
                    modality_probabilities[0]
                )
            )


        with prob_col2:

            st.metric(
                "CT",
                f"{modality_probabilities[1] * 100:.1f}%"
            )

            st.progress(
                float(
                    modality_probabilities[1]
                )
            )


        with prob_col3:

            st.metric(
                "MRI",
                f"{modality_probabilities[2] * 100:.1f}%"
            )

            st.progress(
                float(
                    modality_probabilities[2]
                )
            )


        # ====================================================
        # ACCEPT ONLY CT
        # ====================================================

        if (
            modality == "CT"
            and
            modality_confidence >= MODALITY_THRESHOLD
        ):

            st.success(
                f"CT scan verified "
                f"({modality_confidence * 100:.2f}% confidence)"
            )


        # ====================================================
        # REJECT X-RAY
        # ====================================================

        elif modality == "CHEST_XRAY":

            progress_bar.empty()

            status_text.empty()

            st.error(
                "Chest X-ray detected."
            )

            st.warning(
                "The image could not be processed "
                "by the lung cancer classifier."
            )

            st.stop()


        # ====================================================
        # REJECT MRI
        # ====================================================

        elif modality == "MRI":

            progress_bar.empty()

            status_text.empty()

            st.error(
                "MRI image detected."
            )

            st.warning(
                "The image could not be processed "
                "by the lung cancer classifier."
            )

            st.stop()


        # ====================================================
        # LOW CONFIDENCE
        # ====================================================

        else:

            progress_bar.empty()

            status_text.empty()

            st.error(
                "The image could not be confidently "
                "verified for analysis."
            )

            st.stop()


        # ====================================================
        # STEP 2 — GSFF-SVM CLASSIFICATION
        # ====================================================

        progress_bar.progress(
            65
        )


        status_text.write(
            "Extracting deep statistical features..."
        )


        with st.spinner(
            "Extracting features and performing classification..."
        ):

            try:

                (
                    predicted_class,
                    confidence,
                    probabilities
                ) = predict_lung_cancer(
                    image
                )

            except Exception as e:

                progress_bar.empty()

                status_text.empty()

                st.error(
                    "Lung cancer prediction failed."
                )

                st.exception(e)

                st.stop()


        progress_bar.progress(
            100
        )


        status_text.write(
            "Analysis completed successfully."
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">🎯 Detection Result</div>',
            unsafe_allow_html=True
        )


        if predicted_class == "Normal":

            result_icon = "🟢"

            result_message = "Normal"

            result_box = "success"


        elif predicted_class == "Benign":

            result_icon = "🟡"

            result_message = "Benign"

            result_box = "warning"


        else:

            result_icon = "🔴"

            result_message = "Malignant"

            result_box = "error"


        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Predicted Condition
                </div>

                <div class="result-value">
                    {result_icon}
                    {result_message}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # STREAMLIT RESULT MESSAGE
        # ====================================================

        if predicted_class == "Normal":

            st.success(
                "The model classified the image as Normal."
            )

        elif predicted_class == "Benign":

            st.warning(
                "The model classified the image as Benign."
            )

        else:

            st.error(
                "The model classified the image as Malignant."
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.markdown(
            "### 📈 Prediction Confidence"
        )


        confidence_col1, confidence_col2 = (
            st.columns([1, 2])
        )


        with confidence_col1:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


        with confidence_col2:

            st.progress(
                float(confidence)
            )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.markdown(
            "### 📊 Class Probability Estimates"
        )


        probability_cols = st.columns(3)


        for i, class_name in enumerate(
            class_names
        ):

            probability = float(
                probabilities[i]
            )


            with probability_cols[i]:

                st.metric(
                    class_name,
                    f"{probability * 100:.2f}%"
                )


                st.progress(
                    probability
                )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        with st.expander(
            "🔧 Technical Prediction Details"
        ):

            detail_col1, detail_col2 = (
                st.columns(2)
            )


            with detail_col1:

                st.write(
                    "Verified modality: CT"
                )

                st.write(
                    f"CT verification confidence: "
                    f"{modality_confidence * 100:.2f}%"
                )

                st.write(
                    "Feature extractor: EfficientNetB0"
                )

                st.write(
                    "Truncation: Block5c"
                )


            with detail_col2:

                st.write(
                    "Pooling: GAP + GSDP"
                )

                st.write(
                    "Fusion: GSFF"
                )

                st.write(
                    "Feature dimension: 224"
                )

                st.write(
                    "Classifier: RBF-SVM"
                )


        # ====================================================
        # MEDICAL DISCLAIMER
        # ====================================================

        st.info(
            "⚠️ This system is a research prototype "
            "developed for educational and research purposes. "
            "It is not intended to provide clinical diagnosis "
            "or replace professional medical evaluation."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🫁 <b>PulmoVision</b>
        <br>
        <b>AI-Powered Lung Cancer Detection System</b>

        <br><br>

        EfficientNetB0 • GSFF • RobustScaler • RBF-SVM

        <br><br>

        Research Prototype

    </div>
    """,
    unsafe_allow_html=True
)
