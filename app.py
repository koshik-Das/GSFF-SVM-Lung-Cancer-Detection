# ============================================================
# app.py
# PulmoVision
# AI-Powered Lung Cancer Detection System
#
# Pipeline:
#
# Uploaded Image
#       ↓
# Image Validation
#       ↓
# CT / X-ray / MRI Modality Verification
#       ↓
# CT
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
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PulmoVision",
    page_icon="🫁",
    layout="centered"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LUNG_ICON_PATH = BASE_DIR / "Lung_Icon.png"


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
#
# Your trained GSFF model contains a custom GSDP layer.
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
# This is NOT GMP.
#
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

    "CT Verifier": CT_VERIFIER_PATH,

    "GSFF Feature Extractor":
        FEATURE_EXTRACTOR_PATH,

    "RobustScaler": SCALER_PATH,

    "SVM Classifier": SVM_PATH

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
# LOAD ROBUST SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    scaler = joblib.load(
        SCALER_PATH
    )

    return scaler


# ============================================================
# LOAD SVM
# ============================================================

@st.cache_resource
def load_svm():

    model = joblib.load(
        SVM_PATH
    )

    return model


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

    st.error(
        "The saved GSFF model contains the custom GSDP layer."
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

    """
    Determines whether an uploaded image is effectively
    grayscale.

    A grayscale CT image may be stored as RGB, but if all
    three channels are almost identical, it is treated as
    grayscale.

    Genuine coloured images are rejected.
    """

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
                "❌ Colour images are not supported. "
                "Please upload a grayscale CT image."
            )


        return (
            True,
            "✅ Grayscale image detected."
        )


    except Exception as e:

        return (
            False,
            f"❌ Unable to validate image: {e}"
        )


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_image(image):

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    width, height = image.size


    if width < 64 or height < 64:

        return (
            False,
            "❌ Image resolution is too small."
        )


    # --------------------------------------------------------
    # Check grayscale
    # --------------------------------------------------------

    is_grayscale, message = (
        check_grayscale_image(image)
    )


    if not is_grayscale:

        return (
            False,
            message
        )


    # --------------------------------------------------------
    # Convert to grayscale for basic statistics
    # --------------------------------------------------------

    gray = image.convert("L")


    gray_array = np.asarray(
        gray,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Blank image check
    # --------------------------------------------------------

    standard_deviation = np.std(
        gray_array
    )


    if standard_deviation < 8:

        return (
            False,
            "❌ Image appears blank or invalid."
        )


    # --------------------------------------------------------
    # Almost completely black
    # --------------------------------------------------------

    dark_ratio = np.mean(
        gray_array < 10
    )


    if dark_ratio > 0.98:

        return (
            False,
            "❌ Image is almost completely black."
        )


    # --------------------------------------------------------
    # Almost completely white
    # --------------------------------------------------------

    bright_ratio = np.mean(
        gray_array > 245
    )


    if bright_ratio > 0.98:

        return (
            False,
            "❌ Image is almost completely white."
        )


    return (
        True,
        "✅ Image passed basic validation."
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


    # --------------------------------------------------------
    # Same preprocessing used during modality training
    #
    # ImageDataGenerator:
    #
    # rescale = 1 / 255
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # EfficientNet preprocessing
    #
    # This matches the training code:
    #
    # X = preprocess_input(...)
    # --------------------------------------------------------

    image_array = preprocess_input(
        image_array
    )


    return image_array


# ============================================================
# CT / X-RAY / MRI VERIFICATION
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


    # --------------------------------------------------------
    # Safety normalization
    # --------------------------------------------------------

    probability_sum = np.sum(
        probabilities
    )


    if probability_sum > 0:

        probabilities = (
            probabilities / probability_sum
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

    # --------------------------------------------------------
    # Extract GSFF feature
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
    # RBF-SVM prediction
    # --------------------------------------------------------

    prediction = svm.predict(
        scaled_features
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
# HEADER
# ============================================================

header_col1, header_col2 = st.columns([1, 5])

with header_col1:
    if LUNG_ICON_PATH.exists():
        st.image(
            str(LUNG_ICON_PATH),
            width=110
        )

with header_col2:
    st.markdown(
        """
        <h1 style="
            margin-top: 15px;
            margin-bottom: 5px;
            font-size: 42px;
        ">
            PulmoVision
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h4 style="
            margin-top: 0px;
            font-weight: 500;
        ">
            AI-Powered Lung Cancer Detection System
        </h4>
        """,
        unsafe_allow_html=True
    )

st.write(
    "An AI-powered research prototype for "
    "lung CT image classification using deep "
    "feature extraction and machine learning."
)

st.info(
    "Upload a grayscale lung CT image for analysis."
)





# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🫁 PulmoVision"
    )


    st.write(
        "AI-Powered Lung Cancer Detection System"
    )


    st.divider()


    st.write(
        "**AI Pipeline:**"
    )


    st.write(
        """
        Image
        ↓
        Modality Verification
        ↓
        CT
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


    st.write(
        "**Classification Classes:**"
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
# MODEL STATUS
# ============================================================

with st.expander(
    "⚙️ Model Information"
):

    st.write(
        f"CT verifier: "
        f"{CT_VERIFIER_PATH.name}"
    )


    st.write(
        f"GSFF extractor: "
        f"{FEATURE_EXTRACTOR_PATH.name}"
    )


    st.write(
        "Feature extractor: EfficientNetB0"
    )


    st.write(
        "Truncation: Block5c"
    )


    st.write(
        "Feature fusion: GAP + GSDP"
    )


    st.write(
        "Feature dimension: 224"
    )


    st.write(
        "Scaler: RobustScaler"
    )


    st.write(
        "Classifier: RBF-SVM"
    )


    st.write(
        f"Modality confidence threshold: "
        f"{MODALITY_THRESHOLD:.2f}"
    )


# ============================================================
# FILE UPLOADER
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
            "❌ Unable to read the uploaded image."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    st.subheader(
        "🖼️ Uploaded Image"
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

        st.stop()


    st.success(
        validation_message
    )


    # ========================================================
    # DETECTION BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze CT Image",
        use_container_width=True
    ):

        # ====================================================
        # STEP 1 — MODALITY VERIFICATION
        # ====================================================

        st.subheader(
            "Step 1 — Image Modality Verification"
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
                    "❌ CT modality verification failed."
                )

                st.exception(e)

                st.stop()


        # ====================================================
        # DISPLAY MODALITY PROBABILITIES
        # ====================================================

        st.write(
            "**Modality probabilities:**"
        )


        st.write(
            f"Chest X-ray: "
            f"{modality_probabilities[0] * 100:.2f}%"
        )


        st.progress(
            float(
                modality_probabilities[0]
            )
        )


        st.write(
            f"CT: "
            f"{modality_probabilities[1] * 100:.2f}%"
        )


        st.progress(
            float(
                modality_probabilities[1]
            )
        )


        st.write(
            f"MRI: "
            f"{modality_probabilities[2] * 100:.2f}%"
        )


        st.progress(
            float(
                modality_probabilities[2]
            )
        )


        # ====================================================
        # ACCEPT CT
        # ====================================================

        if (
            modality == "CT"
            and
            modality_confidence >= MODALITY_THRESHOLD
        ):

            st.success(
                f"✅ CT scan detected "
                f"({modality_confidence * 100:.2f}% confidence)"
            )


        # ====================================================
        # REJECT X-RAY
        # ====================================================

        elif modality == "CHEST_XRAY":

            st.error(
                "❌ Chest X-ray detected."
            )


            st.warning(
                "Lung cancer classification was not "
                "performed because the analysis pipeline "
                "is designed for CT images."
            )


            st.stop()


        # ====================================================
        # REJECT MRI
        # ====================================================

        elif modality == "MRI":

            st.error(
                "❌ MRI image detected."
            )


            st.warning(
                "Lung cancer classification was not "
                "performed because the analysis pipeline "
                "is designed for CT images."
            )


            st.stop()


        # ====================================================
        # LOW CONFIDENCE
        # ====================================================

        else:

            st.error(
                "❌ Image modality could not be verified "
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
                "Please upload a clear grayscale lung "
                "CT image."
            )


            st.stop()


        # ====================================================
        # STEP 2 — LUNG CANCER CLASSIFICATION
        # ====================================================

        st.subheader(
            "Step 2 — Lung Cancer Classification"
        )


        with st.spinner(
            "Extracting features and classifying the CT image..."
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

                st.error(
                    "❌ Lung cancer prediction failed."
                )

                st.exception(e)

                st.stop()


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.subheader(
            "🎯 Prediction Result"
        )


        if predicted_class == "Normal":

            st.success(
                f"Prediction: **{predicted_class}**"
            )


        elif predicted_class == "Benign":

            st.warning(
                f"Prediction: **{predicted_class}**"
            )


        else:

            st.error(
                f"Prediction: **{predicted_class}**"
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )


        st.progress(
            float(
                confidence
            )
        )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.subheader(
            "📊 Class Probability Estimates"
        )


        for i, class_name in enumerate(
            class_names
        ):

            probability = float(
                probabilities[i]
            )


            st.write(
                f"**{class_name}: "
                f"{probability * 100:.2f}%**"
            )


            st.progress(
                probability
            )


        # ====================================================
        # MODEL PIPELINE INFORMATION
        # ====================================================

        with st.expander(
            "🔧 Technical Prediction Details"
        ):

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
                "Feature scaling: RobustScaler"
            )


            st.write(
                "Classifier: RBF-SVM"
            )


        # ====================================================
        # MEDICAL DISCLAIMER
        # ====================================================

        st.info(
            "⚠️ This system is a research prototype "
            "for educational and research purposes. "
            "It is not intended to provide clinical "
            "diagnosis or replace professional medical "
            "evaluation."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "🫁 PulmoVision — AI-Powered Lung Cancer Detection System"
)


st.caption(
    "Research Prototype • EfficientNetB0 • GSFF • RBF-SVM"
)
