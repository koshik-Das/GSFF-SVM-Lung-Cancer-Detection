# ============================================================
# app.py
# GSFF-SVM Lung Cancer Detection
#
# OPTION 2:
# Image-based validation without a separate CT verifier model
#
# Pipeline:
#
# Uploaded Image
#       ↓
# Colour Image Check
#       ↓
# Basic Image Validation
#       ↓
# CT-like Image Heuristic Check
#       ↓
# EfficientNetB0
#       ↓
# block5c_add
#       ↓
# GAP + GSDP
#       ↓
# GSFF
#       ↓
# RobustScaler
#       ↓
# RBF-SVM
#       ↓
# Normal / Benign / Malignant
#
# NO Lambda
# NO keras.backend.K
# ============================================================


import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import joblib

from PIL import Image

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.layers import (
    Layer,
    GlobalAveragePooling2D,
    Concatenate
)

from tensorflow.keras.models import Model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GSFF-SVM Lung Cancer Detection",
    page_icon="🫁",
    layout="centered"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL FILES
# ============================================================

SCALER_FILE = BASE_DIR / "RobustScaler.pkl"

SVM_FILE = BASE_DIR / "SVM_Classifier.pkl"


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Normal",
    "Benign",
    "Malignant"
]


# ============================================================
# VALIDATION SETTINGS
# ============================================================

MIN_IMAGE_SIZE = 64

MIN_STD = 8.0

MAX_DARK_RATIO = 0.98

MAX_BRIGHT_RATIO = 0.98

MAX_ASPECT_RATIO = 3.0

MIN_ASPECT_RATIO = 1 / 3


# ============================================================
# CHECK REQUIRED FILE
# ============================================================

def check_model_file(
    path,
    model_name
):

    if not path.exists():

        st.error(
            f"❌ {model_name} was not found."
        )

        st.code(
            str(path)
        )

        st.stop()


    if path.stat().st_size == 0:

        st.error(
            f"❌ {model_name} is empty."
        )

        st.code(
            str(path)
        )

        st.stop()


# ============================================================
# CHECK REQUIRED MODEL FILES
# ============================================================

check_model_file(
    SCALER_FILE,
    "RobustScaler.pkl"
)

check_model_file(
    SVM_FILE,
    "SVM_Classifier.pkl"
)


# ============================================================
# GSDP LAYER
# ============================================================
#
# Global Standard Deviation Pooling
#
# Input:
# 14 × 14 × 112
#
# Output:
# 112
#
# IMPORTANT:
# No Lambda
# No K
# ============================================================

class GSDP(Layer):

    def call(
        self,
        inputs
    ):

        return tf.math.reduce_std(
            inputs,
            axis=[1, 2]
        )


# ============================================================
# BUILD GSFF FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def build_feature_extractor():

    # --------------------------------------------------------
    # EfficientNetB0
    # --------------------------------------------------------

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )


    # --------------------------------------------------------
    # Freeze all layers
    # --------------------------------------------------------

    for layer in base_model.layers:

        layer.trainable = False


    # --------------------------------------------------------
    # Truncate at block5c_add
    # --------------------------------------------------------

    feature_map = base_model.get_layer(
        "block5c_add"
    ).output


    # --------------------------------------------------------
    # GAP
    # --------------------------------------------------------

    gap = GlobalAveragePooling2D(
        name="GAP"
    )(feature_map)


    # --------------------------------------------------------
    # GSDP
    # --------------------------------------------------------

    gsdp = GSDP(
        name="GSDP"
    )(feature_map)


    # --------------------------------------------------------
    # GSFF Fusion
    #
    # GAP  = 112
    # GSDP = 112
    #
    # Total = 224
    # --------------------------------------------------------

    gsff = Concatenate(
        name="GSFF_Fusion"
    )(
        [
            gap,
            gsdp
        ]
    )


    # --------------------------------------------------------
    # Final feature extractor
    # --------------------------------------------------------

    feature_extractor = Model(
        inputs=base_model.input,
        outputs=gsff,
        name="GSFF_Feature_Extractor"
    )


    return feature_extractor


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    scaler = joblib.load(
        str(SCALER_FILE)
    )

    return scaler


# ============================================================
# LOAD SVM
# ============================================================

@st.cache_resource
def load_svm():

    svm = joblib.load(
        str(SVM_FILE)
    )

    return svm


# ============================================================
# LOAD MODELS
# ============================================================

try:

    feature_extractor = (
        build_feature_extractor()
    )

    scaler = load_scaler()

    svm = load_svm()


except Exception as e:

    st.error(
        "❌ Model loading failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# VERIFY FEATURE DIMENSION
# ============================================================

feature_dimension = (
    feature_extractor.output_shape[-1]
)


if feature_dimension != 224:

    st.error(
        "❌ GSFF feature dimension mismatch."
    )

    st.write(
        f"Expected: 224"
    )

    st.write(
        f"Actual: {feature_dimension}"
    )

    st.stop()


# ============================================================
# VERIFY SCALER
# ============================================================

if hasattr(
    scaler,
    "n_features_in_"
):

    if scaler.n_features_in_ != 224:

        st.error(
            "❌ RobustScaler feature dimension mismatch."
        )

        st.write(
            f"Expected: 224"
        )

        st.write(
            f"Actual: {scaler.n_features_in_}"
        )

        st.stop()


# ============================================================
# VERIFY SVM
# ============================================================

if hasattr(
    svm,
    "n_features_in_"
):

    if svm.n_features_in_ != 224:

        st.error(
            "❌ SVM feature dimension mismatch."
        )

        st.write(
            f"Expected: 224"
        )

        st.write(
            f"Actual: {svm.n_features_in_}"
        )

        st.stop()


# ============================================================
# COLOUR IMAGE DETECTION
# ============================================================

def is_colour_image(
    image
):

    # --------------------------------------------------------
    # Direct colour modes
    # --------------------------------------------------------

    if image.mode in [
        "RGB",
        "RGBA",
        "CMYK",
        "HSV",
        "YCbCr"
    ]:

        rgb_image = image.convert(
            "RGB"
        )

        rgb_array = np.asarray(
            rgb_image,
            dtype=np.float32
        )


        r = rgb_array[:, :, 0]

        g = rgb_array[:, :, 1]

        b = rgb_array[:, :, 2]


        # ----------------------------------------------------
        # Channel difference
        # ----------------------------------------------------

        channel_difference = np.mean(
            np.abs(r - g)
            +
            np.abs(g - b)
            +
            np.abs(r - b)
        )


        if channel_difference > 3.0:

            return True


    # --------------------------------------------------------
    # Palette images
    # --------------------------------------------------------

    if image.mode == "P":

        rgb_image = image.convert(
            "RGB"
        )

        rgb_array = np.asarray(
            rgb_image,
            dtype=np.float32
        )


        r = rgb_array[:, :, 0]

        g = rgb_array[:, :, 1]

        b = rgb_array[:, :, 2]


        channel_difference = np.mean(
            np.abs(r - g)
            +
            np.abs(g - b)
            +
            np.abs(r - b)
        )


        if channel_difference > 3.0:

            return True


    return False


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def basic_image_validation(
    image
):

    # --------------------------------------------------------
    # Colour check
    # --------------------------------------------------------

    if is_colour_image(
        image
    ):

        return (
            False,
            "colour",
            None
        )


    # --------------------------------------------------------
    # Grayscale conversion
    # --------------------------------------------------------

    gray = image.convert(
        "L"
    )


    gray_array = np.asarray(
        gray,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    width, height = gray.size


    if (
        width < MIN_IMAGE_SIZE
        or
        height < MIN_IMAGE_SIZE
    ):

        return (
            False,
            "resolution",
            None
        )


    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    aspect_ratio = (
        width / height
    )


    if (
        aspect_ratio > MAX_ASPECT_RATIO
        or
        aspect_ratio < MIN_ASPECT_RATIO
    ):

        return (
            False,
            "aspect_ratio",
            None
        )


    # --------------------------------------------------------
    # Intensity statistics
    # --------------------------------------------------------

    mean_intensity = float(
        np.mean(
            gray_array
        )
    )


    std_intensity = float(
        np.std(
            gray_array
        )
    )


    minimum = float(
        np.min(
            gray_array
        )
    )


    maximum = float(
        np.max(
            gray_array
        )
    )


    # --------------------------------------------------------
    # Blank image
    # --------------------------------------------------------

    if std_intensity < MIN_STD:

        return (
            False,
            "blank",
            None
        )


    # --------------------------------------------------------
    # Dark / bright ratio
    # --------------------------------------------------------

    dark_ratio = float(
        np.mean(
            gray_array < 10
        )
    )


    bright_ratio = float(
        np.mean(
            gray_array > 245
        )
    )


    if dark_ratio > MAX_DARK_RATIO:

        return (
            False,
            "black",
            None
        )


    if bright_ratio > MAX_BRIGHT_RATIO:

        return (
            False,
            "white",
            None
        )


    # --------------------------------------------------------
    # Return statistics
    # --------------------------------------------------------

    statistics = {

        "width": width,

        "height": height,

        "aspect_ratio": aspect_ratio,

        "mean": mean_intensity,

        "std": std_intensity,

        "minimum": minimum,

        "maximum": maximum,

        "dark_ratio": dark_ratio,

        "bright_ratio": bright_ratio
    }


    return (
        True,
        "valid",
        statistics
    )


# ============================================================
# CT-LIKE IMAGE HEURISTIC
# ============================================================
#
# IMPORTANT:
#
# This is NOT a medical modality classifier.
#
# It only attempts to reject obviously incompatible images.
#
# MRI/X-ray images can still pass this test.
#
# ============================================================

def ct_like_validation(
    image
):

    gray = image.convert(
        "L"
    )


    # --------------------------------------------------------
    # Resize for stable statistics
    # --------------------------------------------------------

    gray = gray.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )


    array = np.asarray(
        gray,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = (
        array / 255.0
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_value = float(
        np.mean(
            normalized
        )
    )


    std_value = float(
        np.std(
            normalized
        )
    )


    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    p5 = float(
        np.percentile(
            normalized,
            5
        )
    )


    p25 = float(
        np.percentile(
            normalized,
            25
        )
    )


    p50 = float(
        np.percentile(
            normalized,
            50
        )
    )


    p75 = float(
        np.percentile(
            normalized,
            75
        )
    )


    p95 = float(
        np.percentile(
            normalized,
            95
        )
    )


    # --------------------------------------------------------
    # Dynamic range
    # --------------------------------------------------------

    dynamic_range = (
        p95 - p5
    )


    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    histogram, _ = np.histogram(
        normalized,
        bins=32,
        range=(0.0, 1.0),
        density=True
    )


    histogram = (
        histogram /
        (
            np.sum(histogram)
            + 1e-8
        )
    )


    # --------------------------------------------------------
    # Histogram entropy
    # --------------------------------------------------------

    entropy = float(
        -np.sum(
            histogram
            *
            np.log(
                histogram + 1e-8
            )
        )
    )


    # ========================================================
    # HEURISTIC SCORE
    # ========================================================
    #
    # These values are deliberately permissive.
    # We do NOT want to reject valid CT images unnecessarily.
    #
    # ========================================================

    score = 0


    # --------------------------------------------------------
    # Reasonable intensity variation
    # --------------------------------------------------------

    if (
        0.08 <= std_value <= 0.45
    ):

        score += 1


    # --------------------------------------------------------
    # Reasonable dynamic range
    # --------------------------------------------------------

    if dynamic_range > 0.20:

        score += 1


    # --------------------------------------------------------
    # Not excessively dark
    # --------------------------------------------------------

    if p50 > 0.05:

        score += 1


    # --------------------------------------------------------
    # Not completely compressed
    # --------------------------------------------------------

    if (
        p95 - p25 > 0.15
    ):

        score += 1


    # --------------------------------------------------------
    # Histogram contains sufficient information
    # --------------------------------------------------------

    if entropy > 2.0:

        score += 1


    # --------------------------------------------------------
    # Final decision
    #
    # At least 3 of 5 conditions must pass.
    # --------------------------------------------------------

    is_ct_like = (
        score >= 3
    )


    statistics = {

        "mean": mean_value,

        "std": std_value,

        "p5": p5,

        "p25": p25,

        "median": p50,

        "p75": p75,

        "p95": p95,

        "dynamic_range": dynamic_range,

        "entropy": entropy,

        "heuristic_score": score,

        "maximum_score": 5
    }


    return (
        is_ct_like,
        statistics
    )


# ============================================================
# PREPROCESS IMAGE FOR GSFF
# ============================================================

def preprocess_for_gsff(
    image
):

    # --------------------------------------------------------
    # Convert grayscale
    # --------------------------------------------------------

    gray = image.convert(
        "L"
    )


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    gray = gray.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    array = np.asarray(
        gray,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Grayscale → RGB
    #
    # EfficientNet expects 3 channels.
    #
    # All channels remain identical.
    # --------------------------------------------------------

    array = np.stack(
        [
            array,
            array,
            array
        ],
        axis=-1
    )


    # --------------------------------------------------------
    # Batch dimension
    # --------------------------------------------------------

    array = np.expand_dims(
        array,
        axis=0
    )


    # --------------------------------------------------------
    # EfficientNet preprocessing
    # --------------------------------------------------------

    array = preprocess_input(
        array
    )


    return array


# ============================================================
# LUNG CANCER PREDICTION
# ============================================================

def predict_image(
    image
):

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_array = (
        preprocess_for_gsff(
            image
        )
    )


    # --------------------------------------------------------
    # GSFF feature extraction
    # --------------------------------------------------------

    features = (
        feature_extractor.predict(
            image_array,
            verbose=0
        )
    )


    # --------------------------------------------------------
    # Verify feature dimension
    # --------------------------------------------------------

    if features.shape[1] != 224:

        raise ValueError(
            "GSFF feature dimension is "
            f"{features.shape[1]}, "
            "but expected 224."
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
    # SVM prediction
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


    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_class = (
        class_names[
            predicted_index
        ]
    )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(
        probabilities[
            predicted_index
        ] * 100
    )


    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🫁 GSFF-SVM Lung Cancer Detection"
)


st.write(
    "Deep learning and machine learning based "
    "lung CT image classification system."
)


st.info(
    "The system accepts grayscale medical images "
    "that pass the CT-domain validation checks. "
    "Colour images and clearly invalid images are rejected."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "🔬 Model Information"
):

    st.write(
        "Backbone: EfficientNetB0"
    )

    st.write(
        "Truncation: block5c_add"
    )

    st.write(
        "Feature Map: 14 × 14 × 112"
    )

    st.write(
        "GAP: 112 features"
    )

    st.write(
        "GSDP: 112 features"
    )

    st.write(
        "GSFF: 224 features"
    )

    st.write(
        "Scaler: RobustScaler"
    )

    st.write(
        "Classifier: RBF-SVM"
    )

    st.write(
        "Classes: Normal / Benign / Malignant"
    )


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Lung CT Image",
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

    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        )


        image.load()


        # ----------------------------------------------------
        # Display image
        # ----------------------------------------------------

        st.subheader(
            "Uploaded Image"
        )


        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


        # ====================================================
        # STEP 1
        # BASIC IMAGE VALIDATION
        # ====================================================

        st.subheader(
            "Step 1 — Image Validation"
        )


        (
            valid,
            reason,
            statistics
        ) = basic_image_validation(
            image
        )


        # ----------------------------------------------------
        # Reject invalid images
        # ----------------------------------------------------

        if not valid:

            if reason == "colour":

                st.error(
                    "❌ REJECTED — Colour images "
                    "are not supported."
                )


            elif reason == "resolution":

                st.error(
                    "❌ REJECTED — Image resolution "
                    "is too small."
                )


            elif reason == "aspect_ratio":

                st.error(
                    "❌ REJECTED — Unusual image "
                    "aspect ratio."
                )


            elif reason == "blank":

                st.error(
                    "❌ REJECTED — Image appears "
                    "blank or has insufficient "
                    "intensity variation."
                )


            elif reason == "black":

                st.error(
                    "❌ REJECTED — Image is almost "
                    "completely black."
                )


            elif reason == "white":

                st.error(
                    "❌ REJECTED — Image is almost "
                    "completely white."
                )


            st.warning(
                "Please upload a valid grayscale "
                "lung CT image."
            )


            st.stop()


        # ----------------------------------------------------
        # Basic validation passed
        # ----------------------------------------------------

        st.success(
            "✅ Basic image validation passed."
        )


        # ====================================================
        # DISPLAY BASIC STATISTICS
        # ====================================================

        with st.expander(
            "Image Statistics"
        ):

            col1, col2 = st.columns(
                2
            )


            with col1:

                st.write(
                    f"Width: "
                    f"{statistics['width']} px"
                )

                st.write(
                    f"Height: "
                    f"{statistics['height']} px"
                )

                st.write(
                    f"Mean intensity: "
                    f"{statistics['mean']:.2f}"
                )


            with col2:

                st.write(
                    f"Std. deviation: "
                    f"{statistics['std']:.2f}"
                )

                st.write(
                    f"Aspect ratio: "
                    f"{statistics['aspect_ratio']:.2f}"
                )

                st.write(
                    f"Dark ratio: "
                    f"{statistics['dark_ratio'] * 100:.2f}%"
                )


        # ====================================================
        # STEP 2
        # CT-LIKE IMAGE HEURISTIC
        # ====================================================

        st.subheader(
            "Step 2 — CT-Domain Validation"
        )


        with st.spinner(
            "Checking image characteristics..."
        ):

            (
                ct_like,
                ct_statistics
            ) = ct_like_validation(
                image
            )


        # ----------------------------------------------------
        # Display heuristic information
        # ----------------------------------------------------

        with st.expander(
            "CT-Domain Validation Details"
        ):

            st.write(
                f"Mean: "
                f"{ct_statistics['mean']:.4f}"
            )

            st.write(
                f"Standard deviation: "
                f"{ct_statistics['std']:.4f}"
            )

            st.write(
                f"5th percentile: "
                f"{ct_statistics['p5']:.4f}"
            )

            st.write(
                f"25th percentile: "
                f"{ct_statistics['p25']:.4f}"
            )

            st.write(
                f"Median: "
                f"{ct_statistics['median']:.4f}"
            )

            st.write(
                f"75th percentile: "
                f"{ct_statistics['p75']:.4f}"
            )

            st.write(
                f"95th percentile: "
                f"{ct_statistics['p95']:.4f}"
            )

            st.write(
                f"Dynamic range: "
                f"{ct_statistics['dynamic_range']:.4f}"
            )

            st.write(
                f"Histogram entropy: "
                f"{ct_statistics['entropy']:.4f}"
            )

            st.write(
                f"Heuristic score: "
                f"{ct_statistics['heuristic_score']}/"
                f"{ct_statistics['maximum_score']}"
            )


        # ====================================================
        # REJECT IMAGE
        # ====================================================

        if not ct_like:

            st.error(
                "❌ REJECTED — Image does not "
                "meet the CT-domain validation "
                "criteria."
            )


            st.warning(
                "Please upload a lung CT image "
                "similar to the images used during "
                "model development."
            )


            st.stop()


        # ====================================================
        # CT-LIKE PASSED
        # ====================================================

        st.success(
            "✅ Image passed the CT-domain "
            "validation."
        )


        # ====================================================
        # STEP 3
        # GSFF-SVM CLASSIFICATION
        # ====================================================

        st.subheader(
            "Step 3 — Lung Cancer Classification"
        )


        with st.spinner(
            "Analyzing CT image..."
        ):

            (
                predicted_class,
                confidence,
                probabilities
            ) = predict_image(
                image
            )


        # ====================================================
        # RESULT
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
            f"{confidence:.2f}%"
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
        # DISCLAIMER
        # ====================================================

        st.info(
            "This system is intended for research "
            "and educational purposes only. It is "
            "not a substitute for professional "
            "medical diagnosis."
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🫁 GSFF-SVM Framework"
    )


    st.write(
        """
        Uploaded Image
              ↓
        Colour Check
              ↓
        Image Validation
              ↓
        CT-Domain Check
              ↓
        EfficientNetB0
              ↓
        block5c_add
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
        "**Accepted:**"
    )

    st.write(
        "✅ Grayscale CT-like images"
    )


    st.write(
        "**Rejected:**"
    )

    st.write(
        "❌ Colour images"
    )

    st.write(
        "❌ Blank images"
    )

    st.write(
        "❌ Extremely small images"
    )

    st.write(
        "❌ Extremely unusual images"
    )


    st.divider()


    st.write(
        "**Classes:**"
    )

    st.write(
        "Normal"
    )

    st.write(
        "Benign"
    )

    st.write(
        "Malignant"
    )


    st.divider()


    st.caption(
        "Research prototype. "
        "Not intended for clinical diagnosis."
    )
