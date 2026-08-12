# ============================================================
# app.py
# GSFF-SVM Lung Cancer Detection
#
# INPUT VALIDATION PIPELINE
#
# Uploaded Image
#       ↓
# Colour Image Check
#       ↓
# CT / Non-CT Verification
#       ↓
#       ├── Non-CT → REJECT
#       │
#       └── CT → GSFF-SVM
#                    ↓
#              EfficientNetB0
#                    ↓
#                block5c_add
#                    ↓
#               GAP + GSDP
#                    ↓
#                   GSFF
#                    ↓
#              RobustScaler
#                    ↓
#                  RBF-SVM
#                    ↓
#        Normal / Benign / Malignant
#
# IMPORTANT:
# No Lambda layer
# No keras.backend.K
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
# STREAMLIT CONFIGURATION
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

SCALER_FILE = (
    BASE_DIR / "RobustScaler.pkl"
)

SVM_FILE = (
    BASE_DIR / "SVM_Classifier.pkl"
)

# ------------------------------------------------------------
# CT VERIFIER MODEL
#
# This must be a model trained to distinguish:
#
# 0 = CT
# 1 = Non-CT
#
# Non-CT should include X-ray and MRI images.
# ------------------------------------------------------------

CT_VERIFIER_FILE = (
    BASE_DIR / "CT_Verifier.keras"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

CT_VERIFIER_SIZE = (224, 224)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Normal",
    "Benign",
    "Malignant"
]


# ============================================================
# CT VERIFIER SETTINGS
# ============================================================
#
# IMPORTANT:
#
# This threshold depends on how your CT verifier was trained.
#
# We assume:
#
# probability >= 0.50 → CT
# probability <  0.50 → Non-CT
#
# If your verifier uses a different threshold,
# change this value.
# ============================================================

CT_THRESHOLD = 0.50


# ============================================================
# CHECK REQUIRED FILE
# ============================================================

def check_file(path, name):

    if not path.exists():

        st.error(
            f"❌ {name} was not found."
        )

        st.code(
            str(path)
        )

        st.stop()


    if path.stat().st_size == 0:

        st.error(
            f"❌ {name} is empty."
        )

        st.code(
            str(path)
        )

        st.stop()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

check_file(
    SCALER_FILE,
    "RobustScaler.pkl"
)

check_file(
    SVM_FILE,
    "SVM_Classifier.pkl"
)

check_file(
    CT_VERIFIER_FILE,
    "CT_Verifier.keras"
)


# ============================================================
# GSDP LAYER
# ============================================================
#
# Global Standard Deviation Pooling
#
# Input:
#     14 × 14 × 112
#
# Output:
#     112
#
# No Lambda.
# No K.
# ============================================================

class GSDP(Layer):

    def call(self, inputs):

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
    # Freeze EfficientNet
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
    # GSFF
    # --------------------------------------------------------

    gsff = Concatenate(
        name="GSFF_Fusion"
    )([
        gap,
        gsdp
    ])


    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    model = Model(
        inputs=base_model.input,
        outputs=gsff,
        name="GSFF_Feature_Extractor"
    )


    return model


# ============================================================
# LOAD CT VERIFIER
# ============================================================

@st.cache_resource
def load_ct_verifier():

    model = tf.keras.models.load_model(
        str(CT_VERIFIER_FILE),
        compile=False
    )

    return model


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    return joblib.load(
        str(SCALER_FILE)
    )


# ============================================================
# LOAD SVM
# ============================================================

@st.cache_resource
def load_svm():

    return joblib.load(
        str(SVM_FILE)
    )


# ============================================================
# LOAD MODELS
# ============================================================

try:

    feature_extractor = (
        build_feature_extractor()
    )

    ct_verifier = (
        load_ct_verifier()
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
# VERIFY GSFF FEATURE DIMENSION
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
# VERIFY SCALER DIMENSION
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
# VERIFY SVM DIMENSION
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

def is_colour_image(image):

    # --------------------------------------------------------
    # Explicit colour modes
    # --------------------------------------------------------

    if image.mode in [
        "RGB",
        "RGBA",
        "CMYK",
        "HSV",
        "YCbCr"
    ]:

        rgb = image.convert(
            "RGB"
        )

        array = np.asarray(
            rgb,
            dtype=np.float32
        )

        r = array[:, :, 0]
        g = array[:, :, 1]
        b = array[:, :, 2]


        # ----------------------------------------------------
        # Difference between channels
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
    # Palette / indexed images
    # --------------------------------------------------------

    if image.mode == "P":

        rgb = image.convert(
            "RGB"
        )

        array = np.asarray(
            rgb,
            dtype=np.float32
        )

        r = array[:, :, 0]
        g = array[:, :, 1]
        b = array[:, :, 2]


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

def validate_basic_image(image):

    # --------------------------------------------------------
    # Colour check
    # --------------------------------------------------------

    if is_colour_image(image):

        return (
            False,
            "colour"
        )


    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = image.convert(
        "L"
    )


    gray_array = np.asarray(
        gray,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    width, height = gray.size


    if width < 64 or height < 64:

        return (
            False,
            "resolution"
        )


    # --------------------------------------------------------
    # Standard deviation
    # --------------------------------------------------------

    std_intensity = np.std(
        gray_array
    )


    if std_intensity < 8:

        return (
            False,
            "blank"
        )


    # --------------------------------------------------------
    # Almost black
    # --------------------------------------------------------

    dark_ratio = np.mean(
        gray_array < 10
    )


    if dark_ratio > 0.98:

        return (
            False,
            "black"
        )


    # --------------------------------------------------------
    # Almost white
    # --------------------------------------------------------

    bright_ratio = np.mean(
        gray_array > 245
    )


    if bright_ratio > 0.98:

        return (
            False,
            "white"
        )


    return (
        True,
        "valid"
    )


# ============================================================
# PREPROCESS FOR CT VERIFIER
# ============================================================

def preprocess_for_ct_verifier(image):

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
        CT_VERIFIER_SIZE,
        Image.Resampling.LANCZOS
    )


    # --------------------------------------------------------
    # Convert to RGB
    #
    # The three channels contain identical grayscale values.
    # --------------------------------------------------------

    array = np.asarray(
        gray,
        dtype=np.float32
    )


    array = np.stack(
        [
            array,
            array,
            array
        ],
        axis=-1
    )


    # --------------------------------------------------------
    # Normalize
    #
    # This assumes your CT verifier was trained with
    # 0-1 normalization.
    # --------------------------------------------------------

    array /= 255.0


    # --------------------------------------------------------
    # Batch dimension
    # --------------------------------------------------------

    array = np.expand_dims(
        array,
        axis=0
    )


    return array


# ============================================================
# CT VERIFICATION
# ============================================================

def verify_ct_image(image):

    image_array = (
        preprocess_for_ct_verifier(
            image
        )
    )


    prediction = (
        ct_verifier.predict(
            image_array,
            verbose=0
        )
    )


    prediction = np.asarray(
        prediction
    )


    # --------------------------------------------------------
    # HANDLE COMMON MODEL OUTPUTS
    # --------------------------------------------------------

    # Case 1:
    # Dense(1, sigmoid)
    #
    # [probability]
    #
    if prediction.ndim == 2 and prediction.shape[1] == 1:

        non_ct_probability = float(
            prediction[0][0]
        )

        # IMPORTANT:
        # This assumes:
        #
        # 0 = CT
        # 1 = Non-CT
        #
        ct_probability = (
            1.0 - non_ct_probability
        )


    # Case 2:
    # Dense(2, softmax)
    #
    # [CT probability, Non-CT probability]
    #
    elif (
        prediction.ndim == 2
        and prediction.shape[1] == 2
    ):

        ct_probability = float(
            prediction[0][0]
        )

        non_ct_probability = float(
            prediction[0][1]
        )


    else:

        raise ValueError(
            "Unsupported CT verifier output shape: "
            f"{prediction.shape}. "
            "Expected (1,1) sigmoid or (1,2) softmax."
        )


    ct_probability = float(
        np.clip(
            ct_probability,
            0.0,
            1.0
        )
    )


    non_ct_probability = float(
        np.clip(
            non_ct_probability,
            0.0,
            1.0
        )
    )


    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if ct_probability >= CT_THRESHOLD:

        result = "CT"

    else:

        result = "NON_CT"


    return (
        result,
        ct_probability,
        non_ct_probability
    )


# ============================================================
# PREPROCESS FOR GSFF
# ============================================================

def preprocess_for_gsff(image):

    # --------------------------------------------------------
    # Convert to grayscale
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
    # Convert grayscale → RGB
    # --------------------------------------------------------

    array = np.asarray(
        gray,
        dtype=np.float32
    )


    array = np.stack(
        [
            array,
            array,
            array
        ],
        axis=-1
    )


    # --------------------------------------------------------
    # Add batch
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

def predict_lung_condition(image):

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_array = (
        preprocess_for_gsff(
            image
        )
    )


    # --------------------------------------------------------
    # GSFF FEATURE EXTRACTION
    # --------------------------------------------------------

    features = (
        feature_extractor.predict(
            image_array,
            verbose=0
        )
    )


    # --------------------------------------------------------
    # Verify 224 features
    # --------------------------------------------------------

    if features.shape[1] != 224:

        raise ValueError(
            f"Expected 224 GSFF features, "
            f"but received {features.shape[1]}."
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
    # Class name
    # --------------------------------------------------------

    predicted_class = (
        class_names[
            predicted_index
        ]
    )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = (
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
    "A deep learning and machine learning based "
    "lung CT image classification system."
)


st.info(
    "Only grayscale CT images are supported. "
    "Colour images, X-ray images, MRI images, "
    "and other non-CT images are rejected."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "🔬 Model Information"
):

    st.write(
        "Modality Verification: CT vs Non-CT"
    )

    st.write(
        "Backbone: EfficientNetB0"
    )

    st.write(
        "Truncation Layer: block5c_add"
    )

    st.write(
        "Feature Map: 14 × 14 × 112"
    )

    st.write(
        "Fusion: GAP + GSDP"
    )

    st.write(
        "GSFF Feature Dimension: 224"
    )

    st.write(
        "Feature Scaling: RobustScaler"
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
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        )


        image.load()


        # ----------------------------------------------------
        # Display
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
        # STEP 1 — BASIC VALIDATION
        # ====================================================

        valid, reason = (
            validate_basic_image(
                image
            )
        )


        if not valid:

            if reason == "colour":

                st.error(
                    "❌ Rejected: Colour images are "
                    "not supported."
                )

            elif reason == "resolution":

                st.error(
                    "❌ Rejected: Image resolution "
                    "is too small."
                )

            elif reason == "blank":

                st.error(
                    "❌ Rejected: Image appears "
                    "blank or invalid."
                )

            elif reason == "black":

                st.error(
                    "❌ Rejected: Image is almost "
                    "completely black."
                )

            elif reason == "white":

                st.error(
                    "❌ Rejected: Image is almost "
                    "completely white."
                )

            st.warning(
                "Please upload a valid grayscale "
                "lung CT image."
            )

            st.stop()


        st.success(
            "✅ Grayscale image accepted for "
            "modality verification."
        )


        # ====================================================
        # STEP 2 — CT VERIFICATION
        # ====================================================

        st.subheader(
            "Step 1 — CT Scan Verification"
        )


        with st.spinner(
            "Checking whether the image is a CT scan..."
        ):

            (
                modality_result,
                ct_probability,
                non_ct_probability
            ) = verify_ct_image(
                image
            )


        # ====================================================
        # NON-CT → REJECT
        # ====================================================

        if modality_result == "NON_CT":

            st.error(
                "❌ Image rejected: This image "
                "does not appear to be a CT scan."
            )


            st.write(
                f"CT probability: "
                f"{ct_probability * 100:.2f}%"
            )


            st.write(
                f"Non-CT probability: "
                f"{non_ct_probability * 100:.2f}%"
            )


            st.warning(
                "X-ray, MRI, and other non-CT images "
                "are not accepted by this system."
            )


            st.stop()


        # ====================================================
        # CT ACCEPTED
        # ====================================================

        st.success(
            f"✅ CT scan detected "
            f"(confidence: "
            f"{ct_probability * 100:.2f}%)"
        )


        st.write(
            f"CT probability: "
            f"{ct_probability * 100:.2f}%"
        )


        st.write(
            f"Non-CT probability: "
            f"{non_ct_probability * 100:.2f}%"
        )


        # ====================================================
        # STEP 3 — LUNG CANCER CLASSIFICATION
        # ====================================================

        st.subheader(
            "Step 2 — Lung Cancer Classification"
        )


        with st.spinner(
            "Analyzing CT image..."
        ):

            (
                predicted_class,
                confidence,
                probabilities
            ) = predict_lung_condition(
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
            "and educational purposes only and is "
            "not intended for clinical diagnosis."
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
        INPUT IMAGE
             ↓
        Colour Check
             ↓
        CT Verification
             ↓
        ┌───────────────┐
        │               │
       CT            Non-CT
        │               │
        ↓               ↓
    GSFF-SVM          REJECT
        │
        ↓
    Normal
    Benign
    Malignant
        """
    )


    st.divider()


    st.write(
        "**Supported Input:**"
    )


    st.write(
        "✅ Grayscale CT"
    )


    st.write(
        "**Rejected Input:**"
    )


    st.write(
        "❌ Colour images"
    )


    st.write(
        "❌ X-ray"
    )


    st.write(
        "❌ MRI"
    )


    st.write(
        "❌ Other non-CT images"
    )


    st.divider()


    st.write(
        "**GSFF:**"
    )


    st.write(
        "GAP + GSDP"
    )


    st.write(
        "**Classifier:**"
    )


    st.write(
        "RBF-SVM"
    )


    st.divider()


    st.caption(
        "Research prototype. "
        "Not intended for clinical diagnosis."
    )
