# ============================================================
# app.py
# GSFF-SVM Lung Cancer Detection
#
# OPTION 3:
# CT Verification Model + GSFF-SVM Classifier
#
# Pipeline:
#
# Uploaded Image
#       ↓
# Colour Image Check
#       ↓
# CT Verifier
#       ↓
# ┌─────────────────────────────┐
# │ CT image?                   │
# └─────────────────────────────┘
#       ↓ YES
# EfficientNetB0
#       ↓
# block5c_add
#       ↓
# GAP + GSDP
#       ↓
# GSFF (224 features)
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

CT_VERIFIER_FILE = (
    BASE_DIR / "CT_Verifier.keras"
)

SCALER_FILE = (
    BASE_DIR / "RobustScaler.pkl"
)

SVM_FILE = (
    BASE_DIR / "SVM_Classifier.pkl"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

CT_VERIFIER_SIZE = (
    224,
    224
)

GSFF_IMAGE_SIZE = (
    224,
    224
)


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
# This assumes your CT_Verifier.keras is a binary classifier.
#
# Expected:
#
# class 0 = CT
# class 1 = Non-CT
#
# If your verifier was trained with the opposite labels,
# change CT_CLASS_INDEX below.
#
# ============================================================

CT_CLASS_INDEX = 0

CT_VERIFIER_THRESHOLD = 0.50


# ============================================================
# CHECK REQUIRED FILE
# ============================================================

def check_file(
    path,
    name
):

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
# CHECK MODEL FILES
# ============================================================

check_file(
    CT_VERIFIER_FILE,
    "CT_Verifier.keras"
)

check_file(
    SCALER_FILE,
    "RobustScaler.pkl"
)

check_file(
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
#     14 × 14 × 112
#
# Output:
#     112
#
# NO Lambda
# NO K
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
        input_shape=(
            224,
            224,
            3
        )
    )


    # --------------------------------------------------------
    # Freeze EfficientNet
    # --------------------------------------------------------

    for layer in base_model.layers:

        layer.trainable = False


    # --------------------------------------------------------
    # block5c_add
    # --------------------------------------------------------

    feature_map = (
        base_model
        .get_layer(
            "block5c_add"
        )
        .output
    )


    # --------------------------------------------------------
    # GAP
    # --------------------------------------------------------

    gap = GlobalAveragePooling2D(
        name="GAP"
    )(
        feature_map
    )


    # --------------------------------------------------------
    # GSDP
    # --------------------------------------------------------

    gsdp = GSDP(
        name="GSDP"
    )(
        feature_map
    )


    # --------------------------------------------------------
    # GSFF
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
    # Model
    # --------------------------------------------------------

    feature_extractor = Model(
        inputs=base_model.input,
        outputs=gsff,
        name="GSFF_Feature_Extractor"
    )


    return feature_extractor


# ============================================================
# LOAD CT VERIFIER
# ============================================================

@st.cache_resource
def load_ct_verifier():

    model = tf.keras.models.load_model(
        str(CT_VERIFIER_FILE),
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
# LOAD ALL MODELS
# ============================================================

try:

    ct_verifier = (
        load_ct_verifier()
    )

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
# VERIFY GSFF DIMENSION
# ============================================================

if feature_extractor.output_shape[-1] != 224:

    st.error(
        "❌ GSFF feature dimension mismatch."
    )

    st.write(
        "Expected: 224"
    )

    st.write(
        f"Actual: "
        f"{feature_extractor.output_shape[-1]}"
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
            "❌ RobustScaler dimension mismatch."
        )

        st.write(
            "Expected: 224"
        )

        st.write(
            f"Actual: "
            f"{scaler.n_features_in_}"
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
            "❌ SVM dimension mismatch."
        )

        st.write(
            "Expected: 224"
        )

        st.write(
            f"Actual: "
            f"{svm.n_features_in_}"
        )

        st.stop()


# ============================================================
# COLOUR IMAGE DETECTION
# ============================================================

def is_colour_image(
    image
):

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

        difference = np.mean(
            np.abs(r - g)
            +
            np.abs(g - b)
            +
            np.abs(r - b)
        )

        if difference > 3.0:

            return True


    # --------------------------------------------------------
    # Palette image
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

        difference = np.mean(
            np.abs(r - g)
            +
            np.abs(g - b)
            +
            np.abs(r - b)
        )

        if difference > 3.0:

            return True


    return False


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(
    image
):

    # --------------------------------------------------------
    # COLOUR CHECK
    # --------------------------------------------------------

    if is_colour_image(
        image
    ):

        return (
            False,
            "colour"
        )


    # --------------------------------------------------------
    # SIZE CHECK
    # --------------------------------------------------------

    width, height = (
        image.size
    )


    if (
        width < 64
        or
        height < 64
    ):

        return (
            False,
            "small"
        )


    # --------------------------------------------------------
    # GRAYSCALE CHECK
    # --------------------------------------------------------

    gray = image.convert(
        "L"
    )


    array = np.asarray(
        gray,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # STANDARD DEVIATION
    # --------------------------------------------------------

    std = np.std(
        array
    )


    if std < 5:

        return (
            False,
            "blank"
        )


    # --------------------------------------------------------
    # ALMOST COMPLETELY BLACK
    # --------------------------------------------------------

    dark_ratio = np.mean(
        array < 10
    )


    if dark_ratio > 0.98:

        return (
            False,
            "black"
        )


    # --------------------------------------------------------
    # ALMOST COMPLETELY WHITE
    # --------------------------------------------------------

    bright_ratio = np.mean(
        array > 245
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

def preprocess_for_ct_verifier(
    image
):

    # --------------------------------------------------------
    # CT verifier receives grayscale replicated into RGB
    # --------------------------------------------------------

    gray = image.convert(
        "L"
    )


    gray = gray.resize(
        CT_VERIFIER_SIZE,
        Image.Resampling.LANCZOS
    )


    array = np.asarray(
        gray,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Grayscale → RGB
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
    # IMPORTANT:
    #
    # Do NOT divide by 255 unless your CT verifier
    # was trained with /255 preprocessing.
    #
    # Change this only if your verifier training code
    # explicitly used image_array / 255.0.
    # --------------------------------------------------------

    return array


# ============================================================
# PREPROCESS FOR GSFF
# ============================================================

def preprocess_for_gsff(
    image
):

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = image.convert(
        "L"
    )


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    gray = gray.resize(
        GSFF_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )


    # --------------------------------------------------------
    # NumPy
    # --------------------------------------------------------

    array = np.asarray(
        gray,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Grayscale → RGB
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
    # IMPORTANT:
    #
    # EfficientNetB0 in current TensorFlow/Keras includes
    # its own Rescaling layer and expects [0,255] inputs.
    #
    # Therefore no /255 here.
    # --------------------------------------------------------

    return array


# ============================================================
# CT VERIFICATION
# ============================================================

def verify_ct_image(
    image
):

    array = (
        preprocess_for_ct_verifier(
            image
        )
    )


    prediction = (
        ct_verifier.predict(
            array,
            verbose=0
        )
    )


    prediction = np.asarray(
        prediction
    )


    # ========================================================
    # CASE 1
    # Binary sigmoid output
    #
    # [[0.83]]
    # ========================================================

    if (
        prediction.ndim == 2
        and
        prediction.shape[1] == 1
    ):

        probability = float(
            prediction[0][0]
        )


        probability = float(
            np.clip(
                probability,
                0.0,
                1.0
            )
        )


        # ----------------------------------------------------
        # ASSUMPTION:
        #
        # sigmoid 1 = CT
        # sigmoid 0 = Non-CT
        #
        # If your training labels are reversed,
        # change CT_SIGMOID_MEANS_CT below.
        # ----------------------------------------------------

        CT_SIGMOID_MEANS_CT = True


        if CT_SIGMOID_MEANS_CT:

            ct_probability = (
                probability
            )

            non_ct_probability = (
                1.0 - probability
            )

        else:

            non_ct_probability = (
                probability
            )

            ct_probability = (
                1.0 - probability
            )


        is_ct = (
            ct_probability
            >=
            CT_VERIFIER_THRESHOLD
        )


        return (
            is_ct,
            ct_probability,
            non_ct_probability
        )


    # ========================================================
    # CASE 2
    # Two-class softmax
    #
    # [[0.90, 0.10]]
    # ========================================================

    if (
        prediction.ndim == 2
        and
        prediction.shape[1] == 2
    ):

        probabilities = (
            prediction[0]
        )


        probabilities = np.asarray(
            probabilities,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Class 0 = CT
        # Class 1 = Non-CT
        # ----------------------------------------------------

        ct_probability = float(
            probabilities[
                CT_CLASS_INDEX
            ]
        )


        non_ct_index = (
            1
            -
            CT_CLASS_INDEX
        )


        non_ct_probability = float(
            probabilities[
                non_ct_index
            ]
        )


        is_ct = (
            ct_probability
            >=
            CT_VERIFIER_THRESHOLD
        )


        return (
            is_ct,
            ct_probability,
            non_ct_probability
        )


    # ========================================================
    # Unsupported output
    # ========================================================

    raise ValueError(
        "Unsupported CT verifier output shape: "
        f"{prediction.shape}. "
        "Expected (1,1) sigmoid or (1,2) softmax."
    )


# ============================================================
# GSFF-SVM PREDICTION
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
    # GSFF FEATURE EXTRACTION
    # --------------------------------------------------------

    features = (
        feature_extractor.predict(
            image_array,
            verbose=0
        )
    )


    # --------------------------------------------------------
    # Confirm 224 features
    # --------------------------------------------------------

    if features.shape[1] != 224:

        raise ValueError(
            "Expected 224 GSFF features, "
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

    predicted_index = int(
        svm.predict(
            scaled_features
        )[0]
    )


    # --------------------------------------------------------
    # SVM probabilities
    # --------------------------------------------------------

    probabilities = (
        svm.predict_proba(
            scaled_features
        )[0]
    )


    # --------------------------------------------------------
    # Class
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
        ]
        *
        100
    )


    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🫁 GSFF-SVM Lung Cancer Detection"
)


st.write(
    "A lightweight deep learning and machine learning "
    "framework for classification of lung CT images."
)


st.info(
    "Only grayscale CT images are accepted. "
    "Colour images and images classified as non-CT "
    "by the CT verification model are rejected."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "🔬 Model Information"
):

    st.write(
        "**CT Verification:** "
        "CT_Verifier.keras"
    )

    st.write(
        "**Feature Extractor:** EfficientNetB0"
    )

    st.write(
        "**Truncation:** block5c_add"
    )

    st.write(
        "**Feature Map:** 14 × 14 × 112"
    )

    st.write(
        "**Pooling:** GAP + GSDP"
    )

    st.write(
        "**GSFF Dimension:** 224"
    )

    st.write(
        "**Feature Scaling:** RobustScaler"
    )

    st.write(
        "**Classifier:** RBF-SVM"
    )

    st.write(
        "**Classes:** Normal, Benign, Malignant"
    )


# ============================================================
# UPLOAD IMAGE
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
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        )


        image.load()


        # ----------------------------------------------------
        # DISPLAY IMAGE
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
        # COLOUR + BASIC VALIDATION
        # ====================================================

        st.subheader(
            "Step 1 — Image Validation"
        )


        (
            valid,
            reason
        ) = validate_image(
            image
        )


        if not valid:

            if reason == "colour":

                st.error(
                    "❌ REJECTED: Colour images "
                    "are not supported."
                )


            elif reason == "small":

                st.error(
                    "❌ REJECTED: Image resolution "
                    "is too small."
                )


            elif reason == "blank":

                st.error(
                    "❌ REJECTED: Image appears "
                    "blank or invalid."
                )


            elif reason == "black":

                st.error(
                    "❌ REJECTED: Image is almost "
                    "completely black."
                )


            elif reason == "white":

                st.error(
                    "❌ REJECTED: Image is almost "
                    "completely white."
                )


            st.warning(
                "Please upload a grayscale lung CT image."
            )


            st.stop()


        st.success(
            "✅ Grayscale image validation passed."
        )


        # ====================================================
        # STEP 2
        # CT VERIFICATION
        # ====================================================

        st.subheader(
            "Step 2 — CT Image Verification"
        )


        with st.spinner(
            "Verifying whether the image is a CT scan..."
        ):

            (
                is_ct,
                ct_probability,
                non_ct_probability
            ) = verify_ct_image(
                image
            )


        # ----------------------------------------------------
        # DISPLAY CT PROBABILITIES
        # ----------------------------------------------------

        col1, col2 = st.columns(
            2
        )


        with col1:

            st.metric(
                "CT Probability",
                f"{ct_probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Non-CT Probability",
                f"{non_ct_probability * 100:.2f}%"
            )


        # ====================================================
        # NON-CT → REJECT
        # ====================================================

        if not is_ct:

            st.error(
                "❌ REJECTED — The uploaded image "
                "was classified as NON-CT."
            )


            st.warning(
                "X-ray, MRI, or other non-CT images "
                "are not accepted by this system."
            )


            st.stop()


        # ====================================================
        # CT VERIFIED
        # ====================================================

        st.success(
            "✅ CT image verified."
        )


        # ====================================================
        # STEP 3
        # GSFF-SVM CLASSIFICATION
        # ====================================================

        st.subheader(
            "Step 3 — Lung Cancer Classification"
        )


        if st.button(
            "🔍 Detect Lung Condition",
            use_container_width=True
        ):

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


            # =================================================
            # RESULT
            # =================================================

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


            # =================================================
            # CONFIDENCE
            # =================================================

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )


            # =================================================
            # PROBABILITIES
            # =================================================

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


            # =================================================
            # DISCLAIMER
            # =================================================

            st.info(
                "This system is intended for research "
                "and educational purposes only and is "
                "not a substitute for professional "
                "medical diagnosis."
            )


    except Exception as e:

        st.error(
            "❌ An error occurred while processing "
            "the uploaded image."
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
        CT Verifier
              ↓
        CT / Non-CT
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
        Normal / Benign / Malignant
        """
    )


    st.divider()


    st.write(
        "**Input:**"
    )

    st.write(
        "224 × 224 grayscale"
    )


    st.write(
        "**Feature extraction:**"
    )

    st.write(
        "EfficientNetB0 block5c_add"
    )


    st.write(
        "**Feature fusion:**"
    )

    st.write(
        "GAP + GSDP = 224 features"
    )


    st.write(
        "**Classifier:**"
    )

    st.write(
        "RBF-SVM"
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
