# ============================================================
# app.py
# GSFF-SVM Lung Cancer Detection
#
# EfficientNetB0
#      ↓
# block5c_add
#      ↓
# GAP + GSDP
#      ↓
# GSFF Fusion (224 features)
#      ↓
# RobustScaler
#      ↓
# RBF-SVM
#      ↓
# Normal / Benign / Malignant
#
# IMPORTANT:
# NO Lambda layer
# NO keras.backend.K
# NO .keras feature extractor loading
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
# CUSTOM GSDP LAYER
# ============================================================
#
# GSDP = Global Standard Deviation Pooling
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

    def call(self, inputs):

        return tf.math.reduce_std(
            inputs,
            axis=[1, 2]
        )


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
# CHECK SCALER AND SVM
# ============================================================

check_file(
    SCALER_FILE,
    "RobustScaler.pkl"
)

check_file(
    SVM_FILE,
    "SVM_Classifier.pkl"
)


# ============================================================
# BUILD GSFF FEATURE EXTRACTOR
# ============================================================
#
# This completely bypasses the old problematic
# GSFF_Feature_Extractor.keras file.
#
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
    # GSFF FUSION
    # --------------------------------------------------------

    gsff = Concatenate(
        name="GSFF_Fusion"
    )([
        gap,
        gsdp
    ])


    # --------------------------------------------------------
    # Final feature extractor
    # --------------------------------------------------------

    model = Model(
        inputs=base_model.input,
        outputs=gsff,
        name="GSFF_Feature_Extractor"
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
# LOAD EVERYTHING
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

try:

    feature_dimension = (
        feature_extractor.output_shape[-1]
    )

except Exception as e:

    st.error(
        "❌ Could not determine feature dimension."
    )

    st.exception(e)

    st.stop()


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
# IMAGE VALIDATION
# ============================================================

def validate_image(image):

    # --------------------------------------------------------
    # Reject actual colour images
    # --------------------------------------------------------

    if image.mode in [
        "RGB",
        "RGBA",
        "CMYK",
        "P",
        "HSV"
    ]:

        rgb = image.convert(
            "RGB"
        )

        arr = np.asarray(
            rgb,
            dtype=np.float32
        )


        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]


        channel_difference = np.mean(
            np.abs(r - g)
            +
            np.abs(g - b)
            +
            np.abs(r - b)
        )


        # If channels differ meaningfully,
        # treat as a colour image.

        if channel_difference > 3.0:

            return (
                False,
                "❌ Colour images are not supported. "
                "Please upload a grayscale lung CT image."
            )


    # --------------------------------------------------------
    # Convert to grayscale for validation
    # --------------------------------------------------------

    gray = image.convert(
        "L"
    )


    gray_array = np.asarray(
        gray,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Resolution check
    # --------------------------------------------------------

    width, height = gray.size


    if width < 64 or height < 64:

        return (
            False,
            "❌ Image resolution is too small."
        )


    # --------------------------------------------------------
    # Intensity statistics
    # --------------------------------------------------------

    mean_intensity = np.mean(
        gray_array
    )

    std_intensity = np.std(
        gray_array
    )


    # --------------------------------------------------------
    # Blank image check
    # --------------------------------------------------------

    if std_intensity < 8:

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
            "❌ Image appears almost completely black."
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
            "❌ Image appears almost completely white."
        )


    # --------------------------------------------------------
    # Passed
    # --------------------------------------------------------

    return (
        True,
        "✅ Image passed basic validation."
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # Convert grayscale to RGB
    #
    # IMPORTANT:
    #
    # This does NOT mean colour images are accepted.
    #
    # Validation happens BEFORE this function.
    #
    # The trained EfficientNet requires 3 channels.
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
    #
    # All three channels contain the SAME grayscale values.
    # --------------------------------------------------------

    image_array = np.asarray(
        gray,
        dtype=np.float32
    )


    image_array = np.stack(
        [
            image_array,
            image_array,
            image_array
        ],
        axis=-1
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # EfficientNet preprocessing
    # --------------------------------------------------------

    image_array = preprocess_input(
        image_array
    )


    return image_array


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_array = preprocess_image(
        image
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
    # Safety check
    # --------------------------------------------------------

    if features.shape[1] != 224:

        raise ValueError(
            f"GSFF produced {features.shape[1]} "
            f"features instead of 224."
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
    )[0]


    # --------------------------------------------------------
    # SVM probabilities
    # --------------------------------------------------------

    probabilities = svm.predict_proba(
        scaled_features
    )[0]


    # --------------------------------------------------------
    # Class
    # --------------------------------------------------------

    predicted_class = class_names[
        int(prediction)
    ]


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = (
        float(
            probabilities[
                int(prediction)
            ]
        )
        * 100
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
    "EfficientNetB0-based deep feature extraction "
    "combined with GAP + GSDP fusion and an RBF-SVM "
    "classifier for lung CT image classification."
)


st.info(
    "⚠️ Only grayscale lung CT images are supported. "
    "Colour images are rejected."
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
        "Pooling: GAP + GSDP"
    )

    st.write(
        "GSFF Feature Dimension: 224"
    )

    st.write(
        "Scaler: RobustScaler"
    )

    st.write(
        "Classifier: RBF-SVM"
    )

    st.write(
        "Classes: Normal, Benign, Malignant"
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
        # Validate BEFORE prediction
        # ----------------------------------------------------

        is_valid, message = (
            validate_image(
                image
            )
        )


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


        # ----------------------------------------------------
        # Reject invalid image
        # ----------------------------------------------------

        if not is_valid:

            st.error(
                message
            )

            st.warning(
                "Please upload a grayscale lung CT image."
            )

            st.stop()


        # ----------------------------------------------------
        # Valid image
        # ----------------------------------------------------

        st.success(
            message
        )


        # ====================================================
        # DETECTION BUTTON
        # ====================================================

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

                probability = (
                    float(
                        probabilities[i]
                    )
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
        EfficientNetB0
              ↓
        block5c_add
              ↓
        GAP + GSDP
              ↓
        GSFF
              ↓
        224 Features
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
        "**Classes:**"
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


    st.write(
        "**Input:**"
    )

    st.write(
        "Grayscale lung CT image"
    )


    st.write(
        "**Colour images:**"
    )

    st.write(
        "❌ Not supported"
    )


    st.divider()


    st.caption(
        "Research prototype. "
        "Not intended for clinical diagnosis."
    )
