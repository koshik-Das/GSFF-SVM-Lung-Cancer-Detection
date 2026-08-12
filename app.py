# ============================================================
# app.py
# GSFF-SVM Lung Cancer Detection
#
# Pipeline:
# 1. Reject clearly colored images
# 2. CT/MRI/X-ray modality verification
# 3. Only CT images are accepted
# 4. GSFF feature extraction
# 5. RobustScaler
# 6. RBF-SVM classification
# ============================================================

import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import joblib

from PIL import Image


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

CT_VERIFIER_PATH = BASE_DIR / "CT_Verifier.keras"

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
# IMAGE SETTINGS
# ============================================================

MODALITY_IMAGE_SIZE = (128, 128)

GSFF_IMAGE_SIZE = (224, 224)


# ============================================================
# CLASS NAMES
# ============================================================

LUNG_CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]


# IMPORTANT:
# This MUST match the class_mapping printed during training.
#
# Your trained CT_Verifier showed:
#
# 0 = CHEST_XRAY
# 1 = CT
# 2 = MRI

MODALITY_CLASS_NAMES = [
    "CHEST_XRAY",
    "CT",
    "MRI"
]


# ============================================================
# CT ACCEPTANCE THRESHOLD
# ============================================================

# The modality classifier must identify the image as CT
# with at least this confidence.
#
# 0.70 = 70%

CT_THRESHOLD = 0.70


# ============================================================
# CHECK REQUIRED FILE
# ============================================================

def check_file(path, file_description):

    if not path.exists():

        st.error(
            f"❌ {file_description} was not found."
        )

        st.code(
            str(path)
        )

        st.stop()


    if path.stat().st_size == 0:

        st.error(
            f"❌ {file_description} is empty."
        )

        st.code(
            str(path)
        )

        st.stop()


# ============================================================
# CHECK ALL MODEL FILES
# ============================================================

check_file(
    CT_VERIFIER_PATH,
    "CT modality verifier"
)

check_file(
    FEATURE_EXTRACTOR_PATH,
    "GSFF feature extractor"
)

check_file(
    SCALER_PATH,
    "RobustScaler"
)

check_file(
    SVM_PATH,
    "SVM classifier"
)


# ============================================================
# LOAD CT VERIFIER
# ============================================================

@st.cache_resource
def load_ct_verifier():

    model = tf.keras.models.load_model(
        str(CT_VERIFIER_PATH),
        compile=False
    )

    return model


# ============================================================
# LOAD GSFF FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def load_feature_extractor():

    # IMPORTANT:
    #
    # This file was already trained and saved.
    # We DO NOT create any Lambda layer here.
    #
    # Therefore app.py does not depend on K.

    model = tf.keras.models.load_model(
        str(FEATURE_EXTRACTOR_PATH),
        compile=False
    )

    return model


# ============================================================
# LOAD ROBUST SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    scaler = joblib.load(
        str(SCALER_PATH)
    )

    return scaler


# ============================================================
# LOAD SVM
# ============================================================

@st.cache_resource
def load_svm():

    svm = joblib.load(
        str(SVM_PATH)
    )

    return svm


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    ct_verifier = load_ct_verifier()

except Exception as e:

    st.error(
        "❌ CT verifier model loading failed."
    )

    st.error(
        "Please make sure CT_Verifier.keras "
        "was uploaded correctly to GitHub."
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
        "Please make sure GSFF_Feature_Extractor.keras "
        "is the same model used during training."
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

    modality_output_shape = (
        ct_verifier.output_shape
    )

    if modality_output_shape[-1] != 3:

        st.error(
            "❌ CT verifier output mismatch."
        )

        st.write(
            "Expected output: 3 classes"
        )

        st.write(
            f"Actual output: {modality_output_shape}"
        )

        st.stop()

except Exception as e:

    st.error(
        "❌ Could not verify CT verifier output."
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

    feature_dimension = (
        feature_output_shape[-1]
    )

    st.sidebar.success(
        f"GSFF feature dimension: "
        f"{feature_dimension}"
    )

except Exception:

    feature_dimension = None


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image):
    """
    Performs basic input validation.

    The system rejects:
    1. Clearly colored images
    2. Very small images
    3. Blank / nearly blank images
    4. Almost completely black images
    5. Almost completely white images

    CT/MRI/X-ray discrimination is performed separately
    by CT_Verifier.keras.
    """

    # --------------------------------------------------------
    # Original image information
    # --------------------------------------------------------

    original_mode = image.mode

    # --------------------------------------------------------
    # Check image channels
    # --------------------------------------------------------

    rgb_image = image.convert("RGB")

    rgb_array = np.asarray(
        rgb_image,
        dtype=np.float32
    )

    r = rgb_array[:, :, 0]
    g = rgb_array[:, :, 1]
    b = rgb_array[:, :, 2]

    # --------------------------------------------------------
    # Calculate color difference
    # --------------------------------------------------------

    rg_difference = np.mean(
        np.abs(r - g)
    )

    gb_difference = np.mean(
        np.abs(g - b)
    )

    rb_difference = np.mean(
        np.abs(r - b)
    )

    color_difference = (
        rg_difference
        + gb_difference
        + rb_difference
    )

    # --------------------------------------------------------
    # Reject clearly colored image
    # --------------------------------------------------------

    if color_difference > 10:

        return (
            False,
            "❌ Rejected: Colored images are not supported."
        )

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = image.convert("L")

    gray_array = np.asarray(
        gray,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    width, height = gray.size

    if width < 64 or height < 64:

        return (
            False,
            "❌ Rejected: Image resolution is too small."
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
            "❌ Rejected: Image appears blank or invalid."
        )

    # --------------------------------------------------------
    # Dark / bright ratio
    # --------------------------------------------------------

    dark_ratio = np.mean(
        gray_array < 10
    )

    bright_ratio = np.mean(
        gray_array > 245
    )

    # --------------------------------------------------------
    # Almost completely black/white
    # --------------------------------------------------------

    if dark_ratio > 0.98:

        return (
            False,
            "❌ Rejected: Image is almost completely black."
        )

    if bright_ratio > 0.98:

        return (
            False,
            "❌ Rejected: Image is almost completely white."
        )

    # --------------------------------------------------------
    # Passed basic validation
    # --------------------------------------------------------

    return (
        True,
        "✅ Image passed basic image validation."
    )


# ============================================================
# PREPROCESS FOR MODALITY CLASSIFIER
# ============================================================

def preprocess_for_modality(image):

    # Convert grayscale-like input to RGB.
    #
    # This does NOT turn a colored image into an accepted
    # image because validate_image() already rejects
    # clearly colored images.

    image = image.convert("RGB")

    image = image.resize(
        MODALITY_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Same preprocessing used during modality training:
    #
    # rescale = 1 / 255

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
        GSFF_IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Your training code used:
    #
    # X = preprocess_input(...)
    #
    # Therefore we reproduce that preprocessing here.
    # EfficientNet's preprocessing is imported directly
    # instead of creating any Lambda layer.
    # --------------------------------------------------------

    from tensorflow.keras.applications.efficientnet import (
        preprocess_input
    )

    image_array = preprocess_input(
        image_array
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# CT / MRI / X-RAY VERIFICATION
# ============================================================

def verify_modality(image):

    image_array = preprocess_for_modality(
        image
    )

    predictions = ct_verifier.predict(
        image_array,
        verbose=0
    )

    probabilities = np.asarray(
        predictions[0],
        dtype=np.float32
    )

    # Safety normalization

    probability_sum = np.sum(
        probabilities
    )

    if probability_sum > 0:

        probabilities = (
            probabilities
            / probability_sum
        )

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    predicted_modality = (
        MODALITY_CLASS_NAMES[
            predicted_index
        ]
    )

    return (
        predicted_modality,
        predicted_index,
        confidence,
        probabilities
    )


# ============================================================
# GSFF-SVM PREDICTION
# ============================================================

def predict_lung_cancer(image):

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image_array = preprocess_for_gsff(
        image
    )

    # --------------------------------------------------------
    # GSFF FEATURE EXTRACTION
    # --------------------------------------------------------

    features = feature_extractor.predict(
        image_array,
        verbose=0
    )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # RobustScaler
    # --------------------------------------------------------

    scaled_features = scaler.transform(
        features
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
    # SVM probabilities
    # --------------------------------------------------------

    probabilities = svm.predict_proba(
        scaled_features
    )[0]

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    predicted_class = (
        LUNG_CLASS_NAMES[
            predicted_index
        ]
    )

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        predicted_index,
        confidence,
        probabilities,
        features
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
    "Only grayscale CT images are accepted. "
    "Chest X-ray, MRI, and colored images are rejected."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🫁 GSFF-SVM Framework"
    )

    st.write(
        """
        Input Image
        ↓
        Image Validation
        ↓
        CT / X-ray / MRI Verification
        ↓
        CT Only
        ↓
        EfficientNetB0
        ↓
        Block5c
        ↓
        GAP + GSDP
        ↓
        GSFF Fusion
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
        "**Modality Classes:**"
    )

    st.write(
        "0 — Chest X-ray"
    )

    st.write(
        "1 — CT"
    )

    st.write(
        "2 — MRI"
    )

    st.divider()

    st.write(
        f"**CT acceptance threshold:** "
        f"{CT_THRESHOLD * 100:.0f}%"
    )

    st.divider()

    st.write(
        "**Lung Classes:**"
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


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "ℹ️ Model Information"
):

    st.write(
        "CT verifier: 3-class modality classifier"
    )

    st.write(
        "Modality input: 128 × 128 × 3"
    )

    st.write(
        "Modality classes: Chest X-ray, CT, MRI"
    )

    st.write(
        "GSFF input: 224 × 224 × 3"
    )

    st.write(
        "EfficientNetB0 truncation: block5c_add"
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
# PROCESS IMAGE
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
        "📷 Uploaded Image"
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
        validate_image(image)
    )


    if not is_valid:

        st.error(
            validation_message
        )

        st.warning(
            "Please upload a grayscale medical CT image."
        )

        st.stop()


    st.success(
        validation_message
    )


    # ========================================================
    # DETECTION BUTTON
    # ========================================================

    if st.button(
        "🔍 Detect Lung Condition",
        use_container_width=True
    ):

        # ====================================================
        # STEP 1 — MODALITY VERIFICATION
        # ====================================================

        st.subheader(
            "Step 1 — Image Modality Verification"
        )

        with st.spinner(
            "Checking whether the image is CT, X-ray, or MRI..."
        ):

            try:

                (
                    modality,
                    modality_index,
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
        # SHOW MODALITY PROBABILITIES
        # ====================================================

        st.write(
            "**Modality probability estimates:**"
        )

        for i, modality_name in enumerate(
            MODALITY_CLASS_NAMES
        ):

            probability = float(
                modality_probabilities[i]
            )

            st.write(
                f"**{modality_name}: "
                f"{probability * 100:.2f}%**"
            )

            st.progress(
                min(
                    max(
                        probability,
                        0.0
                    ),
                    1.0
                )
            )


        # ====================================================
        # CT ACCEPTANCE DECISION
        # ====================================================

        if (
            modality == "CT"
            and modality_confidence >= CT_THRESHOLD
        ):

            st.success(
                f"✅ CT scan verified "
                f"({modality_confidence * 100:.2f}% confidence)"
            )

            st.write(
                "The image has passed the modality "
                "verification stage."
            )


            # ================================================
            # STEP 2 — GSFF-SVM
            # ================================================

            st.subheader(
                "Step 2 — GSFF-SVM Lung Cancer Classification"
            )

            with st.spinner(
                "Extracting GSFF features and analyzing CT image..."
            ):

                try:

                    (
                        predicted_class,
                        predicted_index,
                        confidence,
                        probabilities,
                        features
                    ) = predict_lung_cancer(
                        image
                    )

                except Exception as e:

                    st.error(
                        "❌ Lung cancer prediction failed."
                    )

                    st.exception(e)

                    st.stop()


            # ================================================
            # RESULT
            # ================================================

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


            # ================================================
            # CONFIDENCE
            # ================================================

            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )


            # ================================================
            # CLASS PROBABILITIES
            # ================================================

            st.subheader(
                "📊 Class Probability Estimates"
            )


            for i, class_name in enumerate(
                LUNG_CLASS_NAMES
            ):

                probability = float(
                    probabilities[i]
                )

                st.write(
                    f"**{class_name}: "
                    f"{probability * 100:.2f}%**"
                )

                st.progress(
                    min(
                        max(
                            probability,
                            0.0
                        ),
                        1.0
                    )
                )


            # ================================================
            # TECHNICAL INFORMATION
            # ================================================

            with st.expander(
                "🔬 Technical Information"
            ):

                st.write(
                    f"Verified modality: **{modality}**"
                )

                st.write(
                    f"Modality confidence: "
                    f"**{modality_confidence * 100:.2f}%**"
                )

                st.write(
                    f"GSFF feature shape: "
                    f"`{features.shape}`"
                )

                st.write(
                    "Feature fusion: GAP + GSDP"
                )

                st.write(
                    "Classifier: RBF-SVM"
                )


            # ================================================
            # DISCLAIMER
            # ================================================

            st.info(
                "⚠️ This system is a research prototype "
                "for educational and research purposes. "
                "It is not intended to replace professional "
                "medical diagnosis."
            )


        else:

            # =================================================
            # REJECT NON-CT IMAGE
            # =================================================

            st.error(
                "❌ Image Rejected"
            )


            if modality == "CHEST_XRAY":

                st.warning(
                    "The uploaded image appears to be "
                    "a Chest X-ray. Pneumonia/X-ray images "
                    "are not supported by this system."
                )

            elif modality == "MRI":

                st.warning(
                    "The uploaded image appears to be "
                    "an MRI scan. MRI images are not "
                    "supported by this system."
                )

            elif modality == "CT":

                st.warning(
                    f"The image was classified as CT, "
                    f"but the confidence "
                    f"({modality_confidence * 100:.2f}%) "
                    f"is below the required "
                    f"{CT_THRESHOLD * 100:.0f}% threshold."
                )

            else:

                st.warning(
                    "The uploaded image could not be "
                    "verified as a valid CT scan."
                )


            st.write(
                "🚫 Lung cancer classification was "
                "not performed."
            )

            st.info(
                "Please upload a grayscale lung CT image."
            )
