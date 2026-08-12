# ============================================================
# app.py
#
# GSFF-SVM Lung Cancer Detection System
# ============================================================

import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow.keras.backend as K
import joblib

from PIL import Image
from pathlib import Path

from tensorflow.keras.applications.efficientnet import preprocess_input


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

INPUT_SIZE = (224, 224)


# ============================================================
# IMAGE VALIDATION THRESHOLDS
# ============================================================
#
# IMPORTANT:
#
# These are IMAGE-QUALITY / INPUT-REJECTION thresholds.
#
# They are NOT trained X-ray/MRI modality thresholds.
# Reliable X-ray-vs-MRI-vs-CT classification requires
# a separate trained modality classifier.
#
# ============================================================

MIN_IMAGE_SIZE = 64

MIN_STD = 8.0

MAX_DARK_RATIO = 0.98

MAX_BRIGHT_RATIO = 0.98

MIN_DYNAMIC_RANGE = 20.0


# ============================================================
# MODEL FILE CHECK
# ============================================================

def check_model_file(path, model_name):

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
            f"❌ {model_name} file is empty."
        )

        st.code(
            str(path)
        )

        st.stop()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

check_model_file(
    FEATURE_EXTRACTOR_PATH,
    "GSFF feature extractor"
)

check_model_file(
    SCALER_PATH,
    "RobustScaler"
)

check_model_file(
    SVM_PATH,
    "SVM classifier"
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # GSFF FEATURE EXTRACTOR
    #
    # custom_objects={"K": K}
    # is required because the saved Lambda layer uses K.
    # --------------------------------------------------------

    feature_extractor = tf.keras.models.load_model(
        str(FEATURE_EXTRACTOR_PATH),

        custom_objects={
            "K": K
        },

        safe_mode=False,

        compile=False
    )

    # --------------------------------------------------------
    # RobustScaler
    # --------------------------------------------------------

    scaler = joblib.load(
        str(SCALER_PATH)
    )

    # --------------------------------------------------------
    # RBF-SVM
    # --------------------------------------------------------

    svm = joblib.load(
        str(SVM_PATH)
    )

    return (
        feature_extractor,
        scaler,
        svm
    )


# ============================================================
# LOAD MODELS
# ============================================================

try:

    (
        feature_extractor,
        scaler,
        svm
    ) = load_models()

except Exception as e:

    st.error(
        "❌ Model loading failed."
    )

    st.error(
        "Please verify that the GSFF feature extractor, "
        "RobustScaler and SVM files match the trained model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Normal",
    "Benign",
    "Malignant"
]


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_ct_image(image):

    """
    Performs basic input validation before the image
    reaches the GSFF-SVM model.

    The validation checks:

    1. Colour image rejection
    2. Minimum image resolution
    3. Blank image rejection
    4. Extremely dark image rejection
    5. Extremely bright image rejection
    6. Extremely low dynamic-range rejection

    IMPORTANT:
    These checks cannot mathematically prove that an image
    is CT. A separate trained modality classifier is needed
    for reliable CT/X-ray/MRI discrimination.
    """

    # ========================================================
    # STEP 1 — COLOUR IMAGE CHECK
    # ========================================================

    original_mode = image.mode

    # --------------------------------------------------------
    # RGB / RGBA / CMYK etc.
    # --------------------------------------------------------

    if original_mode not in [
        "L",
        "I",
        "I;16",
        "F"
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

        channel_difference = np.mean(
            np.abs(r - g)
            +
            np.abs(g - b)
        )

        # ----------------------------------------------------
        # Clearly coloured image
        # ----------------------------------------------------

        if channel_difference > 3.0:

            return (
                False,
                "❌ Colour images are not supported. "
                "Please upload a grayscale lung CT image."
            )

    # ========================================================
    # STEP 2 — CONVERT TO GRAYSCALE
    # ========================================================

    gray = image.convert(
        "L"
    )

    gray_array = np.asarray(
        gray,
        dtype=np.float32
    )

    # ========================================================
    # STEP 3 — IMAGE SIZE
    # ========================================================

    width, height = gray.size

    if (
        width < MIN_IMAGE_SIZE
        or
        height < MIN_IMAGE_SIZE
    ):

        return (
            False,
            "❌ Image resolution is too small. "
            "Please upload a higher-resolution medical image."
        )

    # ========================================================
    # STEP 4 — INTENSITY STATISTICS
    # ========================================================

    mean_intensity = float(
        np.mean(gray_array)
    )

    std_intensity = float(
        np.std(gray_array)
    )

    minimum_intensity = float(
        np.min(gray_array)
    )

    maximum_intensity = float(
        np.max(gray_array)
    )

    dynamic_range = (
        maximum_intensity
        -
        minimum_intensity
    )

    # ========================================================
    # STEP 5 — BLANK / LOW-CONTRAST IMAGE
    # ========================================================

    if std_intensity < MIN_STD:

        return (
            False,
            "❌ The uploaded image has extremely low "
            "contrast and may be blank or invalid."
        )

    # ========================================================
    # STEP 6 — DYNAMIC RANGE CHECK
    # ========================================================

    if dynamic_range < MIN_DYNAMIC_RANGE:

        return (
            False,
            "❌ The uploaded image has an unusually "
            "small intensity range."
        )

    # ========================================================
    # STEP 7 — EXTREME BLACK/WHITE CHECK
    # ========================================================

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
            "❌ The image is almost completely black "
            "and does not appear to be a valid CT image."
        )

    if bright_ratio > MAX_BRIGHT_RATIO:

        return (
            False,
            "❌ The image is almost completely white "
            "and does not appear to be a valid CT image."
        )

    # ========================================================
    # PASSED BASIC VALIDATION
    # ========================================================

    return (
        True,
        "✅ Image passed the basic input validation."
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # Convert grayscale image to RGB
    #
    # IMPORTANT:
    # This happens ONLY after validation.
    #
    # EfficientNet expects 3 channels.
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = image.resize(
        INPUT_SIZE,
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # NumPy conversion
    # --------------------------------------------------------

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Add batch dimension
    #
    # (224,224,3)
    # ->
    # (1,224,224,3)
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
# GSFF-SVM PREDICTION
# ============================================================

def predict_image(image):

    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    image_array = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # GSFF FEATURE EXTRACTION
    # --------------------------------------------------------

    features = feature_extractor.predict(
        image_array,
        verbose=0
    )

    # --------------------------------------------------------
    # RobustScaler
    # --------------------------------------------------------

    scaled_features = scaler.transform(
        features
    )

    # --------------------------------------------------------
    # RBF-SVM CLASSIFICATION
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
    # Convert prediction to integer
    # --------------------------------------------------------

    prediction = int(
        prediction
    )

    # --------------------------------------------------------
    # Class name
    # --------------------------------------------------------

    predicted_class = class_names[
        prediction
    ]

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(
        probabilities[prediction]
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
    "Deep learning and machine learning based "
    "lung CT image classification system."
)

st.info(
    "Only grayscale medical images are supported. "
    "Colour images are automatically rejected."
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
        "Feature extraction stage: Block5c"
    )

    st.write(
        "Feature fusion: GAP + GMP + STD"
    )

    st.write(
        "Feature normalization: RobustScaler"
    )

    st.write(
        "Classifier: RBF-SVM"
    )

    st.write(
        "Input size: 224 × 224 × 3"
    )

    st.write(
        "Classes: Normal, Benign, Malignant"
    )


# ============================================================
# INPUT VALIDATION INFORMATION
# ============================================================

with st.expander(
    "⚙️ Input Validation Thresholds"
):

    st.write(
        f"Minimum image size: "
        f"{MIN_IMAGE_SIZE} × {MIN_IMAGE_SIZE}"
    )

    st.write(
        f"Minimum grayscale standard deviation: "
        f"{MIN_STD}"
    )

    st.write(
        f"Maximum dark-pixel ratio: "
        f"{MAX_DARK_RATIO * 100:.0f}%"
    )

    st.write(
        f"Maximum bright-pixel ratio: "
        f"{MAX_BRIGHT_RATIO * 100:.0f}%"
    )

    st.write(
        f"Minimum dynamic range: "
        f"{MIN_DYNAMIC_RANGE}"
    )

    st.warning(
        "These are image-quality thresholds, not "
        "X-ray/MRI/CT modality-classification thresholds."
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

        # Force image loading
        image.load()

        # ----------------------------------------------------
        # Display uploaded image
        # ----------------------------------------------------

        st.subheader(
            "📷 Uploaded Image"
        )

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # ====================================================
        # STEP 1 — IMAGE VALIDATION
        # ====================================================

        st.subheader(
            "Step 1 — Image Validation"
        )

        (
            is_valid,
            validation_message
        ) = validate_ct_image(
            image
        )

        if not is_valid:

            st.error(
                validation_message
            )

            st.warning(
                "Prediction was stopped because the "
                "uploaded image did not pass validation."
            )

            st.stop()

        st.success(
            validation_message
        )

        # ====================================================
        # IMAGE STATISTICS
        # ====================================================

        gray = image.convert(
            "L"
        )

        gray_array = np.asarray(
            gray,
            dtype=np.float32
        )

        mean_intensity = float(
            np.mean(gray_array)
        )

        std_intensity = float(
            np.std(gray_array)
        )

        dynamic_range = float(
            np.max(gray_array)
            -
            np.min(gray_array)
        )

        with st.expander(
            "Image Statistics"
        ):

            st.write(
                f"Mean intensity: "
                f"{mean_intensity:.2f}"
            )

            st.write(
                f"Standard deviation: "
                f"{std_intensity:.2f}"
            )

            st.write(
                f"Dynamic range: "
                f"{dynamic_range:.2f}"
            )

        # ====================================================
        # STEP 2 — DETECTION
        # ====================================================

        st.subheader(
            "Step 2 — Lung Condition Detection"
        )

        if st.button(
            "🔍 Detect Lung Condition",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing CT image..."
            ):

                try:

                    (
                        predicted_class,
                        confidence,
                        probabilities
                    ) = predict_image(
                        image
                    )

                except Exception as e:

                    st.error(
                        "❌ Prediction failed."
                    )

                    st.exception(
                        e
                    )

                    st.stop()

            # =================================================
            # PREDICTION RESULT
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
                f"{confidence * 100:.2f}%"
            )

            # =================================================
            # CLASS PROBABILITIES
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
                "⚠️ This system is a research prototype "
                "and is not intended for clinical diagnosis. "
                "Predictions should not replace evaluation "
                "by a qualified medical professional."
            )

    except Exception as e:

        st.error(
            "❌ An error occurred while processing the image."
        )

        st.exception(
            e
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
        EfficientNetB0
        ↓
        Block5c
        ↓
        GAP + GMP + STD
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
        "Grayscale lung CT images only"
    )

    st.divider()

    st.caption(
        "Research prototype. "
        "Not intended for clinical diagnosis."
    )
