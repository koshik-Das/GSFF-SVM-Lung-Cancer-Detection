import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import keras.backend as K

from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="GSFF-SVM Lung Cancer Detection",
    page_icon="🫁",
    layout="centered"
)


# =========================================================
# LOAD TRAINED MODELS
# =========================================================

@st.cache_resource
def load_models():

    # -----------------------------------------------------
    # IMPORTANT:
    # The saved GSFF feature extractor contains a Lambda
    # layer that uses K (keras.backend).
    #
    # Passing K through custom_objects allows Keras to
    # correctly reconstruct that Lambda layer.
    # -----------------------------------------------------

    feature_extractor = tf.keras.models.load_model(
        "GSFF_Feature_Extractor.keras",
        custom_objects={
            "K": K
        },
        safe_mode=False,
        compile=False
    )

    # -----------------------------------------------------
    # Load RobustScaler
    # -----------------------------------------------------

    scaler = joblib.load(
        "RobustScaler.pkl"
    )

    # -----------------------------------------------------
    # Load trained RBF-SVM
    # -----------------------------------------------------

    svm = joblib.load(
        "SVM_Classifier.pkl"
    )

    return (
        feature_extractor,
        scaler,
        svm
    )


# =========================================================
# INITIALIZE MODELS
# =========================================================

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

    st.exception(e)

    st.stop()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Normal",
    "Benign",
    "Malignant"
]


# =========================================================
# IMAGE VALIDATION
# =========================================================

def validate_image(image):

    """
    Basic input validation.

    Checks:
    1. Whether the image is grayscale
    2. Whether the image is large enough
    3. Whether the image is blank
    4. Whether the image has extreme intensity distribution

    NOTE:
    JPG/PNG files do not contain reliable modality metadata.
    Therefore, this function cannot guarantee that an image
    is a CT scan.
    """

    # -----------------------------------------------------
    # Check image mode
    # -----------------------------------------------------

    if image.mode not in [
        "L",
        "I",
        "I;16",
        "F"
    ]:

        # Check whether RGB image is actually grayscale
        rgb_image = image.convert("RGB")

        rgb_array = np.array(
            rgb_image
        )

        r = rgb_array[:, :, 0]
        g = rgb_array[:, :, 1]
        b = rgb_array[:, :, 2]

        channel_difference = np.mean(
            np.abs(
                r.astype(np.float32)
                -
                g.astype(np.float32)
            )
            +
            np.abs(
                g.astype(np.float32)
                -
                b.astype(np.float32)
            )
        )

        # Clearly colored image
        if channel_difference > 3.0:

            return (
                False,
                "❌ Rejected: Please upload a grayscale medical image."
            )

    # -----------------------------------------------------
    # Convert to grayscale
    # -----------------------------------------------------

    gray = image.convert("L")

    gray_array = np.array(
        gray
    ).astype(np.float32)

    # -----------------------------------------------------
    # Image size check
    # -----------------------------------------------------

    width, height = gray.size

    if width < 64 or height < 64:

        return (
            False,
            "❌ Rejected: Image resolution is too small."
        )

    # -----------------------------------------------------
    # Intensity statistics
    # -----------------------------------------------------

    mean_intensity = np.mean(
        gray_array
    )

    std_intensity = np.std(
        gray_array
    )

    # -----------------------------------------------------
    # Blank / nearly blank image
    # -----------------------------------------------------

    if std_intensity < 8:

        return (
            False,
            "❌ Rejected: Image appears blank or invalid."
        )

    # -----------------------------------------------------
    # Extreme intensity check
    # -----------------------------------------------------

    dark_ratio = np.mean(
        gray_array < 10
    )

    bright_ratio = np.mean(
        gray_array > 245
    )

    if (
        dark_ratio > 0.98
        or
        bright_ratio > 0.98
    ):

        return (
            False,
            "❌ Rejected: Image does not appear to be a valid CT scan."
        )

    # -----------------------------------------------------
    # Passed validation
    # -----------------------------------------------------

    return (
        True,
        "✅ Image passed the basic medical-image validation."
    )


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    # -----------------------------------------------------
    # Convert image to RGB
    # -----------------------------------------------------
    # EfficientNetB0 expects 3 channels.
    # -----------------------------------------------------

    image = image.convert(
        "RGB"
    )

    # -----------------------------------------------------
    # Resize to model input size
    # -----------------------------------------------------

    image = image.resize(
        (224, 224)
    )

    # -----------------------------------------------------
    # Convert to NumPy array
    # -----------------------------------------------------

    image_array = np.array(
        image
    ).astype(
        np.float32
    )

    # -----------------------------------------------------
    # Add batch dimension
    # Shape:
    # (224, 224, 3)
    # ->
    # (1, 224, 224, 3)
    # -----------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # -----------------------------------------------------
    # EfficientNet preprocessing
    # -----------------------------------------------------

    image_array = preprocess_input(
        image_array
    )

    # -----------------------------------------------------
    # GSFF FEATURE EXTRACTION
    # -----------------------------------------------------

    features = feature_extractor.predict(
        image_array,
        verbose=0
    )

    # -----------------------------------------------------
    # RobustScaler
    # -----------------------------------------------------

    scaled_features = scaler.transform(
        features
    )

    # -----------------------------------------------------
    # RBF-SVM prediction
    # -----------------------------------------------------

    prediction = svm.predict(
        scaled_features
    )[0]

    # -----------------------------------------------------
    # Probability prediction
    # -----------------------------------------------------

    probabilities = svm.predict_proba(
        scaled_features
    )[0]

    # -----------------------------------------------------
    # Predicted class
    # -----------------------------------------------------

    predicted_class = class_names[
        int(prediction)
    ]

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence = (
        probabilities[int(prediction)]
        * 100
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "🫁 GSFF-SVM Lung Cancer Detection"
)

st.write(
    "Deep learning and machine learning based "
    "lung CT image classification system."
)

st.info(
    "Upload a grayscale lung CT image for classification."
)


# =========================================================
# UPLOAD IMAGE
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Lung CT Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# PROCESS UPLOADED IMAGE
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        )

    except Exception:

        st.error(
            "❌ Unable to read the uploaded image."
        )

        st.stop()

    # -----------------------------------------------------
    # Display uploaded image
    # -----------------------------------------------------

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------------------------------
    # Validate image
    # -----------------------------------------------------

    (
        is_valid,
        validation_message
    ) = validate_image(
        image
    )

    if not is_valid:

        st.error(
            validation_message
        )

        st.warning(
            "Please upload a grayscale lung CT image."
        )

        st.stop()

    st.success(
        validation_message
    )

    # =====================================================
    # DETECTION BUTTON
    # =====================================================

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
            f"{confidence:.2f}%"
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

            probability = (
                probabilities[i]
                * 100
            )

            st.write(
                f"**{class_name}: "
                f"{probability:.2f}%**"
            )

            st.progress(
                float(
                    probabilities[i]
                )
            )


# =========================================================
# SIDEBAR
# =========================================================

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
        Classification
        """
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
