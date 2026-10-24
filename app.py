# ============================================================
# app.py — PulmoVision
# AI-Powered Lung Cancer Detection System
# ============================================================

# ============================================================
# 1. IMPORTS
# ============================================================

import io
import hashlib
from pathlib import Path
from datetime import datetime

import joblib
import keras
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PulmoVision",
    page_icon="🫁",
    layout="wide",
)

# ============================================================
# BASE DIRECTORY AND MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

LUNG_ICON_PATH = BASE_DIR / "Lung_Icon.png"

CT_VERIFIER_PATH = BASE_DIR / "CT_Verifier.keras"
FEATURE_EXTRACTOR_PATH = BASE_DIR / "GSFF_Feature_Extractor.keras"
SCALER_PATH = BASE_DIR / "RobustScaler.pkl"
SVM_PATH = BASE_DIR / "SVM_Classifier.pkl"

# ============================================================
# IMAGE SETTINGS
# ============================================================

CT_VERIFIER_IMAGE_SIZE = (128, 128)
FEATURE_EXTRACTOR_IMAGE_SIZE = (224, 224)

MODALITY_THRESHOLD = 0.60
COLOR_TOLERANCE = 8.0

class_names = ["Normal", "Benign", "Malignant"]

modality_names = [
    "CHEST_XRAY",
    "CT",
    "MRI",
]

# ============================================================
# CUSTOM GSDP LAYER — FIXED
#
# Must match the custom layer used when the model was saved.
# Register this class BEFORE loading the feature extractor.
# ============================================================

@keras.saving.register_keras_serializable(
    package="PulmoVision"
)
class SpatialStdPooling(keras.layers.Layer):

    def call(self, inputs):
        mean = tf.reduce_mean(
            inputs,
            axis=[1, 2],
            keepdims=True,
        )

        variance = tf.reduce_mean(
            tf.square(inputs - mean),
            axis=[1, 2],
        )

        return tf.sqrt(
            tf.maximum(variance, 0.0)
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super().get_config()


# ============================================================
# MODEL FILE VALIDATION
# ============================================================

def validate_model_file(path, model_name):

    if not path.exists():
        st.error(f"{model_name} was not found.")
        st.code(str(path))
        st.stop()

    if path.stat().st_size == 0:
        st.error(f"{model_name} file is empty.")
        st.stop()


required_files = {
    "CT Verifier": CT_VERIFIER_PATH,
    "GSFF Feature Extractor": FEATURE_EXTRACTOR_PATH,
    "RobustScaler": SCALER_PATH,
    "SVM Classifier": SVM_PATH,
}

for model_name, model_path in required_files.items():
    validate_model_file(model_path, model_name)


# ============================================================
# LOAD CT VERIFIER
# ============================================================

@st.cache_resource
def load_ct_verifier():

    return keras.models.load_model(
        CT_VERIFIER_PATH,
        compile=False,
    )


# ============================================================
# LOAD GSFF FEATURE EXTRACTOR — FIXED
# ============================================================

@st.cache_resource
def load_feature_extractor():

    model = keras.models.load_model(
        FEATURE_EXTRACTOR_PATH,
        custom_objects={
            "SpatialStdPooling": SpatialStdPooling,
            "PulmoVision>SpatialStdPooling": SpatialStdPooling,
        },
        compile=False,
        safe_mode=True,
    )

    return model


# ============================================================
# LOAD ROBUST SCALER
# ============================================================

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


# ============================================================
# LOAD SVM CLASSIFIER
# ============================================================

@st.cache_resource
def load_svm():
    return joblib.load(SVM_PATH)


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:
    ct_verifier = load_ct_verifier()
except Exception as e:
    st.error("CT verifier loading failed.")
    st.exception(e)
    st.stop()

try:
    feature_extractor = load_feature_extractor()
except Exception as e:
    st.error("GSFF feature extractor loading failed.")
    st.error(
        "Check that GSFF_Feature_Extractor.keras was saved "
        "using the SpatialStdPooling implementation defined "
        "in this app and that the uploaded model is valid."
    )
    st.exception(e)
    st.stop()

try:
    scaler = load_scaler()
except Exception as e:
    st.error("RobustScaler loading failed.")
    st.exception(e)
    st.stop()

try:
    svm = load_svm()
except Exception as e:
    st.error("SVM classifier loading failed.")
    st.exception(e)
    st.stop()


# ============================================================
# VERIFY MODEL OUTPUTS
# ============================================================

try:
    ct_output_shape = ct_verifier.output_shape

    if ct_output_shape[-1] != 3:
        st.error("CT verifier output mismatch.")
        st.write("Expected: 3 classes")
        st.write(f"Actual: {ct_output_shape}")
        st.stop()

except Exception as e:
    st.error("Unable to verify CT verifier output.")
    st.exception(e)
    st.stop()


try:
    feature_output_shape = feature_extractor.output_shape
    expected_feature_dimension = 224

    if feature_output_shape[-1] != expected_feature_dimension:
        st.error("GSFF feature dimension mismatch.")
        st.write(f"Expected: {expected_feature_dimension}")
        st.write(f"Actual: {feature_output_shape}")
        st.stop()

    if getattr(scaler, "n_features_in_", 224) != 224:
        st.error("RobustScaler expects a different feature dimension.")
        st.write(f"Scaler feature count: {scaler.n_features_in_}")
        st.stop()

    if getattr(svm, "n_features_in_", 224) != 224:
        st.error("SVM expects a different feature dimension.")
        st.write(f"SVM feature count: {svm.n_features_in_}")
        st.stop()

except Exception as e:
    st.error("Unable to verify GSFF feature dimension.")
    st.exception(e)
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "pdf_report" not in st.session_state:
    st.session_state.pdf_report = None

if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2.5rem;
        max-width: 100%;
    }

    h1, h2, h3 {
        color: var(--text-color);
    }

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 750;
        line-height: 1.25;
        margin-top: 0px;
        margin-bottom: 10px;
        padding-top: 4px;
        color: var(--text-color);
    }

    .brand-name {
        text-align: center;
        font-size: 17px;
        font-weight: 650;
        line-height: 1.4;
        margin-top: 2px;
        margin-bottom: 10px;
        color: var(--text-color);
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        line-height: 1.6;
        margin-top: 8px;
        margin-bottom: 10px;
        color: var(--text-color);
    }

    .header-divider {
        border: none;
        border-top: 1px solid rgba(128, 128, 128, 0.35);
        margin-top: 20px;
        margin-bottom: 30px;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        min-height: 42px;
        border: 0;
        border-radius: 8px;
        background-color: #ff4b4b;
        color: white;
        font-weight: 600;
    }

    div[data-testid="stButton"] > button:hover {
        background-color: #f13f3f;
        color: white;
        border: 0;
    }

    div[data-testid="stButton"] > button:focus {
        color: white;
        border: 0;
        box-shadow: none;
    }

    .modality-text {
        font-size: 26px;
        font-weight: 650;
        margin-top: 4px;
        margin-bottom: 22px;
        color: var(--text-color);
    }

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
        background-color: #e4f6e9;
        color: #008b3d;
    }

    .final-benign {
        background-color: #fff4d6;
        color: #9a6700;
    }

    .final-malignant {
        background-color: #fde8e8;
        color: #c62828;
    }

    .report-divider {
        border: none;
        border-top: 1px solid rgba(128, 128, 128, 0.35);
        margin-top: 30px;
        margin-bottom: 36px;
    }

    .research-note {
        margin-top: 24px;
        color: var(--text-color);
        opacity: 0.65;
        font-size: 13px;
        line-height: 1.5;
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
# COLOUR IMAGE CHECK
# ============================================================

def check_color_image(image):

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    rg_difference = np.mean(np.abs(red - green))
    gb_difference = np.mean(np.abs(green - blue))
    rb_difference = np.mean(np.abs(red - blue))

    average_difference = (
        rg_difference + gb_difference + rb_difference
    ) / 3.0

    is_color = average_difference > COLOR_TOLERANCE

    return is_color, float(average_difference)


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_image(image):

    width, height = image.size

    if width < 64 or height < 64:
        return False, "Image resolution is too small."

    array = np.asarray(image)

    if array.size == 0:
        return False, "Image is empty."

    is_color, _ = check_color_image(image)

    if is_color:
        return (
            False,
            "Color image detected. Please input a lung CT image.",
        )

    gray = np.asarray(
        image.convert("L"),
        dtype=np.float32,
    )

    if np.std(gray) < 8:
        return False, "Image appears blank or invalid."

    dark_ratio = np.mean(gray < 10)

    if dark_ratio > 0.98:
        return False, "Image is almost completely black."

    bright_ratio = np.mean(gray > 245)

    if bright_ratio > 0.98:
        return False, "Image is almost completely white."

    return True, "Image passed basic validation."


# ============================================================
# PREPROCESS FOR CT VERIFIER
# ============================================================

def preprocess_for_ct_verifier(image):

    image = image.convert("RGB")
    image = image.resize(
        CT_VERIFIER_IMAGE_SIZE,
        Image.Resampling.LANCZOS,
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array /= 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# PREPROCESS FOR GSFF
# ============================================================

def preprocess_for_gsff(image):

    image = image.convert("RGB")
    image = image.resize(
        FEATURE_EXTRACTOR_IMAGE_SIZE,
        Image.Resampling.LANCZOS,
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    return image_array


# ============================================================
# MODALITY VERIFICATION
# ============================================================

def verify_modality(image):

    image_array = preprocess_for_ct_verifier(image)

    predictions = ct_verifier.predict(
        image_array,
        verbose=0,
    )

    probabilities = np.asarray(
        predictions[0],
        dtype=np.float32,
    )

    probability_sum = np.sum(probabilities)

    if probability_sum > 0:
        probabilities = probabilities / probability_sum

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    modality = modality_names[predicted_class]

    return modality, confidence, probabilities


# ============================================================
# GSFF FEATURE EXTRACTION
# ============================================================

def extract_gsff_features(image):

    image_array = preprocess_for_gsff(image)

    features = feature_extractor.predict(
        image_array,
        verbose=0,
    )

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    return features


# ============================================================
# LUNG CANCER PREDICTION
# ============================================================

def predict_lung_cancer(image):

    features = extract_gsff_features(image)
    scaled_features = scaler.transform(features)

    prediction = svm.predict(scaled_features)
    predicted_index = int(prediction[0])

    probabilities = svm.predict_proba(scaled_features)[0]

    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, probabilities


# ============================================================
# FRIENDLY MODALITY NAME
# ============================================================

def friendly_modality_name(modality):

    if modality == "CHEST_XRAY":
        return "X-ray"

    if modality == "CT":
        return "CT"

    if modality == "MRI":
        return "MRI"

    return modality


# ============================================================
# PDF REPORT
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

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        spaceAfter=2,
        textColor=colors.black,
    )

    system_title_style = ParagraphStyle(
        "SystemTitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        spaceAfter=10,
        textColor=colors.black,
    )

    report_title_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        spaceAfter=12,
        textColor=colors.black,
    )

    date_style = ParagraphStyle(
        "ReportDate",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        spaceAfter=10,
        textColor=colors.black,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=0,
        spaceAfter=7,
        textColor=colors.black,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        spaceBefore=10,
        textColor=colors.black,
    )

    story = [
        Paragraph("PulmoVision", title_style),
        Paragraph(
            "AI-Powered Lung Cancer Detection System",
            system_title_style,
        ),
        Paragraph(
            "Medical Image Analysis Report",
            report_title_style,
        ),
    ]

    report_time = datetime.now().strftime(
        "%d %B %Y, %I:%M:%S %p"
    )

    story.append(
        Paragraph(
            f"<b>Analysis Date:</b> {report_time}",
            date_style,
        )
    )

    image_buffer = io.BytesIO()

    image.convert("RGB").save(
        image_buffer,
        format="PNG",
    )

    image_buffer.seek(0)

    image_width, image_height = image.size

    max_image_width = 100 * mm
    max_image_height = 92 * mm

    image_scale = min(
        max_image_width / image_width,
        max_image_height / image_height,
    )

    report_image = RLImage(
        image_buffer,
        width=image_width * image_scale,
        height=image_height * image_scale,
    )

    report_image.hAlign = "CENTER"

    story.append(report_image)
    story.append(Spacer(1, 10))

    def apply_report_table_style(table):

        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )

    # 1. Medical image modality
    story.append(
        Paragraph("1. Medical Image Modality", heading_style)
    )

    modality_data = [
        ["Parameter", "Result"],
        ["Detected Modality", friendly_modality_name(modality)],
        ["Modality Confidence", f"{modality_confidence * 100:.2f}%"],
    ]

    modality_table = Table(
        modality_data,
        colWidths=[84 * mm, 84 * mm],
        hAlign="CENTER",
    )

    apply_report_table_style(modality_table)
    story.append(modality_table)
    story.append(Spacer(1, 10))

    # 2. CT verification
    story.append(
        Paragraph("2. CT Verification", heading_style)
    )

    ct_verification_data = [
        ["Parameter", "Result"],
        ["CT Verification", "CT"],
        ["CT Probability", f"{modality_probabilities[1] * 100:.2f}%"],
    ]

    ct_verification_table = Table(
        ct_verification_data,
        colWidths=[84 * mm, 84 * mm],
        hAlign="CENTER",
    )

    apply_report_table_style(ct_verification_table)
    story.append(ct_verification_table)
    story.append(Spacer(1, 10))

    # 3. Lung cancer detection
    story.append(
        Paragraph("3. Lung Cancer Detection", heading_style)
    )

    result_data = [
        ["Parameter", "Result"],
        ["Predicted Class", predicted_class],
        ["Prediction Confidence", f"{prediction_confidence * 100:.2f}%"],
    ]

    result_table = Table(
        result_data,
        colWidths=[84 * mm, 84 * mm],
        hAlign="CENTER",
    )

    apply_report_table_style(result_table)
    story.append(result_table)

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This application is a research "
            "prototype and is not intended to provide clinical "
            "diagnosis or replace professional medical evaluation.",
            disclaimer_style,
        )
    )

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# FINAL RESULT DISPLAY
# ============================================================

def render_final_result(predicted_class):

    if predicted_class == "Normal":
        css_class = "final-normal"
    elif predicted_class == "Benign":
        css_class = "final-benign"
    else:
        css_class = "final-malignant"

    st.markdown(
        f"""
        <div class="final-result {css_class}">
            Final Result: {predicted_class}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RESET ANALYSIS STATE
# ============================================================

def reset_analysis_state():

    st.session_state.analysis_result = None
    st.session_state.pdf_report = None


# ============================================================
# PULMOVISION HEADER
# ============================================================

header_left, header_center, header_right = st.columns(
    [1.20, 8.60, 1.20],
    vertical_alignment="center",
)

with header_left:

    if LUNG_ICON_PATH.exists():
        st.image(str(LUNG_ICON_PATH), width=105)
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

with header_right:
    st.write("")

st.markdown(
    '<hr class="header-divider">',
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD MEDICAL IMAGE
# ============================================================

st.header("Upload Medical Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    uploaded_bytes = uploaded_file.getvalue()

    file_signature = hashlib.sha256(
        uploaded_bytes
    ).hexdigest()

    if st.session_state.uploaded_signature != file_signature:

        st.session_state.uploaded_signature = file_signature
        reset_analysis_state()

    try:
        image = Image.open(io.BytesIO(uploaded_bytes))
        image.load()
    except Exception as e:
        st.error("Unable to read the uploaded image.")
        st.exception(e)
        st.stop()

    st.header("Uploaded Medical Image")

    image_left, image_center, image_right = st.columns([1, 2, 1])

    with image_center:
        st.image(
            image,
            caption="Uploaded Medical Image",
            use_container_width=True,
        )

    analyze_clicked = st.button(
        "Check Your Image By Initiating AI Analysis",
        use_container_width=True,
    )

    if analyze_clicked:

        reset_analysis_state()

        # STEP 1: Basic validation
        is_valid, validation_message = validate_image(image)

        if not is_valid:
            st.error(validation_message)
            st.stop()

        st.success(validation_message)

        # STEP 2: Modality classification
        try:
            with st.spinner("Detecting medical image type..."):
                (
                    modality,
                    modality_confidence,
                    modality_probabilities,
                ) = verify_modality(image)

        except Exception as e:
            st.error("Medical image type detection failed.")
            st.exception(e)
            st.stop()

        st.header("Detected Medical Image Type")

        st.markdown(
            f"""
            <div class="modality-text">
                {friendly_modality_name(modality)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.header("CT Verification")

        # CT accepted
        if (
            modality == "CT"
            and modality_confidence >= MODALITY_THRESHOLD
        ):
            st.success("Chest CT verified successfully.")

        # X-ray rejected
        elif modality == "CHEST_XRAY":
            st.error("Chest X-ray detected.")
            st.warning(
                "Lung cancer classification was not performed "
                "because this pipeline is designed for CT images."
            )
            st.stop()

        # MRI rejected
        elif modality == "MRI":
            st.error("MRI image detected.")
            st.warning(
                "Lung cancer classification was not performed "
                "because this pipeline is designed for CT images."
            )
            st.stop()

        # Low confidence
        else:
            st.error(
                "The image could not be verified as a CT scan "
                "with sufficient confidence."
            )
            st.warning("Please upload a clear lung CT image.")
            st.stop()

        # STEP 3: Lung cancer classification
        try:
            with st.spinner("Running lung cancer analysis..."):
                (
                    predicted_class,
                    prediction_confidence,
                    class_probabilities,
                ) = predict_lung_cancer(image)

        except Exception as e:
            st.error("Lung cancer prediction failed.")
            st.exception(e)
            st.stop()

        # STEP 4: Create PDF
        try:
            pdf_bytes = create_pdf_report(
                image=image,
                modality=modality,
                modality_confidence=modality_confidence,
                modality_probabilities=modality_probabilities,
                predicted_class=predicted_class,
                prediction_confidence=prediction_confidence,
                class_probabilities=class_probabilities,
            )

        except Exception as e:
            pdf_bytes = None
            st.warning("PDF report could not be generated.")
            st.exception(e)

        # Save result
        st.session_state.analysis_result = {
            "modality": modality,
            "modality_confidence": modality_confidence,
            "modality_probabilities": modality_probabilities,
            "predicted_class": predicted_class,
            "prediction_confidence": prediction_confidence,
            "class_probabilities": class_probabilities,
        }

        st.session_state.pdf_report = pdf_bytes

    # ========================================================
    # KEEP RESULTS VISIBLE AFTER RERUN
    # ========================================================

    result = st.session_state.analysis_result

    if result is not None:

        if not analyze_clicked:

            st.success("Image passed basic validation.")

            st.header("Detected Medical Image Type")

            st.markdown(
                f"""
                <div class="modality-text">
                    {friendly_modality_name(result["modality"])}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.header("CT Verification")
            st.success("Chest CT verified successfully.")

        st.header("Lung Cancer Detection")

        render_final_result(result["predicted_class"])

        st.markdown(
            '<hr class="report-divider">',
            unsafe_allow_html=True,
        )

        st.header("Analysis Report")

        if st.session_state.pdf_report is not None:

            st.download_button(
                label="Download Final Report (PDF)",
                data=st.session_state.pdf_report,
                file_name="PulmoVision_Lung_Cancer_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        else:
            st.warning("The PDF report is currently unavailable.")

        st.markdown(
            """
            <div class="research-note">
                Research prototype for educational and research
                use only. It is not a clinical diagnosis and
                does not replace professional medical evaluation.
            </div>
            """,
            unsafe_allow_html=True,
        )
