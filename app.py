# ============================================================
# PulmoVision - AI-Powered Lung Cancer Detection System
# ============================================================
# Pipeline:
# Image -> Basic validation -> CT/X-ray/MRI verifier
# -> EfficientNetB0 Block5c GSFF extractor
# -> RobustScaler -> SVM -> Output -> PDF report
#
# Required files beside app.py:
#   CT_Verifier.keras
#   GSFF_Feature_Extractor.keras
#   RobustScaler.pkl
#   SVM_Classifier.pkl
#   Lung_Icon.png (optional)
#
# IMPORTANT:
# Verify the SVM label encoding against your training notebook.
# A modality verifier is NOT an out-of-distribution detector.
# This research application is not a clinical diagnostic tool.
# ============================================================

import io
import hashlib
from pathlib import Path
from datetime import datetime

import joblib
import keras
import numpy as np
import streamlit as st

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
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PulmoVision",
    page_icon="🫁",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

LUNG_ICON_PATH = BASE_DIR / "Lung_Icon.png"
CT_VERIFIER_PATH = BASE_DIR / "CT_Verifier.keras"
FEATURE_EXTRACTOR_PATH = BASE_DIR / "GSFF_Feature_Extractor.keras"
SCALER_PATH = BASE_DIR / "RobustScaler.pkl"
SVM_PATH = BASE_DIR / "SVM_Classifier.pkl"


# ============================================================
# 2. SETTINGS
# ============================================================

CT_VERIFIER_IMAGE_SIZE = (128, 128)
FEATURE_EXTRACTOR_IMAGE_SIZE = (224, 224)

MODALITY_THRESHOLD = 0.60
COLOR_TOLERANCE = 8.0

# Review threshold only; not clinically validated.
# A score above this threshold can still be incorrect.
CLASSIFICATION_MIN_SCORE = 0.60

# Verify these mappings against your original training code.
SVM_LABEL_TO_NAME = {
    0: "Normal",
    1: "Benign",
    2: "Malignant",
    "Normal": "Normal",
    "Benign": "Benign",
    "Malignant": "Malignant",
}

CLASS_NAMES = ["Normal", "Benign", "Malignant"]

# Verify this order against the CT verifier training labels.
MODALITY_NAMES = ["CHEST_XRAY", "CT", "MRI"]


# ============================================================
# 3. CUSTOM GSDP LAYER
# ============================================================

@keras.saving.register_keras_serializable(package="GSFF")
class GSDP(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return keras.ops.std(inputs, axis=(1, 2))

    def get_config(self):
        return super().get_config()


# ============================================================
# 4. MODEL FILE CHECKS
# ============================================================

def validate_model_file(path, model_name):
    if not path.exists():
        st.error(f"{model_name} was not found.")
        st.code(str(path))
        st.stop()

    if not path.is_file() or path.stat().st_size == 0:
        st.error(f"{model_name} is missing or empty.")
        st.stop()


REQUIRED_FILES = {
    "CT Verifier": CT_VERIFIER_PATH,
    "GSFF Feature Extractor": FEATURE_EXTRACTOR_PATH,
    "RobustScaler": SCALER_PATH,
    "SVM Classifier": SVM_PATH,
}

for model_name, model_path in REQUIRED_FILES.items():
    validate_model_file(model_path, model_name)


# ============================================================
# 5. LOAD MODELS
# ============================================================

@st.cache_resource
def load_ct_verifier():
    return keras.models.load_model(
        CT_VERIFIER_PATH,
        compile=False,
    )


@st.cache_resource
def load_feature_extractor():
    return keras.models.load_model(
        FEATURE_EXTRACTOR_PATH,
        custom_objects={"GSDP": GSDP},
        compile=False,
        safe_mode=False,
    )


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_svm():
    return joblib.load(SVM_PATH)


def load_all_models():
    try:
        ct_model = load_ct_verifier()
        feature_model = load_feature_extractor()
        loaded_scaler = load_scaler()
        loaded_svm = load_svm()

        if ct_model.output_shape[-1] != 3:
            raise ValueError(
                f"Expected 3 modality outputs, got "
                f"{ct_model.output_shape}."
            )

        if feature_model.output_shape[-1] != 224:
            raise ValueError(
                f"Expected 224 GSFF features, got "
                f"{feature_model.output_shape}."
            )

        scaler_dim = getattr(
            loaded_scaler, "n_features_in_", None
        )
        if scaler_dim is not None and scaler_dim != 224:
            raise ValueError(
                f"Scaler expects {scaler_dim} features, not 224."
            )

        svm_dim = getattr(
            loaded_svm, "n_features_in_", None
        )
        if svm_dim is not None and svm_dim != 224:
            raise ValueError(
                f"SVM expects {svm_dim} features, not 224."
            )

        if not hasattr(loaded_svm, "classes_"):
            raise ValueError("SVM has no classes_ attribute.")

        if len(loaded_svm.classes_) != 3:
            raise ValueError(
                f"Expected a three-class SVM. "
                f"Found classes: {loaded_svm.classes_}"
            )

        return ct_model, feature_model, loaded_scaler, loaded_svm

    except Exception as exc:
        st.error("Could not load or validate the model artifacts.")
        st.exception(exc)
        st.stop()


ct_verifier, feature_extractor, scaler, svm = load_all_models()


# ============================================================
# 6. SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "pdf_report" not in st.session_state:
    st.session_state.pdf_report = None

if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None


def reset_analysis_state():
    st.session_state.analysis_result = None
    st.session_state.pdf_report = None


# ============================================================
# 7. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem;
        max-width: 100%;
    }

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 750;
        line-height: 1.25;
        margin-bottom: 8px;
        color: var(--text-color);
    }

    .brand-name {
        text-align: center;
        font-size: 17px;
        font-weight: 650;
        margin-bottom: 10px;
        color: var(--text-color);
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 10px;
        color: var(--text-color);
    }

    .header-divider, .report-divider {
        border: none;
        border-top: 1px solid rgba(128,128,128,0.35);
        margin-top: 20px;
        margin-bottom: 28px;
    }

    .modality-text {
        font-size: 26px;
        font-weight: 650;
        margin-bottom: 22px;
        color: var(--text-color);
    }

    .final-result {
        width: 100%;
        box-sizing: border-box;
        border-radius: 8px;
        padding: 24px 16px;
        margin: 8px 0 24px;
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

    .research-note {
        margin-top: 24px;
        color: var(--text-color);
        opacity: 0.75;
        font-size: 13px;
        line-height: 1.5;
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
# 8. IMAGE UTILITIES AND VALIDATION
# ============================================================

def image_to_rgb(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image.")
    return image.convert("RGB")


def get_image_hash(image):
    image = image_to_rgb(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def check_color_image(image):
    rgb = np.asarray(
        image_to_rgb(image),
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

    return (
        average_difference > COLOR_TOLERANCE,
        float(average_difference),
    )


def validate_image(image):
    if image is None:
        return False, "No image was provided."

    try:
        image = image_to_rgb(image)
    except Exception:
        return False, "The uploaded file is not a readable image."

    width, height = image.size

    if width < 64 or height < 64:
        return False, "Image resolution is too small."

    array = np.asarray(image, dtype=np.float32)

    if array.size == 0 or not np.isfinite(array).all():
        return False, "The image is empty or contains invalid values."

    is_color, _ = check_color_image(image)

    if is_color:
        return (
            False,
            "A colour image was detected. Please check that you "
            "uploaded the intended lung CT image.",
        )

    gray = np.asarray(image.convert("L"), dtype=np.float32)

    if float(np.std(gray)) < 8.0:
        return False, "The image appears blank or nearly uniform."

    if float(np.mean(gray < 10)) > 0.98:
        return False, "The image is almost completely black."

    if float(np.mean(gray > 245)) > 0.98:
        return False, "The image is almost completely white."

    return True, "Image passed basic technical validation."


# ============================================================
# 9. IMAGE PREPROCESSING
# ============================================================

def preprocess_for_ct_verifier(image):
    image = image_to_rgb(image)
    image = image.resize(
        CT_VERIFIER_IMAGE_SIZE,
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(image, dtype=np.float32)
    array /= 255.0
    return np.expand_dims(array, axis=0)


def preprocess_for_gsff(image):
    image = image_to_rgb(image)
    image = image.resize(
        FEATURE_EXTRACTOR_IMAGE_SIZE,
        Image.Resampling.LANCZOS,
    )

    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    return preprocess_input(array)


# ============================================================
# 10. CT / X-RAY / MRI VERIFICATION
# ============================================================

def verify_modality(image):
    image_array = preprocess_for_ct_verifier(image)

    predictions = ct_verifier.predict(
        image_array,
        verbose=0,
    )

    probabilities = np.asarray(
        predictions[0],
        dtype=np.float64,
    )

    if probabilities.size != len(MODALITY_NAMES):
        raise ValueError(
            "The modality model returned an unexpected number of outputs."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("The modality model returned invalid scores.")

    total = probabilities.sum()

    if total <= 0:
        raise ValueError("The modality model returned zero total probability.")

    probabilities = probabilities / total

    predicted_index = int(np.argmax(probabilities))
    modality = MODALITY_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    return modality, confidence, probabilities


def friendly_modality_name(modality):
    names = {
        "CHEST_XRAY": "X-ray",
        "CT": "CT",
        "MRI": "MRI",
    }
    return names.get(modality, modality)


# ============================================================
# 11. GSFF FEATURE EXTRACTION
# ============================================================

def extract_gsff_features(image):
    image_array = preprocess_for_gsff(image)

    features = feature_extractor.predict(
        image_array,
        verbose=0,
    )

    features = np.asarray(features, dtype=np.float32)

    if features.shape != (1, 224):
        raise ValueError(
            f"Expected GSFF features with shape (1, 224); "
            f"received {features.shape}."
        )

    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced invalid values.")

    return features


# ============================================================
# 12. LUNG CANCER CLASSIFICATION
# ============================================================

def predict_lung_cancer(image):
    """
    Return:
        predicted_class
        prediction_score
        probabilities ordered as [Normal, Benign, Malignant]

    The label mapping must match the original training labels.
    A low-score prediction is marked Needs review.
    This is not a full out-of-distribution detector.
    """

    features = extract_gsff_features(image)
    scaled_features = scaler.transform(features)

    svm_classes = list(svm.classes_)

    label_to_name = {}

    for label in svm_classes:
        key = label.item() if isinstance(label, np.generic) else label

        if key not in SVM_LABEL_TO_NAME:
            raise ValueError(
                f"Unknown SVM label {key!r}. Update SVM_LABEL_TO_NAME "
                "to match the training labels."
            )

        label_to_name[label] = SVM_LABEL_TO_NAME[key]

    mapped_names = list(label_to_name.values())

    if (
        len(mapped_names) != 3
        or set(mapped_names) != set(CLASS_NAMES)
    ):
        raise ValueError(
            "The SVM class mapping must represent Normal, Benign, "
            "and Malignant exactly once each."
        )

    if not hasattr(svm, "predict_proba"):
        raise ValueError(
            "SVM has no predict_proba method. Check that the trained "
            "SVC was configured with probability=True."
        )

    prediction_label = svm.predict(scaled_features)[0]

    raw_probabilities = np.asarray(
        svm.predict_proba(scaled_features)[0],
        dtype=float,
    )

    if raw_probabilities.size != len(svm_classes):
        raise ValueError(
            "The SVM probabilities do not align with svm.classes_."
        )

    # Explicitly associate each probability with its semantic class.
    probability_by_name = {
        label_to_name[label]: float(raw_probabilities[index])
        for index, label in enumerate(svm_classes)
    }

    # Fixed display order for UI and report.
    probabilities = np.asarray(
        [probability_by_name[name] for name in CLASS_NAMES],
        dtype=float,
    )

    prediction_key = (
        prediction_label.item()
        if isinstance(prediction_label, np.generic)
        else prediction_label
    )

    if prediction_key not in SVM_LABEL_TO_NAME:
        raise ValueError(
            f"Predicted label {prediction_key!r} has no class mapping."
        )

    predicted_class = SVM_LABEL_TO_NAME[prediction_key]
    prediction_score = probability_by_name[predicted_class]

    # Low score -> abstain from making a class prediction.
    if prediction_score < CLASSIFICATION_MIN_SCORE:
        predicted_class = "Needs review"

    return predicted_class, float(prediction_score), probabilities


# ============================================================
# 13. PDF REPORT GENERATION
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
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PVTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "PVSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=11,
        leading=15,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "PVHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=8,
        spaceAfter=6,
    )

    disclaimer_style = ParagraphStyle(
        "PVDisclaimer",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        spaceBefore=12,
    )

    story = [
        Paragraph("PulmoVision", title_style),
        Paragraph(
            "AI-Powered Lung Cancer Detection System",
            subtitle_style,
        ),
        Paragraph(
            "Medical Image Analysis Report",
            subtitle_style,
        ),
        Paragraph(
            f"<b>Analysis date:</b> "
            f"{datetime.now().strftime('%d %B %Y, %I:%M:%S %p')}",
            styles["Normal"],
        ),
        Spacer(1, 8),
    ]

    # Embed uploaded image in the PDF.
    image_buffer = io.BytesIO()
    image_to_rgb(image).save(image_buffer, format="PNG")
    image_buffer.seek(0)

    image_width, image_height = image.size
    max_width = 100 * mm
    max_height = 85 * mm

    scale = min(
        max_width / image_width,
        max_height / image_height,
    )

    report_image = RLImage(
        image_buffer,
        width=image_width * scale,
        height=image_height * scale,
    )

    report_image.hAlign = "CENTER"
    story.extend([report_image, Spacer(1, 10)])

    def make_table(data):
        table = Table(
            data,
            colWidths=[82 * mm, 86 * mm],
            hAlign="CENTER",
        )

        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])
        )

        return table

    story.append(
        Paragraph("1. Medical Image Modality", heading_style)
    )

    story.append(
        make_table([
            ["Parameter", "Result"],
            ["Detected modality", friendly_modality_name(modality)],
            ["Top modality score", f"{modality_confidence * 100:.2f}%"],
        ])
    )

    story.append(
        Paragraph("2. CT Verification", heading_style)
    )

    ct_score = (
        float(modality_probabilities[1]) * 100
        if len(modality_probabilities) > 1
        else 0.0
    )

    story.append(
        make_table([
            ["Parameter", "Result"],
            ["CT class score", f"{ct_score:.2f}%"],
            ["Modality verification", "Passed the app's modality threshold"],
        ])
    )

    story.append(
        Paragraph("3. Lung Cancer Classification", heading_style)
    )

    story.append(
        make_table([
            ["Parameter", "Result"],
            ["Model output (not a diagnosis)", predicted_class],
            [
                "Top SVM class score (not clinical confidence)",
                f"{prediction_confidence * 100:.2f}%",
            ],
            ["Normal class score", f"{class_probabilities[0] * 100:.2f}%"],
            ["Benign class score", f"{class_probabilities[1] * 100:.2f}%"],
            [
                "Malignant class score",
                f"{class_probabilities[2] * 100:.2f}%",
            ],
        ])
    )

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This application is a research prototype "
            "for educational and research use. Its output is not a clinical "
            "diagnosis, and its scores are not validated measures of clinical "
            "certainty. The model may misclassify images from different "
            "datasets, scanners, hospitals, or image-processing pipelines. "
            "Consult a qualified medical professional for clinical evaluation.",
            disclaimer_style,
        )
    )

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# 14. RESULT DISPLAY
# ============================================================

def render_final_result(predicted_class):
    css_classes = {
        "Normal": "final-normal",
        "Benign": "final-benign",
        "Malignant": "final-malignant",
    }

    if predicted_class not in css_classes:
        st.warning(
            "Needs review: the model did not produce a sufficiently "
            "reliable classification score. This is not evidence that "
            "the scan is normal or malignant. Please seek qualified "
            "clinical review."
        )
        return

    css_class = css_classes[predicted_class]

    st.markdown(
        f"""
        <div class="final-result {css_class}">
            Model Output: {predicted_class}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 15. HEADER
# ============================================================

header_left, header_center, header_right = st.columns(
    [1.2, 8.6, 1.2],
    vertical_alignment="center",
)

with header_left:
    if LUNG_ICON_PATH.exists():
        st.image(str(LUNG_ICON_PATH), width=100)
    else:
        st.markdown(
            "<div style='font-size:70px;text-align:center'>🫁</div>",
            unsafe_allow_html=True,
        )

with header_center:
    st.markdown(
        """
        <div class="main-title">PulmoVision</div>
        <div class="brand-name">
            AI-Powered Lung Cancer Detection System
        </div>
        <div class="subtitle">
            Intelligent analysis of lung CT images using deep feature
            extraction and machine learning.
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
# 16. IMAGE UPLOAD
# ============================================================

st.header("Upload Medical Image")

uploaded_file = st.file_uploader(
    "Choose a lung CT image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.markdown(
        """
        <div class="research-note">
            Supported inputs are individual image files. This application
            does not load or analyse a complete DICOM CT series.
            An image being classified as CT does not guarantee that it
            is suitable for this model.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# 17. READ UPLOADED IMAGE
# ============================================================

uploaded_bytes = uploaded_file.getvalue()

file_signature = hashlib.sha256(uploaded_bytes).hexdigest()

if st.session_state.uploaded_signature != file_signature:
    st.session_state.uploaded_signature = file_signature
    reset_analysis_state()

try:
    image = Image.open(io.BytesIO(uploaded_bytes))
    image.load()
    image = image.convert("RGB")

except Exception as exc:
    st.error("Unable to read the uploaded image.")
    st.exception(exc)
    st.stop()


st.header("Uploaded Medical Image")

image_left, image_center, image_right = st.columns([1, 2, 1])

with image_center:
    st.image(
        image,
        caption="Uploaded Medical Image",
        use_container_width=True,
    )


# ============================================================
# 18. RUN ANALYSIS
# ============================================================

analyze_clicked = st.button(
    "Check Your Image By Initiating AI Analysis",
    use_container_width=True,
)

if analyze_clicked:
    reset_analysis_state()

    # --------------------------------------------------------
    # STEP 1: BASIC IMAGE VALIDATION
    # --------------------------------------------------------

    is_valid, validation_message = validate_image(image)

    if not is_valid:
        st.error(validation_message)
        st.stop()

    st.success(validation_message)

    # --------------------------------------------------------
    # STEP 2: MODALITY CLASSIFICATION
    # --------------------------------------------------------

    try:
        with st.spinner("Detecting medical image type..."):
            (
                modality,
                modality_confidence,
                modality_probabilities,
            ) = verify_modality(image)

    except Exception as exc:
        st.error("Medical image type detection failed.")
        st.exception(exc)
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

    # --------------------------------------------------------
    # STEP 3: CT VERIFICATION
    # --------------------------------------------------------

    st.header("CT Verification")

    if modality != "CT":
        st.error(
            f"{friendly_modality_name(modality)} detected. "
            "The lung cancer classification stage was not performed "
            "because this pipeline is designed for CT images."
        )
        st.stop()

    if modality_confidence < MODALITY_THRESHOLD:
        st.error(
            "The modality verifier did not identify CT with sufficient "
            "score. Classification was stopped."
        )
        st.warning(
            "A CT verifier can still accept unsuitable or unfamiliar "
            "images. It does not establish that an image is from the "
            "model's training distribution."
        )
        st.stop()

    st.success("The modality model classified this image as CT.")

    # --------------------------------------------------------
    # STEP 4: CLASSIFICATION
    # --------------------------------------------------------

    try:
        with st.spinner("Running GSFF + SVM analysis..."):
            (
                predicted_class,
                prediction_confidence,
                class_probabilities,
            ) = predict_lung_cancer(image)

    except Exception as exc:
        st.error("Lung cancer prediction failed.")
        st.exception(exc)
        st.stop()

    # --------------------------------------------------------
    # STEP 5: PDF REPORT
    # --------------------------------------------------------

    pdf_bytes = None

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

    except Exception as exc:
        st.warning("The PDF report could not be generated.")
        st.exception(exc)

    # --------------------------------------------------------
    # STEP 6: SAVE RESULT
    # --------------------------------------------------------

    st.session_state.analysis_result = {
        "modality": modality,
        "modality_confidence": modality_confidence,
        "modality_probabilities": modality_probabilities,
        "predicted_class": predicted_class,
        "prediction_confidence": prediction_confidence,
        "class_probabilities": class_probabilities,
    }

    st.session_state.pdf_report = pdf_bytes


# ============================================================
# 19. DISPLAY SAVED RESULTS
# ============================================================

result = st.session_state.analysis_result

if result is not None:
    st.header("Lung Cancer Detection")

    render_final_result(result["predicted_class"])

    st.markdown(
        '<hr class="report-divider">',
        unsafe_allow_html=True,
    )

    st.header("Analysis Report")

    st.write(
        "**Detected modality:**",
        friendly_modality_name(result["modality"]),
    )

    st.write(
        "**Modality model score:**",
        f"{result['modality_confidence'] * 100:.2f}%",
    )

    st.write(
        "**Model output:**",
        result["predicted_class"],
    )

    st.write(
        "**Top SVM class score (not clinical confidence):**",
        f"{result['prediction_confidence'] * 100:.2f}%",
    )

    st.subheader("Class Scores")

    score_col1, score_col2, score_col3 = st.columns(3)

    with score_col1:
        st.metric(
            "Normal",
            f"{result['class_probabilities'][0] * 100:.2f}%",
        )

    with score_col2:
        st.metric(
            "Benign",
            f"{result['class_probabilities'][1] * 100:.2f}%",
        )

    with score_col3:
        st.metric(
            "Malignant",
            f"{result['class_probabilities'][2] * 100:.2f}%",
        )

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
            Research prototype for educational and research use only.
            The model was developed using a particular dataset and may
            misclassify CT images from other websites, scanners, hospitals,
            or image-processing pipelines. A CT modality check does not
            confirm that an image belongs to the model's training
            distribution. SVM class scores are not clinical confidence.
            This application is not a clinical diagnosis and does not
            replace professional medical evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )
