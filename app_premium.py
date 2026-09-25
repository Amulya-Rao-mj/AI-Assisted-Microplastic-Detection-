import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Microplastic Screening",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# YOLO MODEL
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "results",
    "microplastic_detection-2",
    "weights",
    "best.pt"
)


if not os.path.exists(MODEL_PATH):

    st.error(
        "YOLO model not found.\n\n"
        f"Expected location:\n{MODEL_PATH}"
    )

    st.stop()


@st.cache_resource
def load_yolo_model():

    return YOLO(MODEL_PATH)


model = load_yolo_model()


# ============================================================
# IMAGE PREPROCESSING / CLEANING
# ============================================================

def preprocess_image(image_array):
    """
    Explicit image-processing pipeline shown in the UI.

    Steps:
    1. Convert RGB -> grayscale for noise/contrast processing
    2. Gaussian denoising
    3. CLAHE local contrast enhancement
    4. Convert back to RGB for display/model use
    """
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Noise reduction
    denoised = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Local contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced_gray = clahe.apply(denoised)

    # Convert grayscale processing result back to RGB
    enhanced_rgb = cv2.cvtColor(
        enhanced_gray,
        cv2.COLOR_GRAY2RGB
    )

    return gray, denoised, enhanced_rgb


def rgb_to_pil(image_array):
    return Image.fromarray(
        image_array.astype(np.uint8)
    )


# ============================================================
# CLASSIFIER MODEL
# ============================================================

CLASSIFIER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "microplastic_classifier_4class.pth"
)


CLASS_NAMES = [
    "fibre",
    "film",
    "fragment",
    "microbeads"
]


@st.cache_resource
def load_classifier():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    classifier = models.resnet18(
        weights=None
    )

    num_features = classifier.fc.in_features

    classifier.fc = nn.Linear(
        num_features,
        4
    )

    classifier.load_state_dict(
        torch.load(
            CLASSIFIER_PATH,
            map_location=device
        )
    )

    classifier = classifier.to(device)

    classifier.eval()

    return classifier, device


if not os.path.exists(CLASSIFIER_PATH):

    st.error(
        "Microplastic classifier not found.\n\n"
        f"Expected location:\n{CLASSIFIER_PATH}\n\n"
        "Train the classifier first using "
        "train_classifier.py."
    )

    st.stop()


classifier, classifier_device = load_classifier()


# ============================================================
# CLASSIFIER IMAGE TRANSFORM
# ============================================================

classifier_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# CLASSIFY PARTICLE
# ============================================================

def classify_particle(crop):

    if crop is None:
        return "unknown", 0.0

    if crop.size == 0:
        return "unknown", 0.0

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        crop_rgb
    )

    image_tensor = classifier_transform(
        pil_image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        classifier_device
    )


    with torch.no_grad():

        outputs = classifier(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )


    class_index = predicted.item()

    confidence_value = confidence.item()

    class_name = CLASS_NAMES[
        class_index
    ]


    return (
        class_name,
        confidence_value
    )


# ============================================================
# RISK SCORE CALCULATION
# ============================================================

def calculate_risk_score(
    particle_count,
    average_confidence,
    bounding_boxes,
    image_width,
    image_height,
    classifications=None
):

    if particle_count == 0:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        )


    # --------------------------------------------------------
    # 1. Particle Density - 40%
    # --------------------------------------------------------

    image_area = (
        image_width *
        image_height
    )

    density = (
        particle_count /
        image_area *
        1_000_000
    )


    density_score = min(
        (density / 50.0) * 100,
        100
    )


    # --------------------------------------------------------
    # 2. Detection Confidence - 30%
    # --------------------------------------------------------

    confidence_score = (
        average_confidence * 100
    )


    # --------------------------------------------------------
    # 3. Relative Particle Size - 20%
    # --------------------------------------------------------

    relative_sizes = []


    for box in bounding_boxes:

        x1, y1, x2, y2 = box

        width = max(
            x2 - x1,
            0
        )

        height = max(
            y2 - y1,
            0
        )

        particle_area = (
            width *
            height
        )

        relative_area = (
            particle_area /
            image_area
        )

        relative_sizes.append(
            relative_area
        )


    if relative_sizes:

        average_relative_area = np.mean(
            relative_sizes
        )

        size_score = min(
            (
                average_relative_area /
                0.01
            ) * 100,
            100
        )

    else:

        size_score = 0.0


    # --------------------------------------------------------
    # 4. Spatial Clustering - 10%
    # --------------------------------------------------------

    centers = []


    for box in bounding_boxes:

        x1, y1, x2, y2 = box

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2

        centers.append(
            [
                center_x,
                center_y
            ]
        )


    if len(centers) >= 2:

        centers = np.array(
            centers
        )

        center_mean = np.mean(
            centers,
            axis=0
        )

        distances = np.linalg.norm(
            centers -
            center_mean,
            axis=1
        )

        average_distance = np.mean(
            distances
        )

        image_diagonal = np.sqrt(
            image_width ** 2 +
            image_height ** 2
        )

        spread_ratio = (
            average_distance /
            image_diagonal
        )

        clustering_score = max(
            0,
            min(
                (1 - spread_ratio) *
                100,
                100
            )
        )

    else:

        clustering_score = 0.0


    # --------------------------------------------------------
    # Final Risk Score
    # --------------------------------------------------------

    risk_score = (

        0.40 * density_score +

        0.30 * confidence_score +

        0.20 * size_score +

        0.10 * clustering_score

    )


    risk_score = max(
        0,
        min(
            risk_score,
            100
        )
    )


    return (
        risk_score,
        density_score,
        confidence_score,
        size_score,
        clustering_score
    )


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(risk_score):

    if risk_score <= 30:

        return "Low Risk"

    elif risk_score <= 60:

        return "Moderate Risk"

    elif risk_score <= 80:

        return "High Risk"

    else:

        return "Very High Risk"


# ============================================================
# GENERATE HEATMAP
# ============================================================

def generate_heatmap(
    image_array,
    bounding_boxes
):

    if len(image_array.shape) == 2:

        image_array = cv2.cvtColor(
            image_array,
            cv2.COLOR_GRAY2RGB
        )


    height, width = (
        image_array.shape[:2]
    )


    density_map = np.zeros(
        (height, width),
        dtype=np.float32
    )


    for box in bounding_boxes:

        x1, y1, x2, y2 = box

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )


        center_x = max(
            0,
            min(
                center_x,
                width - 1
            )
        )

        center_y = max(
            0,
            min(
                center_y,
                height - 1
            )
        )


        density_map[
            center_y,
            center_x
        ] += 1


    kernel_size = max(
        21,
        int(
            min(
                height,
                width
            ) * 0.08
        )
    )


    if kernel_size % 2 == 0:

        kernel_size += 1


    blurred = cv2.GaussianBlur(
        density_map,
        (
            kernel_size,
            kernel_size
        ),
        0
    )


    if np.max(blurred) > 0:

        normalized = cv2.normalize(
            blurred,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(
            np.uint8
        )

    else:

        normalized = np.zeros_like(
            blurred,
            dtype=np.uint8
        )


    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )


    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    overlay = cv2.addWeighted(
        image_array,
        0.55,
        heatmap,
        0.45,
        0
    )


    return overlay



# ============================================================
# PREMIUM FRONTEND
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #071018;
    --panel: #0d1822;
    --panel-2: #101f2b;
    --line: rgba(148, 163, 184, .16);
    --text: #eef6fb;
    --muted: #91a5b5;
    --accent: #5eead4;
    --accent-2: #38bdf8;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(56,189,248,.10), transparent 28%),
        radial-gradient(circle at 90% 8%, rgba(94,234,212,.08), transparent 24%),
        var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 1380px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -.03em;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 38px 42px;
    border: 1px solid rgba(94,234,212,.18);
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(16,31,43,.96), rgba(7,16,24,.96)),
        radial-gradient(circle at 85% 20%, rgba(56,189,248,.18), transparent 35%);
    box-shadow: 0 24px 80px rgba(0,0,0,.24);
    margin-bottom: 22px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -80px;
    top: -110px;
    border-radius: 50%;
    border: 1px solid rgba(94,234,212,.12);
    box-shadow: 0 0 0 35px rgba(94,234,212,.025), 0 0 0 70px rgba(94,234,212,.018);
}

.brand {
    color: var(--accent);
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.1rem, 5vw, 4.3rem);
    line-height: .98;
    font-weight: 700;
    max-width: 820px;
    margin-bottom: 16px;
}

.hero-title span { color: var(--accent); }

.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    line-height: 1.7;
    max-width: 760px;
}

.pills {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 22px;
}

.pill {
    padding: 8px 12px;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: #c7d5df;
    background: rgba(255,255,255,.025);
    font-size: .78rem;
    font-weight: 600;
}

.section {
    margin: 30px 0 12px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.section .dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 18px rgba(94,234,212,.65);
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.18rem;
    font-weight: 700;
}

.section-note {
    color: var(--muted);
    font-size: .82rem;
    margin-left: auto;
}

.upload-card {
    padding: 28px;
    border: 1px dashed rgba(94,234,212,.35);
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(16,31,43,.86), rgba(13,24,34,.72));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
    margin-bottom: 18px;
}

[data-testid="stFileUploader"] {
    border-radius: 18px !important;
}

[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,.018) !important;
    border: 1px dashed rgba(148,163,184,.25) !important;
    border-radius: 18px !important;
    padding: 18px !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: rgba(94,234,212,.55) !important;
    background: rgba(94,234,212,.035) !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(16,31,43,.94), rgba(13,24,34,.94));
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 18px 18px 16px;
    box-shadow: 0 12px 32px rgba(0,0,0,.16);
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: .72rem !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif;
}

div.stButton > button,
.stDownloadButton > button {
    width: 100%;
    border: 1px solid rgba(94,234,212,.32);
    border-radius: 14px;
    min-height: 48px;
    background: linear-gradient(135deg, rgba(94,234,212,.16), rgba(56,189,248,.12));
    color: var(--text);
    font-weight: 700;
    transition: all .2s ease;
}

div.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(94,234,212,.65);
    box-shadow: 0 10px 28px rgba(94,234,212,.10);
}

[data-testid="stAlert"] {
    border-radius: 16px !important;
    border: 1px solid var(--line) !important;
}

.stProgress > div > div > div > div {
    border-radius: 999px;
}

.image-card {
    padding: 14px;
    border: 1px solid var(--line);
    border-radius: 22px;
    background: rgba(13,24,34,.75);
    overflow: hidden;
}

.image-label {
    color: var(--muted);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .10em;
    text-transform: uppercase;
    margin: 0 0 10px 4px;
}

.risk-card {
    text-align: center;
    padding: 34px 22px;
    border-radius: 24px;
    border: 1px solid var(--line);
    background: linear-gradient(145deg, rgba(16,31,43,.96), rgba(13,24,34,.92));
}

.risk-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.5rem;
    line-height: 1;
    font-weight: 700;
}

.risk-caption {
    color: var(--muted);
    margin-top: 8px;
}

.risk-badge {
    display: inline-block;
    margin-top: 18px;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .06em;
    text-transform: uppercase;
    background: rgba(255,255,255,.07);
}

.class-card {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px 18px;
    background: rgba(16,31,43,.78);
    margin-bottom: 10px;
}

.class-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
}

.class-name {
    font-weight: 700;
    color: var(--text);
}

.class-confidence {
    color: var(--accent);
    font-weight: 700;
}

.class-meta {
    color: var(--muted);
    font-size: .78rem;
    margin-top: 6px;
}

.meter {
    height: 6px;
    border-radius: 999px;
    background: rgba(148,163,184,.12);
    margin-top: 12px;
    overflow: hidden;
}

.meter > div {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
}

.footer-note {
    margin-top: 34px;
    padding: 18px 20px;
    border-top: 1px solid var(--line);
    color: #718595;
    font-size: .78rem;
    line-height: 1.7;
    text-align: center;
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(13,24,34,.55);
}

@media (max-width: 900px) {
    .hero { padding: 28px 24px; border-radius: 22px; }
    .hero-title { font-size: 2.4rem; }
    .section-note { display: none; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <div class="brand">Microplastic AI · Intelligent Screening Platform</div>
    <div class="hero-title">See what is hidden in your <span>sample.</span></div>
    <div class="hero-sub">
        AI-assisted microscopic analysis for particle detection,
        morphology classification, spatial density mapping and
        preliminary screening-risk assessment.
    </div>
    <div class="pills">
        <div class="pill">YOLO Detection</div>
        <div class="pill">ResNet18 Classification</div>
        <div class="pill">Risk Analytics</div>
        <div class="pill">Density Heatmap</div>
        <div class="pill">Analysis Report</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# UPLOAD
# ============================================================

st.markdown("""
<div class="section">
    <div class="dot"></div>
    <div class="section-title">Analyze a microscopic sample</div>
    <div class="section-note">JPG · JPEG · PNG</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="upload-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your microscope image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:

    # --------------------------------------------------------
    # Read Image
    # --------------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
    image_height, image_width = image_array.shape[:2]

    # --------------------------------------------------------
    # IMAGE PREPROCESSING / CLEANING
    # --------------------------------------------------------

    gray_image, denoised_image, enhanced_image = preprocess_image(
        image_array
    )

    st.markdown("""
    <div class="section">
        <div class="dot"></div>
        <div class="section-title">Image processing pipeline</div>
        <div class="section-note">Cleaning and enhancement before AI analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "The uploaded microscopic image is converted to grayscale, "
        "denoised using Gaussian filtering, and locally enhanced using CLAHE. "
        "The processed image is shown here as the image-cleaning stage."
    )

    prep1, prep2, prep3 = st.columns(3)

    with prep1:
        st.markdown(
            '<div class="image-card"><div class="image-label">1 · Grayscale</div>',
            unsafe_allow_html=True
        )
        st.image(
            gray_image,
            use_container_width=True,
            clamp=True
        )
        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with prep2:
        st.markdown(
            '<div class="image-card"><div class="image-label">2 · Noise Reduction</div>',
            unsafe_allow_html=True
        )
        st.image(
            denoised_image,
            use_container_width=True,
            clamp=True
        )
        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with prep3:
        st.markdown(
            '<div class="image-card"><div class="image-label">3 · Contrast Enhancement</div>',
            unsafe_allow_html=True
        )
        st.image(
            enhanced_image,
            use_container_width=True,
            clamp=True
        )
        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section">'
        '<div class="dot"></div>'
        '<div class="section-title">AI analysis pipeline</div>'
        '<div class="section-note">Detection → cropping → morphology classification</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # The YOLO detector is kept on the original RGB image because
    # the detector was trained on the original image distribution.
    # The preprocessing above is an explicit image-processing/cleaning
    # stage shown to the user without changing the trained detector's input.
    with st.spinner("Running AI detection and morphology analysis..."):
        results = model(image_array, conf=0.20)

    result = results[0]
    boxes = result.boxes
    particle_count = len(boxes)

    if particle_count > 0:
        confidences = boxes.conf.cpu().numpy()
        average_confidence = float(np.mean(confidences))
        bounding_boxes = boxes.xyxy.cpu().numpy()
    else:
        confidences = np.array([])
        average_confidence = 0.0
        bounding_boxes = np.empty((0, 4))

    # --------------------------------------------------------
    # Overview metrics
    # --------------------------------------------------------
    # Classification first so the particle number shown on each
    # YOLO box matches the Particle number shown below.
    # --------------------------------------------------------

    st.markdown("""
    <div class="section">
        <div class="dot"></div>
        <div class="section-title">Analysis overview</div>
        <div class="section-note">AI-generated sample summary</div>
    </div>
    """, unsafe_allow_html=True)

    # Classification first so the metrics can use its counts
    classifications = []
    classification_confidences = []

    if particle_count > 0:
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image_width, int(x2))
            y2 = min(image_height, int(y2))

            crop = image_array[y1:y2, x1:x2]
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

            class_name, class_confidence = classify_particle(crop_bgr)
            classifications.append(class_name)
            classification_confidences.append(class_confidence)

    fibre_count = classifications.count("fibre")
    film_count = classifications.count("film")
    fragment_count = classifications.count("fragment")
    microbeads_count = classifications.count("microbeads")

    # --------------------------------------------------------
    # CUSTOM YOLO ANNOTATIONS WITH PARTICLE NUMBERS
    # --------------------------------------------------------
    # The numbering follows the same order as bounding_boxes and
    # classifications, so Particle 1 on the image is the same
    # Particle 1 shown in the classification section/table/report.

    annotated_image = image_array.copy()

    for i, (box, class_name, class_confidence) in enumerate(
        zip(bounding_boxes, classifications, classification_confidences),
        start=1
    ):
        x1, y1, x2, y2 = map(int, box)

        # Keep coordinates inside the image.
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        # Green YOLO-style bounding box.
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Example:
        # Particle 1 | Fragment | 99.4%
        label = (
            f"Particle {i} | "
            f"{class_name.title()} | "
            f"{class_confidence * 100:.1f}%"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        # Put the label above the box whenever possible.
        label_y2 = y1
        label_y1 = y1 - text_height - baseline - 10

        # If there is not enough room above the box, put it inside
        # the top of the box instead.
        if label_y1 < 0:
            label_y1 = y1
            label_y2 = y1 + text_height + baseline + 10
            text_y = y1 + text_height + 2
        else:
            text_y = y1 - baseline - 5

        # Label background.
        cv2.rectangle(
            annotated_image,
            (x1, label_y1),
            (min(image_width - 1, x1 + text_width + 10), label_y2),
            (0, 255, 0),
            -1
        )

        # Label text.
        cv2.putText(
            annotated_image,
            label,
            (x1 + 5, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    (
        risk_score,
        density_score,
        confidence_score,
        size_score,
        clustering_score
    ) = calculate_risk_score(
        particle_count,
        average_confidence,
        bounding_boxes,
        image_width,
        image_height,
        classifications
    )

    risk_level = get_risk_level(risk_score)

    metric_cols = st.columns(7)

    metric_data = [
        ("Detected Particles", particle_count),
        ("Fibre", fibre_count),
        ("Film", film_count),
        ("Fragment", fragment_count),
        ("Microbeads", microbeads_count),
        ("Detection Confidence", f"{average_confidence * 100:.1f}%"),
        ("Risk Score", f"{risk_score:.1f}/100"),
    ]

    for col, (label, value) in zip(metric_cols, metric_data):
        with col:
            st.metric(label, value)

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    st.markdown("""
    <div class="section">
        <div class="dot"></div>
        <div class="section-title">Visual analysis</div>
        <div class="section-note">Original sample vs AI detection</div>
    </div>
    """, unsafe_allow_html=True)

    img_left, img_right = st.columns(2)

    with img_left:
        st.markdown('<div class="image-card"><div class="image-label">Original Sample</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with img_right:
        st.markdown('<div class="image-card"><div class="image-label">AI Detection</div>', unsafe_allow_html=True)
        st.image(annotated_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    st.markdown("""
    <div class="section">
        <div class="dot"></div>
        <div class="section-title">Morphology classification</div>
        <div class="section-note">ResNet18 particle-level predictions</div>
    </div>
    """, unsafe_allow_html=True)

    if particle_count > 0:
        for i, (class_name, class_confidence) in enumerate(
            zip(classifications, classification_confidences), start=1
        ):
            pct = class_confidence * 100
            st.markdown(f"""
            <div class="class-card">
                <div class="class-row">
                    <div class="class-name">Particle {i:02d} · {class_name.title()}</div>
                    <div class="class-confidence">{pct:.1f}%</div>
                </div>
                <div class="class-meta">Classification confidence</div>
                <div class="meter"><div style="width:{max(0,min(pct,100)):.1f}%"></div></div>
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    st.markdown("""
    <div class="section">
        <div class="dot"></div>
        <div class="section-title">Screening risk assessment</div>
        <div class="section-note">Preliminary image-based indicator</div>
    </div>
    """, unsafe_allow_html=True)

    if risk_level == "Low Risk":
        badge = "LOW RISK"
        badge_style = "color:#86efac; border:1px solid rgba(134,239,172,.25);"
    elif risk_level == "Moderate Risk":
        badge = "MODERATE RISK"
        badge_style = "color:#fde68a; border:1px solid rgba(253,230,138,.25);"
    elif risk_level == "High Risk":
        badge = "HIGH RISK"
        badge_style = "color:#fdba74; border:1px solid rgba(253,186,116,.25);"
    else:
        badge = "VERY HIGH RISK"
        badge_style = "color:#fca5a5; border:1px solid rgba(252,165,165,.25);"

    risk_left, risk_right = st.columns([1, 2])

    with risk_left:
        st.markdown(f"""
        <div class="risk-card">
            <div class="risk-number">{risk_score:.1f}</div>
            <div class="risk-caption">out of 100</div>
            <div class="risk-badge" style="{badge_style}">{badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with risk_right:
        factors = [
            ("Particle Density", density_score),
            ("Detection Confidence", confidence_score),
            ("Relative Particle Size", size_score),
            ("Spatial Clustering", clustering_score),
        ]

        for label, score in factors:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;margin:10px 2px 6px;">'
                f'<span style="color:#91a5b5;font-size:.82rem;">{label}</span>'
                f'<span style="color:#eef6fb;font-size:.82rem;font-weight:700;">{score:.1f}</span></div>',
                unsafe_allow_html=True
            )
            st.progress(int(max(0, min(score, 100))))

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    if particle_count > 0:
        st.markdown("""
        <div class="section">
            <div class="dot"></div>
            <div class="section-title">Particle density map</div>
            <div class="section-note">Spatial concentration visualization</div>
        </div>
        """, unsafe_allow_html=True)

        heatmap_image = generate_heatmap(image_array, bounding_boxes)
        # Keep the heatmap compact enough to fit comfortably on a laptop screen.
        # A fixed display width prevents very large source images from creating a
        # long vertical section that requires excessive scrolling.
        heatmap_display_width = min(900, image_width)

        st.image(
            heatmap_image,
            caption="Brighter regions indicate higher concentrations of detected particles.",
            width=heatmap_display_width
        )

    # --------------------------------------------------------
    # Particle inspection
    # --------------------------------------------------------

    if particle_count > 0:
        st.markdown("""
        <div class="section">
            <div class="dot"></div>
            <div class="section-title">Particle inspection</div>
            <div class="section-note">Detailed detection measurements</div>
        </div>
        """, unsafe_allow_html=True)

        detection_data = []

        for i, (box, detection_confidence) in enumerate(
            zip(bounding_boxes, confidences), start=1
        ):
            x1, y1, x2, y2 = box
            width_pixels = x2 - x1
            height_pixels = y2 - y1
            particle_area = width_pixels * height_pixels

            detection_data.append({
                "Particle": f"#{i:02d}",
                "Classification": classifications[i - 1].title(),
                "Class Confidence": f"{classification_confidences[i - 1] * 100:.1f}%",
                "Detection Confidence": f"{float(detection_confidence) * 100:.1f}%",
                "Width (px)": f"{width_pixels:.1f}",
                "Height (px)": f"{height_pixels:.1f}",
                "Area (px²)": f"{particle_area:.1f}",
            })

        st.dataframe(
            detection_data,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning("No potential microplastic particles were detected in this sample.")

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    if particle_count > 0:
        report = ""
        report += "AI-ASSISTED MICROPLASTIC SCREENING REPORT\n"
        report += "=" * 60 + "\n\n"
        report += f"Image: {uploaded_file.name}\n"
        report += f"Image Dimensions: {image_width} x {image_height} pixels\n"
        report += f"Total Particles Detected: {particle_count}\n"
        report += f"Average Detection Confidence: {average_confidence * 100:.1f}%\n\n"
        report += "MORPHOLOGY CLASSIFICATION\n"
        report += "-" * 60 + "\n"
        report += f"Fibre: {fibre_count}\n"
        report += f"Film: {film_count}\n"
        report += f"Fragment: {fragment_count}\n"
        report += f"Microbeads: {microbeads_count}\n\n"
        report += "SCREENING RISK ASSESSMENT\n"
        report += "-" * 60 + "\n"
        report += f"Risk Score: {risk_score:.1f}/100\n"
        report += f"Risk Level: {risk_level}\n\n"
        report += "RISK FACTOR BREAKDOWN\n"
        report += "-" * 60 + "\n"
        report += f"Particle Density Score: {density_score:.1f}/100\n"
        report += f"Detection Confidence Score: {confidence_score:.1f}/100\n"
        report += f"Relative Particle Size Score: {size_score:.1f}/100\n"
        report += f"Spatial Clustering Score: {clustering_score:.1f}/100\n\n"
        report += "PARTICLE CLASSIFICATION DETAILS\n"
        report += "-" * 60 + "\n"

        for i, (
            box,
            detection_confidence
        ) in enumerate(zip(bounding_boxes, confidences), start=1):
            x1, y1, x2, y2 = box
            width_pixels = x2 - x1
            height_pixels = y2 - y1
            particle_area = width_pixels * height_pixels

            report += f"Particle {i}: {classifications[i-1].title()}\n"
            report += f"Classification Confidence: {classification_confidences[i-1] * 100:.1f}%\n"
            report += f"Detection Confidence: {float(detection_confidence) * 100:.1f}%\n"
            report += f"Width: {width_pixels:.1f} pixels\n"
            report += f"Height: {height_pixels:.1f} pixels\n"
            report += f"Bounding Box Area: {particle_area:.1f} pixels²\n\n"

        report += "IMPORTANT NOTE\n"
        report += "-" * 60 + "\n"
        report += (
            "The risk score is a preliminary AI-assisted screening indicator "
            "calculated from image-based features such as detected particle "
            "density, model confidence, relative particle size and spatial "
            "distribution. It is not a measurement of toxicity, human health "
            "risk or environmental risk. Laboratory confirmation using "
            "techniques such as FTIR or Raman spectroscopy is required for "
            "definitive identification and characterization.\n"
        )

        st.markdown("""
        <div class="section">
            <div class="dot"></div>
            <div class="section-title">Analysis report</div>
            <div class="section-note">Export your screening summary</div>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            label="Download Full Analysis Report",
            data=report,
            file_name="microplastic_analysis_report.txt",
            mime="text/plain"
        )

st.markdown("""
<div class="footer-note">
    <strong>Microplastic AI</strong> · AI-assisted microscopic screening platform.<br>
    Results are preliminary image-based screening outputs and should be confirmed
    using appropriate laboratory methods such as FTIR or Raman spectroscopy.
</div>
""", unsafe_allow_html=True)
