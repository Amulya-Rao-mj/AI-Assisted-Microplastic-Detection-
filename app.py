import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
<<<<<<< HEAD
=======
import cv2
import os
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Microplastic Screening",
    page_icon="🔬",
    layout="wide"
)


# --------------------------------------------------
# Load trained YOLO model
# --------------------------------------------------

<<<<<<< HEAD
MODEL_PATH = r"D:\MAJORPROJECT\runs\detect\results\microplastic_detection-2\weights\best.pt"
=======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7

model = YOLO(MODEL_PATH)


# --------------------------------------------------
<<<<<<< HEAD
=======
# Risk Score Calculation
# --------------------------------------------------

def calculate_risk_score(
    particle_count,
    average_confidence,
    bounding_boxes,
    image_width,
    image_height
):
    """
    Calculate a preliminary AI-assisted screening risk score
    between 0 and 100.

    Factors:
    1. Particle density       -> 40%
    2. Detection confidence   -> 30%
    3. Relative particle size -> 20%
    4. Spatial clustering     -> 10%
    """

    if particle_count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # --------------------------------------------------
    # 1. Particle Density Score - 40%
    # --------------------------------------------------

    image_area = image_width * image_height

    density = particle_count / image_area * 1_000_000

    # Normalize density.
    # 50 particles per million pixels is considered
    # a high-density reference point for this screening score.
    density_score = min((density / 50.0) * 100, 100)


    # --------------------------------------------------
    # 2. Confidence Score - 30%
    # --------------------------------------------------

    confidence_score = average_confidence * 100


    # --------------------------------------------------
    # 3. Relative Particle Size Score - 20%
    # --------------------------------------------------

    relative_sizes = []

    for box in bounding_boxes:

        x1, y1, x2, y2 = box

        width = max(x2 - x1, 0)
        height = max(y2 - y1, 0)

        particle_area = width * height

        relative_area = particle_area / image_area

        relative_sizes.append(relative_area)


    if relative_sizes:

        average_relative_area = np.mean(relative_sizes)

        # Reference value for normalization.
        # This represents 1% of the image area.
        size_score = min(
            (average_relative_area / 0.01) * 100,
            100
        )

    else:

        size_score = 0.0


    # --------------------------------------------------
    # 4. Spatial Clustering Score - 10%
    # --------------------------------------------------

    centers = []

    for box in bounding_boxes:

        x1, y1, x2, y2 = box

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        centers.append([center_x, center_y])


    if len(centers) >= 2:

        centers = np.array(centers)

        # Calculate distance of each particle from
        # the average particle location.
        center_mean = np.mean(centers, axis=0)

        distances = np.linalg.norm(
            centers - center_mean,
            axis=1
        )

        average_distance = np.mean(distances)

        image_diagonal = np.sqrt(
            image_width ** 2 +
            image_height ** 2
        )

        # Smaller average distance means particles
        # are more concentrated in one region.
        spread_ratio = average_distance / image_diagonal

        clustering_score = max(
            0,
            min((1 - spread_ratio) * 100, 100)
        )

    else:

        clustering_score = 0.0


    # --------------------------------------------------
    # Final Weighted Risk Score
    # --------------------------------------------------

    risk_score = (
        0.40 * density_score +
        0.30 * confidence_score +
        0.20 * size_score +
        0.10 * clustering_score
    )

    risk_score = max(
        0,
        min(risk_score, 100)
    )


    return (
        risk_score,
        density_score,
        confidence_score,
        size_score,
        clustering_score
    )


# --------------------------------------------------
# Risk Level
# --------------------------------------------------

def get_risk_level(risk_score):

    if risk_score <= 30:
        return "Low Risk"

    elif risk_score <= 60:
        return "Moderate Risk"

    elif risk_score <= 80:
        return "High Risk"

    else:
        return "Very High Risk"


# --------------------------------------------------
# Generate Heatmap
# --------------------------------------------------

def generate_heatmap(image_array, bounding_boxes):

    # Make sure image is RGB
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(
            image_array,
            cv2.COLOR_GRAY2RGB
        )

    height, width = image_array.shape[:2]

    # Create empty density map
    density_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # Add each particle location
    for box in bounding_boxes:

        x1, y1, x2, y2 = box

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Make sure coordinates are inside image
        center_x = max(0, min(center_x, width - 1))
        center_y = max(0, min(center_y, height - 1))

        density_map[center_y, center_x] += 1


    # Smooth the density map
    # This creates the heatmap effect.
    kernel_size = max(
        21,
        int(min(height, width) * 0.08)
    )

    # Kernel size must be odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(
        density_map,
        (kernel_size, kernel_size),
        0
    )


    # Normalize
    if np.max(blurred) > 0:

        normalized = cv2.normalize(
            blurred,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    else:

        normalized = np.zeros_like(
            blurred,
            dtype=np.uint8
        )


    # Apply OpenCV colormap
    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    # Convert BGR -> RGB
    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    # Overlay heatmap on original image
    overlay = cv2.addWeighted(
        image_array,
        0.55,
        heatmap,
        0.45,
        0
    )

    return overlay


# --------------------------------------------------
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
# Title
# --------------------------------------------------

st.title("🔬 AI-Assisted Microplastic Screening System")

st.write(
<<<<<<< HEAD
    "Upload a microscopic image to detect potential microplastic particles."
=======
    "Upload a microscopic image to detect potential "
    "microplastic particles and estimate a preliminary "
    "screening risk score."
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
)

st.info(
    "This system provides preliminary AI-assisted screening "
    "and does not replace laboratory confirmation."
)


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a microscopic image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Process uploaded image
# --------------------------------------------------

if uploaded_file is not None:

<<<<<<< HEAD
    # Read image
    image = Image.open(uploaded_file)
=======
    # --------------------------------------------------
    # Read image
    # --------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    image_array = np.array(image)

    image_height, image_width = image_array.shape[:2]

>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7

    # --------------------------------------------------
    # Original Image
    # --------------------------------------------------

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True
    )


<<<<<<< HEAD
    # Convert image to NumPy
    image_array = np.array(image)


    # --------------------------------------------------
    # Run YOLO detection
    # --------------------------------------------------

    results = model(
        image_array,
        conf=0.50
    )
=======
    # --------------------------------------------------
    # Run YOLO Detection
    # --------------------------------------------------

    with st.spinner("Detecting potential microplastic particles..."):

        results = model(
            image_array,
            conf=0.50
        )
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7

    result = results[0]


    # --------------------------------------------------
<<<<<<< HEAD
    # Detection information
=======
    # Detection Information
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
    # --------------------------------------------------

    boxes = result.boxes

<<<<<<< HEAD
    # Number of detected particles
    particle_count = len(boxes)


    # Confidence values
    if particle_count > 0:

        confidences = boxes.conf.cpu().numpy()
=======
    particle_count = len(boxes)


    # --------------------------------------------------
    # Confidence Values
    # --------------------------------------------------

    if particle_count > 0:

        confidences = (
            boxes.conf
            .cpu()
            .numpy()
        )
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7

        average_confidence = float(
            np.mean(confidences)
        )

    else:

<<<<<<< HEAD
        confidences = []
=======
        confidences = np.array([])
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7

        average_confidence = 0.0


    # --------------------------------------------------
<<<<<<< HEAD
    # Annotated image
=======
    # Bounding Boxes
    # --------------------------------------------------

    if particle_count > 0:

        bounding_boxes = (
            boxes.xyxy
            .cpu()
            .numpy()
        )

    else:

        bounding_boxes = np.empty(
            (0, 4)
        )


    # --------------------------------------------------
    # Annotated Image
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
    # --------------------------------------------------

    annotated_image = result.plot()

    st.subheader("Detected Particles")

    st.image(
        annotated_image,
        use_container_width=True
    )


    # --------------------------------------------------
<<<<<<< HEAD
    # Analysis Results
    # --------------------------------------------------

    st.subheader("Analysis Results")

    col1, col2 = st.columns(2)
=======
    # Risk Score Calculation
    # --------------------------------------------------

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
        image_height
    )


    risk_level = get_risk_level(
        risk_score
    )


    # --------------------------------------------------
    # Analysis Results
    # --------------------------------------------------

    st.subheader("📊 Analysis Results")


    col1, col2, col3 = st.columns(3)
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7


    with col1:

        st.metric(
<<<<<<< HEAD
            "Total Particles Detected",
=======
            "Total Particles",
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
            particle_count
        )


    with col2:

        st.metric(
            "Average Confidence",
            f"{average_confidence * 100:.1f}%"
        )


<<<<<<< HEAD
=======
    with col3:

        st.metric(
            "Risk Score",
            f"{risk_score:.1f}/100"
        )


    # --------------------------------------------------
    # Risk Level
    # --------------------------------------------------

    st.subheader("⚠️ Screening Risk Assessment")


    if risk_level == "Low Risk":

        st.success(
            f"🟢 {risk_level} — "
            f"Screening Score: {risk_score:.1f}/100"
        )

    elif risk_level == "Moderate Risk":

        st.warning(
            f"🟡 {risk_level} — "
            f"Screening Score: {risk_score:.1f}/100"
        )

    elif risk_level == "High Risk":

        st.warning(
            f"🟠 {risk_level} — "
            f"Screening Score: {risk_score:.1f}/100"
        )

    else:

        st.error(
            f"🔴 {risk_level} — "
            f"Screening Score: {risk_score:.1f}/100"
        )


    # --------------------------------------------------
    # Risk Factor Breakdown
    # --------------------------------------------------

    if particle_count > 0:

        st.subheader("Risk Factor Breakdown")


        risk_col1, risk_col2 = st.columns(2)


        with risk_col1:

            st.write(
                f"**Particle Density:** "
                f"{density_score:.1f}/100"
            )

            st.progress(
                int(density_score)
            )


            st.write(
                f"**Detection Confidence:** "
                f"{confidence_score:.1f}/100"
            )

            st.progress(
                int(confidence_score)
            )


        with risk_col2:

            st.write(
                f"**Relative Particle Size:** "
                f"{size_score:.1f}/100"
            )

            st.progress(
                int(size_score)
            )


            st.write(
                f"**Spatial Clustering:** "
                f"{clustering_score:.1f}/100"
            )

            st.progress(
                int(clustering_score)
            )


    # --------------------------------------------------
    # Heatmap
    # --------------------------------------------------

    if particle_count > 0:

        st.subheader("🔥 Particle Density Heatmap")

        heatmap_image = generate_heatmap(
            image_array,
            bounding_boxes
        )

        st.image(
            heatmap_image,
            caption=(
                "Heatmap showing spatial concentration "
                "of detected particles"
            ),
            use_container_width=True
        )

        st.caption(
            "Brighter regions indicate areas with a higher "
            "concentration of detected particles."
        )


>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
    # --------------------------------------------------
    # Detection Details
    # --------------------------------------------------

    if particle_count > 0:

<<<<<<< HEAD
        st.subheader("Detection Details")


        # Get bounding boxes
        bounding_boxes = boxes.xyxy.cpu().numpy()


        # Store detection information for report
=======
        st.subheader("🔍 Detection Details")


>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
        detection_data = []


        for i, (box, confidence) in enumerate(
<<<<<<< HEAD
            zip(bounding_boxes, confidences),
            start=1
        ):

            # Bounding box coordinates
            x1, y1, x2, y2 = box


            # Calculate dimensions in pixels
            width_pixels = x2 - x1

            height_pixels = y2 - y1


            # Save information
            detection_data.append({
                "particle": i,
                "confidence": confidence,
                "width": width_pixels,
                "height": height_pixels
            })


            # Display information
=======
            zip(
                bounding_boxes,
                confidences
            ),
            start=1
        ):

            x1, y1, x2, y2 = box


            width_pixels = x2 - x1
            height_pixels = y2 - y1


            particle_area = (
                width_pixels *
                height_pixels
            )


            detection_data.append({
                "particle": i,
                "confidence": float(confidence),
                "width": float(width_pixels),
                "height": float(height_pixels),
                "area": float(particle_area)
            })


>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
            st.write(
                f"**Particle {i}**"
            )

            st.write(
<<<<<<< HEAD
                f"Confidence: {confidence * 100:.1f}%"
            )

            st.write(
                f"Width: {width_pixels:.1f} pixels"
            )

            st.write(
                f"Height: {height_pixels:.1f} pixels"
=======
                f"Confidence: "
                f"{confidence * 100:.1f}%"
            )

            st.write(
                f"Width: "
                f"{width_pixels:.1f} pixels"
            )

            st.write(
                f"Height: "
                f"{height_pixels:.1f} pixels"
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
            )

            st.divider()


    else:

        st.warning(
<<<<<<< HEAD
            "No potential microplastic particles were detected."
=======
            "No potential microplastic particles "
            "were detected."
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
        )


    # --------------------------------------------------
    # Generate Analysis Report
    # --------------------------------------------------

    if particle_count > 0:

        st.subheader("📄 Analysis Report")


        report = ""

        report += (
            "AI-ASSISTED MICROPLASTIC SCREENING REPORT\n"
        )

<<<<<<< HEAD
        report += "=" * 55 + "\n\n"
=======
        report += "=" * 60 + "\n\n"
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7


        # Image information
        report += (
            f"Image: {uploaded_file.name}\n"
        )

        report += (
<<<<<<< HEAD
=======
            f"Image Dimensions: "
            f"{image_width} x {image_height} pixels\n"
        )

        report += (
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
            f"Total Particles Detected: "
            f"{particle_count}\n"
        )

        report += (
            f"Average Confidence: "
            f"{average_confidence * 100:.1f}%\n\n"
        )


<<<<<<< HEAD
=======
        # Risk assessment
        report += (
            "SCREENING RISK ASSESSMENT\n"
        )

        report += "-" * 60 + "\n"

        report += (
            f"Risk Score: "
            f"{risk_score:.1f}/100\n"
        )

        report += (
            f"Risk Level: "
            f"{risk_level}\n\n"
        )


        # Risk factors
        report += (
            "RISK FACTOR BREAKDOWN\n"
        )

        report += "-" * 60 + "\n"

        report += (
            f"Particle Density Score: "
            f"{density_score:.1f}/100\n"
        )

        report += (
            f"Detection Confidence Score: "
            f"{confidence_score:.1f}/100\n"
        )

        report += (
            f"Relative Particle Size Score: "
            f"{size_score:.1f}/100\n"
        )

        report += (
            f"Spatial Clustering Score: "
            f"{clustering_score:.1f}/100\n\n"
        )


>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
        # Detection details
        report += (
            "DETECTION DETAILS\n"
        )

<<<<<<< HEAD
        report += "-" * 55 + "\n"
=======
        report += "-" * 60 + "\n"
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7


        for data in detection_data:

            report += (
                f"Particle {data['particle']}\n"
            )

            report += (
                f"Confidence: "
                f"{data['confidence'] * 100:.1f}%\n"
            )

            report += (
                f"Width: "
                f"{data['width']:.1f} pixels\n"
            )

            report += (
                f"Height: "
<<<<<<< HEAD
                f"{data['height']:.1f} pixels\n\n"
=======
                f"{data['height']:.1f} pixels\n"
            )

            report += (
                f"Bounding Box Area: "
                f"{data['area']:.1f} pixels²\n\n"
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
            )


        # Scientific disclaimer
<<<<<<< HEAD
        report += "\n"
        report += "IMPORTANT NOTE\n"
        report += "-" * 55 + "\n"

        report += (
            "This system provides preliminary AI-assisted "
            "screening of potential microplastic particles. "
            "It does not replace laboratory confirmation "
            "techniques such as FTIR or Raman spectroscopy.\n"
=======
        report += (
            "IMPORTANT NOTE\n"
        )

        report += "-" * 60 + "\n"

        report += (
            "The risk score is a preliminary AI-assisted "
            "screening indicator calculated from image-based "
            "features such as detected particle density, "
            "model confidence, relative particle size and "
            "spatial distribution. It is not a measurement "
            "of toxicity, human health risk or environmental "
            "risk. Laboratory confirmation using techniques "
            "such as FTIR or Raman spectroscopy is required "
            "for definitive identification and characterization.\n"
>>>>>>> 0e6bc09f9af3408e46d6c9e55190ab6d9107e0b7
        )


        # Download button
        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name="microplastic_analysis_report.txt",
            mime="text/plain"
        )