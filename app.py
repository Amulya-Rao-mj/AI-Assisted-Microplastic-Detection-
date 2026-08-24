import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


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

MODEL_PATH = r"D:\MAJORPROJECT\runs\detect\results\microplastic_detection-2\weights\best.pt"

model = YOLO(MODEL_PATH)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🔬 AI-Assisted Microplastic Screening System")

st.write(
    "Upload a microscopic image to detect potential microplastic particles."
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

    # Read image
    image = Image.open(uploaded_file)

    # --------------------------------------------------
    # Original Image
    # --------------------------------------------------

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True
    )


    # Convert image to NumPy
    image_array = np.array(image)


    # --------------------------------------------------
    # Run YOLO detection
    # --------------------------------------------------

    results = model(
        image_array,
        conf=0.50
    )

    result = results[0]


    # --------------------------------------------------
    # Detection information
    # --------------------------------------------------

    boxes = result.boxes

    # Number of detected particles
    particle_count = len(boxes)


    # Confidence values
    if particle_count > 0:

        confidences = boxes.conf.cpu().numpy()

        average_confidence = float(
            np.mean(confidences)
        )

    else:

        confidences = []

        average_confidence = 0.0


    # --------------------------------------------------
    # Annotated image
    # --------------------------------------------------

    annotated_image = result.plot()

    st.subheader("Detected Particles")

    st.image(
        annotated_image,
        use_container_width=True
    )


    # --------------------------------------------------
    # Analysis Results
    # --------------------------------------------------

    st.subheader("Analysis Results")

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Total Particles Detected",
            particle_count
        )


    with col2:

        st.metric(
            "Average Confidence",
            f"{average_confidence * 100:.1f}%"
        )


    # --------------------------------------------------
    # Detection Details
    # --------------------------------------------------

    if particle_count > 0:

        st.subheader("Detection Details")


        # Get bounding boxes
        bounding_boxes = boxes.xyxy.cpu().numpy()


        # Store detection information for report
        detection_data = []


        for i, (box, confidence) in enumerate(
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
            st.write(
                f"**Particle {i}**"
            )

            st.write(
                f"Confidence: {confidence * 100:.1f}%"
            )

            st.write(
                f"Width: {width_pixels:.1f} pixels"
            )

            st.write(
                f"Height: {height_pixels:.1f} pixels"
            )

            st.divider()


    else:

        st.warning(
            "No potential microplastic particles were detected."
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

        report += "=" * 55 + "\n\n"


        # Image information
        report += (
            f"Image: {uploaded_file.name}\n"
        )

        report += (
            f"Total Particles Detected: "
            f"{particle_count}\n"
        )

        report += (
            f"Average Confidence: "
            f"{average_confidence * 100:.1f}%\n\n"
        )


        # Detection details
        report += (
            "DETECTION DETAILS\n"
        )

        report += "-" * 55 + "\n"


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
                f"{data['height']:.1f} pixels\n\n"
            )


        # Scientific disclaimer
        report += "\n"
        report += "IMPORTANT NOTE\n"
        report += "-" * 55 + "\n"

        report += (
            "This system provides preliminary AI-assisted "
            "screening of potential microplastic particles. "
            "It does not replace laboratory confirmation "
            "techniques such as FTIR or Raman spectroscopy.\n"
        )


        # Download button
        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name="microplastic_analysis_report.txt",
            mime="text/plain"
        )