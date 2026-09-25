from pathlib import Path
from PIL import Image
import yaml

# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = Path(r"D:\MAJORPROJECT\microplastic_4class_source")
OUTPUT_DIR = Path(r"D:\MAJORPROJECT\classification_dataset_4class")

# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAP = {
    0: "fibre",
    1: "film",
    2: "fragment",
    3: "microbeads"
}

# ============================================================
# LOAD YAML
# ============================================================

yaml_path = SOURCE_DIR / "data.yaml"

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

print("Classes from data.yaml:")
print(data["names"])

# ============================================================
# PROCESS FUNCTION
# ============================================================

def process_split(split_name, output_split):

    image_dir = SOURCE_DIR / split_name / "images"
    label_dir = SOURCE_DIR / split_name / "labels"

    output_dir = OUTPUT_DIR / output_split

    print("\n" + "=" * 60)
    print(f"PROCESSING: {split_name} -> {output_split}")
    print("=" * 60)

    if not image_dir.exists():
        print(f"ERROR: Image directory not found:")
        print(image_dir)
        return

    if not label_dir.exists():
        print(f"ERROR: Label directory not found:")
        print(label_dir)
        return

    image_files = list(image_dir.glob("*"))

    print(f"Images found: {len(image_files)}")

    # Create class directories
    for class_name in CLASS_MAP.values():
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    total_annotations = 0
    class_counts = {
        "fibre": 0,
        "film": 0,
        "fragment": 0,
        "microbeads": 0
    }

    processed_images = 0

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for image_path in image_files:

        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
        except Exception as e:
            print(f"Could not open {image_path}: {e}")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        image_had_annotation = False

        for annotation_index, line in enumerate(lines):

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            if class_id not in CLASS_MAP:
                continue

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # Convert YOLO normalized coordinates -> pixels

            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Clamp coordinates

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Ignore invalid boxes

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))

            class_name = CLASS_MAP[class_id]

            # Unique filename
            output_name = (
                f"{image_path.stem}_particle_{annotation_index}.jpg"
            )

            output_path = output_dir / class_name / output_name

            crop.save(output_path, quality=95)

            class_counts[class_name] += 1
            total_annotations += 1
            image_had_annotation = True

        if image_had_annotation:
            processed_images += 1

    # ========================================================
    # RESULTS
    # ========================================================

    print("\nResults:")
    print(f"Images with annotations: {processed_images}")
    print(f"Total crops generated: {total_annotations}")

    for class_name, count in class_counts.items():
        print(f"  {class_name:12s}: {count}")


# ============================================================
# CLEAR OLD DATA
# ============================================================

print("\nOutput directory:")
print(OUTPUT_DIR)

# IMPORTANT:
# We are NOT deleting anything automatically.
# Remove the old output manually before running this script
# if you want a completely fresh conversion.

# ============================================================
# PROCESS DATASET
# ============================================================

process_split("train", "train")
process_split("valid", "val")
process_split("test", "test")

print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print(f"\nDataset created at:")
print(OUTPUT_DIR)