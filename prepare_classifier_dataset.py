import os
import shutil
import random
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "Photos_data.xlsx - Sheet1.csv"

IMAGE_FOLDER = r"C:\Users\Akshatha\OneDrive\proj\Microplastic_Data_Portal\data\Microplastic_Images"

OUTPUT_FOLDER = "classification_dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


# ============================================================
# MORPHOLOGY MAPPING
# ============================================================

def map_morphology(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    # Fibre
    if value in ["fiber", "fibre", "fibers", "fibres"]:
        return "fibre"

    # Fragment
    if "fragment" in value:
        return "fragment"

    # Ignore film, shaving and other categories
    return None


# ============================================================
# CHECK IMAGE FOLDER
# ============================================================

print("\nChecking image folder...")

if not os.path.exists(IMAGE_FOLDER):

    print("\nERROR: Image folder does not exist:")
    print(IMAGE_FOLDER)
    exit()

print("Image folder found!")


# ============================================================
# FIND IMAGES
# ============================================================

print("\nSearching for images...")

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

image_lookup = {}

for root, dirs, files in os.walk(IMAGE_FOLDER):

    for filename in files:

        extension = os.path.splitext(filename)[1].lower()

        if extension not in image_extensions:
            continue

        full_path = os.path.join(root, filename)

        # Full filename
        image_lookup[filename.lower()] = full_path

        # Filename without extension
        stem = os.path.splitext(filename)[0].lower()

        if stem not in image_lookup:
            image_lookup[stem] = full_path


print(
    f"Found {len(image_lookup)} image references."
)


# ============================================================
# READ CSV
# ============================================================

print("\nReading metadata...")

if not os.path.exists(CSV_PATH):

    print("\nERROR: Metadata file not found:")
    print(os.path.abspath(CSV_PATH))
    exit()


df = pd.read_csv(
    CSV_PATH,
    encoding="utf-8-sig"
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.replace("\r", " ", regex=False)
    .str.replace("\n", " ", regex=False)
    .str.strip()
)

print("\nCSV columns:")
print(df.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Image File",
    "Morphology of particle"
]

for column in required_columns:

    if column not in df.columns:

        print(
            f"\nERROR: Missing column: {column}"
        )

        print(
            "\nAvailable columns:"
        )

        print(
            df.columns.tolist()
        )

        exit()


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

classes = [
    "fibre",
    "fragment"
]

print("\nCreating dataset folders...")

for split in [
    "train",
    "val",
    "test"
]:

    for class_name in classes:

        folder = os.path.join(
            OUTPUT_FOLDER,
            split,
            class_name
        )

        os.makedirs(
            folder,
            exist_ok=True
        )


# ============================================================
# MATCH IMAGES WITH LABELS
# ============================================================

dataset_items = []

not_found = []

ignored = []

print("\nMatching images with morphology labels...")


for index, row in df.iterrows():

    # --------------------------------------------------------
    # Get image name
    # --------------------------------------------------------

    image_name = str(
        row["Image File"]
    ).strip()


    # --------------------------------------------------------
    # Get morphology
    # --------------------------------------------------------

    morphology = map_morphology(
        row["Morphology of particle"]
    )


    # --------------------------------------------------------
    # Ignore other morphology types
    # --------------------------------------------------------

    if morphology is None:

        ignored.append(
            (
                image_name,
                row["Morphology of particle"]
            )
        )

        continue


    # --------------------------------------------------------
    # Find image
    # --------------------------------------------------------

    image_path = None

    image_name_lower = image_name.lower().strip()

    image_stem = os.path.splitext(
        image_name_lower
    )[0]


    # --------------------------------------------------------
    # Exact filename match
    # --------------------------------------------------------

    if image_name_lower in image_lookup:

        image_path = image_lookup[
            image_name_lower
        ]


    # --------------------------------------------------------
    # Filename without extension
    # --------------------------------------------------------

    if image_path is None:

        if image_stem in image_lookup:

            image_path = image_lookup[
                image_stem
            ]


    # --------------------------------------------------------
    # Partial filename match
    # --------------------------------------------------------

    if image_path is None:

        for key, path in image_lookup.items():

            if image_stem == key:

                image_path = path
                break

            if key.startswith(image_stem):

                image_path = path
                break


    # --------------------------------------------------------
    # Image not found
    # --------------------------------------------------------

    if image_path is None:

        not_found.append(
            (
                image_name,
                morphology
            )
        )

        continue


    # --------------------------------------------------------
    # Store dataset item
    # --------------------------------------------------------

    dataset_items.append(
        {
            "path": image_path,
            "class": morphology
        }
    )


# ============================================================
# MATCHING RESULTS
# ============================================================

print("\n====================================")
print("DATASET MATCHING RESULTS")
print("====================================")

print(
    "Usable images:",
    len(dataset_items)
)

print(
    "Images not found:",
    len(not_found)
)

print(
    "Ignored morphology:",
    len(ignored)
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

for class_name in classes:

    count = sum(
        1
        for item in dataset_items
        if item["class"] == class_name
    )

    print(
        f"{class_name}: {count}"
    )


# ============================================================
# STOP IF NO DATA
# ============================================================

if len(dataset_items) == 0:

    print(
        "\nERROR: No usable images found."
    )

    exit()


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(
    dataset_items
)


# ============================================================
# SPLIT DATA
# ============================================================

total = len(
    dataset_items
)

train_end = int(
    total * TRAIN_RATIO
)

val_end = int(
    total * (TRAIN_RATIO + VAL_RATIO)
)


train_items = dataset_items[
    :train_end
]

val_items = dataset_items[
    train_end:val_end
]

test_items = dataset_items[
    val_end:
]


print("\n====================================")
print("DATASET SPLIT")
print("====================================")

print(
    "Train:",
    len(train_items)
)

print(
    "Validation:",
    len(val_items)
)

print(
    "Test:",
    len(test_items)
)


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_items(items, split):

    print(
        f"\nCreating {split} dataset..."
    )

    counters = {
        class_name: 0
        for class_name in classes
    }


    for item in items:

        source = item["path"]

        class_name = item["class"]

        extension = os.path.splitext(
            source
        )[1]

        number = counters[
            class_name
        ]

        destination_filename = (
            f"{class_name}_{number}{extension}"
        )

        destination = os.path.join(
            OUTPUT_FOLDER,
            split,
            class_name,
            destination_filename
        )

        shutil.copy2(
            source,
            destination
        )

        counters[
            class_name
        ] += 1


    print(
        f"{split} completed."
    )


# ============================================================
# CREATE DATASET
# ============================================================

copy_items(
    train_items,
    "train"
)

copy_items(
    val_items,
    "val"
)

copy_items(
    test_items,
    "test"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n====================================")
print("DATASET CREATION COMPLETE")
print("====================================")

print(
    "\nDataset created at:"
)

print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)

print("\nFinal dataset:")

for split in [
    "train",
    "val",
    "test"
]:

    print(
        f"\n{split}:"
    )

    for class_name in classes:

        folder = os.path.join(
            OUTPUT_FOLDER,
            split,
            class_name
        )

        count = len(
            os.listdir(folder)
        )

        print(
            f"  {class_name}: {count}"
        )


print("\nFinished!")