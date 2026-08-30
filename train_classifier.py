import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "classification_dataset"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "microplastic_classifier.pth"
)

BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 0.0001

IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing device:", device)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    os.path.join(
        DATASET_DIR,
        "train"
    ),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(
        DATASET_DIR,
        "val"
    ),
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(
        DATASET_DIR,
        "test"
    ),
    transform=val_test_transform
)


print("\nClasses:")
print(train_dataset.class_to_idx)

print(
    "\nTraining images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Test images:",
    len(test_dataset)
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD PRETRAINED RESNET18
# ============================================================

print("\nLoading ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model = model.to(device)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n====================================")
print("STARTING TRAINING")
print("====================================")


best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        100 * correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100 * val_correct / val_total
    )


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "  → Best model saved!"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# TEST
# ============================================================

test_correct = 0

test_total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(images)


        _, predicted = torch.max(
            outputs,
            1
        )


        test_total += labels.size(0)

        test_correct += (
            predicted == labels
        ).sum().item()


test_accuracy = (
    100 * test_correct / test_total
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n====================================")
print("CLASSIFIER TRAINING COMPLETE")
print("====================================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    "\nModel saved at:"
)

print(
    os.path.abspath(
        MODEL_PATH
    )
)