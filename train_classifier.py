import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"D:\MAJORPROJECT\classification_dataset_4class"
MODEL_DIR = r"D:\MAJORPROJECT\models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "microplastic_classifier_4class.pth"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "val"),
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "test"),
    transform=val_transform
)

class_names = train_dataset.classes
num_classes = len(class_names)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i} -> {name}")

print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_dataset.targets,
    minlength=num_classes
)

class_weights = len(train_dataset) / (
    num_classes * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")

for i, weight in enumerate(class_weights):
    print(
        f"{class_names[i]}: "
        f"{weight.item():.4f}"
    )

# ============================================================
# RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze the pretrained feature extractor
for param in model.parameters():
    param.requires_grad = False

# Replace final layer with 4-class classifier
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    num_classes
)

# Only the new FC layer will train
model = model.to(DEVICE)

# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0

print("\n" + "=" * 60)
print("STARTING FAST TRANSFER LEARNING")
print("=" * 60)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    total_batches = len(train_loader)

    for batch_number, (images, labels) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        # Show progress every 25 batches
        if batch_number % 25 == 0 or batch_number == total_batches:

            progress = (
                batch_number /
                total_batches
            ) * 100

            print(
                f"\rEpoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_number}/{total_batches} "
                f"({progress:.0f}%)",
                end="",
                flush=True
            )

    train_accuracy = (
        100 * correct / total
    )

    train_loss = (
        running_loss /
        len(train_loader)
    )

    print()

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_total = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += loss.item()

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

    val_loss = (
        val_loss_total /
        len(val_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"✓ Best model saved: "
            f"{best_val_accuracy:.2f}%"
        )

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("TESTING BEST MODEL")
print("=" * 60)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

all_predictions = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

test_accuracy = (
    100 * correct / total
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print("\nModel saved at:")
print(MODEL_PATH)
13