import os
import torch
import torchvision
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np

class BoneFractureDataset(torch.utils.data.Dataset):
    def __init__(self, labeled_train_file, paths_file, transform=None, augmentations=None):
        self.transform = transform
        self.augmentations = augmentations

        # Load labeled data and paths
        labeled_train = pd.read_csv(labeled_train_file, header=None, names=["directory", "label"])
        paths = pd.read_csv(paths_file, header=None, names=["image_path"])

        # Create a mapping of directories to labels
        self.label_dict = {os.path.normpath(row["directory"]): row["label"] for _, row in labeled_train.iterrows()}

        # Combine image paths with their respective labels
        self.data = []
        for path in paths["image_path"]:
            parent_dir = os.path.normpath(os.path.dirname(path))
            if parent_dir in self.label_dict:
                self.data.append((path, self.label_dict[parent_dir]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB
        if self.transform:
            image = self.transform(image)
            
        if self.augmentations:
            image = self.augmentations(image=np.array(image))["image"]
        
        return image, label, img_path


# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Standard ImageNet normalization
])

albumentations_transforms = A.Compose([
    A.SmallestMaxSize(max_size=256),  # Asigură că imaginea este suficient de mare
    A.RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(p=0.2),
    A.GaussianBlur(p=0.1),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normalizează și convertește în float32
    ToTensorV2()
])

# Initialize datasets
dataset = BoneFractureDataset(
    labeled_train_file='./MURA-v1.1/train_labeled_studies.csv',
    paths_file='./MURA-v1.1/train_image_paths.csv',
    transform=data_transforms,
    augmentations=None
)

augmented_dataset = BoneFractureDataset(
    labeled_train_file='./MURA-v1.1/train_labeled_studies.csv',
    paths_file='./MURA-v1.1/train_image_paths.csv',
    transform=None,  # Albumentations handles ToTensor
    augmentations=albumentations_transforms
)

test_dataset = BoneFractureDataset(
    labeled_train_file='./MURA-v1.1/test_labeled_studies.csv',
    paths_file='./MURA-v1.1/test_image_paths.csv',
    transform=data_transforms,
    augmentations=None
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Testing images: {len(test_dataset)}")

combined_dataset = ConcatDataset([train_dataset, augmented_dataset])
print(f"Combined training images: {len(combined_dataset)}")

# Initialize DataLoaders
train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("Finished dataloader creation\n.")

# Set up model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize EfficientNet-B0
weights = EfficientNet_B0_Weights.DEFAULT  # Pretrained weights for EfficientNet-B0
model = efficientnet_b0(weights=weights)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)  # Ajustăm pentru 2 clase
model = model.to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Validation Loop
num_epochs = 25
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_losses.append(train_loss / train_total)
    train_accuracies.append(train_correct / train_total)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_losses.append(val_loss / val_total)
    val_accuracies.append(val_correct / val_total)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f} - "
          f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

# Test Evaluation
model.eval()
all_labels, all_predictions, all_probs, all_paths = [], [], [], []

with torch.no_grad():
    for inputs, labels, paths in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_paths.extend(paths)

# Save Results
results_df = pd.DataFrame({
    "file_path": all_paths,
    "true_label": all_labels,
    "predicted_label": all_predictions,
    "probability": all_probs,
})
results_df.to_csv("results.csv", index=False)

# Metrics
accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)
roc_auc = roc_auc_score(all_labels, all_probs)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(f"Test ROC-AUC: {roc_auc:.4f}")

# Save the confusion matrix as an image
conf_matrix = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.savefig("pretrained_Confussion_Matrix2.png", bbox_inches="tight")
plt.close()

# Save Training and Validation Loss Over Epochs
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("pretrainedTraining_Validation_Loss2.png", bbox_inches="tight")
plt.close()

# Save Training and Validation Accuracy Over Epochs
plt.figure()
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("pretrainedTraining_Validation_Accuracy2.png", bbox_inches="tight")
plt.close()
print("Plots saved as images: 'pretrained_Confussion_Matrix2.png', 'pretrainedTraining_Validation_Loss.png', 'pretrainedTraining_Validation_Accuracy2.png'")
