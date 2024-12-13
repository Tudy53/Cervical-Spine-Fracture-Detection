import os
import torch
import torchvision
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn.functional import one_hot
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from os import listdir
from random import randint
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Subset
from collections import Counter
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from torch.utils.data import random_split

train_directory_path = './MURA-v1.1/train'
test_directory_path = './MURA-v1.1/test'


# Define Dataset
import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

class BoneFractureDataset(Dataset):
    def __init__(self, labeled_train_file, paths_file, transform=None):
        """
        Args:
            labeled_train_file (str): Fișierul CSV care conține directoarele și etichetele lor.
            paths_file (str): Fișierul care conține căile complete către imagini.
            transform (callable, optional): Transformările aplicate pe imagini.
        """
        self.transform = transform

        # Încarcă fișierele
        labeled_train = pd.read_csv(labeled_train_file, header=None, names=["directory", "label"])
        paths = pd.read_csv(paths_file, header=None, names=["image_path"])

        # Creează un dicționar pentru a lega directoarele de etichete
        self.label_dict = {os.path.normpath(row["directory"]): row["label"] for _, row in labeled_train.iterrows()}

        # Creează o listă cu imaginile și etichetele lor
        self.data = []
        for path in paths["image_path"]:
            parent_dir = os.path.normpath(os.path.dirname(path))  # Extrage directorul părinte
            if parent_dir in self.label_dict:
                self.data.append((path, self.label_dict[parent_dir]))
    
    def __len__(self):
        """Returnează numărul total de imagini."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Return an image, its label, and its file path.
        
        Args:
            idx (int): Index of the image.
        
        Returns:
            image (Tensor): Transformed image.
            label (int): Label of the image.
            img_path (str): File path of the image.
        """
        img_path, label = self.data[idx]

        # Load the image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path



# Define Transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Inițializăm dataset-ul
dataset = BoneFractureDataset(
    labeled_train_file='./MURA-v1.1/train_labeled_studies.csv',
    paths_file='./MURA-v1.1/train_image_paths.csv',
    transform=data_transforms
)

test_dataset = BoneFractureDataset(
    labeled_train_file='./MURA-v1.1/test_labeled_studies.csv',
    paths_file='./MURA-v1.1/test_image_paths.csv',
    transform=data_transforms
)

# Train-validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Testing images: {len(test_dataset)}")

# Inițializăm un DataLoader pentru iterare
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
print("Finished dataloader creation\n.")


# Set up model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification
model = model.to(device)
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

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.title("Confusion Matrix on Test Set")

# Save the confusion matrix as an image
plt.savefig("Resnet50_Confussion_Matrix.jpeg", bbox_inches="tight")
plt.close()  # Close the figure to free up memory

# Plot Training History (Loss Over Epochs)
epochs = range(1, num_epochs + 1)
plt.figure()
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

# Save the loss evolution plot
plt.savefig("Resnet50_Training_Validation_Loss.jpeg", bbox_inches="tight")
plt.close()

# Plot Training History (Accuracy Over Epochs)
plt.figure()
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

# Save the accuracy evolution plot
plt.savefig("Resnet50_Training_Validation_Accuracy.jpeg", bbox_inches="tight")
plt.close()

print("Plots saved as images: 'Resnet50_Confussion_Matrix.jpeg', 'Resnet50_Training_Validation_Loss.jpeg', 'Resnet50_Training_Validation_Accuracy.jpeg'")
