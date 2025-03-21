import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class BoneFractureDataset(Dataset):
    def __init__(self, labeled_train_file, paths_file, transform=None, augmentations=None):
        self.transform = transform
        self.augmentations = augmentations
        labeled_train = pd.read_csv(labeled_train_file, header=None, names=["directory", "label"])
        paths = pd.read_csv(paths_file, header=None, names=["image_path"])
        self.label_dict = {os.path.normpath(row["directory"]): row["label"] for _, row in labeled_train.iterrows()}
        self.data = [(path, self.label_dict[os.path.normpath(os.path.dirname(path))])
                     for path in paths["image_path"] if os.path.normpath(os.path.dirname(path)) in self.label_dict]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        else:
            if self.transform:
                image = Image.fromarray(image)
                image = self.transform(image)

        return image, label


def train_efficientnet(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in train_loader:
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

        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        val_loss = val_loss / val_total

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss / train_total:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    return model


# Define the search space for hyperparameters
search_space = [
    Real(1e-5, 1e-2, name='lr'),
]

# Define the objective function
@use_named_args(search_space)
def objective_function(lr):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Testing with learning rate: {lr:.6f}")
    trained_model = train_efficientnet(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10)

    trained_model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    return -val_accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    augmentations = A.Compose([
        A.Resize(64, 64),
        A.RandomRotate90(),
        A.Flip(),
        A.RandomBrightnessContrast(),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    dataset1 = BoneFractureDataset(
        labeled_train_file='./MURA-v1.1/train_labeled_studies.csv',
        paths_file='./MURA-v1.1/train_image_paths.csv',
        transform=transform, 
        augmentations=None
    )

    dataset2 = BoneFractureDataset(
        labeled_train_file='./synthetic_image_labels.csv',
        paths_file='./synthetic_image_paths.csv',
        transform=transform, 
        augmentations=None
    )

    dataset3 = BoneFractureDataset(
        labeled_train_file='./MURA-v1.1/train_labeled_studies.csv',
        paths_file='./MURA-v1.1/train_image_paths.csv',
        transform=None, 
        augmentations=augmentations
    )

    dataset4 = BoneFractureDataset(
        labeled_train_file='./synthetic_image_labels.csv',
        paths_file='./synthetic_image_paths.csv',
        transform=None, 
        augmentations=augmentations
    )

    datasetob1 = ConcatDataset([dataset1, dataset2])
    datasetob2 = ConcatDataset([dataset3, dataset4])

    dataset = ConcatDataset([datasetob1, datasetob2])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    test_dataset = BoneFractureDataset(
        labeled_train_file='./MURA-v1.1/test_labeled_studies.csv',
        paths_file='./MURA-v1.1/test_image_paths.csv',
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    result = gp_minimize(objective_function, search_space, n_calls=10, random_state=42)
    best_lr = result.x[0]
    print(f"Best Learning Rate: {best_lr:.6f}")

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training EfficientNet with best hyperparameters...")
    model = train_efficientnet(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]


            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
 
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
 
    conf_matrix = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    print("Confusion matrix saved as 'confusion_matrix.png'.")
    