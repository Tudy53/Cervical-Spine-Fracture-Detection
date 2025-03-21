import os
import torch
import torchvision
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, inception_v3
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from torch.nn.functional import adaptive_avg_pool2d
import random

class BoneFractureDataset(Dataset):
    def __init__(self, labeled_train_file, paths_file, transform=None):
        self.transform = transform
        labeled_train = pd.read_csv(labeled_train_file, header=None, names=["directory", "label"])
        paths = pd.read_csv(paths_file, header=None, names=["image_path"])
        self.label_dict = {os.path.normpath(row["directory"]): row["label"] for _, row in labeled_train.iterrows()}
        self.data = [(path, self.label_dict[os.path.normpath(os.path.dirname(path))])
                     for path in paths["image_path"] if os.path.normpath(os.path.dirname(path)) in self.label_dict]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class DCGANGenerator(torch.nn.Module):
    def __init__(self, noise_dim, img_channels, feature_dim):
        super(DCGANGenerator, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(noise_dim, feature_dim * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(feature_dim * 8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_dim * 4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_dim * 2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_dim),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(feature_dim, img_channels, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class DCGANDiscriminator(torch.nn.Module):
    def __init__(self, img_channels, feature_dim):
        super(DCGANDiscriminator, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(img_channels, feature_dim, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_dim * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_dim * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(feature_dim * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(feature_dim * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)


def train_dcgan(train_loader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device):
    for images, labels in train_loader:
        images = images.to(device)
        batch_size = images.size(0)

        # Generate fake images
        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        fake_images = generator(noise)

        # Train Discriminator
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        d_real_loss = criterion(discriminator(images), real_labels)
        d_fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
        d_loss = d_real_loss + d_fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        output = discriminator(fake_images)
        g_loss = criterion(output, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()


def generate_synthetic_images(generator, noise_dim, num_images, output_folder, csv_path, train_dataset, transform):
    os.makedirs(output_folder, exist_ok=True)
    generator.eval()

    # Select 15000 random samples from the train dataset
    indices = np.random.choice(len(train_dataset), num_images, replace=False)
    sampled_dataset = torch.utils.data.Subset(train_dataset, indices)
    image_paths = []
    labels = []

    for i, (image, label) in enumerate(sampled_dataset):
        image = image.unsqueeze(0).to(device)
        noise = torch.randn(1, noise_dim, 1, 1, device=device)
        fake_image = generator(noise).detach().cpu().squeeze(0)

        # Save synthetic image
        image_path = os.path.join(output_folder, f"img_{i}.png")
        torchvision.utils.save_image((fake_image + 1) / 2, image_path)
        image_paths.append(image_path)
        labels.append(label)

    # Save paths and labels to CSV
    df = pd.DataFrame({"image_path": image_paths, "label": labels})
    df.to_csv(csv_path, index=False)
    print(f"Generated {num_images} synthetic images and saved paths and labels to {csv_path}.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_dim = 100
    img_channels = 3
    feature_dim = 64

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset and DataLoader
    dataset = BoneFractureDataset('./MURA-v1.1/train_labeled_studies.csv',
                                  './MURA-v1.1/train_image_paths.csv',
                                  transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # DCGAN Setup
    generator = DCGANGenerator(noise_dim, img_channels, feature_dim).to(device)
    discriminator = DCGANDiscriminator(img_channels, feature_dim).to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = torch.nn.BCELoss()

    # Train DCGAN
    print("Training DCGAN...")
    for epoch in range(25):
        train_dcgan(train_loader, generator, discriminator, g_optimizer, d_optimizer, criterion, noise_dim, device)
        print(f"Epoch {epoch+1}/25 completed")

    # Generate 15000 Synthetic Images
    print("Generating synthetic data...")
    generate_synthetic_images(generator, noise_dim, 15000, './synthetic_images', './synthetic_image_labels.csv', dataset, transform)
    print("Synthetic images generation completed.")
