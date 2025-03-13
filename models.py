import torch
import torch.nn as nn
from torch.utils.data import Dataset 
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
from PIL import Image

import os
import polars as pl

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # Grayscale image input, 16 output channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Flattened size after pooling, 32 channels of 32x32 spatial size
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (adjust for your problem)

    def forward(self, x):
        # Apply the first convolution, ReLU activation, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply the second convolution, ReLU activation, and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output to feed into the fully connected layer
        x = x.view(-1, 32 * 32 * 32)  # Flatten the output to a 1D tensor
        # Apply the first fully connected layer
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer (output)
        x = self.fc2(x)
        return x

class EmotionsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, device="cuda", transform=None, target_transform=None):
        self.img_labels = pl.read_csv(annotations_file).drop("user.id")
        self.img_labels.sort("image")
        self.img_dir = img_dir
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

        self.img_labels = self.img_labels.with_columns(
            self.img_labels[:, 1].str.to_lowercase().alias("emotion")
        )
        self.emotions = self.img_labels[:, 1].unique().to_list()
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}

        if not self.transform:
            self.transform = Compose([
                Grayscale(num_output_channels=1),
                Resize((128, 128)),  # Resize all images to 128x128 (can be changed)
                Normalize(mean=[0.485], std=[0.229]),  # Normalize with a single value for grayscale
            ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx, 0])
        # image = Image.open(img_path)
        image = read_image(img_path).float()

        label = self.img_labels[idx, 1]
        label = self.emotion_to_idx[label]

        image = self.transform(image).to(self.device)

        if self.target_transform:
            label = self.target_transform(label)

        image = image.to(self.device)
        label = torch.tensor(label).to(self.device)
        return image, label
