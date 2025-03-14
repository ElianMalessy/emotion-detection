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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Shape: [64, 16, 175, 175]
        x = self.pool(F.relu(self.conv2(x)))  # Shape: [64, 32, 87, 87]
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EmotionsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, device="cuda"):
        self.img_labels = pl.read_csv(annotations_file).drop("user.id")
        self.img_labels.sort("image")
        self.img_dir = img_dir
        self.device = device

        self.img_labels = self.img_labels.with_columns(
            self.img_labels[:, 1].str.to_lowercase().alias("emotion")
        )
        self.emotions = self.img_labels[:, 1].unique().to_list()
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}

        self.transform = Compose([
            Grayscale(num_output_channels=1), # Most images are already grayscale
            Resize((128, 128)),  # Most images are 350x350, but it takes very long to train
            Normalize(mean=[0.4736], std=[0.2079]), # mean and std of the dataset
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx, 0])
        # image = Image.open(img_path)
        image = read_image(img_path).float()

        label = self.img_labels[idx, 1]
        label = self.emotion_to_idx[label]
        label = torch.tensor(label)

        image = self.transform(image)


        return image, label
