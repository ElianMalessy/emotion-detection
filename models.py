import torch
import torch.nn as nn
from torch.utils.data import Dataset 
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, Grayscale, RandomHorizontalFlip, RandomRotation, RandomResizedCrop

import os

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
    def __init__(self, img_labels, emotion_to_idx, img_dir, device="cuda", train=True):
        self.img_labels = img_labels
        self.emotion_to_idx = emotion_to_idx

        base_transform = [
            Grayscale(num_output_channels=1), # Most images are already grayscale
            Resize((128, 128)),  # Most images are 350x350, but it takes very long to train
            Normalize(mean=[0.4736], std=[0.2079]), # mean and std of the dataset
        ]

        augmentation_transforms = [
            RandomHorizontalFlip(p=0.5),  # Flip images with 50% probability
            RandomRotation(degrees=10),  # Rotate images by ±10 degrees
            RandomResizedCrop(128, scale=(0.8, 1.0)),  # Random crop and resize
            Grayscale(num_output_channels=1), # Most images are already grayscale
            Normalize(mean=[0.4736], std=[0.2079]), # mean and std of the dataset
        ]

        if train:
            self.transform = Compose(augmentation_transforms)
            # self.transform = Compose(base_transform)
        else:
            self.transform = Compose(base_transform)

        self.img_dir = img_dir
        self.device = device


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx, 0])
        image = read_image(img_path).float()

        label = self.img_labels[idx, 1]
        label = self.emotion_to_idx[label]
        label = torch.tensor(label)

        image = self.transform(image)

        return image, label
