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
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Adaptive pooling to keep fully connected layer input size fixed
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 8)  # Assuming 8 emotion classes

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  
        
        x = self.adaptive_pool(x)  # Ensure a fixed size before FC layers
        x = x.view(x.size(0), -1)  
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout before the final layers
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

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
            RandomRotation(degrees=10),  # Rotate images by ±20 degrees
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
