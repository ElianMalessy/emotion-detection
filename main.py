import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from models import CNN, EmotionsDataset
import numpy as np
import logging

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

# Set up logging to write to a file
logging.basicConfig(filename='training_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
np.random.seed(42)

def cross_validate(dataset, k=5, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = FocalLoss(alpha=0.25, gamma=2).to(device)
    
    # Split dataset into k folds
    fold_size = len(dataset) // k
    train_results = []
    val_results = []

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    for fold in range(k):
        logging.info(f"Fold {fold + 1}/{k}")
        model = CNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create training and validation sets for this fold
        val_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
        train_indices = list(set(range(len(dataset))) - set(val_indices))
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        train_loss = 0
        train_accuracy = 0

        for epoch in range(num_epochs):
            print(f"  Epoch {epoch + 1}/{num_epochs}")
            model.train()

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                train_accuracy += (predicted == labels).sum().item()

            train_loss /= len(train_loader)
            train_accuracy /= len(train_subset)

            train_results.append((train_loss, train_accuracy))

            # Validation
            model.eval()
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    val_accuracy += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy /= len(val_subset)

            val_results.append((val_loss, val_accuracy))

        train_accuracies = []
        val_accuracies = []
        for epoch in range(num_epochs):
            train_accuracies.append(train_results[epoch][1])
            val_accuracies.append(val_results[epoch][1])

        epochs = list(range(1, num_epochs + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_accuracies, label='Training Accuracy', color='b', marker='o')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='g', marker='x')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy per Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()


    logging.info("Cross-validation results:")
    for fold in range(len(val_results)):
        train_loss, train_accuracy = train_results[fold]
        val_loss, val_accuracy = val_results[fold]
        logging.info(f"Fold {fold + 1}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        logging.info(f"Fold {fold + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


    
    # Calculate and log average results
    avg_val_loss = np.mean([result[0] for result in val_results])
    avg_val_accuracy = np.mean([result[1] for result in val_results])
    logging.info(f"Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_accuracy:.4f}")

    


# Load dataset
dataset = EmotionsDataset("data/legend.csv", "images")

# Perform cross-validation
cross_validate(dataset, k=5, num_epochs=20)
