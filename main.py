import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from models import CNN, EmotionsDataset
import numpy as np
import logging

# Set up logging to write to a file
logging.basicConfig(filename='training_log2.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

def cross_validate(dataset, k=5, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    # Split dataset into k folds
    fold_size = len(dataset) // k
    results = []

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

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

        val_loss = 0
        val_accuracy = 0

        for epoch in range(num_epochs):
            model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss = running_train_loss / len(train_loader)
            train_accuracy = correct_train / total_train

            # Validation
            model.eval()
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    val_accuracy += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy /= len(val_subset)

            # Log the training and validation statistics for each epoch
            logging.info(f"  Epoch {epoch + 1}/{num_epochs}")
            logging.info(f"    Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
            logging.info(f"    Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        results.append((val_loss, val_accuracy))

    # Save final results to the log file
    logging.info("Cross-validation results:")
    for fold, (val_loss, val_accuracy) in enumerate(results):
        logging.info(f"Fold {fold + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Calculate and log average results
    avg_val_loss = np.mean([result[0] for result in results])
    avg_val_accuracy = np.mean([result[1] for result in results])
    logging.info(f"Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_accuracy:.4f}")


# Load dataset
dataset = EmotionsDataset("data/legend.csv", "images")

# Perform cross-validation
cross_validate(dataset, k=5, num_epochs=20)
