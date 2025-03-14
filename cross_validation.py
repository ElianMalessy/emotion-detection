import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from models import CNN
import numpy as np
import logging
from sklearn.model_selection import KFold


def cross_validate(dataset, k=5, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = FocalLoss(alpha=0.25, gamma=2).to(device)
    
    # Split dataset into k folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_results = []
    val_results = []


    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        logging.info(f"Fold {fold + 1}/{k}")
        model = CNN().to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
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

            print(train_loss, len(train_loader))
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

            print(f"Epoch {epoch + 1}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
            print(f"Epoch {epoch + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            logging.info(f"Epoch {epoch + 1}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
            logging.info(f"Epoch {epoch + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


    
    # Calculate and log average results
    avg_val_loss = np.mean([result[0] for result in val_results])
    avg_val_accuracy = np.mean([result[1] for result in val_results])
    logging.info(f"Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_accuracy:.4f}")
