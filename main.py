import torch
from torch import nn
from torch.utils.data import DataLoader
from models import CNN, EmotionsDataset
import numpy as np
import logging
from torch.utils.data import random_split

# Set up logging to write to a file
logging.basicConfig(filename='training_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
np.random.seed(42)


def test_model(dataset, num_epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = FocalLoss(alpha=0.25, gamma=2).to(device)
    
    model = CNN().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(dataset, [train_size, test_size], generator)
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    results = {}


    for epoch in range(num_epochs):
        train_loss = 0
        train_accuracy = 0

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
        results[epoch] = (train_loss, train_accuracy, val_loss, val_accuracy)

        print(f"Epoch {epoch + 1}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"Epoch {epoch + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        logging.info(f"Epoch {epoch + 1}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        logging.info(f"Epoch {epoch + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    import json
    with open("sgd.json", "w") as f:
        json.dump(results, f)



if __name__ == "__main__":
    dataset = EmotionsDataset("data/legend.csv", "images")
    test_model(dataset, num_epochs=50)
    # cross_validate(dataset, k=5, num_epochs=100)
