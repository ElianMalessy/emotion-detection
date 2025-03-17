import torch
from torch import nn
from torch.utils.data import DataLoader
from models import CNN, EmotionsDataset
import numpy as np
import logging
import polars as pl
from sklearn.model_selection import train_test_split
from cross_validation import cross_validate
from loss import FocalLoss
from stopping import EarlyStopping

# Set up logging to write to a file
logging.basicConfig(filename='training_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

np.random.seed(42)
torch.manual_seed(42) 
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def test_model(data, num_epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = FocalLoss(data, gamma=1).to(device)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    emotions = data[:, 1].unique().sort().to_list()
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}

    train_indices, test_indices = train_test_split(
        data.select(pl.arange(0, data.height)).to_series().to_list(),
        test_size=0.2,
        random_state=42,
        stratify=data["emotion"].to_list()  # Ensures equal label distribution
    )

    training_data = data[train_indices]
    validate_data = data[test_indices]

    train_subset = EmotionsDataset(training_data, emotion_to_idx, "images", train=True)
    val_subset = EmotionsDataset(validate_data, emotion_to_idx, "images", train=False)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    early_stopping = EarlyStopping(patience=5, delta=0.001)

    results = {}
    best_val_accuracy = 0.0  # Track the best validation accuracy

    for epoch in range(num_epochs):
        train_accuracy = 0
        val_accuracy = 0

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

            _, predicted = torch.max(outputs, 1)
            train_accuracy += (predicted == labels).sum().item()

        train_accuracy /= len(train_subset)

        # Validation
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                val_accuracy += (predicted == labels).sum().item()

        val_accuracy /= len(val_subset)
        results[epoch] = (train_accuracy, val_accuracy)

        print(f"Epoch {epoch + 1}: Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        logging.info(f"Epoch {epoch + 1}: Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

<<<<<<< HEAD
        # Save the model if it's the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "emotion_model.pth")
            print(f"Saved new best model with validation accuracy: {val_accuracy:.4f}")
            logging.info(f"Saved new best model with validation accuracy: {val_accuracy:.4f}")

    # Save the final model regardless of performance
    torch.save(model.state_dict(), "emotion_model_final.pth")
    print(f"Saved final model with validation accuracy: {val_accuracy:.4f}")
    logging.info(f"Saved final model with validation accuracy: {val_accuracy:.4f}")
||||||| parent of 0778884 (early stopping)
=======
        early_stopping(val_accuracy, epoch)
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch + 1}")
            break

>>>>>>> 0778884 (early stopping)

    import json
    with open("data.json", "w") as f:
        json.dump(results, f)

    return model, best_val_accuracy


if __name__ == "__main__":
    data = pl.read_csv("data/legend.csv").drop("user.id")
    data = data.with_columns(
        data[:, 1].str.to_lowercase().alias("emotion")
    )
<<<<<<< HEAD
    model, best_accuracy = test_model(data, num_epochs=30)
    print(f"Training complete! Best validation accuracy: {best_accuracy:.4f}")
    # cross_validate(data, k=5, num_epochs=100)
||||||| parent of 0778884 (early stopping)
    test_model(data, num_epochs=30)
    # cross_validate(data, k=5, num_epochs=100)


=======
    test_model(data, num_epochs=100)
    # cross_validate(data, k=5, num_epochs=100)


>>>>>>> 0778884 (early stopping)
