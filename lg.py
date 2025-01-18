import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import wandb  # ייבוא WandB

##############################################################################
# Custom Dataset
##############################################################################
PROJECT_DIR = r"C:\Users\Barak\PycharmProjects\DLproject"
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")
CSV_PATH = os.path.join(ARCHIVE_DIR, "hmnist_28_28_RGB.csv")

class SkinDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

##############################################################################
# Logistic Regression Model
##############################################################################
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

##############################################################################
# Main function
##############################################################################
def main(csv_path=CSV_PATH,
         test_size=0.2,        # 20% test split
         patience=5,           # Early stopping patience
         max_epochs=20,        # Up to 50 epochs
         lr=5e-4,              # Learning rate
         weight_decay=1e-4,    # L2 regularization
         batch_size=64):

    # Initialize WandB
    wandb.init(project="Deep Learning", name="logistic_regression_model")
    wandb.config.update({
        "test_size": test_size,
        "patience": patience,
        "max_epochs": max_epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size
    })

    # 1. Load CSV data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at: {csv_path}")
        return

    if 'label' not in df.columns:
        print("Error: 'label' column not found in the CSV. Please check your dataset.")
        return

    all_features = df.drop('label', axis=1).values.astype(np.float32)
    all_features /= 255.0
    all_labels = df['label'].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_idx, test_idx in sss.split(all_features, all_labels):  # שינוי val ל-test
        train_features_raw, test_features_raw = all_features[train_idx], all_features[test_idx]
        train_labels_raw, test_labels_raw = all_labels[train_idx], all_labels[test_idx]

    train_dataset = SkinDataset(train_features_raw, train_labels_raw)
    test_dataset = SkinDataset(test_features_raw, test_labels_raw)  # test במקום val

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # test במקום val

    input_dim = train_features_raw.shape[1]
    num_classes = len(np.unique(all_labels))
    model = LogisticRegressionModel(input_dim, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs_completed = 0  # Counter for completed epochs

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)

            # Accuracy for train
            _, predicted = torch.max(outputs, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100.0 * correct_train / total_train
        train_error = 100.0 - train_accuracy  # Calculate train error

        print(f"Epoch [{epoch}/{max_epochs}] - "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Train Error: {train_error:.2f}%")

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_accuracy": train_accuracy,
            "train_error": train_error
        })

        # Early stopping counter
        epochs_completed += 1

    # Test Evaluation
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for test_features_batch, test_labels_batch in test_loader:
            test_outputs = model(test_features_batch)
            _, predicted = torch.max(test_outputs, 1)
            total_test += test_labels_batch.size(0)
            correct_test += (predicted == test_labels_batch).sum().item()

    test_accuracy = 100.0 * correct_test / total_test
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")

    # Log Test Accuracy and completed epochs to WandB
    wandb.log({
        "test_accuracy": test_accuracy,
        "epochs_completed": epochs_completed
    })
    wandb.finish()

if __name__ == "__main__":
    main(
        csv_path=CSV_PATH,
        test_size=0.2,
        patience=5,
        max_epochs=50,
        lr=1e-4,
        weight_decay=1e-5,
        batch_size=64
    )
