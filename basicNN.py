import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score
import wandb  # ייבוא WandB
import copy


PROJECT_DIR = r"C:\Users\Barak\PycharmProjects\DLproject"
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")
CSV_PATH = os.path.join(ARCHIVE_DIR, "hmnist_28_28_RGB.csv")

# Dataset definition
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

# Model definition
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Main function
def main(csv_path=CSV_PATH):
    # Initialize WandB
    wandb.init(project="Deep Learning", name="neural_network_model")
    wandb.config.update({
        "batch_size": 64,
        "learning_rate": 1e-3,
        "num_epochs": 30,
    })

    # Load CSV data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at: {csv_path}")
        return

    if 'label' not in df.columns:
        print("Error: 'label' column not found in the CSV.")
        return

    features = df.drop('label', axis=1).values
    labels = df['label'].values
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    dataset = SkinDataset(features, labels)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_dim = features.shape[1]
    num_classes = 10

    model = NeuralNetworkModel(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30
    best_model = None
    best_val_accuracy = 0.0  # Track best validation accuracy

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        all_preds_train = []
        all_labels_train = []

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_features.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()

            all_preds_train.extend(predicted.cpu().numpy())
            all_labels_train.extend(batch_labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        train_error = 100 - train_accuracy

        train_precision = precision_score(all_labels_train, all_preds_train, average='macro')
        train_recall = recall_score(all_labels_train, all_preds_train, average='macro')

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        all_preds_val = []
        all_labels_val = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_features.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += batch_labels.size(0)
                correct_val += (predicted == batch_labels).sum().item()

                all_preds_val.extend(predicted.cpu().numpy())
                all_labels_val.extend(batch_labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct_val / total_val

        val_precision = precision_score(all_labels_val, all_preds_val, average='macro')
        val_recall = recall_score(all_labels_val, all_preds_val, average='macro')

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Error: {train_error:.2f}%, "
              f"Train Precision: {train_precision:.4f}, "
              f"Train Recall: {train_recall:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%, "
              f"Val Precision: {val_precision:.4f}, "
              f"Val Recall: {val_recall:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_error": train_error,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
        })

        # Track best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(model)

    # Log Best Validation Accuracy
    wandb.log({"best_val_accuracy": best_val_accuracy})

    # Test
    model = best_model
    correct_test = 0
    total_test = 0

    all_preds_test = []
    all_labels_test = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            total_test += batch_labels.size(0)
            correct_test += (predicted == batch_labels).sum().item()

            all_preds_test.extend(predicted.cpu().numpy())
            all_labels_test.extend(batch_labels.cpu().numpy())

    test_accuracy = 100 * correct_test / total_test
    test_error = 100 - test_accuracy

    test_precision = precision_score(all_labels_test, all_preds_test, average='macro')
    test_recall = recall_score(all_labels_test, all_preds_test, average='macro')

    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Error: {test_error:.2f}%")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    wandb.log({
        "test_error": test_error,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
    })

    wandb.finish()

if __name__ == "__main__":
    main()
