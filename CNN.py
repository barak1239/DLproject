import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
import wandb
import copy
from torchvision import transforms

PROJECT_DIR = r"C:\Users\Barak\PycharmProjects\DLproject"
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")
CSV_PATH = os.path.join(ARCHIVE_DIR, "hmnist_28_28_RGB.csv")

# Dataset definition

class SkinDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features.reshape(-1, 3, 28, 28)  # Reshape to 3x28x28 for RGB
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# CNN Model definition

class ImprovedCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Main function
def main(csv_path=CSV_PATH):
    # Initialize WandB
    wandb.init(project="Deep Learning", name="improved_cnn_model")
    wandb.config.update({
        "batch_size": 64,
        "learning_rate": 1e-3,
        "num_epochs": 50,
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

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = SkinDataset(features, labels, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_classes = 10
    model = ImprovedCNNModel(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    num_epochs = 50
    best_model = None
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

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

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_features.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += batch_labels.size(0)
                correct_val += (predicted == batch_labels).sum().item()

                all_labels.extend(batch_labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct_val / total_val
        val_precision = precision_score(all_labels, all_predictions, average="macro")
        val_recall = recall_score(all_labels, all_predictions, average="macro")
        scheduler.step(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%, "
              f"Val Precision: {val_precision:.2f}, "
              f"Val Recall: {val_recall:.2f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = copy.deepcopy(model)

    wandb.log({"best_val_accuracy": best_val_accuracy})

    # Test
    model = best_model
    correct_test = 0
    total_test = 0
    all_test_labels = []
    all_test_predictions = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            total_test += batch_labels.size(0)
            correct_test += (predicted == batch_labels).sum().item()

            all_test_labels.extend(batch_labels.cpu().numpy())
            all_test_predictions.extend(predicted.cpu().numpy())

    test_accuracy = 100 * correct_test / total_test
    test_precision = precision_score(all_test_labels, all_test_predictions, average="macro")
    test_recall = recall_score(all_test_labels, all_test_predictions, average="macro")

    print(f"Test Accuracy: {test_accuracy:.2f}%, "
          f"Test Precision: {test_precision:.2f}, "
          f"Test Recall: {test_recall:.2f}")

    wandb.log({
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall
    })

    wandb.finish()

if __name__ == "__main__":
    main()
