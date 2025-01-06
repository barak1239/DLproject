import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
import copy

##############################################################################
# Define your absolute or well-constructed path here:
##############################################################################
PROJECT_DIR = r"C:\Users\Barak\PycharmProjects\DLproject"
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")
CSV_PATH = os.path.join(ARCHIVE_DIR, "hmnist_28_28_L.csv")

# ^ Make sure this file actually exists at this location.

##############################################################################
# Dataset definition
##############################################################################
class SkinDataset(Dataset):
    """
    A PyTorch Dataset for loading features & labels from arrays or a DataFrame.
    """

    def __init__(self, features, labels):
        """
        Args:
            features (ndarray): shape (num_samples, num_features).
            labels   (ndarray): shape (num_samples,).
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

##############################################################################
# Model definition
##############################################################################
class NeuralNetworkModel(nn.Module):
    """
    A neural network with three layers and dropout.
    """

    def __init__(self, input_dim, num_classes):
        super(NeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # Input layer to first hidden layer
        self.relu1 = nn.ReLU()  # Activation function for first layer
        self.dropout1 = nn.Dropout(0.5)  # Dropout for first hidden layer
        self.fc2 = nn.Linear(256, 64)  # First hidden layer to second hidden layer
        self.relu2 = nn.ReLU()  # Activation function for second layer
        self.dropout2 = nn.Dropout(0.5)  # Dropout for second hidden layer
        self.fc3 = nn.Linear(64, num_classes)  # Second hidden layer to output layer

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

##############################################################################
# Main function (called from your main.py)
##############################################################################
def main(csv_path=CSV_PATH):
    """
    Main entry point for training and testing a neural network model.
    """

    # ------------------------------------------------------------
    # 1. Load your CSV data using the provided or default csv_path
    # ------------------------------------------------------------
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at: {csv_path}")
        return

    # Check if 'label' column exists
    if 'label' not in df.columns:
        print("Error: 'label' column not found in the CSV. Please check your dataset.")
        return

    # Separate features and labels
    features = df.drop('label', axis=1).values
    labels = df['label'].values

    # Normalize features manually
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    print(f"Loaded dataset from: {csv_path}")
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    # ------------------------------------------------------------
    # 2. Create Dataset and DataLoader
    # ------------------------------------------------------------
    dataset = SkinDataset(features, labels)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------
    # 3. Define the Neural Network model
    # ------------------------------------------------------------
    input_dim = features.shape[1]  # e.g., 784 for 28x28 grayscale
    num_classes = 10  # Assuming 10 output classes

    model = NeuralNetworkModel(input_dim, num_classes)

    # ------------------------------------------------------------
    # 4. Define loss function & optimizer
    # ------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------------------------------------------
    # 5. Training loop with early stopping
    # ------------------------------------------------------------
    num_epochs = 30
    print("\nStarting training...")
    best_model = None
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_features.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_features.size(0)
                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

                all_labels.extend(batch_labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        val_recall = recall_score(all_labels, all_predictions, average="macro")

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%, "
              f"Val Recall: {val_recall:.2f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            print ("model got better !!!!")
    # ------------------------------------------------------------
    # 6. Testing on the test dataset
    # ------------------------------------------------------------
    print("\nStarting testing...")
    model = best_model  # Load the best model
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    recall = recall_score(all_labels, all_predictions, average="macro")

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Recall: {recall:.2f}")

    print("\nNeural Network run completed.")
