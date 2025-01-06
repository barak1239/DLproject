import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# For method (6) - Stratified split
from sklearn.model_selection import StratifiedShuffleSplit

##############################################################################
# Define paths
##############################################################################
PROJECT_DIR = r"C:\Users\Barak\PycharmProjects\DLproject"
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")
CSV_PATH = os.path.join(ARCHIVE_DIR, "hmnist_28_28_RGB.csv")

##############################################################################
# Custom Dataset
##############################################################################
class SkinDataset(Dataset):
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
# Logistic Regression Model
##############################################################################
class LogisticRegressionModel(nn.Module):
    """Single linear layer: input_dim -> num_classes."""
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

##############################################################################
# Main function
##############################################################################
def main(csv_path=CSV_PATH,
         test_size=0.2,        # 20% val split
         patience=5,           # Early stopping patience
         max_epochs=20,        # Up to 50 epochs
         lr=5e-4,              # Learning rate
         weight_decay=1e-4,    # L2 regularization
         batch_size=64):

    # 1. Load CSV data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at: {csv_path}")
        return

    if 'label' not in df.columns:
        print("Error: 'label' column not found in the CSV. Please check your dataset.")
        return

    # Separate features & labels
    all_features = df.drop('label', axis=1).values.astype(np.float32)
    all_features /= 255.0  # normalize pixel values to [0, 1]

    all_labels = df['label'].values

    print(f"Loaded dataset from: {csv_path}")
    print("Features shape:", all_features.shape)  # e.g. (N, 2352) for 28×28 RGB
    print("Labels shape:", all_labels.shape)

    # 2. Stratified train/val split (method 6)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_idx, val_idx in sss.split(all_features, all_labels):
        train_features_raw, val_features_raw = all_features[train_idx], all_features[val_idx]
        train_labels_raw, val_labels_raw = all_labels[train_idx], all_labels[val_idx]

    # Create PyTorch Datasets from the raw features (no PCA)
    train_dataset = SkinDataset(train_features_raw, train_labels_raw)
    val_dataset   = SkinDataset(val_features_raw,   val_labels_raw)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # 3. Define logistic regression model
    input_dim = train_features_raw.shape[1]  # e.g. 2352 for 28×28 RGB
    num_classes = len(np.unique(all_labels))
    model = LogisticRegressionModel(input_dim, num_classes)

    # Define loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Training with early stopping (method 4)
    print("\nStarting training with early stopping (no PCA)...")

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
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
        with torch.no_grad():
            for val_features_batch, val_labels_batch in val_loader:
                val_outputs = model(val_features_batch)
                _, predicted = torch.max(val_outputs, 1)
                total += val_labels_batch.size(0)
                correct += (predicted == val_labels_batch).sum().item()

        val_accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch}/{max_epochs}] - "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")

        # Early stopping check
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

    # 5. Single sample test (optional)
    if len(val_dataset) > 0:
        model.eval()
        sample_idx = 0
        sample_features, sample_label = val_dataset[sample_idx]
        sample_features = sample_features.unsqueeze(0)

        with torch.no_grad():
            output = model(sample_features)
            _, predicted_class = torch.max(output, 1)

        print(f"\nValidation sample index [{sample_idx}]")
        print("Predicted class:", predicted_class.item())
        print("Ground truth class:", sample_label.item())

    print(f"\nBest validation accuracy achieved: {best_val_acc:.2f}%")
    print("Training + early stopping run completed.")

if __name__ == "__main__":
    # Example usage:
    main(
        csv_path=CSV_PATH,
        test_size=0.2,
        patience=5,
        max_epochs=50,
        lr=1e-4,
        weight_decay=1e-5,
        batch_size=64
    )
