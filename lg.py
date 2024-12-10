import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from PIL import Image

# Define paths
PROJECT_DIR = r"C:\Users\Barak\PycharmProjects\DLproject"
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")
METADATA_PATH = os.path.join(ARCHIVE_DIR, "HAM10000_metadata.csv")
IMAGES_PART1_DIR = os.path.join(ARCHIVE_DIR, "HAM10000_images_part_1")
IMAGES_PART2_DIR = os.path.join(ARCHIVE_DIR, "HAM10000_images_part_2")

def load_metadata():
    """Load metadata and return as a DataFrame."""
    try:
        df = pd.read_csv(METADATA_PATH)
        print("Metadata loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

def load_images(df, image_size=(64, 64)):
    """Load images as flattened arrays and return feature matrix."""
    print("Loading images...")
    image_data = []
    labels = []

    for index, row in df.iterrows():
        image_id = row['image_id']
        dx = row['dx']

        # Check both image directories
        img_path = os.path.join(IMAGES_PART1_DIR, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGES_PART2_DIR, f"{image_id}.jpg")

        if os.path.exists(img_path):
            # Open, resize, and flatten the image
            img = Image.open(img_path).resize(image_size)
            image_data.append(np.array(img).flatten())
            labels.append(dx)
        else:
            print(f"Image {image_id} not found.")

    print(f"Loaded {len(image_data)} images.")
    return np.array(image_data), np.array(labels)

def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model."""
    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=2000, solver='saga')
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def apply_pca(X_train, X_test, n_components=500):
    """Apply PCA to reduce dimensionality."""
    print(f"Applying PCA to reduce dimensionality to1 {n_components} components...")
    pca = PCA(n_components=1024)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    return X_train_pca, X_test_pca

def main():
    print("Loading metadata...")
    df = load_metadata()
    if df is None:
        return

    # Split the dataset
    print("Splitting data into training and testing sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

    # Load images and labels
    X_train, y_train = load_images(train_df, image_size=(32, 32))  # Higher resolution
    X_test, y_test = load_images(test_df, image_size=(32, 32))

    # Normalize image data
    X_train = X_train / 255.0  # Scale pixel values to [0, 1]
    X_test = X_test / 255.0

    # Apply PCA for dimensionality reduction
    X_train_pca, X_test_pca = apply_pca(X_train, X_test, n_components=100)

    # Train and evaluate logistic regression
    model = train_logistic_regression(X_train_pca, y_train)
    evaluate_model(model, X_test_pca, y_test)

if __name__ == "__main__":
    main()
