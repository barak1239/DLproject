import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Define absolute paths
PROJECT_DIR = r"C:\Users\Barak\PycharmProjects\DLproject"
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")
METADATA_PATH = os.path.join(ARCHIVE_DIR, "HAM10000_metadata.csv")
IMAGES_PART1_DIR = os.path.join(ARCHIVE_DIR, "HAM10000_images_part_1")
IMAGES_PART2_DIR = os.path.join(ARCHIVE_DIR, "HAM10000_images_part_2")

def verify_paths():
    """Verify all required paths exist"""
    paths = {
        "Project Directory": PROJECT_DIR,
        "Archive Directory": ARCHIVE_DIR,
        "Metadata File": METADATA_PATH,
        "Images Part 1": IMAGES_PART1_DIR,
        "Images Part 2": IMAGES_PART2_DIR,
    }

    for name, path in paths.items():
        exists = os.path.exists(path)
        print(f"{name}: {path} {'✓' if exists else '✗'}")

    return all(os.path.exists(path) for path in paths.values())

def load_metadata():
    """Load the HAM10000 metadata."""
    try:
        df = pd.read_csv(METADATA_PATH)
        print("Metadata loaded successfully.")
        print("\nDiagnosis distribution:")
        print(df['dx'].value_counts())
        return df
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return None

def plot_class_distribution(df):
    """Plot class distribution."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='dx')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def run_baseline(df):
    """Run the baseline model pipeline."""
    print("\nDetermining the majority class...")
    majority_class = df['dx'].value_counts().idxmax()
    print(f"Majority Class: {majority_class}")

    print("\nSplitting data into training and testing sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
    print(f"Training Set: {len(train_df)} samples")
    print(f"Testing Set: {len(test_df)} samples")

    print("\nMaking baseline predictions...")
    y_true = test_df['dx']
    y_pred = [majority_class] * len(test_df)

    print("\nEvaluating baseline performance...")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=y_true.unique()))

def main():
    print("Verifying paths...")
    paths_valid = verify_paths()

    if paths_valid:
        print("\nAll paths verified successfully.")

        print("\nLoading metadata...")
        df = load_metadata()
        if df is None:
            return

        print("\nVisualizing data...")
        plot_class_distribution(df)

        print("\nRunning the baseline model...")
        run_baseline(df)
    else:
        print("\nError: One or more paths are invalid. Please check your file structure.")

if __name__ == "__main__":
    main()
