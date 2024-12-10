import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

def load_metadata(metadata_path):
    """Load the HAM10000 metadata."""
    try:
        df = pd.read_csv(metadata_path)
        print("Metadata loaded successfully.")
        print("\nDiagnosis distribution:")
        print(df['dx'].value_counts())
        return df
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return None

def get_majority_class(df):
    """Get the majority class in the dataset."""
    majority_class = df['dx'].value_counts().idxmax()
    print(f"Majority Class: {majority_class}")
    return majority_class

def split_data(df):
    """Split the dataset into training and testing sets."""
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
    return train_df, test_df

def run_baseline(df):
    """Run the baseline model pipeline."""
    print("\nDetermining the majority class...")
    majority_class = get_majority_class(df)

    print("\nSplitting data into training and testing sets...")
    train_df, test_df = split_data(df)
    print(f"Training Set: {len(train_df)} samples")
    print(f"Testing Set: {len(test_df)} samples")

    # Check class distribution in training and testing sets
    print("\nClass distribution in training set:")
    print(train_df['dx'].value_counts())
    print("\nClass distribution in testing set:")
    print(test_df['dx'].value_counts())

    print("\nMaking baseline predictions...")
    y_true = test_df['dx']
    y_pred = [majority_class] * len(test_df)

    # Debug: Verify predictions
    print(f"\nExample predictions: {y_pred[:10]}")
    print(f"Predicted class distribution: {pd.Series(y_pred).value_counts()}")

    print("\nEvaluating baseline performance...")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=test_df['dx'].unique(),  # Ensure labels match true test labels
        target_names=test_df['dx'].unique(),  # Align target names with test labels
        zero_division=0
    ))

def main(metadata_path):
    print("\nLoading metadata...")
    df = load_metadata(metadata_path)
    if df is None:
        return

    print("\nRunning the baseline model...")
    run_baseline(df)
