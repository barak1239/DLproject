import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import wandb

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
    # Initialize wandb for this run
    wandb.init(project="Deep Learning", name="baseline_model")

    print("\nDetermining the majority class...")
    majority_class = get_majority_class(df)

    print("\nSplitting data into training and testing sets...")
    train_df, test_df = split_data(df)
    print(f"Training Set: {len(train_df)} samples")
    print(f"Testing Set: {len(test_df)} samples")

    # Log dataset information
    wandb.config.update({
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "majority_class": majority_class
    })

    print("\nMaking baseline predictions...")
    y_true = test_df['dx']
    y_pred = [majority_class] * len(test_df)

    print("\nEvaluating baseline performance...")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Log accuracy to wandb
    wandb.log({"accuracy": accuracy})

    print("\nClassification Report:")
    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
        output_dict=True  # Get results as a dictionary for logging
    )
    print(classification_report(y_true, y_pred, zero_division=0))

    # Log metrics for `nv` and `bcc` (or any other category)
    if "nv" in report:
        wandb.log({
            "nv_precision": report["nv"]["precision"],
            "nv_recall": report["nv"]["recall"],
            "nv_f1-score": report["nv"]["f1-score"]
        })

    if "bcc" in report:
        wandb.log({
            "bcc_recall": report["bcc"]["recall"]
        })


    wandb.finish()

def main(metadata_path):
    print("\nLoading metadata...")
    df = load_metadata(metadata_path)
    if df is None:
        return

    print("\nRunning the baseline model...")
    run_baseline(df)
