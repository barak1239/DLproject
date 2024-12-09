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
    return df['dx'].value_counts().idxmax()

def split_data(df):
    """Split the dataset into training and testing sets."""
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
    return train_df, test_df

def baseline_prediction(test_df, majority_class):
    """Predict the majority class for all samples."""
    y_true = test_df['dx']
    y_pred = [majority_class] * len(test
