import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

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


def plot_diagnosis_by_gender(df):
    """Plot the distribution of diagnoses by gender"""
    plt.figure(figsize=(14, 7))
    sns.countplot(data=df, x='dx', hue='sex')
    plt.title('Diagnosis Distribution by Gender')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.legend(title='Gender')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_age_distribution(df):
    """Plot age distribution by diagnosis"""
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df, x='dx', y='age')
    plt.title('Age Distribution by Diagnosis')
    plt.xlabel('Diagnosis')
    plt.ylabel('Age')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_class_imbalance(df):
    """Plot a pie chart showing the class imbalance"""
    plt.figure(figsize=(10, 8))
    df['dx'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='Pastel1')
    plt.title('Class Imbalance of Skin Lesion Types')
    plt.ylabel('')  # Remove default y-axis label
    plt.tight_layout()
    plt.show()


def plot_cancerous_vs_age(df):
    """Plot age distribution for cancerous vs non-cancerous lesions"""
    cancerous_labels = ['mel', 'bcc', 'akiec']
    df['is_cancerous'] = df['dx'].apply(lambda x: 'Cancerous' if x in cancerous_labels else 'Non-Cancerous')

    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df, x='is_cancerous', y='age')
    plt.title('Age Distribution: Cancerous vs Non-Cancerous Lesions')
    plt.xlabel('Lesion Type')
    plt.ylabel('Age')
    plt.tight_layout()
    plt.show()


def plot_missing_data(df):
    """Plot a heatmap of missing data"""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Heatmap of Missing Data')
    plt.tight_layout()
    plt.show()


def display_sample_images(df):
    """Display sample images from the dataset"""
    unique_diagnoses = df['dx'].unique()
    num_samples = min(5, len(unique_diagnoses))

    plt.figure(figsize=(15, 10))
    for idx, diagnosis in enumerate(unique_diagnoses[:num_samples]):
        # Get one sample image for each diagnosis
        sample = df[df['dx'] == diagnosis].iloc[0]
        image_id = sample['image_id']

        # Try both image directories
        img_path = os.path.join(IMAGES_PART1_DIR, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGES_PART2_DIR, f"{image_id}.jpg")

        if os.path.exists(img_path):
            img = Image.open(img_path)
            plt.subplot(2, 3, idx + 1)
            plt.imshow(img)
            plt.title(f"Type: {diagnosis}\nID: {image_id}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def load_and_visualize():
    """Load and visualize the HAM10000 dataset"""
    try:
        # Load metadata
        df = pd.read_csv(METADATA_PATH)
        print("\nDataset Summary:")
        print(f"Total images: {len(df)}")
        print("\nDiagnosis distribution:")
        print(df['dx'].value_counts())

        # Generate visualizations
        plot_diagnosis_by_gender(df)
        plot_age_distribution(df)
        plot_class_imbalance(df)
        plot_cancerous_vs_age(df)
        plot_missing_data(df)

        # Display sample images
        display_sample_images(df)

    except Exception as e:
        print(f"Error occurred: {str(e)}")


def main():
    print("Verifying paths...")
    paths_valid = verify_paths()

    if paths_valid:
        print("\nAll paths verified successfully.")
        print("\nAttempting to load and visualize data...")
        load_and_visualize()
    else:
        print("\nError: One or more paths are invalid. Please check your file structure.")


if __name__ == "__main__":
    main()
