import os
import cv2
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from baseline import run_baseline, load_metadata
from lg import main as run_logistic_regression
from basicNN import main as run_basicNN
from CNN import run_CNN, generate_gradcam

###############################################################################
#                          Project Paths and Settings                         #
###############################################################################

PROJECT_DIR = r"C:\Users\Barak\PycharmProjects\DLproject"
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")
METADATA_PATH = os.path.join(ARCHIVE_DIR, "HAM10000_metadata.csv")
IMAGES_PART1_DIR = os.path.join(ARCHIVE_DIR, "HAM10000_images_part_1")
IMAGES_PART2_DIR = os.path.join(ARCHIVE_DIR, "HAM10000_images_part_2")  # Fixed parenthesis

###############################################################################
#                            Visualization Functions                          #
###############################################################################

def visualize_class_distribution(df):
    """Visualize the diagnosis distribution from metadata."""
    diagnosis_counts = df['dx'].value_counts()
    plt.figure(figsize=(8, 5))
    diagnosis_counts.plot(kind='bar')
    plt.title("Diagnosis Distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.show()

def visualize_sample_images_by_class(df, num_samples=3):
    """
    Display a few sample images per lesion class, each row having a single title
    (suptitle) indicating the class (dx).

    Assumes:
      - df has columns: 'image_id' (filename stem) and 'dx' (lesion class)
      - Images are stored in IMAGES_PART1_DIR or IMAGES_PART2_DIR
    """
    # Group by 'dx' to get each class
    grouped = df.groupby('dx')

    for dx_class, group_df in grouped:
        # Randomly sample up to 'num_samples' images from this class
        sampled_rows = group_df.sample(min(num_samples, len(group_df)), random_state=42)

        # Create a figure with as many subplots as we have sampled images
        fig, axes = plt.subplots(1, len(sampled_rows), figsize=(5 * len(sampled_rows), 4))
        # Add a single suptitle for this row of images
        fig.suptitle(f"Class: {dx_class}", fontsize=14)

        # If there's only one sample, make axes a list so we can iterate uniformly
        if len(sampled_rows) == 1:
            axes = [axes]

        for ax, (_, row) in zip(axes, sampled_rows.iterrows()):
            image_id = row['image_id']
            # Attempt to load the image from part1 or part2
            img_path = os.path.join(IMAGES_PART1_DIR, f"{image_id}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(IMAGES_PART2_DIR, f"{image_id}.jpg")

            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(f"ID: {image_id}")
                ax.axis("off")
            else:
                ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
                ax.axis("off")

        plt.tight_layout()
        plt.show()

def visualize_metadata_distributions(df):
    """
    Show distributions for columns like 'age', 'sex', 'localization', etc.,
    if they exist in the metadata.
    """
    if 'age' in df.columns:
        plt.figure(figsize=(8, 5))
        df['age'].dropna().hist(bins=20, edgecolor='black')
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.show()

    if 'sex' in df.columns:
        plt.figure(figsize=(6, 4))
        df['sex'].value_counts().plot(kind='bar')
        plt.title("Sex Distribution")
        plt.xlabel("Sex")
        plt.ylabel("Count")
        plt.show()

    if 'localization' in df.columns:
        plt.figure(figsize=(10, 5))
        df['localization'].value_counts().plot(kind='bar')
        plt.title("Localization Distribution")
        plt.xlabel("Localization")
        plt.ylabel("Count")
        plt.show()

def display_gradcam(input_image, cam, alpha=0.5):
    """
    Display the original image, Grad-CAM heatmap, and overlay.
    This central visualization supports our hypothesis by showing that the CNN
    attends to clinically relevant regions.
    """
    # Convert input image to numpy array and adjust dimensions (H x W x C)
    image = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    # Create heatmap using OpenCV
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap * alpha + image * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(cam, cmap='jet')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()

def visualize_confusion_matrix(y_true, y_pred, class_labels):
    """
    Plot and display a confusion matrix given the true labels, predicted labels,
    and a list of class names.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix")
    plt.show()

def display_three_images_same_class(image_ids, class_name):
    """
    Displays exactly three images in a single row, each labeled with its image_id,
    and a single suptitle indicating the class.

    :param image_ids: List of exactly 3 image IDs (strings, no '.jpg')
    :param class_name: A string for the class (e.g. "mel" or "Melanoma")
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Class: {class_name}", fontsize=16)

    for ax, image_id in zip(axes, image_ids):
        # Check part1, then part2
        img_path = os.path.join(IMAGES_PART1_DIR, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGES_PART2_DIR, f"{image_id}.jpg")

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f"ID: {image_id}")
        else:
            ax.text(0.5, 0.5, f"Image {image_id} not found", ha='center', va='center')
        ax.axis("off")

    plt.tight_layout()
    plt.show()

###############################################################################
#                          Path Verification & main()                         #
###############################################################################

def verify_paths():
    """Verify that all required paths exist."""
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

def main():
    print("Verifying paths...")
    if not verify_paths():
        print("\nError: One or more paths are invalid. Please check your file structure.")
        return

    print("\nAll paths verified successfully.")

    # Load metadata and visualize the class distribution
    print("\nLoading metadata...")
    df = load_metadata(METADATA_PATH)
    if df is None:
        return

    # 1. Class distribution
    print("\nVisualizing metadata class distribution...")
    visualize_class_distribution(df)

    # 2. Sample images by class (with a single suptitle above each row)
    if 'image_id' in df.columns:
        print("\nVisualizing sample images by class...")
        visualize_sample_images_by_class(df, num_samples=3)

    # 3. Other metadata distributions
    print("\nVisualizing other metadata distributions...")
    visualize_metadata_distributions(df)

    # Print high-level hypothesis explanation
    print("\nHypothesis: Different ML models capture different levels of data representation.")
    print("Simpler models (e.g., logistic regression) capture global linear features, whereas deep models,")
    print("especially CNNs, uncover complex hierarchical features that align with clinically relevant regions.")
    print("These visualizations help us confirm or refute these theoretical preconceptions.\n")

    # Example usage of the new function (uncomment if you want to test):
    # three_ids = ["ISIC_0025438", "ISIC_0029881", "ISIC_0026399"]
    # display_three_images_same_class(three_ids, "mel (Melanoma)")

    # Menu for selecting the model to run
    print("Choose an option:")
    print("1. Run Baseline Model")
    print("2. Run Logistic Regression Model")
    print("3. Run Basic Neural Network")
    print("4. Run Complex CNN Model (with Grad-CAM visualization)")
    choice = input("Enter your choice (1/2/3/4): ")

    if choice == "1":
        print("\nRunning the baseline model...")
        run_baseline(df)

    elif choice == "2":
        print("\nRunning the logistic regression model...")
        run_logistic_regression()

    elif choice == "3":
        print("\nRunning the basic Neural Network model...")
        run_basicNN()

    elif choice == "4":
        print("\nRunning the complex CNN model...")
        best_model, sample_img = run_CNN()
        if best_model is None or sample_img is None:
            print("Error: CNN model training did not return a valid model or sample image.")
            return

        # Use the CNN module's Grad-CAM function for visualization
        sample_img_for_gradcam = sample_img[:1]  # Take one image from the test set
        output = best_model(sample_img_for_gradcam)
        target_class = torch.argmax(output, dim=1).item()
        print(f"\nGrad-CAM: Visualizing for predicted class {target_class}")
        cam = generate_gradcam(best_model, sample_img_for_gradcam, target_class, best_model.conv3)
        display_gradcam(sample_img_for_gradcam, cam)

    else:
        print("\nInvalid choice. Exiting.")

if __name__ == "__main__":
    main()
