import os
from baseline import run_baseline, load_metadata
from lg import main as run_logistic_regression
from basicNN import main as run_basicNN
# Define absolute paths
PROJECT_DIR = r"C:\Users\Barak\PycharmProjects\DLproject"
ARCHIVE_DIR = os.path.join(PROJECT_DIR, "archive")
METADATA_PATH = os.path.join(ARCHIVE_DIR, "HAM10000_metadata.csv")
IMAGES_PART1_DIR = os.path.join(ARCHIVE_DIR, "HAM10000_images_part_1")
IMAGES_PART2_DIR = os.path.join(ARCHIVE_DIR, "HAM10000_images_part_2")

def verify_paths():
    """Verify all required paths exist."""
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
    paths_valid = verify_paths()

    if paths_valid:
        print("\nAll paths verified successfully.")

        print("\nLoading metadata...")
        df = load_metadata(METADATA_PATH)
        if df is None:
            return

        # Menu to choose functionality
        print("\nChoose an option:")
        print("1. Run Baseline Model")
        print("2. Run Logistic Regression Model")
        print("3. Run basic Neural Network ")
        print("4. Run complex Neural Network model ")
        choice = input("Enter your choice (1/4): ")

        if choice == "1":
            print("\nRunning the baseline model")
            run_baseline(df)
        elif choice == "2":
            print("\nRunning the logistic regression model")
            run_logistic_regression()
        elif choice == "3" :
            print("\n Running the basic Neural Network model")
            run_basicNN()
        elif choice == "4":
            print("\n Running the complex Neural Network model")
        else:
            print("\nInvalid choice. Exiting.")
    else:
        print("\nError: One or more paths are invalid. Please check your file structure.")

if __name__ == "__main__":
    main()
