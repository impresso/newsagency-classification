import os
import shutil
import argparse
from tqdm import tqdm


# Function to collect target directories
def collect_dirs(parent_dir):
    target_dirs = []
    for root, dirs, _ in os.walk(parent_dir):
        for dir_name in dirs:
            if dir_name.startswith("model_"):
                target_dirs.append(os.path.join(root, dir_name))
    return target_dirs


# Main function to perform the copying
def copy_files(parent_dir, new_folder):
    os.makedirs(new_folder, exist_ok=True)

    # Collect the directories to process
    dirs_to_process = collect_dirs(parent_dir)

    # Files to skip
    skip_files = {
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "label_map.json",
    }

    # Copy .json and .tsv files to the new folder
    for dir_path in tqdm(dirs_to_process, desc="Processing directories"):
        # Create corresponding directory in new folder
        new_dir = os.path.join(new_folder, os.path.relpath(dir_path, parent_dir))
        os.makedirs(new_dir, exist_ok=True)

        # List all files in the current directory
        for root, _, files in os.walk(dir_path):
            for file in tqdm(
                files,
                desc=f"Copying files in {os.path.basename(dir_path)}",
                leave=False,
            ):
                if (
                    file.endswith(".json") or file.endswith(".tsv")
                ) and file not in skip_files:
                    # Construct full file path
                    file_path = os.path.join(root, file)
                    # Copy file to new directory
                    shutil.copy(file_path, new_dir)

    print("Copying completed successfully.")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Copy .json and .tsv files to a new directory."
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        required=True,
        help="The parent directory to search for target directories.",
    )
    parser.add_argument(
        "--new_folder",
        type=str,
        required=True,
        help="The new folder where the results will be copied.",
    )

    args = parser.parse_args()

    # Run the main function with provided arguments
    copy_files(args.parent_dir, args.new_folder)
