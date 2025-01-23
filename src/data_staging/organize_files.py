import os
import shutil
from .config import source_of_truth_data_folder, data_folder, base_assets

def organize_data_by_multiple_coins(source_folder:str, base_assets:list, target_folder:str, overwrite:bool=False):
    """
    Copies CSV files from a source of truth (SOT) data folder to a target working data folder, organizing them by coin and pair.

    :param source_folder: Path to the source folder containing all CSV files (source of truth).
    :param base_assets: List of base cryptocurrencies to organize (e.g., ["XRP", "ETH"]).
    :param target_folder: Path to the folder where the organized working data will be copied.
    :param overwrite: If True, overwrite existing data in non-empty folders. Defaults to False (skip functionality).
    """
    excluded_assets = ["ETH2"]  # Add coins you want to exclude here

    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    for base_asset in base_assets:
        # Create the base asset folder in the target directory
        base_folder = os.path.join(target_folder, base_asset)

        # Check if the base folder exists
        if not os.path.exists(base_folder):
            print(f"Creating folder for {base_asset}: {base_folder}")
            os.makedirs(base_folder)
        else:
            # Check if base folder is not empty
            if len(os.listdir(base_folder)) > 0 and not overwrite:
                print(f"Folder for {base_asset} exists and is not empty. Skipping.")
                continue  # Skip this base asset if overwrite is not set

        # Iterate through files in the source data folder
        for filename in os.listdir(source_folder):
            if filename.endswith(".csv"):
                # Extract the pair name (e.g., ETHUSD or ETH2BTC)
                pair = filename.split("_")[0]

                # Skip excluded assets
                if any(excluded_asset in pair for excluded_asset in excluded_assets):
                    print(f"Skipping excluded pair: {pair}")
                    continue

                # Ensure the base asset is the FIRST part of the pair
                if pair.startswith(base_asset):  # Only process pairs like ETHCRV, not CRVETH
                    # Create a folder for the pair inside the base folder
                    pair_folder = os.path.join(base_folder, pair)
                    os.makedirs(pair_folder, exist_ok=True)

                    # Copy the file to the pair folder in the target directory
                    src_path = os.path.join(source_folder, filename)
                    dest_path = os.path.join(pair_folder, filename)

                    # Handle overwriting logic
                    if os.path.exists(dest_path) and not overwrite:
                        print(f"File {dest_path} already exists. Skipping.")
                    else:
                        shutil.copy2(src_path, dest_path)  # Use copy2 to preserve metadata
                        print(f"Copied {filename} to {dest_path}")

if __name__ == "__main__":
    organize_data_by_multiple_coins(
        source_folder=source_of_truth_data_folder,
        base_assets=base_assets,
        target_folder=data_folder
    )
