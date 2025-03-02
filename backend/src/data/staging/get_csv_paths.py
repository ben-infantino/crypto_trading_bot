import os

def get_csv_paths(base_folder, coin:str=None, pair:str=None, timeframes:list=None):
    """
    Crawls the data folder and returns a nested dictionary of file paths grouped by coin, pair, and timeframe.

    :param base_folder: Path to the data folder (e.g., "E:/TRADER_1_DATA/data").
    :param coin: The coin to filter by (e.g., "XRP"). If None, includes all coins.
    :param pair: The specific pair to filter by (e.g., "XRPETH"). If None, includes all pairs.
    :param timeframes: List of timeframes to filter by (e.g., [1, 15, 1440]). If None, includes all timeframes.
    :return: Nested dictionary of file paths grouped by coin, pair, and timeframe.
    """
    csv_paths = {}

    if timeframes is not None:
        # Ensure timeframes are strings for comparison
        timeframes = set(str(t) for t in timeframes)

    # Walk through the folder structure
    for coin_folder in os.listdir(base_folder):
        coin_path = os.path.join(base_folder, coin_folder)

        # Filter by coin
        if coin and coin_folder != coin:
            continue

        if os.path.isdir(coin_path):
            csv_paths[coin_folder] = {}

            for pair_folder in os.listdir(coin_path):
                pair_path = os.path.join(coin_path, pair_folder)

                # Filter by pair
                if pair and pair_folder != pair:
                    continue

                if os.path.isdir(pair_path):
                    # Initialize pair dictionary with timeframes key
                    csv_paths[coin_folder][pair_folder] = {'timeframes': {}}

                    for file_name in os.listdir(pair_path):
                        file_path = os.path.join(pair_path, file_name)
                        if "_" in file_name:
                            _, timeframe = file_name.split("_")
                            timeframe = timeframe.split(".")[0]  # Remove ".csv"

                            # Filter by timeframes
                            if timeframes and timeframe not in timeframes:
                                continue

                            # Store the file path under the timeframes key
                            csv_paths[coin_folder][pair_folder]['timeframes'][int(timeframe)] = file_path

    return csv_paths