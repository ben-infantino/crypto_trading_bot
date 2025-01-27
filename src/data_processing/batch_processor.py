from src.data_processing.configs import coin_pairs, timeframes
from src.data_staging.get_csv_paths import get_csv_paths
from src.data_ingestion.ingest_csv import ingest_csv
from src.shared.config import data_folder
from src.data_processing.single_processor import process_single_file

def process_batch():
    for coin, pair in coin_pairs.items():

        csv_paths = get_csv_paths(data_folder, coin, pair, timeframes = [1])

        for timeframe in csv_paths[coin][pair]['timeframes']:
            data = ingest_csv(csv_paths[coin][pair]['timeframes'][timeframe])
            processed_data = process_single_file(data, timeframe)
            print(processed_data.tail())



      # for path in csv_paths:
      #     data = ingest_csv(path)
      #     processed_data = process_single_file(data)
      #     return processed_data

if __name__ == "__main__":
    process_batch()

