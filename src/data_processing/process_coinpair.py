from src.data_processing.configs import timeframes, target_timeframe
from src.data_staging.get_csv_paths import get_csv_paths
from src.data_ingestion.ingest_csv import ingest_csv
from src.shared.config import data_folder
from src.data_processing.process_timeframes import process_timeframe
from src.data_processing.merge_timeframes import merge_timeframes

def process_coinpair(coin, pair):
    data_dict = {}
   
    csv_paths = get_csv_paths(data_folder, coin, pair, timeframes)

    for timeframe in csv_paths[coin][pair]['timeframes']:
        data = ingest_csv(csv_paths[coin][pair]['timeframes'][timeframe])
        processed_data = process_timeframe(data, timeframe, target_timeframe)
        data_dict[timeframe] = processed_data
        
    merged_data = merge_timeframes(data_dict, target_timeframe)

    return merged_data


