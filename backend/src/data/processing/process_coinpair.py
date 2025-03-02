from backend.src.shared.config.data_processing_config import timeframes, target_timeframe
from backend.src.data.staging.get_csv_paths import get_csv_paths
from backend.src.data.ingestion.ingest_csv import ingest_csv
from backend.src.shared.config.global_config import data_folder
from backend.src.data.processing.process_timeframes import process_timeframe
from backend.src.data.processing.merge_timeframes import merge_timeframes

def process_coinpair(coin, pair):
    data_dict = {}
   
    csv_paths = get_csv_paths(data_folder, coin, pair, timeframes)

    for timeframe in csv_paths[coin][pair]['timeframes']:
        data = ingest_csv(csv_paths[coin][pair]['timeframes'][timeframe])
        processed_data = process_timeframe(data, timeframe, target_timeframe)
        data_dict[timeframe] = processed_data
        

    merged_data = merge_timeframes(data_dict, target_timeframe)

    return merged_data


