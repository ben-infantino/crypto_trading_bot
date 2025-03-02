from backend.src.data.processing.process_coinpair import process_coinpair
from backend.src.shared.config.data_processing_config import coin_pairs
from backend.src.models.xgboost.train_tune import run_optuna_study_timeseries
import pandas as pd
if __name__ == "__main__":
    all_results = []  # List to accumulate all folds' backtest summaries

    for coin, pair in coin_pairs.items():
        print(f"Processing coin: {coin}")
        data = process_coinpair(coin, pair)
        # remove the last 10% of data from pandas df to save for out of sample testing
        test_data = data[-int(len(data) * 0.1):]
        train_data = data[:-int(len(data) * 0.1)]
        # XGBoost Model
        study = run_optuna_study_timeseries(train_data)

        from backend.src.models.xgboost.train_tune import train_and_test_XGBoost
        
        print(f"Finished tuning and now testing the model on out of sample data.")
        train_and_test_XGBoost(train_data, test_data, extra_params=study.best_params)

