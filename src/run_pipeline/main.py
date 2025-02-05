from src.data_processing.process_coinpair import process_coinpair
from src.data_processing.configs import coin_pairs
from src.data_modeling.XGBoost.train_tune import run_optuna_study_timeseries

if __name__ == "__main__":
    all_results = []  # List to accumulate all folds' backtest summaries

    for coin, pair in coin_pairs.items():
        print(f"Processing coin: {coin}")
        data = process_coinpair(coin, pair)
        study = run_optuna_study_timeseries(data)

    # Trigger visualizations for key backtest results
    # from src.result_visualizations.visualizations import plot_backtest_results
    # plot_backtest_results(all_results)
  