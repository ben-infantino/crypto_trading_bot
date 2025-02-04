from src.data_processing.process_coinpair import process_coinpair
from src.data_processing.utility.train_val_test_split import time_series_folds
from src.data_processing.configs import coin_pairs
from src.data_modeling.XGBoost.XGBoost import train_XGBoost, predict_XGBoost, plot_XGBoost
from src.backtesting.simulate_backtest import backtest_model

if __name__ == "__main__":
    all_results = []  # List to accumulate all folds' backtest summaries

    for coin, pair in coin_pairs.items():
        print(f"Processing coin: {coin}")
        data = process_coinpair(coin, pair)
        folds = time_series_folds(data, n_folds=5, initial_train_frac=0.3)
        for fold_idx, (train_data, test_data) in enumerate(folds):
            print(f"Coin: {coin}, processing fold {fold_idx+1}/{len(folds)}...")
            # Pass the test_data as the second argument to train_XGBoost (used as validation data)
            model = train_XGBoost(train_data, test_data)
            summary, trade_logs_df = backtest_model(model, test_data)
            # Augment the summary with coin and fold information
            summary.update({
                'coin': coin,
                'fold': fold_idx + 1
            })
            all_results.append(summary)

    # Trigger visualizations for key backtest results
    from src.result_visualizations.visualizations import plot_backtest_results
    plot_backtest_results(all_results)
  