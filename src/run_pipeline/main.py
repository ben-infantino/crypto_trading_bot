from src.data_processing.process_coinpair import process_coinpair
from src.data_processing.utility.train_val_test_split import train_validate_test_split
from src.data_processing.configs import coin_pairs
from src.data_modeling.XGBoost.XGBoost import train_XGBoost, predict_XGBoost, plot_XGBoost
from src.backtesting.simulate_backtest import backtest_model

if __name__ == "__main__":
    for coin, pair in coin_pairs.items():
        data = process_coinpair(coin, pair)
        train_data, validate_data, test_data = train_validate_test_split(data)
        model = train_XGBoost(train_data, validate_data)
        val_preds, test_preds = predict_XGBoost(model, validate_data, test_data)
        #plot_XGBoost(validate_data, test_data, val_preds, test_preds)
        backtest_model(model, test_data)
  