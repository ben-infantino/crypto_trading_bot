import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
#from src.data_modeling.XGBoost.configs import param_grid
from sklearn.metrics import mean_squared_error
import time


# def grid_search_XGBoost(train_data, val_data):

#     params = {
#         'n_estimators': param_grid['n_estimators'],
#         'max_depth': param_grid['max_depth'],
#         'learning_rate': param_grid['learning_rate'],
#         'random_state': param_grid['random_state'],
#         'eval_metric': param_grid['eval_metric'],
#         'early_stopping_rounds': param_grid['early_stopping_rounds']
#     }


def train_XGBoost(train_data, val_data):
    params = {
        "tree_method": "hist",
        "subsample": 0.8,  
        "colsample_bytree": 0.8,  
        "learning_rate": 0.05,
        "max_depth": 10,
        "eval_metric": "rmse",
        "random_state": 42, 
        "device": "cuda", 
        "early_stopping_rounds": 20, 
        "n_estimators": 6000, 
        "max_bin": 1024
    }

    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(
        train_data["features"],
        train_data["targets"],
        eval_set=[(val_data["features"], val_data["targets"])], 
        verbose=False
    )

    print(f"\nModel RMSE at Best Iteration: {xgb_model.best_score}")
    print(f"\nBest Iteration: {xgb_model.best_iteration}")

    return xgb_model


def predict_XGBoost(model, test_data):
    test_preds = model.predict(test_data["features"], iteration_range=(0, model.best_iteration))
    test_mse = mean_squared_error(test_data["targets"], test_preds)
    print(f"\nTest RMSE: {test_mse**0.5}")    
    return test_preds


def plot_XGBoost(test_data, test_preds):

    test_comparison = pd.DataFrame({
        "actual_return": test_data["targets"],
        "predicted_return": test_preds
    })

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 2)
    plt.plot(test_comparison["actual_return"].values, label="Actual Return", alpha=0.7)
    plt.plot(test_comparison["predicted_return"].values, label="Predicted Return", alpha=0.7)
    plt.title("Test Set: Actual vs. Predicted Returns")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
