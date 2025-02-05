import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
from src.data_processing.utility.train_val_test_split import time_series_folds
import numpy as np  # Added for use in calculating the mean metric
from src.backtesting.backtest import backtest_model


def objective_time_series(trial, data, n_folds=5):
    """
    Objective function using time series folds.
    

    For each hyperparameter set, the function:
      - Splits the data via TimeSeriesSplit.
      - Trains an XGBoost model on each training fold.
      - Evaluates on the corresponding validation fold using a backtest simulator.
      - Aggregates the results (here via average metric over folds).
    
    Args:
        trial (optuna.trial.Trial): Optuna trial for suggesting hyperparameters.
        data (dict): Data containing keys "features" and "targets" or structured appropriately.
        n_folds (int): Number of cross validation folds.
        
    Returns:
        float: The average performance metric across the time series folds.
    """
    print("Starting a new trial evaluation.")  # New print statement for trial start

    folds = time_series_folds(data, n_folds=n_folds)
    fold_metrics = []  # collect metric for each fold
    

    # Suggest early stopping rounds for the XGBoost fit, to be passed to model.fit()
    
    # Expanded and optimized hyperparameter search based on XGBoost docs
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    params = {
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "tree_method": "hist",  # Fixed for speed
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0, step=0.1),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0, step=0.1),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),  # Updated
        "max_depth": trial.suggest_int("max_depth", 3, 6, step=1),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0, step=0.1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),         # Updated
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),         # Updated
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_bin": trial.suggest_int("max_bin", 256, 2048, step=128),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 10, 100, step=10),
        "grow_policy": grow_policy,
        "random_state": 42,
        "device": "cuda",  # Leverage GPU if available
        "eval_metric": "rmse"
    }
    if grow_policy == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 256)
    
    # print parameters in an easy to read format - I want 4 parameters per line
    print()
    print("Parameters for this trial:")
    i = 0
    for key, value in params.items():
        print(f"{key}: {value}", end="\t")
        if (i + 1) % 4 == 0:
            print()
        i += 1
    print()




    # Delay pruning until at least a minimum number of folds have been processed (to reduce noise)
    min_folds_before_prune = 2
    for fold_idx, (train_data, val_data, test_data) in enumerate(folds):
        print()
        print(f"Trial {trial.number if hasattr(trial, 'number') else ''} - Fold {fold_idx+1}/{n_folds}: Starting training.")
        print()
        model = xgb.XGBRegressor(**params)
        model.fit(
            train_data["features"],
            train_data["targets"],
            eval_set=[(val_data["features"], val_data["targets"])],
            verbose=False
        )
        print()
        print(f"Trial {trial.number if hasattr(trial, 'number') else ''} - Fold {fold_idx+1}/{n_folds}: Training completed. Starting backtesting.")
        print()

        # Call the optimized backtest function which returns a summary and a performance metric
        backtest_summary, backtest_metric = backtest_model(model, test_data)
        fold_metrics.append(backtest_metric)
        print(f"Trial {trial.number if hasattr(trial, 'number') else ''} - Fold {fold_idx+1}/{n_folds}: Backtesting metric: {backtest_metric:.4f}")

        # Report intermediate results after each fold for early pruning
        intermediate_value = np.mean(fold_metrics)
        trial.report(intermediate_value, fold_idx)
        if fold_idx >= min_folds_before_prune and trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    avg_metric = np.mean(fold_metrics)
    print()
    print(f"Trial {trial.number if hasattr(trial, 'number') else ''} completed with average metric: {avg_metric:.4f}")
    print()
    # Return the average performance metric across all folds.
    return avg_metric



def run_optuna_study_timeseries(data, n_trials=5, n_folds=5, use_pruner=False):
    """
    Runs an Optuna study that performs hyperparameter tuning with time series cross validation.
    

    Args:
        data (dict): Data containing time series details (features/targets).
        n_trials (int): Number of hyperparameter trials.
        n_folds (int): Number of folds for TimeSeriesSplit.
        use_pruner (bool): Whether to use a bandit-based pruner (e.g., Successive Halving).
    
    Returns:
        Tuple[optuna.study.Study, list]: The completed study including the best hyperparameter set
                                          and a list of all trial results (parameters and score).
    """
    print()
    print(f"Starting Optuna study with {n_trials} trials and {n_folds} folds.")
    print()
    if use_pruner:

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=2)
        )
    else:
        study = optuna.create_study(direction="maximize")
    
    study.optimize(
        lambda trial: objective_time_series(trial, data, n_folds=n_folds),   
        n_trials=n_trials
    )
    print()
    print(f"Optuna study completed. Best trial: {study.best_trial.number} with value: {study.best_trial.value:.4f}")
    print()
    

    # # Gather results from all trials (for later saving).
    # all_results = [
    #     {"trial_number": trial.number, "value": trial.value, "params": trial.params}
    #     for trial in study.trials if trial.value is not None
    # ]
    
    return study





# def train_XGBoost(train_data, val_data):
#     params = {
#         "tree_method": "hist",
#         "subsample": 0.8,  
#         "colsample_bytree": 0.8,  
#         "learning_rate": 0.05,
#         "max_depth": 10,
#         "eval_metric": "rmse",
#         "random_state": 42, 
#         "device": "cuda", 
#         "early_stopping_rounds": 20, 
#         "n_estimators": 6000, 
#         "max_bin": 1024
#     }

#     xgb_model = xgb.XGBRegressor(**params)
#     xgb_model.fit(
#         train_data["features"],
#         train_data["targets"],
#         eval_set=[(val_data["features"], val_data["targets"])], 
#         verbose=False
#     )

#     print(f"\nModel RMSE at Best Iteration: {xgb_model.best_score}")
#     print(f"\nBest Iteration: {xgb_model.best_iteration}")

#     return xgb_model


# def predict_XGBoost(model, test_data):
#     test_preds = model.predict(test_data["features"], iteration_range=(0, model.best_iteration))
#     test_mse = mean_squared_error(test_data["targets"], test_preds)
#     print(f"\nTest RMSE: {test_mse**0.5}")    
#     return test_preds


# def plot_XGBoost(test_data, test_preds):

#     test_comparison = pd.DataFrame({
#         "actual_return": test_data["targets"],
#         "predicted_return": test_preds
#     })

#     plt.figure(figsize=(15, 10))

#     plt.subplot(2, 1, 2)
#     plt.plot(test_comparison["actual_return"].values, label="Actual Return", alpha=0.7)
#     plt.plot(test_comparison["predicted_return"].values, label="Predicted Return", alpha=0.7)
#     plt.title("Test Set: Actual vs. Predicted Returns")
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()
