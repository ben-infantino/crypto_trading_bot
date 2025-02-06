import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
from src.data_processing.utility.train_val_test_split import time_series_folds
import numpy as np  # Added for use in calculating the mean metric
from src.backtesting.backtest import backtest_model
from matplotlib.backends.backend_pdf import PdfPages

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

    folds = time_series_folds(data, n_folds=n_folds, val_set=False)
    fold_metrics = []  # collect metric for each fold
    
    # Suggest early stopping rounds for the XGBoost fit, to be passed to model.fit()
    
    # Expanded and optimized hyperparameter search based on XGBoost docs
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    params = {
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "device": "cuda",  # GPU usage
        "eval_metric": "rmse",
        "n_jobs": 1,   # Added: Ensures sequential execution using CPU resources to avoid contention with GPU
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0, step=0.1),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0, step=0.1),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6, step=1),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0, step=0.1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_bin": trial.suggest_int("max_bin", 256, 2048, step=128),
        "grow_policy": grow_policy,  # Added: Ensures sequential execution using CPU resources to avoid contention with GPU
    }

    if grow_policy == "lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 256)
    
    # Delay pruning until at least a minimum number of folds have been processed (to reduce noise)
    min_folds_before_prune = 2
    for fold_idx, (train_data, test_data) in enumerate(folds):
        model = xgb.XGBRegressor(**params)

        model.fit(
            train_data["features"],
            train_data["targets"],
            verbose=False
        )

        # Call the optimized backtest function which returns a summary and a performance metric
        backtest_summary, backtest_metric = backtest_model(model, test_data)
        print(f"\nfold {fold_idx} backtest metric: {backtest_metric}")
        fold_metrics.append(backtest_metric)

        # Report intermediate results after each fold for early pruning
        intermediate_value = np.mean(fold_metrics)
        trial.report(intermediate_value, fold_idx)
        if fold_idx >= min_folds_before_prune and trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # Compute a weighted average where the weight for fold i is (i+1),
    # thereby giving more recent (higher-index) folds more influence.
    weights = np.arange(1, len(fold_metrics) + 1)
    weighted_avg_metric = np.sum(weights * np.array(fold_metrics)) / np.sum(weights)
    
    avg_metric = weighted_avg_metric

    # Return the average performance metric across all folds.
    return avg_metric

def run_optuna_study_timeseries(data, n_trials=2, n_folds=5, use_pruner=False):
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
    
    #save_hyperparameter_diagnostics_pdf(study)
    
    return study

def save_hyperparameter_diagnostics_pdf(study, pdf_filename="hyperparameter_diagnostics.pdf"):
    """
    Saves hyperparameter diagnostic plots from an Optuna study into a single PDF file.

    Args:
        study (optuna.study.Study): The Optuna study containing trial results.
        pdf_filename (str): The output PDF file name.
    """
    import matplotlib.pyplot as plt

    # Gather all hyperparameter names from all trials with valid objective values.
    param_names = set()
    for trial in study.trials:
        if trial.value is not None:
            param_names.update(trial.params.keys())

    sorted_params = sorted(param_names)
    figs_per_page = 4  # 4 plots per page (2 rows x 2 columns)

    with PdfPages(pdf_filename) as pdf:
        for i in range(0, len(sorted_params), figs_per_page):
            page_params = sorted_params[i:i+figs_per_page]
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            for j, param in enumerate(page_params):
                ax = axes[j]
                values, scores, trial_nums = [], [], []
                for trial in study.trials:
                    if param in trial.params and trial.value is not None:
                        values.append(trial.params[param])
                        scores.append(trial.value)
                        trial_nums.append(trial.number)
                if not values:
                    ax.set_visible(False)
                    continue

                if isinstance(values[0], str):
                    # Categorical: create a boxplot.
                    cat_data = {}
                    for v, score in zip(values, scores):
                        cat_data.setdefault(v, []).append(score)
                    categories = list(cat_data.keys())
                    data_to_plot = [cat_data[cat] for cat in categories]
                    ax.boxplot(data_to_plot, labels=categories)
                    ax.set_xlabel(param)
                    ax.set_ylabel("Objective Score")
                    ax.set_title(f"{param} (Categorical)")
                else:
                    # Numerical: create a scatter plot.
                    sc = ax.scatter(values, scores, c=trial_nums, cmap="viridis", s=50)
                    ax.set_xlabel(param)
                    ax.set_ylabel("Objective Score")
                    ax.set_title(f"{param} (Numerical)")
                    fig.colorbar(sc, ax=ax, label="Trial Number")

            # Hide any unused subplots on the page if fewer than 4 parameters remain.
            for k in range(len(page_params), figs_per_page):
                axes[k].set_visible(False)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Diagnostic plots saved to {pdf_filename}")

def train_and_test_XGBoost(train_data, test_data, extra_params=None):
    """
    Trains an XGBoost model on the full training data without a validation set.
    Uses the provided extra_params (e.g., best hyperparameters from Optuna)
    and overrides n_estimators with final_n_estimators if provided.
    """

    train_features = train_data.drop(columns=['timestamp', 'target'])
    train_targets = train_data['target']

    test_data_dict = {
        'features': test_data.drop(columns=['timestamp', 'target']),
        'targets': test_data['target']
    }

    base_params = {
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,

        "device": "cuda",
        "eval_metric": "rmse",
        "n_jobs": 1,
        "subsample": None,
        "colsample_bytree": None,
        "colsample_bylevel": None,
        "colsample_bynode": None,
        "learning_rate": None,
        "max_depth": None,
        "gamma": None,
        "min_child_weight": None,
        "reg_alpha": None,
        "reg_lambda": None,
        "n_estimators": None,
        "max_bin": None,
        "grow_policy": None,
        "max_leaves": None, 
        "n_estimators": None
    }
    
    if extra_params:
        base_params.update(extra_params)

    # Remove keys with None values so that only defined parameters are passed to XGBRegressor.
    final_params = {k: v for k, v in base_params.items() if v is not None}
    print(f"\n\nFinal parameters:\n {final_params}")
    model = xgb.XGBRegressor(**final_params)

    model.fit(
        train_features,
        train_targets,
        verbose=False
    )

    backtest_summary, backtest_metric = backtest_model(model, test_data_dict)

    print(f"\n\nBacktest summary: {backtest_summary}")
    print(f"\n\nBacktest metric: {backtest_metric}")

