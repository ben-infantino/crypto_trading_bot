from data_processing.single_processor import generate_data_splits
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

def save_grid_search_results(results, filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'model_results_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    return filename

if __name__ == "__main__":
    coin="XRP"
    pair="XRPUSD"
    timeframes=[15]

    train_data, val_data, test_data = generate_data_splits(coin, pair, timeframes)

    param_grid = {
        'n_estimators': [500, 1000, 2000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    # Try all parameter combinations
    results = []
    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                params = {
                    'n_estimators': n_est,
                    'max_depth': depth,
                    'learning_rate': lr,
                    'random_state': 42,
                    'eval_metric': 'rmse'
                }
                
                # Train model with current parameters
                model = xgb.XGBRegressor(**params)
                model.fit(train_data["features"], train_data["targets"])
                
                # Evaluate on validation set
                val_preds = model.predict(val_data["features"])
                val_mse = mean_squared_error(val_data["targets"], val_preds)
                
                results.append({
                    'parameters': {
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'learning_rate': lr
                    },
                    'validation_mse': float(val_mse),
                    'validation_rmse': float(val_mse ** 0.5)
                })
                
                print(f"Params: n_est={n_est}, depth={depth}, lr={lr}, val_rmse={val_mse**0.5:.6f}")

    # Sort results by validation score
    results.sort(key=lambda x: x['validation_mse'])
    
    # Save results
    results_file = save_grid_search_results(results, 'model_results.json')
    print(f"\nDetailed results saved to: {results_file}")

    # Get best parameters
    best_params = results[0]['parameters']
    print("\nBest parameters:", best_params)

    # Train final model with best parameters
    best_model = xgb.XGBRegressor(
        **best_params,
        random_state=42,
        eval_metric='rmse'
    )

    # Fit the model on training data
    best_model.fit(
        train_data["features"],
        train_data["targets"],
        eval_set=[(val_data["features"], val_data["targets"])],
        verbose=True
    )

    # Predictions on validation
    val_preds = best_model.predict(val_data["features"])
    val_mse = mean_squared_error(val_data["targets"], val_preds)
    print("\nValidation MSE:", val_mse)
    print("Validation RMSE:", val_mse**0.5)

    # Predictions on test
    test_preds = best_model.predict(test_data["features"])
    test_mse = mean_squared_error(test_data["targets"], test_preds)
    print("\nTest MSE:", test_mse)
    print("Test RMSE:", test_mse**0.5)

    # Create comparison DataFrames
    val_comparison = pd.DataFrame({
        "actual_return": val_data["targets"],
        "predicted_return": val_preds
    })

    test_comparison = pd.DataFrame({
        "actual_return": test_data["targets"],
        "predicted_return": test_preds
    })

    # Create plots
    plt.figure(figsize=(15, 10))

    # Validation set plot
    plt.subplot(2, 1, 1)
    plt.plot(val_comparison["actual_return"].values, label="Actual Return", alpha=0.7)
    plt.plot(val_comparison["predicted_return"].values, label="Predicted Return", alpha=0.7)
    plt.title("Validation Set: Actual vs. Predicted Returns")
    plt.legend()
    plt.grid(True)

    # Test set plot
    plt.subplot(2, 1, 2)
    plt.plot(test_comparison["actual_return"].values, label="Actual Return", alpha=0.7)
    plt.plot(test_comparison["predicted_return"].values, label="Predicted Return", alpha=0.7)
    plt.title("Test Set: Actual vs. Predicted Returns")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Create scatter plots
    plt.figure(figsize=(15, 10))

    # Validation set scatter
    plt.subplot(2, 1, 1)
    plt.scatter(val_comparison["actual_return"], val_comparison["predicted_return"], alpha=0.5)
    plt.plot([val_comparison["actual_return"].min(), val_comparison["actual_return"].max()], 
             [val_comparison["actual_return"].min(), val_comparison["actual_return"].max()], 
             'r--', lw=2)
    plt.title("Validation Set: Predicted vs Actual Returns")
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.grid(True)

    # Test set scatter
    plt.subplot(2, 1, 2)
    plt.scatter(test_comparison["actual_return"], test_comparison["predicted_return"], alpha=0.5)
    plt.plot([test_comparison["actual_return"].min(), test_comparison["actual_return"].max()], 
             [test_comparison["actual_return"].min(), test_comparison["actual_return"].max()], 
             'r--', lw=2)
    plt.title("Test Set: Predicted vs Actual Returns")
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.grid(True)

    plt.tight_layout()
    plt.show()