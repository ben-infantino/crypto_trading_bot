def train_validate_test_split(data, train_size=0.7, validate_size=0.15, test_size=0.15):
    # Indices (assuming sums to 1.0):
    train_end = int(len(data) * train_size)
    validate_end = train_end + int(len(data) * validate_size)
    # test starts at validate_end automatically

    train_data = data.iloc[:train_end]
    validate_data = data.iloc[train_end:validate_end]
    test_data = data.iloc[validate_end:]

    train_data = {
        'features': train_data.drop(columns=['timestamp','target']),
        'targets': train_data['target']
    }
    validate_data = {
        'features': validate_data.drop(columns=['timestamp','target']),
        'targets': validate_data['target']
    }
    test_data = {
        'features': test_data.drop(columns=['timestamp','target']),
        'targets': test_data['target']
    }

    return train_data, validate_data, test_data


# New function: Create expanding-window time series folds.
def time_series_folds(data, n_folds=5, initial_train_frac=0.3):
    """
    Splits the data into expanding-window time series folds.
    
    Parameters:
      - data: pandas DataFrame sorted chronologically by 'timestamp'.
      - n_folds: Number of folds to create.
      - initial_train_frac: Fraction of the data to use for the initial training set.
      
    Returns:
      - List of tuples [(train_fold, test_fold), ...] where each fold is a dict with:
          • 'features': DataFrame of features (all columns except 'timestamp' and 'target')
          • 'targets': Series for the target ('target' column)
    """
    N = len(data)
    initial_train_end = int(N * initial_train_frac)
    if initial_train_end >= N:
        raise ValueError("initial_train_frac is too high; no data left for testing.")
    remaining = N - initial_train_end
    test_window = int(remaining / n_folds)
    if test_window == 0:
        test_window = 1  # Ensure at least one row per test fold

    folds = []
    for i in range(n_folds):
        train_end = initial_train_end + i * test_window
        test_start = train_end
        test_end = test_start + test_window if i < n_folds - 1 else N

        train_fold = data.iloc[:train_end]
        test_fold = data.iloc[test_start:test_end]

        # Only include the fold if both training and testing parts are non-empty.
        if len(train_fold) > 0 and len(test_fold) > 0:
            train_dict = {
                'features': train_fold.drop(columns=['timestamp', 'target']),
                'targets': train_fold['target']
            }
            test_dict = {
                'features': test_fold.drop(columns=['timestamp', 'target']),
                'targets': test_fold['target']
            }
            folds.append((train_dict, test_dict))
    return folds