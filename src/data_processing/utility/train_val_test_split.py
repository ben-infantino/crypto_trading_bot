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