def simulate_backtest(data, model):
    #start with 1000 dollars
    balance = 1000
    slippage = 0.001
    buy_amount = .1 #10% of balance
    owned_coins = 0
    min_balance = float('inf')
    max_balance = float('-inf')

    features = data['features'] #pd.DataFrame
    targets = data['targets'] #pd.Series

    #no fees

    for feature_row, target in zip(features.iterrows(), targets):
        feature_row = feature_row[1]

        prediction = round(model.predict([feature_row], iteration_range=(0, model.best_iteration))[0], 5)

        if owned_coins != 0:
            # sell all coins at current target price accounting for slippage
            amount_to_sell = owned_coins * (1 - slippage)
            balance += amount_to_sell * target
            owned_coins = 0
            min_balance = min(min_balance, balance)
            max_balance = max(max_balance, balance)
        
        if prediction > feature_row['previous_close']:
            print("precited price:", prediction, "last close:", feature_row['previous_close'], "delta:", prediction - feature_row['previous_close'])
            #buy
            # amount of money to spend
            amount_to_spend = balance * buy_amount
            # amount of coins to buy accounting for slippage
            amount_of_coins = amount_to_spend * (1 - slippage) / feature_row['previous_close']
            # update balance
            balance -= amount_to_spend
            # update owned coins
            owned_coins += amount_of_coins

    print(f"Min balance: {min_balance}, Max balance: {max_balance}, Final balance: {balance}")

