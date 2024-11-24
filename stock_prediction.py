import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def get_data(ticker):
    """
    Fetches historical data for the given ticker using yfinance.
    """
    data = yf.Ticker(ticker).history(period="max")
    data = data.drop(columns=["Dividends", "Stock Splits"])
    return data

def add_target_variable(data, start_date):
    """
    Adds 'Tomorrow' and 'Target' columns to the dataframe.
    Filters data starting from the specified start date.
    """
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    data = data.dropna(subset=["Tomorrow"])
    data = data.loc[start_date:].copy()
    return data

def add_technical_indicators(data, horizons):
    """
    Adds technical indicator columns based on the specified horizons.
    """
    new_predictors = []
    for horizon in horizons:
        rolling_avg = data["Close"].rolling(window=horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_avg

        trend_column = f"Trend_{horizon}"
        data[trend_column] = data["Target"].shift(1).rolling(window=horizon).sum()

        new_predictors += [ratio_column, trend_column]
    data = data.dropna()
    return data, new_predictors

def predict(train, test, predictors, model):
    """
    Trains the model and makes predictions on the test set.
    """
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    """
    Simulates backtesting by training and testing the model over rolling time windows.
    """
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:i+step].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def main():
    # Fetch and prepare data
    sp500 = get_data("^GSPC")
    sp500 = add_target_variable(sp500, start_date="1990-01-01")

    # Add technical indicators
    horizons = [2, 5, 60, 250, 1000]
    sp500, new_predictors = add_technical_indicators(sp500, horizons)

    # Initialize and backtest the model
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    predictions = backtest(sp500, model, new_predictors)

    # Output the results
    print(predictions["Predictions"].value_counts())
    print(precision_score(predictions["Target"], predictions["Predictions"]))

if __name__ == "__main__":
    main()
