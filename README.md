# Stock Market Prediction Using Machine Learning

This project predicts daily stock market movements for the S&P 500 index using a machine learning model. It implements a Random Forest Classifier and evaluates the model's performance using backtesting techniques.

## Features
- Fetches historical stock market data using the `yfinance` library.
- Creates predictive features such as rolling averages and trend indicators.
- Trains a Random Forest Classifier to predict daily market movements.
- Simulates backtesting to evaluate the model's performance over rolling time windows.

## Requirements
- Python 3.8+
- Libraries listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/adiarora/stockPredictor.git
   cd stockPredictor
