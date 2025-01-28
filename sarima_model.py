# sarima_model.py

import pandas as pd
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import config
import numpy as np

def train_sarima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), symbol=None):
    """
    Train a SARIMA model on the provided data. We can optionally detect the symbol's timeframe
    and set the frequency so statsmodels won't warn about 'no associated freq'.

    Args:
        data (pd.DataFrame): DataFrame containing 'close' prices with datetime index.
        order (tuple): (p,d,q) for ARIMA
        seasonal_order (tuple): seasonal (P,D,Q,m)
        symbol (str): e.g. 'EURUSD' to detect timeframe from config symbol settings

    Returns:
        SARIMAXResultsWrapper: Trained SARIMA model, or None if fails.
    """
    try:
        # 1) If we know the symbol’s timeframe, set freq
        timeframe_map = {
            "M1": "1min",
            "M5": "5min",
            "M15": "15min",
            "M30": "30min",
            "H1": "1h",
            "H4": "4h"
        }

        # If the symbol is known, get its timeframe from config
        if symbol:
            symbol_tf = config.symbol_configs.get(symbol, {}).get("timeframe", config.timeframe)
            freq_str = timeframe_map.get(symbol_tf)  # e.g., "1H" for "H1"
            if freq_str and not data.index.freq:
                # Set frequency, handling potential missing data
                data = data.asfreq(freq_str)
                # Forward-fill and back-fill any introduced NaNs
                data = data.ffill().bfill()
                logging.info(f"Set data.index.freq to '{freq_str}' for {symbol} before SARIMA training.")

        # 2) Build the SARIMAX model
        model = SARIMAX(
            data['close'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        # Increase maxiter to help convergence
        sarima_model = model.fit(disp=False, maxiter=2000, method='powell') # method can be 'powell' or 'bfgs' or 'nm' or 'lbfgs'
        logging.info("SARIMA model trained successfully (maxiter=1000).")

        # Save the model
        sarima_model_path = config.BASE_DIR / "models" / f"sarima_model_{symbol or 'generic'}.joblib"
        sarima_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(sarima_model, sarima_model_path)
        logging.info(f"SARIMA model saved at {sarima_model_path}")

        return sarima_model

    except Exception as e:
        logging.error(f"Error training SARIMA model for {symbol or 'unknown'}: {e}")
        return None

def predict_sarima(sarima_model, steps=1):
    """
    Generate SARIMA forecasts.

    Args:
        sarima_model (SARIMAX): Trained SARIMA model.
        steps (int, optional): Number of steps to forecast. Defaults to 1.

    Returns:
        np.array: Forecasted values.
    """
    try:
        forecast = sarima_model.forecast(steps=steps)
        logging.debug(f"SARIMA forecast for next {steps} steps: {forecast.values}")
        return forecast.values
    except Exception as e:
        logging.error(f"Error during SARIMA prediction: {e}")
        return np.array([0] * steps)
