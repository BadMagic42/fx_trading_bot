# file: backtest.py

import logging
import json
import pandas as pd
import numpy as np
import config
import joblib
from mt5_utils import fetch_data, get_timeframe_constant
from data_processing import process_symbol_data
from ml_model import train_ml_model
from model_manager import load_models, get_signals_for_symbol
from trade_execution import calculate_risk_levels  # or your ATR-based approach
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    filename=config.log_file,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load symbols configuration
symbols_config_path = Path(config.BASE_DIR) / "symbol_configs.json"
with open(symbols_config_path, "r") as file:
    symbols_config = json.load(file)

###############################################################################
# 2) HELPER: PROCESS & LABEL DATA FOR TRAINING
###############################################################################
def process_symbol_data_for_training(symbol, timeframe, total_bars):
    """
    Example: fetch data from MT5, process w/ indicators, pivot points, label for training.
    """
    # 1) We can fetch a large chunk
    df = fetch_data(symbol, timeframe, count=total_bars)
    if df.empty:
        logging.warning(f"No data for {symbol} at {timeframe}.")
        return pd.DataFrame()

    # 2) Use your existing pipeline logic
    #    Mode='training' ensures the 'label' is created
    processed = process_symbol_data(symbol, timeframe, mode='training', count=total_bars)
    if processed.empty:
        return pd.DataFrame()

    return processed

###############################################################################
# 3) WALK-FORWARD BACKTEST
###############################################################################
def walkforward_backtest(symbol, timeframe="H1", train_size=20000, test_size=1000, total_bars=40000):
    """
    Modified walk-forward backtest:
    - Train the model once on the first `train_size` bars.
    - Perform inference on subsequent `test_size` blocks without retraining.
    """
    logging.info(f"Fetching data for backtest (single training). Symbol={symbol}, Timeframe={timeframe}, TotalBars={total_bars}")
    
    try:
        # Validate MT5 initialization
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

        # Validate symbol availability
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Symbol {symbol} is not available in MT5.")

        # Fetch data
        full_df = fetch_data(symbol, timeframe, count=total_bars)
        if full_df.empty:
            raise RuntimeError(f"No data fetched for {symbol} with {timeframe} and {total_bars} bars.")

    except Exception as e:
        logging.error(f"Exception during data fetching for {symbol}: {e}")
        raise

    if len(full_df) < train_size + test_size:
        logging.error(f"Insufficient data for backtesting. "
                      f"Required={train_size + test_size}, Available={len(full_df)}")
        return pd.DataFrame()

    logging.info(f"Fetched data successfully. Rows={len(full_df)}")

    # Step 1: Train once on the first `train_size` bars
    train_data_raw = full_df.iloc[:train_size].copy()
    training_df = process_symbol_data_for_training(symbol, timeframe, total_bars=train_size)

    if training_df.empty:
        logging.error("Training dataset is empty. Aborting backtest.")
        return pd.DataFrame()

    symbol_dict = {symbol: training_df}
    try:
        models, feature_names, fi_df = train_ml_model(symbol_dict)
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return pd.DataFrame()

    # Step 2: Perform inference on the next `test_size` blocks
    all_closed_trades = []
    start_idx = train_size
    iteration = 0

    while (start_idx + test_size) <= len(full_df):
        iteration += 1
        logging.info(f"Iteration {iteration}: Performing inference on index range {start_idx} to {start_idx + test_size - 1}.")
        
        # Test set for this block
        test_data_raw = full_df.iloc[start_idx : start_idx + test_size].copy()

        # Run inference-only backtest
        test_closed_trades = run_backtest_simulation(symbol, timeframe, test_data_raw, models)
        all_closed_trades.extend(test_closed_trades)

        # Move to the next block
        start_idx += test_size

    # Summarize results
    trades_df = pd.DataFrame(all_closed_trades)
    if not trades_df.empty:
        total_pnl = trades_df['profit'].sum()
        logging.info(f"Backtest complete for {symbol}. Total iterations={iteration}, #Trades={len(trades_df)}, PnL={total_pnl:.2f}")
    else:
        logging.warning(f"No trades executed during backtest for {symbol}.")

    return trades_df



###############################################################################
# 4) BAR-BY-BAR SIMULATION
###############################################################################
def run_backtest_simulation(symbol, timeframe, test_data, models):
    """
    Step through each bar in 'test_data':
      - call your existing get_signals_for_symbol(...) to get a 'signal' for that bar
      - if signal in {0,2}, open a trade
      - track open trades: see if bar's high/low triggers SL or TP
      - close trades, record final outcome
    """
    # 1) We'll keep a list of open trades, plus closed trades
    open_positions = []
    closed_positions = []

    # For convenience, we’ll store columns to help with SL/TP checks
    test_data = test_data.reset_index(drop=True)
    if 'time' not in test_data.columns:
        # If your data has time in some other place
        test_data['time'] = pd.to_datetime(datetime.now())

    # We want a bar-by-bar approach
    for i in range(len(test_data)):
        current_row = test_data.iloc[i]
        bar_time = current_row['time']
        bar_open = current_row['open']
        bar_high = current_row['high']
        bar_low  = current_row['low']
        bar_close= current_row['close']

        # (a) Generate a signal for THIS bar
        #     We can create a small DataFrame up to i-th row for get_signals_for_symbol, or
        #     We can cheat and pass the entire test_data but rely on the last row's signal.
        #     Example below calls get_signals_for_symbol once per bar, which might be slow but is realistic:
        partial_df = test_data.iloc[:i+1].copy()
        partial_df['time'] = pd.to_datetime(partial_df['time'])

        # Temporarily store partial_df in a dict so get_signals_for_symbol can interpret it
        # But your function also fetches data from process_symbol_data. We can override that if we do:
        #   => We'll do a direct approach that modifies 'mode'='inference' inside your pipeline
        # or define a small custom version that doesn't re-fetch from MT5. For brevity:
        partial_signals, model_contrib = do_inference_locally(symbol, partial_df, models)

        # The last row's signal
        signal = partial_signals.iloc[-1] if not partial_signals.empty else 1

        # (b) If we have a buy/sell signal, open a new position
        if signal == 0 or signal == 2:
            action = 'sell' if signal == 0 else 'buy'
            entry_price = bar_close
            # use your existing “calculate_risk_levels” or “ATR-based approach”
            sl, tp = calculate_risk_levels(
                price=entry_price,
                action=action,
                sl_percent=config.sl_percent,
                tp_percent=config.tp_percent,
                symbol=symbol
            )
            # store the position
            pos = {
                'symbol': symbol,
                'action': action,
                'open_idx': i,
                'open_time': bar_time,
                'open_price': entry_price,
                'sl': sl,
                'tp': tp
            }
            open_positions.append(pos)

        # (c) Check all open trades: did bar_high/low cross SL or TP?
        # We'll check each open position’s direction
        newly_closed = []
        for pos in open_positions:
            direction = pos['action']  # 'buy' or 'sell'
            entry_price = pos['open_price']

            # For a BUY:
            #   if bar_low <= SL => stopped out
            #   elif bar_high >= TP => profit target
            # For a SELL:
            #   if bar_high >= SL => stopped out
            #   elif bar_low <= TP => profit target

            if direction == 'buy':
                if bar_low <= pos['sl']:
                    # Stopped out
                    exit_price = pos['sl']
                    profit = exit_price - entry_price
                    newly_closed.append((pos, i, bar_time, exit_price, profit))
                elif bar_high >= pos['tp']:
                    # Target
                    exit_price = pos['tp']
                    profit = exit_price - entry_price
                    newly_closed.append((pos, i, bar_time, exit_price, profit))

            else:  # 'sell'
                if bar_high >= pos['sl']:
                    exit_price = pos['sl']
                    profit = entry_price - exit_price
                    newly_closed.append((pos, i, bar_time, exit_price, profit))
                elif bar_low <= pos['tp']:
                    exit_price = pos['tp']
                    profit = entry_price - exit_price
                    newly_closed.append((pos, i, bar_time, exit_price, profit))

        # (d) finalize newly_closed
        for (pos, close_idx, close_time, close_price, profit) in newly_closed:
            open_positions.remove(pos)
            closed_positions.append({
                'symbol': pos['symbol'],
                'open_time': pos['open_time'],
                'close_time': close_time,
                'action': pos['action'],
                'open_price': pos['open_price'],
                'close_price': close_price,
                'profit': profit
            })

    # If at the end of test_data you still have open positions, you can close them at the last bar’s close if you want:
    for pos in open_positions:
        # forced exit
        exit_price = test_data.iloc[-1]['close']
        profit = exit_price - pos['open_price'] if pos['action'] == 'buy' else pos['open_price'] - exit_price
        closed_positions.append({
            'symbol': pos['symbol'],
            'open_time': pos['open_time'],
            'close_time': test_data.iloc[-1]['time'],
            'action': pos['action'],
            'open_price': pos['open_price'],
            'close_price': exit_price,
            'profit': profit
        })

    return closed_positions

###############################################################################
# 5) A SMALL HELPER FOR "INFERENCE"
###############################################################################
def do_inference_locally(symbol, partial_df, models):
    """
    Perform inference locally without re-fetching data from MT5. This method ensures consistency 
    with the training process by reusing existing functions and structures.

    Args:
        symbol (str): The symbol for which inference is being performed.
        partial_df (pd.DataFrame): DataFrame containing feature data up to the current bar.
        models (dict): Dictionary of preloaded models.

    Returns:
        signals (pd.Series): Final signals (0=Sell, 1=Hold, 2=Buy).
        model_contrib (list): Model contributions for each bar.
    """
    from model_manager import prepare_dataset_for_prediction, ensemble_predict, post_prediction_validation, load_scaler
    from sklearn.exceptions import NotFittedError

    # Ensure data is filled
    working = partial_df.copy()
    working = working.ffill().bfill().fillna(0)

    # Load trained feature names and scaler
    scaler_path = config.BASE_DIR / "models/scaler.joblib"
    feature_names_path = config.BASE_DIR / "models/trained_feature_names.joblib"

    if not scaler_path.exists() or not feature_names_path.exists():
        raise RuntimeError("Scaler or trained feature names not found. Ensure models are properly trained and saved.")

    scaler = load_scaler()
    trained_feature_names = joblib.load(feature_names_path)

    # Ensure all required features exist in the dataset
    for col in trained_feature_names:
        if col not in working.columns:
            working[col] = 0.0

    # Remove extraneous columns and reindex to match trained features
    working = working.reindex(columns=trained_feature_names, fill_value=0)

    # Scale the features
    try:
        working_X = scaler.transform(working.values)
    except NotFittedError as e:
        raise RuntimeError("The scaler is not fitted. Please check the training process.") from e

    # Perform ensemble prediction
    raw_signals, model_contrib = ensemble_predict(models, working_X)

    # Post-prediction validation
    final_signals = post_prediction_validation(working, raw_signals, symbol)

    # Convert final signals into a pandas Series with the same index as input data
    signals_series = pd.Series(final_signals, index=working.index)

    return signals_series, model_contrib



###############################################################################
# 6) MAIN (EXAMPLE)
###############################################################################
if __name__ == "__main__":
    output_format = "csv"  # Change to "json" if desired
    
    for symbol in config.symbols:
        logging.info(f"Starting backtest for symbol: {symbol}")
        if symbol not in symbols_config:
            logging.error(f"Symbol {symbol} not found in symbols_config.json. Skipping.")
            continue

        # Fetch configuration for this symbol
        config_params = symbols_config[symbol]
        timeframe = config_params.get("timeframe", "M30")
        total_bars = 60000

        try:
            # Run the backtest
            results = walkforward_backtest(symbol, timeframe, train_size=20000, test_size=1000, total_bars=total_bars)

            # Save results
            if results.empty:
                logging.warning(f"No results for {symbol}. Skipping save.")
                continue

            output_file = Path(config.BASE_DIR) / f"backtest_results_{symbol}_{timeframe}.{output_format}"
            if output_format == "csv":
                results.to_csv(output_file, index=False)
                logging.info(f"Results for {symbol} saved to {output_file}")
            elif output_format == "json":
                results.to_json(output_file, orient="records", lines=True)
                logging.info(f"Results for {symbol} saved to {output_file}")
            else:
                logging.error(f"Unsupported output format: {output_format}")
        except Exception as e:
            logging.error(f"Backtest failed for {symbol}: {e}")
