import logging
import joblib
import numpy as np
import pandas as pd
import config

from datetime import datetime
from data_processing import process_symbol_data
from lstm_model import predict_lstm
from sarima_model import predict_sarima
import json

model_performance = {}
model_weights = {}
recent_trades_for_performance = []  # Stores recent trades for rolling window performance

def load_models():
    """
    Load all calibrated models from disk as specified in config.models_available.
    Returns a dict: {model_name: model_object}.
    """
    logging.info("Loading calibrated models...")
    models = {}
    for model_name, active in config.models_available.items():
        if not active:
            continue
        model_path = config.BASE_DIR / f"models/{model_name.lower().replace(' ', '_')}_calibrated_model.joblib"
        if model_path.exists():
            try:
                models[model_name] = joblib.load(model_path)
                logging.info(f"Loaded calibrated {model_name}")
            except Exception as e:
                logging.error(f"Error loading calibrated {model_name}: {e}")
        else:
            logging.warning(f"Calibrated model file not found for {model_name}")
    if not models:
        logging.error("No calibrated models loaded.")
    return models


def load_scaler():
    """
    Load the pre-trained scaler from disk, if available.
    """
    scaler_path = config.BASE_DIR / "models/scaler.joblib"
    if scaler_path.exists():
        try:
            return joblib.load(scaler_path)
        except Exception as e:
            logging.error(f"Error loading scaler: {e}")
    else:
        logging.warning("No scaler found.")
    return None


def prepare_dataset_for_prediction(data):
    """
    Basic fill & NaN handling for inference data. 
    """
    if data.empty:
        return pd.DataFrame()
    return data.ffill().bfill().fillna(0)


def get_model_weights(model_names):
    """
    Return model weights, either manual or performance_based.
    """
    if config.model_weighting_mode == "manual":
        return {m: config.manual_model_weights.get(m, 1.0) for m in model_names}
    elif config.model_weighting_mode == "performance_based":
        # dynamic => from global model_weights
        return {m: model_weights.get(m, 1.0) for m in model_names}
    else:
        # default => all 1.0
        return {m: 1.0 for m in model_names}


def ensemble_predict(models, X, lstm_model=None, sarima_model=None, lstm_data=None, sarima_steps=1):
    """
    Perform ensemble prediction across all loaded models (except we handle LSTM & SARIMA separately).
    Returns:
      signals (np.array[int]): final integer signals (0=Sell,1=Hold,2=Buy)
      model_contributions (list): For each row => list of (model_name, class, confidence)
    
    If LSTM & SARIMA are passed in, they are included in the final voting with weight=1.0 each by default.
    """
    logging.debug(f"Starting ensemble prediction with X shape: {X.shape}")
    n_samples = X.shape[0]

    # Prepare final signals array, default=Hold
    signals = np.ones(n_samples, dtype=int)

    # For each row, we store a list of (model_name, predicted_class, confidence).
    model_contributions = [[] for _ in range(n_samples)]

    # 1) Gather base model predictions for classical models.
    base_model_probs = {}
    for model_name, model_obj in models.items():
        # skip LSTM & SARIMA if named as such
        if model_name.startswith("SARIMA") or model_name == "LSTM":
            continue
        try:
            probs = model_obj.predict_proba(X)
            base_model_probs[model_name] = probs
        except Exception as e:
            logging.error(f"Error predicting with {model_name}: {e}")
            # fallback => all hold
            fallback = np.zeros((n_samples, len(model_obj.classes_)))
            fallback[:, 1] = 1.0
            base_model_probs[model_name] = fallback

    # 2) If LSTM is provided, do a single pass for all rows:
    lstm_probs = None
    if lstm_model and (lstm_data is not None):
        try:
            preds, probs = predict_lstm(lstm_model, lstm_data)
            lstm_probs = probs  # shape (n_samples,3)
        except Exception as e:
            logging.error(f"Error predicting with LSTM: {e}")

    # 3) If sarima_model is provided, do a single pass for each row
    sarima_preds = None
    if sarima_model:
        sarima_preds = np.ones(n_samples, dtype=int)
        for i in range(n_samples):
            try:
                forecast_values = predict_sarima(sarima_model, steps=sarima_steps)
                # naive approach => if forecast>current => buy(2)
                current_close = X[i, 0]  # 'close' is col=0
                predicted_class = 2 if (forecast_values[-1] > current_close) else 0
                sarima_preds[i] = predicted_class
            except Exception as e:
                logging.error(f"SARIMA error row {i}: {e}")
                sarima_preds[i] = 1

    # 4) Combine all predictions into final signals
    current_weights = get_model_weights(base_model_probs.keys())  # e.g. {'LightGBM':1.0,...}

    for i in range(n_samples):
        # Tally votes
        votes = {0:0.0, 1:0.0, 2:0.0}

        # 4.1) Base classical models
        for model_name, probs in base_model_probs.items():
            model = models[model_name]
            threshold = config.model_confidence_thresholds.get(model_name, config.min_confidence_threshold)
            weight = current_weights.get(model_name, 1.0)

            row_probs = probs[i]
            class_idx = np.argmax(row_probs)
            max_prob = row_probs[class_idx]
            predicted_class = model.classes_[class_idx]

            # If confidence < threshold => treat as hold
            if max_prob < threshold:
                predicted_class = 1
                max_prob = threshold

            votes[predicted_class] += weight

            # record
            logging.debug(f"Model {model_name} => row {i} => class={predicted_class}, prob={max_prob:.4f}")
            model_contributions[i].append((model_name, predicted_class, float(max_prob)))

        # 4.2) LSTM if available
        if lstm_probs is not None:
            row_probs = lstm_probs[i]
            lstm_idx = np.argmax(row_probs)
            lstm_conf = row_probs[lstm_idx]
            threshold_lstm = config.model_confidence_thresholds.get("LSTM", config.min_confidence_threshold)

            if lstm_conf < threshold_lstm:
                final_class = 1
            else:
                final_class = lstm_idx

            # Weighted by 1.0
            votes[final_class] += 1.0
            model_contributions[i].append(("LSTM", final_class, float(lstm_conf)))

        # 4.3) SARIMA if available
        if sarima_preds is not None:
            pred_class = sarima_preds[i]
            votes[pred_class] += 1.0
            model_contributions[i].append(("SARIMA", pred_class, 1.0))

        # 4.4) Final majority (or weighting)
        best_class = max(votes.keys(), key=lambda c: votes[c])
        signals[i] = best_class

    return signals, model_contributions


def post_prediction_validation(data, signals, symbol):
    """
    Validate post-prediction signals based on:
      - trading hours
      - RSI thresholds
      - Bollinger
      - ATR-based dynamic filter
    """
    # Grab the symbol-specific config
    symbol_config = config.symbol_configs.get(symbol, {})

    # 1) dynamic ATR check
    atr_mode = symbol_config.get('dynamic_atr', 'static')
    filter_direction = symbol_config.get('dynamic_atr_filter_direction', 'above')
    atr_band = symbol_config.get('dynamic_atr_band', 0.0)

    dynamic_atr = None
    if atr_mode == 'dynamic' and 'atr' in data.columns:
        dynamic_atr = data['atr'].rolling(window=config.dynamic_atr_window).mean().iloc[-1]
        logging.debug(f"[{symbol}] dynamic ATR => {dynamic_atr:.5f}")

    if dynamic_atr is not None:
        atr_threshold = dynamic_atr
    else:
        atr_threshold = symbol_config.get('post_pred_atr_threshold', config.post_pred_atr_val)

    # Counters
    filters = {'time':0,'rsi':0,'bollinger':0,'atr':0}

    for i in range(len(data)):
        s = signals[i]

        # Time-of-day check
        now_utc_hour = datetime.utcnow().hour
        if not (config.start_trading_hour <= now_utc_hour < config.end_trading_hour):
            s = 1
            filters['time'] += 1
        else:
            # RSI check
            if 'rsi' in data.columns:
                rsi_val = data['rsi'].iloc[i]
                if s == 2 and rsi_val > config.rsi_sell_limit:
                    s = 1
                    filters['rsi'] += 1
                if s == 0 and rsi_val < config.rsi_buy_limit:
                    s = 1
                    filters['rsi'] += 1

            # Bollinger check
            if 'upper_band' in data.columns and 'lower_band' in data.columns:
                price = data['close'].iloc[i]
                upper = data['upper_band'].iloc[i]
                lower = data['lower_band'].iloc[i]
                if s == 2 and price >= upper:
                    s = 1
                    filters['bollinger'] += 1
                if s == 0 and price <= lower:
                    s = 1
                    filters['bollinger'] += 1

            # ATR-based
            if 'atr' in data.columns:
                atr_val = data['atr'].iloc[i]
                if pd.notna(atr_val):
                    if filter_direction == 'above':
                        if atr_val > atr_threshold:
                            s = 1
                            filters['atr'] += 1
                    elif filter_direction == 'below':
                        if atr_val < atr_threshold:
                            s = 1
                            filters['atr'] += 1
                    elif filter_direction == 'band':
                        band_low = atr_threshold*(1.0 - atr_band)
                        band_high= atr_threshold*(1.0 + atr_band)
                        if not (band_low <= atr_val <= band_high):
                            s = 1
                            filters['atr'] += 1
                    else:
                        logging.warning(f"[{symbol}] Unknown ATR filter direction: {filter_direction}")

        signals[i] = s

    logging.info(f"[{symbol}] Post-prediction filters => {filters}")
    return signals


def get_signals_for_symbol(symbol, timeframe, mode='inference', count=None, models=None):
    """
    Main pipeline to fetch data for 'symbol' & 'timeframe', run inference if mode='inference',
    apply post_pred_validation, then return final dataset with 'signal'.
    """
    # 1) fetch data
    data = process_symbol_data(symbol, timeframe, mode=mode, count=count)
    if data.empty:
        data['signal'] = 1
        return data, []

    save_data = config.save_training_data

    # 2) Optionally save the raw data for debugging
    if save_data:
        output_path = config.BASE_DIR / "debug" / f"{symbol}_{mode}_data.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_serializable = data.replace({np.nan: None, np.inf: None, -np.inf: None})
        for col in json_serializable.select_dtypes(include=[np.datetime64, 'datetime']):
            json_serializable[col] = json_serializable[col].astype(str)
        with open(output_path, 'w') as f:
            json.dump(json_serializable.to_dict(orient="records"), f, indent=4)
        logging.info(f"{mode.capitalize()} data saved => {output_path}")

    # 3) If training, we do not do predictions
    if mode == 'training':
        return data, []

    # 4) If inference => Need 'models' loaded
    if not models:
        logging.error("No preloaded models provided for get_signals_for_symbol (inference).")
        data['signal'] = 1
        return data, []

    # 5) Load the scaler & feature_names
    scaler_path = config.BASE_DIR / "models/scaler.joblib"
    fn_path = config.BASE_DIR / "models/trained_feature_names.joblib"
    if not (scaler_path.exists() and fn_path.exists()):
        logging.warning("Scaler or feature names not found, defaulting signals to HOLD.")
        data['signal'] = 1
        return data, []

    scaler = joblib.load(scaler_path)
    trained_feature_names = joblib.load(fn_path)

    # 6) Prepare dataset for inference
    X_df = prepare_dataset_for_prediction(data)
    if X_df.empty:
        data['signal'] = 1
        return data, []

    # Ensure all needed columns exist, create any missing as zero
    for col in trained_feature_names:
        if col not in X_df.columns:
            X_df[col] = 0.0
    # remove extraneous columns
    drop_cols = [c for c in X_df.columns if c not in trained_feature_names]
    if drop_cols:
        X_df.drop(columns=drop_cols, inplace=True)
    # reindex
    X_df = X_df.reindex(columns=trained_feature_names)

    # scale
    X_scaled = scaler.transform(X_df.values)

    # 7) run ensemble_predict
    raw_signals, model_contributions = ensemble_predict(models, X_scaled)

    # Post-prediction filters
    final_signals = post_prediction_validation(data, raw_signals, symbol)
    data['signal'] = final_signals

    # Optionally save the final data
    if save_data:
        final_path = config.BASE_DIR / "debug" / f"{symbol}_final_{mode}_data.json"
        json_serializable = data.replace({np.nan: None, np.inf: None, -np.inf: None})
        for col in json_serializable.select_dtypes(include=[np.datetime64, 'datetime']):
            json_serializable[col] = json_serializable[col].astype(str)
        try:
            with open(final_path, 'w') as f:
                json.dump(json_serializable.to_dict(orient="records"), f, indent=4)
            logging.info(f"Final inference data saved => {final_path}")
        except Exception as e:
            logging.error(f"Failed to save final inference data => {final_path}: {e}")

    return data, model_contributions


def update_model_performance(results):
    """
    Takes a list of closed trades info => update global model_performance & dynamic thresholds or weights if needed.
    Each trade is a dict: {
      'symbol': str,
      'profit': float,
      'models': [(model_name, predicted_class, confidence), ...]
    }
    """
    global model_performance, recent_trades_for_performance
    if not results:
        return

    # Store them
    for trade in results:
        recent_trades_for_performance.append(trade)

    # Trim
    if len(recent_trades_for_performance) > config.rolling_window_size:
        recent_trades_for_performance = recent_trades_for_performance[-config.rolling_window_size:]

    # Recalc from scratch
    model_performance.clear()
    for trade in recent_trades_for_performance:
        profit = trade.get('profit', 0.0)
        used = trade.get('models', [])
        is_win = (profit > 0)
        for (model_name, _, _) in used:
            if model_name not in model_performance:
                model_performance[model_name] = {'wins':0,'losses':0,'total':0,'profit_sum':0.0}
            mp = model_performance[model_name]
            mp['total'] += 1
            mp['profit_sum'] += profit
            if is_win:
                mp['wins'] += 1
            else:
                mp['losses'] += 1

    # Adjust thresholds or weights if performance_based
    if config.model_weighting_mode == "performance_based":
        for m_name, mp in model_performance.items():
            if mp['total'] == 0:
                continue
            wr = mp['wins'] / mp['total']
            avgp = mp['profit_sum'] / mp['total']

            # Adjust thresholds
            if wr < 0.4:
                old_thresh = config.model_confidence_thresholds.get(m_name, config.min_confidence_threshold)
                new_thresh = min(old_thresh+0.05, 0.99)
                config.model_confidence_thresholds[m_name] = new_thresh
                logging.info(f"Increasing threshold => {m_name}: {old_thresh:.2f} => {new_thresh:.2f}")

            if wr > 0.7 and avgp > 0:
                old_thresh = config.model_confidence_thresholds.get(m_name, config.min_confidence_threshold)
                new_thresh = max(old_thresh - 0.02, config.min_confidence_threshold)
                config.model_confidence_thresholds[m_name] = new_thresh
                logging.info(f"Decreasing threshold => {m_name}: {old_thresh:.2f} => {new_thresh:.2f}")

            # Update weights
            perf_weight = 1.0 + (wr * avgp)
            model_weights[m_name] = perf_weight
    else:
        # if manual => revert to config
        model_weights.clear()
        for m in config.manual_model_weights:
            model_weights[m] = config.manual_model_weights[m]
