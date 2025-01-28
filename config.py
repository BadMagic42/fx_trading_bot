# config.py
"""
Configuration file for the trading strategy.

This file is structured into sections for clarity. Each section is commented,
and each configuration variable has a comment explaining its purpose.

The GUI will load and allow editing of these parameters, and can write them back here.
"""

import os
from pathlib import Path
import json

# -----------------------------
# ENVIRONMENT & PATHS
# -----------------------------
# The base directory and logging setup
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR
LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR / "fx_trading.log"

# -----------------------------
# ACCOUNT & CONNECTION
# -----------------------------
login_str = os.getenv('MT5_LOGIN')
if login_str is None:
    raise RuntimeError("MT5_LOGIN environment variable not set.")
login = int(login_str)

password = os.getenv('MT5_PASSWORD')
if password is None:
    raise RuntimeError("MT5_PASSWORD environment variable not set.")

server = os.getenv('MT5_SERVER')
if server is None:
    raise RuntimeError("MT5_SERVER environment variable not set.")



server_time_offset_hours = 2  # Adjust if broker server time differs from UTC

# -----------------------------
# SYMBOLS & TIMEFRAMES
# -----------------------------
# Primary symbols to trade
symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURGBP"]
currencies = ["EUR", "USD", "GBP", "JPY", "AUD", "CAD"]
# potentially add EURCHF, NZDUSD, USDSEK


# Load symbol-specific configuration if exists
symbol_config_path = BASE_DIR / "symbol_configs.json"
if symbol_config_path.exists():
    with open(symbol_config_path, 'r') as f:
        symbol_configs = json.load(f)
else:
    symbol_configs = {}

# -----------------------------
# TRADING PARAMETERS
# -----------------------------
lot_size = 0.1
sl_percent = 0.1
tp_percent = 0.3
magic_number = 100001      # Unique identifier for this strategy's orders

dynamic_position_sizing = True
risk_per_trade_percent = 1.0
dynamic_signal_type = "Strong"
fixed_risk_percent = 1.0  # Default risk percentage for non-dynamic sizing


min_lot_size = 0.1
max_lot_size = 0.3
volume_step = 0.1

symbol_last_trade_time = {}

# -----------------------------
# RISK & MANAGEMENT LIMITS
# -----------------------------
max_trades_per_day = 30
max_open_trades = 12
min_equity_threshold = 500.0
start_trading_hour = 6
end_trading_hour = 22

trade_cooldown_times = {
    "EURUSD": 15,  # Cooldown in minutes
    "GBPUSD": 15,
    "USDJPY": 30,
    "AUDUSD": 15,
    "USDCAD": 15,
    "EURGBP": 15
}

scale_management_enabled = False

# -----------------------------
# TRAILING STOP & ADVANCED SETTINGS
# -----------------------------
enable_trailing_stop = True
trailing_stop_fraction = 0.6
break_even_factor = 0.15
trailing_mode = "normal"
trailing_atr_timeframe = "M15"
trailing_bars_to_fetch = 500
# ATR-based SL/TP configuration
atr_sl_tp = True



# -----------------------------
# FILTERS & INDICATORS
# -----------------------------
trend_filter_on = False
long_term_ma_period = 200
rsi_buy_limit = 30
rsi_sell_limit = 70
post_pred_atr_val = 0.01

atr_threshold_period = 14

# -----------------------------
# FEATURES & MODEL TRAINING
# -----------------------------

enable_model_training = True
save_training_data = True

save_plots = True

enable_pca = True

features_selected = {   'adx': True,
    'atr': True,
    'atr_multi': True,
    'bollinger_bands': True,    
    'chikou_atr': False,
    'chikou_span': True,
    'cmf': False,
    'keltner_channels': False,
    'kijun_sen': False,
    'lagged_features': False,
    'long_term_ma': False,
    'macd': False,
    'macd_std5': False,
    'obv': True,
    'rsi': False,
    'rsi_ma5': False,
    'senkou_span_a': False,
    'senkou_span_b': False,
    'stochastic': False,
    'tenkan_sen': False,
    'vwap': True,
    'vwap_atr': False,
    'vol_percentile': True,
    'pivot_points': True
    }

lagged_features_selected = {   'adx': False,
    'atr': False,
    'atr_multi': False,
    'bollinger_bands': False,
    'chikou_span': False,
    'keltner_channels': False,
    'kijun_sen': False,
    'long_term_ma': False,
    'macd': False,
    'rsi': False,
    'obv': False,
    'cmf': False,
    'senkou_span_a': False,
    'senkou_span_b': False,
    'stochastic': False,
    'tenkan_sen': False,
    'vwap': False}

max_lag = 3

base_feature_map = {
    'adx': ['adx', '+di', '-di', '+dm', '-dm'],
    'atr': ['atr'],
    'atr_multi': ['atr_21', 'atr_28'],
    'bollinger_bands': ['lower_band', 'upper_band'],
    'chikou_span': ['chikou_span'],
    'keltner_channels': ['keltner_upper', 'keltner_lower'],
    'kijun_sen': ['kijun_sen'],
    'long_term_ma': ['long_term_ma'],
    'macd': ['macd', 'macd_signal'],
    'rsi': ['rsi'],
    'senkou_span_a': ['senkou_span_a'],
    'senkou_span_b': ['senkou_span_b'],
    'stochastic': ['stochastic_k', 'stochastic_d'],
    'tenkan_sen': ['tenkan_sen'],
    'vwap': ['vwap'],
    # === New Features ===
    'obv': ['obv'],
    'cmf': ['cmf'],
    'vwap_atr': ['vwap_atr'],
    'chikou_atr': ['chikou_atr'],
    'rsi_ma5': ['rsi_ma5'],
    'macd_std5': ['macd_std5'],
    'vol_percentile': ['vol_percentile'],
    'pivot_points': ['pivot', 'r1', 's1', 'r2', 's2']
}

feature_scaling = "standard"

# Models available
models_available = {   'LightGBM': True,
    'Neural Network': False,
    'Random Forest': True,
    'XGBoost': True}

enable_model_tuning = True
rf_max_depth = 15
rf_n_estimators = 200

min_confidence_threshold = 0.65
model_confidence_thresholds = {'LightGBM': 0.1, 'Neural Network': 0.1, 'Random Forest': 0.1, 'XGBoost': 0.1}

strong_threshold = 0.75
strong_min_models_agreement = 2


# -----------------------------
# MODEL WEIGHTING & ENSEMBLE
# -----------------------------
# model_weighting_mode = "manual" or "performance_based"
model_weighting_mode = "manual"

# If manual, define weights here
manual_model_weights = {'LightGBM': 1.0, 'Neural Network': 1.0, 'Random Forest': 1.0, 'XGBoost': 1.0}

min_models_agreement = 1

# Rolling window size for performance (number of recent trades to consider)
rolling_window_size = 50

# -----------------------------
# BACKTESTING CONFIG
# -----------------------------
backtest_mode = True
backtest_start_date = '2022-01-01'
backtest_end_date = '2023-01-01'
backtest_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURGBP"]
backtest_timeframe = 'H1'
backtest_initial_capital = 1000
backtest_slippage = 0.0001
backtest_commission = 0.0002
backtest_total_bars = 2000

BTEST_DIR = BASE_DIR
btest_file = BTEST_DIR / "backtest_results/backtest.log"

# -----------------------------
# SCAN INTERVAL & OTHERS
# -----------------------------
scan_interval = 600
recent_trade_window_minutes = 5
deviation = 10 # Acceptable deviation in pips
look_ahead = 2 # Look ahead in bars for prediction
threshold = 0.002
balancing_on_off = True
bars_to_fetch = 3000
timeframe = "H1"
training_bars = 30000
auto_mode = False
dynamic_atr_window = 100
calibrate_perc = 0.2  # 20% of data for calibration
test_perc = 0.2       # 20% of data for test


# -----------------------------
# LLM Related
# -----------------------------

#alpha_key = 'test'
alpha_key = os.getenv('ALPHA_API')
if alpha_key is None:
    raise RuntimeError("ALPHA_API environment variable not set.")

#openai_key = 'test'
openai_key = os.getenv('OPENAI_API')
if openai_key is None:
    raise RuntimeError("OPENAI_API environment variable not set.")

#gemini_key = 'test'
gemini_key = os.getenv('GEMINI_API')
if gemini_key is None:
    raise RuntimeError("GEMINI_API environment variable not set.")

#marketaux_key = 'test'
marketaux_key = os.getenv('MARKETAUX_API')
if marketaux_key is None:
    raise RuntimeError("MARKETAUX_API environment variable not set.")

ALPHAVANTAGE_FOREX_MAP = {
    "EURUSD": "FOREX:EUR,FOREX:USD",
    "GBPUSD": "FOREX:GBP,FOREX:USD",
    "USDJPY": "FOREX:USD,FOREX:JPY",
    "AUDUSD": "FOREX:AUD,FOREX:USD",
    "USDCAD": "FOREX:USD,FOREX:CAD",
    "EURGBP": "FOREX:EUR,FOREX:GBP",
}

MARKETAUX_FOREX_KEYWORDS = {
    "EURUSD": ["EUR", "Euro", "USD", "Dollar", "eu", "us"],
    "GBPUSD": ["GBP", "Pound", "USD", "Dollar", "gb", "us"],
    "USDJPY": ["USD", "Dollar", "JPY", "Yen", "us", "jp"],
    "AUDUSD": ["AUD", "Australian", "USD", "Dollar", "au", "us"],
    "USDCAD": ["USD", "Dollar", "CAD", "Canadian", "us", "ca"],
    "EURGBP": ["EUR", "Euro", "GBP", "Pound", "eu", "gb"],
}


# -----------------------------
# HyperParameter Related
# -----------------------------

# Hyperparameter Tuning Configuration
ENABLE_HYPERPARAMETER_TUNING = True  # Toggle hyperparameter tuning
TUNING_METHOD = 'grid_search'        # Options: 'grid_search', 'random_search', 'tpot', 'bayesian_optimization'
TUNING_PARAMETERS = {
    'Random Forest': {
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', {0: 2, 1: 1, 2: 2}], # maybe delete or change to class_weight='balanced'
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'objective': ['multi:softprob'],
        'eval_metric': ['mlogloss']
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'max_depth': [7, 10, 20, -1], # was 10, 20, -1
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],              # Range of leaves (check overfitting vs. underfitting)
        'min_child_samples': [20, 50, 100],       # Similar to min_data_in_leaf
        # 'bagging_fraction': [0.8, 1.0],           # Stochastic gradient boosting (row subsampling)
        # 'feature_fraction': [0.8, 1.0],   
        'objective': ['multiclass'],
        'num_class': [3],
        'class_weight': ['balanced', {0: 3, 1: 1, 2: 3}], # maybe delete
        'metric': ['multi_logloss'],
        'boosting_type': ['gbdt', 'dart']
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu'],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.0001],
        'momentum': [0.8, 0.9, 0.99]
    },
}

# AutoML Configuration (TPOT)
AUTOML_GENERATIONS = 5
AUTOML_POPULATION_SIZE = 50

# Bayesian Optimization Configuration (skopt)
BAYESIAN_MAX_EVALS = 30

# Resampling Configuration
RESAMPLING_METHOD = 'NONE'  # Options: 'SMOTE', 'SMOTEEN', 'ADASYN', 'NONE'

# LSTM Model Configuration
ENABLE_LSTM = True
lstm_epochs = 50
lstm_batch_size = 64
lstm_hidden_size = 32
lstm_num_layers = 2
lstm_dropout = 0.2
lstm_patience = 10
lstm_learning_rate = 0.001
lstm_bidirectional = False
lstm_sequence_length = 10

# SARIMA Model Configuration
ENABLE_SARIMA = True
sarima_order = (1, 1, 1)
sarima_seasonal_order = (1, 1, 1, 12)

# Meta Model Configuration
ENABLE_META_MODEL = True


# Model Explainability
ENABLE_SHAP = False
ENABLE_LIME = True


