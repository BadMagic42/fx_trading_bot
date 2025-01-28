# main.py

import time
import pandas as pd
import warnings
from datetime import datetime, timedelta
import logging
from datetime import datetime, timezone
from mt5_utils import initialize_mt5, fetch_historical_deals, get_open_trades
from trade_execution import place_trade, review_positions, update_trailing_sl, close_position
from gui import TradingGUI
import config
import MetaTrader5 as mt5
import threading
import joblib
from model_manager import get_signals_for_symbol, update_model_performance, process_symbol_data, load_models
from ml_model import train_ml_model


logging.basicConfig(
    filename=config.log_file,
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress Matplotlib debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# Suppress Numba warnings
logging.getLogger('numba').setLevel(logging.WARNING)

def trailing_stop_loop(strategy):
    while True:
        if config.enable_trailing_stop:
            update_trailing_sl(strategy)
        time.sleep(config.scan_interval)

class TradingStrategy:
    def __init__(self, symbols, models):
        self.symbols = symbols
        self.models = models
        self.feature_names = []
        self.trade_recommendations = {}
        self.is_refreshing = False  # Added to stop duplicate refresh
        self.refresh_lock = threading.Lock()  # Added to stop duplicate refresh
        self.recent_trades = {}
        self.atr_values = {}
        self.trades_today = 0
        self.last_trade_day = None
        self.auto_mode = config.auto_mode
        self.initial_equity = None
        self.gui = None
        self.symbol_settings = {symbol: config.symbol_configs.get(symbol, {}) for symbol in symbols}
        self.symbol_timeframes = {symbol: self.symbol_settings[symbol].get("timeframe", config.timeframe) for symbol in symbols}  # Load symbol-specific timeframes
        # Dictionary to store opened trades: {ticket: {'symbol':..., 'models':..., 'open_time':...}}
        self.live_opened_trades = {}

    def reload_symbol_settings(self):
        self.symbol_settings = {symbol: config.symbol_configs.get(symbol, {}) for symbol in self.symbols}
        logging.info("Symbol-specific settings reloaded.")

    def within_trading_hours(self):
        now_utc = datetime.now(timezone.utc).hour
        return config.start_trading_hour <= now_utc < config.end_trading_hour

    def check_daily_trade_limit(self):
        today = datetime.now(timezone.utc).date()
        if self.last_trade_day != today:
            self.last_trade_day = today
            self.trades_today = 0
        return self.trades_today < config.max_trades_per_day

    def check_open_trades_limit(self):
        open_trades_count = mt5.positions_total()
        return open_trades_count is not None and open_trades_count < config.max_open_trades

    def check_equity_threshold(self):
        account_info = mt5.account_info()
        if account_info and account_info.equity >= config.min_equity_threshold:
            return True
        return False

    def increment_trade_count(self):
        today = datetime.now(timezone.utc).date()
        if self.last_trade_day != today:
            self.last_trade_day = today
            self.trades_today = 0
        self.trades_today += 1
        logging.info(f"Trade count incremented. Trades today: {self.trades_today}")

    def generate_trade_recommendations(self):
        """
        Generate trade recommendations using the ML models and optionally display feature importances,
        while ensuring we don't run this process concurrently from multiple threads.
        """
        # 1) First concurrency check: skip if already refreshing
        if self.is_refreshing:
            logging.warning("Already refreshing; skipping new request...")
            return

        # 2) Acquire the lock to prevent overlapping runs
        with self.refresh_lock:
            self.is_refreshing = True
            try:
                logging.info("Generating trade recommendations...")

                warnings.filterwarnings(
                    "ignore",
                    message=".*Usage of np.ndarray subset.*",
                    category=UserWarning
                )    

                # Check all necessary conditions
                if not self.within_trading_hours():
                    logging.warning("Outside trading hours. No recommendations generated.")
                    self.trade_recommendations.clear()
                    return
                if not self.check_daily_trade_limit():
                    logging.warning("Daily trade limit reached. No recommendations generated.")
                    self.trade_recommendations.clear()
                    return
                if not self.check_open_trades_limit():
                    logging.warning("Open trades limit reached. No recommendations generated.")
                    self.trade_recommendations.clear()
                    return
                if not self.check_equity_threshold():
                    logging.warning("Equity below minimum threshold. No recommendations generated.")
                    self.trade_recommendations.clear()
                    return
                    
                for symbol in self.symbols:
                    # Fetch open trades for this symbol
                    open_trades = [t for t in get_open_trades() if t['symbol'] == symbol]
                    symbol_open_trade_limit = self.symbol_settings.get(symbol, {}).get("max_open_trades", float('inf'))

                    if len(open_trades) >= symbol_open_trade_limit:
                        logging.warning(f"Open trade limit reached for {symbol}. Skipping recommendations.")
                        continue

                # Prepare data for all symbols (conditionally fetch training data)
                symbol_data = {}
                if getattr(config, 'enable_model_training', False):  # Only fetch training data if training is enabled
                    for symbol in self.symbols:
                        logging.info(f"Preparing training data for {symbol}")
                        data, _ = get_signals_for_symbol(symbol, self.symbol_timeframes[symbol], mode='training', models=self.models)
                        if data.empty:
                            logging.warning(f"No training data available for {symbol}")
                            continue
                        logging.debug(f"Training data for {symbol} contains {len(data)} rows.")
                        symbol_data[symbol] = data

                # Generate recommendations for each symbol (inference mode fetch)
                for symbol in self.symbols:
                    logging.info(f"Generating recommendation for {symbol}")
                    data, model_contributions = get_signals_for_symbol(symbol, self.symbol_timeframes[symbol], mode='inference', models=self.models)
                    if data.empty:
                        logging.warning(f"No data available for {symbol}")
                        continue
                    logging.info(f"Signal data for {symbol}: {data[['signal']].tail()}")
                    # logging.debug(f"Model contributions for {symbol}: {model_contributions}")

                    last_signal = data['signal'].iloc[-1]
                    if last_signal == 0:
                        action = 'sell'
                        reason = "SELL signal detected."
                    elif last_signal == 2:
                        action = 'buy'
                        reason = "BUY signal detected."
                    else:
                        action = None

                    if action:
                        # Calculate Signal Strength
                        strong_count = sum(
                            1 for model, _, confidence in model_contributions[-1] 
                            if confidence >= config.strong_threshold
                        )
                        is_strong = strong_count >= config.strong_min_models_agreement
                        signal_strength = "Strong" if is_strong else "Normal"

                        # Store model contributions for this bar
                        used_models = model_contributions[-1] if model_contributions else []

                        self.trade_recommendations[symbol] = {
                            'action': action,
                            'reason': reason,
                            'models': used_models,
                            'strength': signal_strength
                        }
                        logging.info(f"Trade recommendation for {symbol}: {self.trade_recommendations[symbol]}")
                        self.recent_trades[symbol] = datetime.now()
                    else:
                        logging.info(f"Skipping {symbol}. Last signal: {last_signal}")
                        # If no actionable signal, remove existing recommendation if it exists
                        if symbol in self.trade_recommendations:
                            self.trade_recommendations.pop(symbol, None)

                logging.info(f"Final trade recommendations: {self.trade_recommendations}")
                return True
            finally:
                # 3) Release concurrency lock / reset flag
                self.is_refreshing = False
                 
    def generate_open_trade_recommendations(self):
        """
        Assess open trades to decide if they should be held, closed early, or scaled.
        Uses the models and the data processing pipeline for a new prediction.
        """
        open_trades = get_open_trades()
        recommendations = []
        current_time = datetime.now(timezone.utc)

        server_offset = getattr(config, 'server_time_offset_hours', 0)

        if not open_trades:
            logging.info("No open trades to generate recommendations for.")
            return recommendations

        if config.scale_management_enabled and self.models:
            for trade in open_trades:
                entry_time = trade.get('entry_time')
                if entry_time:
                    entry_utc = entry_time - timedelta(hours=server_offset)
                    # entry_utc = datetime.fromtimestamp(entry_time, tz=timezone.utc) - timedelta(hours=server_offset)
                    time_since_open = current_time - entry_utc
                    if time_since_open < timedelta(minutes=config.recent_trade_window_minutes):
                        trade['recommendation'] = "Keep Open (Recently Opened)"
                        recommendations.append(trade)
                        continue

                symbol = trade['symbol']

                # Fetch data and predict signals again for open positions
                data, model_contributions = get_signals_for_symbol(symbol, self.symbol_timeframes[symbol], mode='inference', models=self.models)
                if data.empty:
                    trade['recommendation'] = "Keep Open"
                    recommendations.append(trade)
                    continue

                actionable = data[data['signal'] != 1]
                if actionable.empty:
                    if trade['profit'] > 0:
                        trade['recommendation'] = "CLOSE EARLY (No strong signal, lock profit)"
                    else:
                        trade['recommendation'] = "Keep Open"
                else:
                    last_row = actionable.iloc[-1]
                    new_action = "buy" if last_row['signal'] == 2 else "sell" if last_row['signal'] == 0 else "hold"
                    original_action = 'buy' if trade['type'] == mt5.ORDER_TYPE_BUY else 'sell'
                    if new_action == original_action:
                        trade['recommendation'] = "KEEP OPEN or SCALE IN (Favorable)"
                    else:
                        if trade['profit'] > 0:
                            trade['recommendation'] = "CLOSE NOW (Opposite signal, lock profit)"
                        else:
                            trade['recommendation'] = "CLOSE NOW (Cut losses)"
                recommendations.append(trade)
        else:
            # If scale_management not enabled, just simple logic
            for trade in open_trades:
                entry_time = trade.get('entry_time')
                if entry_time:
                    entry_utc = entry_time - timedelta(hours=server_offset)
                    # entry_utc = datetime.fromtimestamp(entry_time, tz=timezone.utc) - timedelta(hours=server_offset)
                    time_since_open = current_time - entry_utc
                    if time_since_open < timedelta(minutes=config.recent_trade_window_minutes):
                        trade['recommendation'] = "Keep Open (Recently Opened)"
                        recommendations.append(trade)
                        continue
                entry_price = trade['entry_price']
                current_price = trade['current_price']
                trade_type = 'BUY' if trade['type'] == mt5.ORDER_TYPE_BUY else 'SELL'
                sl_level = entry_price - (entry_price * config.sl_percent / 100) if trade_type == 'BUY' else entry_price + (entry_price * config.sl_percent / 100)
                tp_level = entry_price + (entry_price * config.tp_percent / 100) if trade_type == 'BUY' else entry_price - (entry_price * config.tp_percent / 100)
                if trade_type == 'BUY' and (current_price <= sl_level or current_price >= tp_level):
                    trade['recommendation'] = f"CLOSE (SL: {sl_level:.5f}, TP: {tp_level:.5f})"
                elif trade_type == 'SELL' and (current_price >= sl_level or current_price <= tp_level):
                    trade['recommendation'] = f"CLOSE (SL: {sl_level:.5f}, TP: {tp_level:.5f})"
                else:
                    trade['recommendation'] = "Keep Open"
                recommendations.append(trade)

        logging.info("Open trade recommendations generated.")
        return recommendations

    
    def get_feature_importances(self):
        """
        Retrieve feature importances from the trained models.

        Returns:
            pd.DataFrame: DataFrame containing model, feature, and importance.
        """
        if not self.models:
            logging.warning("No models available to retrieve feature importances.")
            return pd.DataFrame()

        feature_importances = []

        # Load trained feature names
        fn_path = config.BASE_DIR / "models/trained_feature_names.joblib"
        if not fn_path.exists():
            logging.error("Trained feature names file not found.")
            return pd.DataFrame()

        try:
            trained_feature_names = joblib.load(fn_path)
            logging.info(f"Loaded trained feature names: {trained_feature_names}")
        except Exception as e:
            logging.error(f"Error loading trained feature names: {e}")
            return pd.DataFrame()

        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                logging.debug(f"{model_name} feature importances: {importances}")
                logging.info(f"Feature names: {trained_feature_names}")
                feature_names = trained_feature_names

                # Check if lengths match
                if len(importances) != len(feature_names):
                    logging.warning(f"Mismatch in lengths of feature names and importances for {model_name}. Skipping.")
                    continue

                feature_importances.append(pd.DataFrame({
                    "model": [model_name] * len(feature_names),
                    "feature": feature_names,
                    "importance": importances
                }))
            else:
                logging.warning(f"Model {model_name} does not have feature_importances_ or feature_names_in_. Skipping.")

        if feature_importances:
            return pd.concat(feature_importances, axis=0)

        logging.warning("No valid feature importances found.")
        return pd.DataFrame()


    def review_positions(self):
        review_positions(self)

    def can_place_trade(self, symbol, action):
        """
        Check if a trade can be placed for a given symbol and action.
        Includes:
        1. Cooldown period check.
        2. Open trade existence check for the same symbol and action, using entry_time.
        Only applies when in auto mode.

        Returns:
            bool: True if the trade can be placed, False otherwise.
        """
        if not config.auto_mode:
            return True  # Allow trades in manual mode without restrictions

        current_time = datetime.now(timezone.utc)
        cooldown_time = config.trade_cooldown_times.get(symbol, 15)  # Default 15 minutes

        # Get open trades and check latest entry time for the symbol
        open_trades = get_open_trades()
        for trade in open_trades:
            if trade["symbol"] == symbol:
                trade_action = 'buy' if trade['type'] == mt5.ORDER_TYPE_BUY else 'sell'
                if trade_action == action:
                    # Cooldown check based on the trade's entry time
                    entry_time = trade["entry_time"]
                    if current_time - entry_time < timedelta(minutes=cooldown_time):
                        logging.info(f"Cooldown active for {symbol} ({action}). Last trade entry time: {entry_time}.")
                        return False
                    
        # If no conflicting trade or cooldown, allow the trade
        return True


    def execute_trades(self, selected_symbols=None):
        if not self.trade_recommendations:
            logging.info("No trade recommendations available. Skipping trade execution.")
            return
        if not self.check_open_trades_limit():
            return
        if not self.check_equity_threshold():
            return
        if selected_symbols:
            trades_to_execute = {s: r for s, r in self.trade_recommendations.items() if s in selected_symbols}
        else:
            trades_to_execute = self.trade_recommendations
            return

        trades_to_execute = {s: r for s, r in self.trade_recommendations.items() if not selected_symbols or s in selected_symbols}
        
        for symbol, recommendation in trades_to_execute.items():
            # Check if within trading hours
            if not self.within_trading_hours():
                continue

            # Check symbol-specific open trade limits
            open_trades = [t for t in get_open_trades() if t['symbol'] == symbol]
            symbol_open_trade_limit = self.symbol_settings.get(symbol, {}).get("max_open_trades", float('inf'))
            if len(open_trades) >= symbol_open_trade_limit:
                logging.info(f"Trade skipped for {symbol} due to open trade limit.")
                continue

            # Cooldown and open trade checks
            action = recommendation['action']
            if not self.can_place_trade(symbol, action):
                logging.info(f"Trade skipped for {symbol} ({action}) due to cooldown or existing trade.")
                continue

            # Place trade
            used_models = recommendation.get('models', [])
            signal_strength = recommendation.get('strength', 'Normal')  # Default to 'Normal'
            ticket = place_trade(symbol, action, config.lot_size, config.sl_percent, config.tp_percent, strategy=self, signal_strength=signal_strength)
            if ticket is not None:
                self.live_opened_trades[ticket] = {
                    'symbol': symbol,
                    'models': used_models,
                    'open_time': datetime.now()
                }
                self.increment_trade_count()                
            else:
                logging.warning(f"Trade for {symbol} could not be placed.")

        # Refresh GUI tabs if applicable
        if self.gui:
            self.gui.after(0, self.gui.refresh_all_tabs)

    def check_closed_trades(self):
        """
        Check historical deals to find if any of our opened trades have closed.
        Once found, call on_trade_closed with the profit and models.
        """
        # Fetch all historical deals
        deals_df = fetch_historical_deals()
        if deals_df.empty:
            return

        # We consider closed trades: entry=1 deals indicate a closing deal of a position in MT5.
        closed_deals = deals_df[deals_df['entry'] == 1]

        # For each opened trade we have, check if there's a closed deal that matches the position ticket.
        # In MT5 deals, the 'position_id' (or 'position' in some versions) matches the position ticket.
        # Check if deals_df has a column like 'position_id' or 'position'.
        # If not sure, we can try 'position_id' or use 'ticket' if it references the position.
        # Typically, 'ticket' in deals_df is the deal ticket, 'position_id' is the position ticket.
        # We'll assume 'position_id' column holds the position ticket.
        if 'position_id' not in closed_deals.columns:
            logging.warning("No 'position_id' column found in deals. Attempting 'position' column.")
            # Some MT5 versions or code may use 'position' or 'position_id' field.
            if 'position' in closed_deals.columns:
                position_col = 'position'
            else:
                logging.error("No position column found in deals. Cannot match closed trades.")
                return
        else:
            position_col = 'position_id'

        for ticket, trade_info in list(self.live_opened_trades.items()):
            # Find deals matching this ticket
            match = closed_deals[closed_deals[position_col] == ticket]
            if not match.empty:
                # A closed deal found for this ticket
                # Profit is sum of 'profit' from matched deals or last deal profit
                # Usually one closing deal per position, so take the first
                final_deal = match.iloc[-1]
                profit = final_deal['profit']
                self.on_trade_closed(ticket, profit)

    def on_trade_closed(self, ticket, profit):
        """
        Call this method when a trade identified by 'ticket' is closed.
        We have the profit and we know which models contributed from live_opened_trades.
        """
        if ticket in self.live_opened_trades:
            trade_data = self.live_opened_trades[ticket]
            results = [{
                'symbol': trade_data['symbol'],
                'profit': profit,
                'models': trade_data['models']
            }]
            update_model_performance(results)
            # Remove from opened trades
            self.live_opened_trades.pop(ticket, None)

    def close_open_trade(self, trade):
        close_position(trade['symbol'], trade['volume'], trade['type'])

    def run(self):
        if not config.auto_mode:
            threading.Thread(target=self.manual_mode_loop, daemon=True).start()
        if self.gui:
            self.gui.mainloop()

    def manual_mode_loop(self):
        while not config.auto_mode:
            try:
                logging.info("Manual mode: Generating trade recommendations...")
                self.generate_trade_recommendations()

                logging.info("Manual mode: Generating open trade recommendations...")
                self.generate_open_trade_recommendations()

                logging.info("Manual mode: Updating trailing stops...")
                update_trailing_sl(self)

                if self.gui:
                    logging.info("Refreshing GUI tabs in manual mode...")
                    self.gui.after(0, self.gui.refresh_all_tabs)
            except Exception as e:
                logging.error(f"Error in manual_mode_loop: {e}")
            
            time.sleep(config.scan_interval)


    def start_auto_mode(self):
        config.auto_mode = True
        self.auto_thread = threading.Thread(target=self.auto_trade_loop)
        self.auto_thread.daemon = True
        self.auto_thread.start()
        logging.info("Auto trading mode started.")

    def stop_auto_mode(self):
        config.auto_mode = False
        logging.info("Auto trading mode stopped.")

    def auto_trade_loop(self):
        while config.auto_mode:
            try:
                logging.info("Auto mode: Generating trade recommendations...")
                self.generate_trade_recommendations()

                logging.info("Auto mode: Reviewing positions...")
                self.review_positions()

                logging.info("Auto mode: Executing trades...")
                self.execute_trades()

                logging.info("Auto mode: Updating trailing stops...")
                update_trailing_sl(self)

                logging.info("Auto mode: Checking closed trades...")
                self.check_closed_trades()

                if self.gui:
                    logging.info("Refreshing GUI tabs in auto mode...")
                    self.gui.after(0, self.gui.refresh_all_tabs)

            except Exception as e:
                logging.error(f"Error in auto_trade_loop: {e}")

            # Wait for the configured interval
            time.sleep(config.scan_interval)

if __name__ == "__main__":
    try:
        logging.info("Starting Trading Strategy Bot...")
        initialize_mt5(login=config.login, password=config.password, server=config.server)

        # ---------------------
        # TRAINING MODE STAGE
        # ---------------------

        # Extract symbol_settings and symbol_timeframes from config.symbol_configs
        symbol_settings = {symbol: config.symbol_configs.get(symbol, {}) for symbol in config.symbols}
        symbol_timeframes = {symbol: symbol_settings[symbol].get("timeframe", config.timeframe) for symbol in config.symbols}
    

        # Check if model training is enabled
        if getattr(config, 'enable_model_training', False):
            logging.info("Training mode is enabled. Preparing to train and calibrate models...")
            symbol_data = {}
            for symbol in config.symbols:
                try:
                    # Fetch and process training data
                    symbol_timeframe = symbol_timeframes.get(symbol, config.timeframe)
                    logging.info(f"Fetching data for training {symbol}.")
                    data = process_symbol_data(symbol, symbol_timeframe, mode='training', count=None)
                    if data.empty:
                        logging.warning(f"No training data for {symbol}. Skipping training for this symbol.")
                    else:
                        logging.info(f"Training data prepared for {symbol}: {len(data)} rows.")
                        symbol_data[symbol] = data
                except Exception as e:
                    logging.error(f"Error fetching training data for {symbol}: {e}")
            
            if symbol_data:
                try:
                    logging.info("Training and calibrating models...")
                    trained_models, feature_names, feature_importances_df = train_ml_model(symbol_data)
                    logging.info("Models trained and calibrated successfully.")
                except Exception as e:
                    logging.critical(f"Error during model training and calibration: {e}")
                    raise
            else:
                logging.warning("No valid training data found. Proceeding without training.")
            
            # After training, load the calibrated models
            models = load_models()
            if not models:
                logging.critical("No models loaded after training. Exiting.")
                exit(1)
        else:
            # Load existing calibrated models
            logging.info("Training mode is disabled. Loading calibrated models...")
            models = load_models()
            if not models:
                logging.critical("No models loaded and training is disabled. Exiting.")
                exit(1)
        
        # Initialize the TradingStrategy with loaded models
        strategy = TradingStrategy(symbols=config.symbols, models=models)

        # ---------------------
        # LIVE/GUI MODE STAGE
        # ---------------------

        # Start Live/GUI Mode
        logging.info("Starting live/GUI mode...")
        gui = TradingGUI(
            strategy=strategy,
            start_auto_callback=strategy.start_auto_mode,
            stop_auto_callback=strategy.stop_auto_mode,
            execute_trades_callback=strategy.execute_trades
        )
        strategy.gui = gui
    
        # Start trailing stop thread
        trailing_thread = threading.Thread(target=trailing_stop_loop, args=(strategy,))
        trailing_thread.daemon = True
        trailing_thread.start()
        logging.info("Trailing stop thread started.")
    
        # Launch GUI
        logging.info("Launching GUI...")
        gui.mainloop()
    
    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}")
        raise
