# trade_execution.py

import MetaTrader5 as mt5
from risk_management import calculate_risk_levels
from config import (dynamic_position_sizing, risk_per_trade_percent, 
                    atr_sl_tp, trailing_atr_timeframe, trailing_bars_to_fetch, volume_step)
import config
import math
import logging
from mt5_utils import fetch_data, get_timeframe_constant, get_open_trades
from indicators import calculate_atr  # Import ATR calculation
from datetime import datetime, timedelta

def calculate_atr_sl_tp(symbol, price, action, strategy):
    """
    Calculate SL and TP based on ATR.

    Args:
        symbol (str): Trading symbol.
        price (float): Current price.
        action (str): 'buy' or 'sell'.
        strategy (TradingStrategy): Instance of the trading strategy.

    Returns:
        tuple: (sl_price, tp_price)
    """
    try:
        # Fetch ATR using smaller timeframe
        atr_tf = get_timeframe_constant(config.trailing_atr_timeframe)
        atr_data = fetch_data(symbol, atr_tf, count=config.trailing_bars_to_fetch)
        if atr_data.empty:
            logging.warning(f"No ATR data available for {symbol} on timeframe {config.trailing_atr_timeframe}. Using default SL/TP.")
            # Fallback to percentage-based SL/TP
            return calculate_risk_levels(price, action, config.sl_percent, config.tp_percent, symbol)

        atr_data = calculate_atr(atr_data)

        atr_value = atr_data['atr'].iloc[-1]
        if math.isnan(atr_value):
            logging.warning(f"ATR value is NaN for {symbol}. Using default SL/TP.")
            return calculate_risk_levels(price, action, config.sl_percent, config.tp_percent, symbol)


        symbol_settings = config.symbol_configs.get(symbol, {})
        atr_multiplier_sl = symbol_settings.get('atr_multiplier_sl', 1.8)
        atr_multiplier_tp = symbol_settings.get('atr_multiplier_tp', 2.5)

        sl_distance = atr_value * atr_multiplier_sl
        tp_distance = atr_value * atr_multiplier_tp

        if action == "buy":
            sl_price = price - sl_distance
            tp_price = price + tp_distance
        else:
            sl_price = price + sl_distance
            tp_price = price - tp_distance

        logging.debug(f"ATR-based SL/TP for {symbol} ({action}): SL={sl_price}, TP={tp_price}")
        return sl_price, tp_price
    except Exception as e:
        logging.error(f"Error calculating ATR-based SL/TP for {symbol}: {e}")
        # Fallback to fixed SL/TP
        return calculate_risk_levels(price, action, config.sl_percent, config.tp_percent, symbol)

def check_open_positions_count(symbol):
    """
    Check the number of open positions for a given symbol.

    Args:
        symbol (str): Trading symbol.

    Returns:
        int: Number of open positions.
    """
    positions = mt5.positions_get(symbol=symbol)
    count = len(positions) if positions else 0
    logging.debug(f"Open positions count for {symbol}: {count}")
    return count

def calculate_dynamic_lot(symbol, sl_price, entry_price, action, signal_strength="Normal"):
    """
    Calculate dynamic lot size based on account equity, risk parameters, Stop Loss distance, and volatility.

    Steps:
    1. Calculate the risk amount based on a percentage of equity.
    2. Determine the number of pips to Stop Loss.
    3. Calculate the initial lot size.
    4. Adjust lot size inversely based on ATR to account for volatility.
    5. Clamp the lot size between minimum and maximum constraints.
    6. Round down to the nearest volume step.

    Args:
        symbol (str): Trading symbol.
        sl_price (float): Stop Loss price.
        entry_price (float): Entry price.
        action (str): 'buy' or 'sell'.

    Returns:
        float: Calculated lot size.
    """
    try:
        # Apply dynamic lot sizing only if applicable
        if config.dynamic_signal_type != "All" and config.dynamic_signal_type != signal_strength:
            logging.info(f"Dynamic lot sizing not applied for signal type: {signal_strength}")
            return config.lot_size  # Default to a static lot size if not applicable
        
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to retrieve account information.")
            return config.min_lot_size

        equity = account_info.equity
        risk_amount = equity * (risk_per_trade_percent / 100.0)
        tick_size = mt5.symbol_info(symbol).point

        if action == "buy":
            pips_to_sl = (entry_price - sl_price) / tick_size
        else:
            pips_to_sl = (sl_price - entry_price) / tick_size

        logging.debug(f"Symbol: {symbol}")
        logging.debug(f"Equity: {equity}")
        logging.debug(f"Risk Amount ({risk_per_trade_percent}%): {risk_amount}")
        logging.debug(f"Pips to SL: {pips_to_sl}")

        pip_value_for_1lot = 10.0  # Typically, pip value for 1 lot is 10 for EURUSD
        lots = risk_amount / (abs(pips_to_sl) * pip_value_for_1lot)
        logging.debug(f"Initial calculated lots before scaling: {lots}")

        # Volatility Adjustment using ATR
        if atr_sl_tp:
            atr_tf = get_timeframe_constant(trailing_atr_timeframe)
            atr_data = fetch_data(symbol, atr_tf, count=trailing_bars_to_fetch)
            if atr_data.empty:
                logging.warning(f"No ATR data available for {symbol} on timeframe {trailing_atr_timeframe}. Skipping volatility adjustment.")
            else:
                atr_data = calculate_atr(atr_data)
                atr = atr_data['atr'].iloc[-1]
                if math.isnan(atr) or atr == 0:
                    logging.warning(f"Invalid ATR value for {symbol}. Skipping volatility adjustment.")
                else:
                    scaling_factor = 1 / atr
                    lots *= scaling_factor
                    logging.debug(f"ATR: {atr}, Scaling Factor: {scaling_factor}, Lots after scaling: {lots}")

        # Ensure lot_size is within min and max constraints
        lots = max(config.min_lot_size, min(lots, config.max_lot_size))
        logging.debug(f"Lot size after clamping between {config.min_lot_size} and {config.max_lot_size}: {lots}")

        # Adjust to nearest volume_step
        if volume_step > 0:
            steps = math.floor(lots / volume_step)
            final_volume = steps * volume_step
            logging.debug(f"Lot size rounded down to nearest step {volume_step}: {final_volume}")
        else:
            final_volume = lots
            logging.debug(f"Lot size without step adjustment: {final_volume}")

        # Final validation
        if final_volume < config.min_lot_size:
            final_volume = config.min_lot_size
            logging.debug(f"Final volume adjusted to min_lot_size: {final_volume}")
        elif final_volume > config.max_lot_size:
            final_volume = config.max_lot_size
            logging.debug(f"Final volume adjusted to max_lot_size: {final_volume}")

        if final_volume < config.min_lot_size or final_volume > config.max_lot_size or final_volume <= 0:
            logging.error(f"Invalid trade volume for {symbol}: {final_volume} lots.")
            return config.min_lot_size

        logging.info(f"Dynamic lot size for {symbol}: {final_volume} lots.")
        return final_volume
    except Exception as e:
        logging.error(f"Error calculating dynamic lot size for {symbol}: {e}")
        return config.min_lot_size  # Fallback to minimum lot size
    
def place_trade(symbol, action, lot_size, sl_percent, tp_percent, strategy=None, signal_strength="Normal"):
    """
    Place a trade order.

    Args:
        symbol (str): Trading symbol.
        action (str): 'buy' or 'sell'.
        lot_size (float): Volume of the trade.
        sl_percent (float): Stop Loss percentage.
        tp_percent (float): Take Profit percentage.
        strategy (TradingStrategy, optional): Instance for accessing ATR. Defaults to None.
    """
    try:
        symbol_tick = mt5.symbol_info_tick(symbol)
        if not symbol_tick:
            logging.error(f"Tick data not available for {symbol}.")
            return

        price = symbol_tick.ask if action == "buy" else symbol_tick.bid

        # Use ATR-based SL/TP if enabled and strategy is provided
        if atr_sl_tp and strategy is not None:
            sl_price, tp_price = calculate_atr_sl_tp(symbol, price, action, strategy)
        else:
            # Fixed SL/TP
            sl_distance = price * (sl_percent / 100)
            tp_distance = price * (tp_percent / 100)
            sl_price = price - sl_distance if action == "buy" else price + sl_distance
            tp_price = price + tp_distance if action == "buy" else price - tp_distance

        # Dynamic position sizing if enabled and strategy is provided
        if dynamic_position_sizing and strategy is not None:
            lot_size = calculate_dynamic_lot(symbol, sl_price, price, action, signal_strength=signal_strength)

        # Adjust lot size to broker constraints
        info = mt5.symbol_info(symbol)
        if info is None:
            logging.error(f"Symbol info not found for {symbol} after volume calculation.")
            return

        volume_min = info.volume_min
        volume_max = info.volume_max
        volume_step = info.volume_step

        if volume_step > 0:
            steps = math.floor(lot_size / volume_step)
            final_volume = steps * volume_step
        else:
            final_volume = lot_size

        # Ensure final_volume is within allowed range
        if final_volume < volume_min:
            final_volume = volume_min
        elif final_volume > volume_max:
            final_volume = volume_max

        if final_volume < volume_min or final_volume > volume_max or final_volume <= 0:
            logging.error(f"Invalid trade volume for {symbol}: {final_volume} lots.")
            return

        # Check open positions count limit
        if check_open_positions_count(symbol) >= config.max_open_trades:
            logging.warning(f"Maximum open positions reached for {symbol}. Skipping trade.")
            return

        # Prepare trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": final_volume,
            "type": mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": config.deviation,
            "magic": config.magic_number,
            "comment": "Automated trade execution",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Trade failed for {symbol} ({action}). Error code: {result.retcode}, Comment: {result.comment}")
        else:
            logging.info(f"Trade placed: {action.upper()} {symbol}, Volume: {final_volume}, SL: {sl_price}, TP: {tp_price}")
    except Exception as e:
        logging.error(f"Exception during trade placement for {symbol}: {e}")

def close_position(symbol, volume, action):
    """
    Close an open trade position.

    Args:
        symbol (str): Trading symbol.
        volume (float): Volume to close.
        action (str): 'buy' or 'sell'.
    """
    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            logging.warning(f"No open positions found for {symbol}.")
            return None

        logging.info(f"Positions found for {symbol}: {len(positions)}")

        for pos in positions:
            if pos.type == action:  # Match directly with MT5 constants
                position_id = pos.ticket
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    logging.error(f"Tick data not available for {symbol}. Cannot close position.")
                    continue

                close_price = tick.ask if pos.type == mt5.ORDER_TYPE_SELL else tick.bid
                order_type = mt5.ORDER_TYPE_BUY if pos.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": order_type,
                    "price": close_price,
                    "deviation": config.deviation,
                    "magic": config.magic_number,
                    "comment": "Close Trade",
                    "position": position_id,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logging.error(f"Failed to close position {position_id} for {symbol}. Error code: {result.retcode}, Comment: {result.comment}")
                else:
                    logging.info(f"Position {position_id} closed for {symbol}.")
                return result

        logging.warning(f"No matching position found for {symbol} with action {action}.")
        return None
    except Exception as e:
        logging.error(f"Exception during position closure for {symbol}: {e}")
        return None

def check_open_positions(symbol):
    """
    Retrieve all currently open positions for a given symbol.

    Args:
        symbol (str): Trading symbol.

    Returns:
        list: List of open positions.
    """
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            logging.debug(f"Retrieved {len(positions)} open positions for {symbol}.")
            return positions
        logging.debug(f"No open positions for {symbol}.")
        return []
    except Exception as e:
        logging.error(f"Error retrieving open positions for {symbol}: {e}")
        return []

def review_positions(self):
    """
    Review open positions and decide actions based on current recommendations.
    """
    for symbol, recommendation in self.trade_recommendations.items():
        positions = check_open_positions(symbol)
        for position in positions:
            if recommendation['action'] == "sell" and position.type == mt5.ORDER_TYPE_BUY:
                close_position(symbol, position.volume, "sell")
            elif recommendation['action'] == "buy" and position.type == mt5.ORDER_TYPE_SELL:
                close_position(symbol, position.volume, "buy")
    logging.debug("Reviewed open positions based on current recommendations.")

def update_trailing_sl(strategy):
    """
    Update Stop Loss (SL) for open trades based on ATR and trailing mode.

    Args:
        strategy (TradingStrategy): Instance of the trading strategy.
    """
    try:
        open_trades = get_open_trades()
        if not open_trades:
            logging.info("No open trades to update trailing stops.")
            return

        for trade in open_trades:
            symbol = trade['symbol']
            action = "buy" if trade['type'] == mt5.ORDER_TYPE_BUY else "sell"
            entry_price = trade['entry_price']
            current_price = trade['current_price']
            ticket = trade['ticket']
            sl = trade['sl']
            tp = trade['tp']

            # Calculate initial TP distance
            initial_tp_distance = abs(tp - entry_price)

            # Calculate profit threshold for breakeven
            profit_threshold = config.break_even_factor * initial_tp_distance

            # Calculate current profit
            if action == "buy":
                current_profit = current_price - entry_price
            else:
                current_profit = entry_price - current_price

            logging.debug(f"Trade {ticket} ({action.upper()} {symbol}): Current Profit: {current_profit}, Profit Threshold: {profit_threshold}")

            # Check if profit threshold is reached
            if current_profit >= profit_threshold:
                if config.trailing_mode in ['breakeven', 'breakeven_then_normal']:
                    # Move SL to entry price if not already
                    if sl != entry_price:
                        logging.info(f"Breakeven triggered for trade {ticket} ({symbol}). Moving SL to entry price {entry_price}.")
                        modify_sl_tp(symbol, ticket, sl=entry_price, tp=tp)

                if config.trailing_mode in ['normal', 'breakeven_then_normal']:
                    # Calculate ATR-based trailing SL
                    atr_tf = get_timeframe_constant(config.trailing_atr_timeframe)
                    atr_data = fetch_data(symbol, atr_tf, count=config.trailing_bars_to_fetch)

                    # Ensure ATR is calculated
                    atr_data = calculate_atr(atr_data)

                    # Log the columns available in atr_data
                    logging.debug(f"ATR Data Columns for {symbol}: {atr_data.columns.tolist()}")

                    if atr_data.empty:
                        logging.warning(f"No ATR data available for {symbol} on timeframe {config.trailing_atr_timeframe}. Skipping trailing SL update.")
                        continue

                    if 'atr' not in atr_data.columns:
                        logging.error(f"'atr' column missing in ATR data for {symbol}. Skipping trailing SL update.")
                        continue

                    atr = atr_data['atr'].iloc[-1]
                    if math.isnan(atr) or atr == 0:
                        logging.warning(f"Invalid ATR value for {symbol}. Skipping trailing SL update.")
                        continue

                    trailing_sl_distance = atr * config.trailing_stop_fraction

                    # Determine new SL based on trailing mode
                    if action == "buy":
                        new_sl = current_price - trailing_sl_distance
                        if new_sl > sl:
                            logging.info(f"Normal trailing SL triggered for trade {ticket} ({symbol}). Moving SL from {sl} to {new_sl}.")
                            modify_sl_tp(symbol, ticket, sl=new_sl, tp=tp)
                    else:
                        new_sl = current_price + trailing_sl_distance
                        if new_sl < sl:
                            logging.info(f"Normal trailing SL triggered for trade {ticket} ({symbol}). Moving SL from {sl} to {new_sl}.")
                            modify_sl_tp(symbol, ticket, sl=new_sl, tp=tp)

        # After updating trailing stops, schedule the GUI update in the main thread
        if strategy.gui:
            strategy.gui.after(0, lambda: strategy.gui.open_trades_tab.refresh_open_trades())
    except Exception as e:
        logging.error(f"Exception during trailing SL update: {e}")

def modify_sl_tp(symbol, ticket, sl, tp):
    """
    Modify the SL and TP for a given position.

    Args:
        symbol (str): Trading symbol.
        ticket (int): Position ticket number.
        sl (float): New Stop Loss price.
        tp (float): New Take Profit price.
    """
    try:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": sl,
            "tp": tp
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to modify SL/TP for ticket {ticket}: retcode={result.retcode}, comment={result.comment}")
        else:
            logging.info(f"SL/TP modified for ticket {ticket} to SL={sl}, TP={tp}")
    except Exception as e:
        logging.error(f"Exception during SL/TP modification for ticket {ticket}: {e}")