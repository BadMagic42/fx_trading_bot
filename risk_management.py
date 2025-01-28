# risk_management.py

import MetaTrader5 as mt5
import logging

def calculate_risk_levels(price, action, sl_percent, tp_percent, symbol):
    """
    Calculate Stop Loss (SL) and Take Profit (TP) levels based on percentage of the current price.

    Args:
        price (float): Current price.
        action (str): 'buy' or 'sell'.
        sl_percent (float): SL as a percentage of the price.
        tp_percent (float): TP as a percentage of the price.
        symbol (str): Trading symbol.

    Returns:
        tuple: (SL price, TP price)
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error(f"Symbol info not available for {symbol}.")
            raise RuntimeError(f"Symbol info not available for {symbol}")

        sl_distance = price * (sl_percent / 100)
        tp_distance = price * (tp_percent / 100)

        if action == "buy":
            sl = price - sl_distance
            tp = price + tp_distance
        elif action == "sell":
            sl = price + sl_distance
            tp = price - tp_distance
        else:
            logging.error(f"Invalid action: {action}. Expected 'buy' or 'sell'.")
            raise ValueError(f"Invalid action: {action}. Expected 'buy' or 'sell'.")

        logging.debug(f"Calculated SL: {sl}, TP: {tp} for {symbol} ({action.upper()})")
        return sl, tp
    except Exception as e:
        logging.error(f"Error calculating risk levels for {symbol}: {e}")
        raise
