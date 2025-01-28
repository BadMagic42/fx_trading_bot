# mt5_utils.py

import MetaTrader5 as mt5
import logging
import pandas as pd
from datetime import datetime, timezone
import config
import os

def initialize_mt5(login=None, password=None, server=None):
    """
    Initialize MT5 connection and log in if credentials are provided.

    Args:
        login (int, optional): MT5 account login.
        password (str, optional): MT5 account password.
        server (str, optional): MT5 server name.
    """
    try:
        if not mt5.initialize():
            logging.critical("MT5 initialization failed.")
            raise RuntimeError("MT5 initialization failed.")

        if login and password and server:
            authorized = mt5.login(login, password, server)
            if not authorized:
                logging.critical(f"Failed to connect to account {login}. Error: {mt5.last_error()}")
                raise RuntimeError(f"Failed to connect to account {login}. Error: {mt5.last_error()}")
            logging.info(f"Successfully logged into account {login}.")
        else:
            logging.info("Using currently logged-in MT5 account.")
    except Exception as e:
        logging.critical(f"Exception during MT5 initialization: {e}")
        raise
    
    
def fetch_data(symbol, timeframe, count=config.bars_to_fetch):
    """
    Fetch historical data for the given symbol and timeframe.

    Args:
        symbol (str): Trading symbol.
        timeframe (int): MT5 timeframe constant.
        count (int, optional): Number of bars to fetch. Defaults to config.bars_to_fetch.

    Returns:
        pd.DataFrame: Historical data as a DataFrame.
    """
    if count is None:
        count = config.bars_to_fetch + config.training_bars  # Calculate total needed bars based on config
    try:
        # Convert string-based timeframe to MT5 constant
        mt5_timeframe = get_timeframe_constant(timeframe)
    except Exception as e:
        raise ValueError(f"Invalid timeframe: {timeframe}. Error: {e}")    
    
    try:
        if not isinstance(count, int) or count <= 0:
            logging.error(f"Invalid count value: {count}. Must be a positive integer.")
            raise ValueError(f"Invalid count value: {count}. Must be a positive integer.")

        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logging.error(f"Failed to fetch data for {symbol} with timeframe {timeframe}.")
            raise RuntimeError(f"Failed to fetch data for symbol {symbol} with timeframe {timeframe}.")

        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        
        # Sort data in ascending order
        data.sort_values('time', inplace=True)
        data.reset_index(drop=True, inplace=True)

        logging.info(f"Fetched {data.shape[0]} rows for {symbol} with timeframe {timeframe}.")
        return data
    except Exception as e:
        logging.error(f"Exception during data fetching for {symbol}: {e}")
        raise


def get_timeframe_constant(timeframe):
    """
    Convert string-based timeframe (e.g., "H1") to MT5 constant.

    Args:
        timeframe (str): Timeframe string.

    Returns:
        int: MT5 timeframe constant.
    """
    timeframes = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    tf_constant = timeframes.get(timeframe, mt5.TIMEFRAME_H1)  # Default to H1 if not found
    logging.debug(f"Converted timeframe '{timeframe}' to MT5 constant '{tf_constant}'.")
    return tf_constant

def get_open_trades():
    """
    Retrieve all currently open trades from MT5.

    Returns:
        list: List of dictionaries containing open trade details.
    """
    open_trades = []
    try:
        positions = mt5.positions_get()
        if positions is None:
            logging.info("No open positions or error retrieving positions.")
            return open_trades

        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                logging.warning(f"Tick data not available for {pos.symbol}. Skipping trade {pos.ticket}.")
                continue

            current_price = tick.bid if pos.type == mt5.ORDER_TYPE_SELL else tick.ask

            open_trades.append({
                "symbol": pos.symbol,
                "volume": pos.volume,
                "entry_price": pos.price_open,
                "type": pos.type,  # 0 for buy, 1 for sell
                "profit": pos.profit,
                "ticket": pos.ticket,
                "entry_time": datetime.fromtimestamp(pos.time, tz=timezone.utc),
                "current_price": current_price,
                "sl": pos.sl,
                "tp": pos.tp
            })

        logging.info(f"Retrieved {len(open_trades)} open trades from MT5.")
        return open_trades
    except Exception as e:
        logging.error(f"Exception while retrieving open trades: {e}")
        return open_trades

def fetch_historical_deals(from_time=None, to_time=None, trade_type=None):
    """
    Fetch all historical deals within a specified timeframe, optionally filtering by trade type.
    If from_time and to_time are not provided, fetches all available historical deals.
    
    Args:
        from_time (datetime, optional): Start time. Defaults to datetime(2000, 1, 1).
        to_time (datetime, optional): End time. Defaults to datetime.now().
        trade_type (str, optional): 'BUY' or 'SELL'. Defaults to None.
    
    Returns:
        pd.DataFrame: Historical deals data.
    """
    try:
        # Set default dates if not provided
        if from_time is None:
            from_time = datetime(2000, 1, 1)  # Adjust as necessary
        if to_time is None:
            to_time = datetime.now()
        
        logging.debug(f"Fetching all deals from {from_time} to {to_time}. Trade Type: {trade_type}")
        
        # Use group='*' to include all symbols
        deals = mt5.history_deals_get(
            from_time,      # Positional argument: datetime object
            to_time,        # Positional argument: datetime object
            group='*'       # Named argument to include all symbols
        )
        
        if deals is None:
            last_error = mt5.last_error()
            logging.error(f"history_deals_get() failed, error code: {last_error}")
            return pd.DataFrame()
        
        if len(deals) == 0:
            logging.info(f"No historical deals found between {from_time} and {to_time}.")
            return pd.DataFrame()
        
        # Convert deals to a list of dictionaries
        deals_list = [deal._asdict() for deal in deals]
        
        # Create a DataFrame from the list
        deals_df = pd.DataFrame(deals_list)
        
        # Convert the 'time' column from timestamp to datetime
        if 'time' in deals_df.columns:
            deals_df['time'] = pd.to_datetime(deals_df['time'], unit='s')
        
        # Optional: Filter by trade_type if provided
        if trade_type:
            trade_type_map = {
                'BUY': mt5.ORDER_TYPE_BUY,
                'SELL': mt5.ORDER_TYPE_SELL
            }
            mt5_type = trade_type_map.get(trade_type.upper())
            if mt5_type is not None:
                deals_df = deals_df[deals_df['type'] == mt5_type]
                logging.info(f"Filtered historical deals with trade type '{trade_type}'.")
            else:
                logging.warning(f"Invalid trade_type '{trade_type}' provided. No filtering applied.")
        
        logging.info(f"Fetched {len(deals_df)} historical deals.")
        return deals_df
    except Exception as e:
        logging.error(f"Error fetching historical deals: {e}")
        return pd.DataFrame()