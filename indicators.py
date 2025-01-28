# indicators.py

import pandas as pd
import logging
import numpy as np
import config

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    """
    Calculate Bollinger Bands for the given data.

    Args:
        data (pd.DataFrame): DataFrame containing 'close' prices.
        window (int, optional): Rolling window size for moving average. Defaults to 20.
        num_std_dev (float, optional): Number of standard deviations for the bands. Defaults to 2.

    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands added.
    """
    try:
        data['ma'] = data['close'].rolling(window).mean()
        data['std_dev'] = data['close'].rolling(window).std()
        data['upper_band'] = data['ma'] + (num_std_dev * data['std_dev'])
        data['lower_band'] = data['ma'] - (num_std_dev * data['std_dev'])
        logging.debug(f"Bollinger Bands calculated for data. Latest values:\n{data[['close', 'lower_band', 'upper_band']].tail()}")
    except Exception as e:
        logging.error(f"Error calculating Bollinger Bands: {e}")
    return data


def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI) for the given data.

    Args:
        data (pd.DataFrame): DataFrame containing 'close' prices.
        period (int, optional): Period for RSI calculation. Defaults to 14.

    Returns:
        pd.DataFrame: DataFrame with RSI added.
    """
    if data.empty:
        logging.warning("Empty data received for RSI calculation.")
        return data

    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Debugging Output
    logging.debug("RSI calculated. Latest values:")
    logging.debug(data[['close', 'rsi']].tail())

    return data


def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) and Signal Line.

    Args:
        data (pd.DataFrame): DataFrame containing 'close' prices.
        fast_period (int, optional): Fast EMA period. Defaults to 12.
        slow_period (int, optional): Slow EMA period. Defaults to 26.
        signal_period (int, optional): Signal line EMA period. Defaults to 9.

    Returns:
        pd.DataFrame: DataFrame with MACD and Signal Line added.
    """
    try:
        data['macd'] = data['close'].ewm(span=fast_period, adjust=False).mean() - data['close'].ewm(span=slow_period, adjust=False).mean()
        data['macd_signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()
        logging.debug("MACD and Signal Line calculated.")
        logging.debug(data[['close', 'macd']].tail())
        logging.debug(data[['macd', 'macd_signal']].tail())
    except Exception as e:
        logging.error(f"Error calculating MACD: {e}")
    return data


def calculate_adx(data, period=14):
    """
    Calculate Average Directional Index (ADX) WITHOUT overwriting 'atr' in the main DataFrame.
    
    We compute an internal ATR-like series just for ADX to avoid duplicating 'data[\"atr\"]'.
    """
    try:
        high = data['high']
        low = data['low']
        close = data['close']

        # True Range for ADX only
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        # Compute this local ATR for ADX logic, do NOT assign to data['atr']
        adx_atr = tr.rolling(window=period).mean()

        # +DM / -DM
        plus_dm = ( (high - high.shift()) > (low.shift() - low) ) & ((high - high.shift()) > 0)
        minus_dm = ( (low.shift() - low) > (high - high.shift()) ) & ((low.shift() - low) > 0)

        data['+dm'] = (high - high.shift()).where(plus_dm, 0.0)
        data['-dm'] = (low.shift() - low).where(minus_dm, 0.0)

        # Rolling sums for +DM, -DM
        rolled_plus_dm = data['+dm'].rolling(window=period).sum()
        rolled_minus_dm = data['-dm'].rolling(window=period).sum()

        data['+di'] = (rolled_plus_dm / adx_atr) * 100.0
        data['-di'] = (rolled_minus_dm / adx_atr) * 100.0

        # ADX
        di_diff = (data['+di'] - data['-di']).abs()
        di_sum  = data['+di'] + data['-di']
        data['adx'] = (di_diff.rolling(window=period).mean() / di_sum) * 100.0

        logging.debug("ADX calculated without overwriting the main 'atr'.")
    except Exception as e:
        logging.error(f"Error calculating ADX: {e}")
    return data


def calculate_stochastic(data, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator (%K and %D).

    Args:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        k_period (int, optional): %K period. Defaults to 14.
        d_period (int, optional): %D period. Defaults to 3.

    Returns:
        pd.DataFrame: DataFrame with Stochastic %K and %D added.
    """
    try:
        data['lowest_low'] = data['low'].rolling(window=k_period).min()
        data['highest_high'] = data['high'].rolling(window=k_period).max()
        data['stochastic_k'] = ((data['close'] - data['lowest_low']) / (data['highest_high'] - data['lowest_low'])) * 100
        data['stochastic_d'] = data['stochastic_k'].rolling(window=d_period).mean()
        logging.debug("Stochastic Oscillator calculated.")
    except Exception as e:
        logging.error(f"Error calculating Stochastic Oscillator: {e}")
    return data


def calculate_atr(data, period=14):
    """
    Calculate Average True Range (ATR) and assign it to the appropriate column.

    Args:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        period (int, optional): Period for ATR calculation. Defaults to 14.

    Returns:
        pd.DataFrame: Modified DataFrame with ATR column added.
    """
    try:
        high = data['high']
        low = data['low']
        close = data['close']
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        # Use min_periods=period for reliability during training/backtesting
        atr = tr.rolling(window=period, min_periods=period).mean()
        logging.debug(f"ATR calculated for period {period}.")
    except Exception as e:
        logging.error(f"Error calculating ATR for period {period}: {e}")
        atr = pd.Series([np.nan]*len(data), index=data.index)
    
    # Assign to 'atr' if period is 14, else to 'atr_{period}'
    if period == 14:
        data['atr'] = atr
    else:
        atr_column = f'atr_{period}'
        data[atr_column] = atr
    return data  

def calculate_atr_multi(data, periods=[21, 28]):
    """
    Calculate ATR over multiple periods without overwriting the 'atr' column.

    Args:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        periods (list, optional): List of periods for ATR calculation. Defaults to [21, 28].

    Returns:
        pd.DataFrame: DataFrame with multiple ATR columns added.
    """
    try:
        for period in periods:
            data = calculate_atr(data, period=period)  # Properly update the DataFrame
            atr_column = f'atr_{period}'
            logging.debug(f"ATR for period {period} added as '{atr_column}'.")
    except Exception as e:
        logging.error(f"Error calculating multiple ATR periods: {e}")
    return data

def calculate_long_term_ma(data, period=200):
    """
    Calculate Long-Term Moving Average.

    Args:
        data (pd.DataFrame): DataFrame containing 'close' prices.
        period (int, optional): Period for moving average. Defaults to 200.

    Returns:
        pd.DataFrame: DataFrame with Long-Term MA added.
    """
    try:
        data['long_term_ma'] = data['close'].rolling(window=period).mean()
        logging.debug("Long-term Moving Average calculated.")
    except Exception as e:
        logging.error(f"Error calculating Long-Term MA: {e}")
    return data


def determine_trend(data, period=200):
    """
    Determine the market trend based on Long-Term Moving Average.

    Args:
        data (pd.DataFrame): DataFrame containing 'close' and 'long_term_ma' prices.
        period (int, optional): Period for moving average. Defaults to 200.

    Returns:
        str: 'up' for uptrend, 'down' for downtrend.
    """
    try:
        latest = data.iloc[-1]
        trend = "up" if latest['close'] > latest['long_term_ma'] else "down"
        logging.debug(f"Determined trend: {trend}")
        return trend
    except Exception as e:
        logging.error(f"Error determining trend: {e}")
        return "unknown"

def calculate_vwap(data):
    """
    Calculate tick_volume Weighted Average Price (VWAP).

    Args:
        data (pd.DataFrame): DataFrame containing 'high', 'low', 'close', and 'tick_volume'.

    Returns:
        pd.DataFrame: DataFrame with VWAP added.
    """
    try:
        # Typical Price
        data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        # Cumulative tick_volume-Weighted Typical Price
        data['cum_tp_vol'] = (data['typical_price'] * data['tick_volume']).cumsum()
        # Cumulative tick_volume
        data['cum_vol'] = data['tick_volume'].cumsum()
        # VWAP
        data['vwap'] = data['cum_tp_vol'] / data['cum_vol']
        # Clean up intermediate columns
        data.drop(['typical_price', 'cum_tp_vol', 'cum_vol'], axis=1, inplace=True)
        logging.debug("VWAP calculated.")
    except Exception as e:
        logging.error(f"Error calculating VWAP: {e}")
    return data

def calculate_ichimoku(data, tenkan_period=9, kijun_period=26, senkou_span_b_period=52):
    """
    Calculate Ichimoku Cloud indicators.

    Args:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        tenkan_period (int, optional): Period for Tenkan-sen. Defaults to 9.
        kijun_period (int, optional): Period for Kijun-sen. Defaults to 26.
        senkou_span_b_period (int, optional): Period for Senkou Span B. Defaults to 52.

    Returns:
        pd.DataFrame: DataFrame with Ichimoku components added.
    """
    try:
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                logging.error(f"Ichimoku calculation failed: '{col}' not in data.")
                return data

        logging.debug("Starting Ichimoku Cloud calculation.")

        # Calculate Tenkan-sen
        data['tenkan_sen'] = (data['high'].rolling(window=tenkan_period).max() + data['low'].rolling(window=tenkan_period).min()) / 2
        logging.debug("Tenkan-sen calculated.")

        # Calculate Kijun-sen
        data['kijun_sen'] = (data['high'].rolling(window=kijun_period).max() + data['low'].rolling(window=kijun_period).min()) / 2
        logging.debug("Kijun-sen calculated.")

        # Calculate Senkou Span A
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(kijun_period)
        logging.debug("Senkou Span A calculated and shifted.")

        # Calculate Senkou Span B
        data['senkou_span_b'] = ((data['high'].rolling(window=senkou_span_b_period).max() + data['low'].rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_period)
        logging.debug("Senkou Span B calculated and shifted.")

        # Calculate Chikou Span
        data['chikou_span'] = data['close'].shift(-kijun_period)
        logging.debug("Chikou Span calculated and shifted.")

        logging.debug("Ichimoku Cloud calculation completed.")

        # Verify Ichimoku columns
        ichimoku_columns = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        missing_columns = [col for col in ichimoku_columns if col not in data.columns or data[col].isnull().all()]
        if missing_columns:
            logging.error(f"Missing Ichimoku columns after calculation: {missing_columns}")
        else:
            logging.debug("All Ichimoku columns are present in the DataFrame.")

    except Exception as e:
        logging.error(f"Error calculating Ichimoku Cloud: {e}")
    return data


def calculate_keltner_channels(data, ema_period=20, atr_period=10, multiplier=1.5):
    """
    Calculate Keltner Channels for the given data.

    Args:
        data (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
        ema_period (int, optional): Period for Exponential Moving Average. Defaults to 20.
        atr_period (int, optional): Period for Average True Range. Defaults to 10.
        multiplier (float, optional): Multiplier for ATR. Defaults to 1.5.

    Returns:
        pd.DataFrame: DataFrame with Keltner Channels added.
    """
    try:
        data['ema'] = data['close'].ewm(span=ema_period, adjust=False).mean()
        data['atr'] = calculate_atr(data, period=atr_period)['atr']
        data['keltner_upper'] = data['ema'] + (multiplier * data['atr'])
        data['keltner_lower'] = data['ema'] - (multiplier * data['atr'])
        logging.debug("Keltner Channels calculated.")
    except Exception as e:
        logging.error(f"Error calculating Keltner Channels: {e}")
    return data

def create_lagged_features(data, features, lags=3):
    """
    Create lagged features for the specified columns.
    Args:
        data (pd.DataFrame): DataFrame containing features.
        features (list): List of feature column names to lag.
        lags (int, optional): Number of lag periods. Defaults to 3.
    Returns:
        pd.DataFrame: DataFrame with lagged features added.
    """
    try:
        lagged_data = {}
        for feature in features:
            if feature not in data.columns:
                logging.warning(f"Feature '{feature}' not found in data. Skipping lagged feature creation.")
                continue
            for lag in range(1, lags + 1):
                lagged_feature = f"{feature}_lag_{lag}"
                lagged_data[lagged_feature] = data[feature].shift(lag)
                logging.debug(f"Created lagged feature: {lagged_feature}")

        # Use pd.concat to add all lagged features to the original DataFrame at once
        lagged_df = pd.concat([data, pd.DataFrame(lagged_data, index=data.index)], axis=1)

        # Fill NaN values to avoid data loss
        lagged_df = lagged_df.ffill().bfill()

        logging.debug("Lagged features created successfully.")
        return lagged_df
    except Exception as e:
        logging.error(f"Error creating lagged features: {e}")
        raise

def calculate_obv(data):
    """
    Calculate On-Balance Volume (OBV) using vectorized operations.

    Args:
        data (pd.DataFrame): DataFrame containing 'close' and 'volume' columns.

    Returns:
        pd.DataFrame: DataFrame with 'obv' column added as float.
    """
    # Ensure 'close' and 'volume' are present
    if 'close' not in data.columns or 'tick_volume' not in data.columns:
        logging.error("Missing 'close' or 'tick_volume' columns for OBV calculation.")
        data['obv'] = np.nan
        return data

    # Calculate direction: 1 if current close > previous close, -1 if <, else 0
    direction = np.sign(data['close'].diff())
    direction.iloc[0] = 0  # First OBV value is zero

    # Multiply direction by volume
    obv_changes = direction * data['tick_volume'].fillna(0).values

    # Cumulative sum to get OBV
    obv = np.cumsum(obv_changes)

    data['obv'] = obv
    return data


def calculate_cmf(data, window=20):
    """
    Calculate Chaikin Money Flow (CMF).

    Args:
        data (pd.DataFrame): DataFrame containing 'high', 'low', 'close', and 'volume' columns.
        window (int): Number of periods to calculate the rolling CMF.

    Returns:
        pd.DataFrame: DataFrame with 'cmf' column added.
    """
    mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    mfm = mfm.fillna(0)  # Replace NaN with 0
    mfv = mfm * data['tick_volume']
    cmf = mfv.rolling(window=window).sum() / data['tick_volume'].rolling(window=window).sum()
    data['cmf'] = cmf
    return data

def calculate_volatility_percentile(data, window=20):
    """
    Calculate a 'rolling volatility percentile' or 'rolling standard deviation percentile.'
    For example:
      - We compute rolling std dev of returns over 'window'
      - Then, for each bar, we see how today's volatility ranks vs. the past 'window' bars.

    Args:
        data (pd.DataFrame): should contain 'close'.
        window (int): rolling window size.

    Returns:
        pd.DataFrame: with a new column like 'vol_percentile'.
    """
    if data.empty or 'close' not in data.columns:
        return data

    # 1) compute log returns or percent returns
    data['returns'] = data['close'].pct_change()
    # 2) compute rolling std dev
    data['rolling_std'] = data['returns'].rolling(window).std()

    # 3) percentile rank of the current rolling_std within that window
    # simplest approach: for each bar, we look back 'window' bars
    # and see what fraction of those is below the current bar’s std dev
    vol_perc = []
    rolling_stds = data['rolling_std'].tolist()

    for i in range(len(data)):
        if i < window:
            vol_perc.append(np.nan)
        else:
            # window slice = rolling_stds[i-window+1 : i+1]
            # We compare rolling_stds[i] to that slice
            window_slice = rolling_stds[i-window+1 : i+1]
            current_val = rolling_stds[i]
            if all(pd.isna(window_slice)) or pd.isna(current_val):
                vol_perc.append(np.nan)
            else:
                # fraction of the slice that is <= current
                rank = sum(x <= current_val for x in window_slice if not pd.isna(x))
                count_valid = sum(not pd.isna(x) for x in window_slice)
                if count_valid == 0:
                    vol_perc.append(np.nan)
                else:
                    percentile = rank / count_valid
                    vol_perc.append(percentile)

    data['vol_percentile'] = vol_perc

    # Optionally drop intermediate columns:
    # data.drop(['returns','rolling_std'], axis=1, inplace=True)

    return data

def calculate_pivot_points(data, method="classic"):
    """
    Calculate pivot points (and possibly R1, R2, S1, S2, etc.) based on the chosen method.
    Typically for 'classic' pivot points:
      Pivot = (High + Low + Close) / 3
      R1    = 2*Pivot - Low
      S1    = 2*Pivot - High
      R2    = Pivot + (High - Low)
      S2    = Pivot - (High - Low)
    ...
    
    Args:
        data (pd.DataFrame): Must contain 'high', 'low', 'close'.
        method (str): "classic", or you could extend with "camarilla", "woodie", etc.

    Returns:
        pd.DataFrame: Original `data` with columns like 'pivot', 'r1', 's1', 'r2', 's2', ...
    """
    if data.empty:
        return data
    if not all(col in data.columns for col in ['high','low','close']):
        return data  # or raise an error

    if method == "classic":
        pivot = (data['high'] + data['low'] + data['close']) / 3
        r1    = (2 * pivot) - data['low']
        s1    = (2 * pivot) - data['high']
        r2    = pivot + (data['high'] - data['low'])
        s2    = pivot - (data['high'] - data['low'])

        data['pivot'] = pivot
        data['r1'] = r1
        data['s1'] = s1
        data['r2'] = r2
        data['s2'] = s2

        # Optional: add R3/S3 or other pivot formulas as desired

    # Could implement other method (camarilla, woodie, etc.)
    # elif method == "woodie": ...
    # elif method == "camarilla": ...
    # else: 
    #   pass or raise an error

    return data