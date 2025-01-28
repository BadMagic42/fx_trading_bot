# data_processing.py

import logging
import pandas as pd
import numpy as np
import config
from mt5_utils import fetch_data
import requests
import datetime

from indicators import (
    calculate_bollinger_bands, calculate_rsi, calculate_macd, calculate_vwap,
    calculate_ichimoku, calculate_keltner_channels, create_lagged_features,
    calculate_atr_multi, calculate_adx, calculate_stochastic, calculate_atr,
    calculate_long_term_ma, calculate_obv, calculate_cmf, calculate_pivot_points, 
    calculate_volatility_percentile
)

import numpy as np
import logging
import config
import pandas as pd

def determine_atr_based_threshold(data, atr_multiplier, atr_period=14, symbol=None):
    """
    Calculate threshold based on ATR and a multiplier, without adding new ATR columns to `data`.
    If `atr_period == 14`, we use the existing 'atr' column in `data`.
    Otherwise, compute a custom ATR inline for that period and use it locally.

    Args:
        data (pd.DataFrame): DataFrame containing columns for 'close' and, if atr_period == 14, 'atr'.
            Otherwise, must contain 'high', 'low', and 'close' to compute custom ATR.
        atr_multiplier (float): Multiplier for the ATR value.
        atr_period (int, optional): Period for ATR calculation. Default is 14.
        symbol (str, optional): Symbol key for symbol-specific config. Defaults to None.

    Returns:
        pd.Series: Threshold values, indexed like `data`.
    """
    # Get symbol-specific or global 'look_ahead'
    symbol_settings = config.symbol_configs.get(symbol, {})
    look_ahead = symbol_settings.get('look_ahead', config.look_ahead)

    # Decide whether to use the DataFrame's existing ATR column (period=14) or compute custom ATR
    if atr_period == 14:
        # Must have 'atr' and 'close' columns
        if 'atr' not in data.columns or 'close' not in data.columns:
            raise ValueError("Data must contain 'atr' and 'close' columns for period=14.")
        local_atr = data['atr']
    else:
        # Compute ATR inline for the given period
        required_cols = {'high', 'low', 'close'}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Data must contain columns {required_cols} to compute a custom ATR.")

        try:
            high = data['high']
            low = data['low']
            close = data['close']

            true_range = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)

            local_atr = true_range.rolling(window=atr_period, min_periods=atr_period).mean()
            logging.debug(f"Inline ATR calculated for period={atr_period}.")
        except Exception as e:
            logging.error(f"Error computing custom ATR with period {atr_period}: {e}")
            local_atr = pd.Series([np.nan]*len(data), index=data.index)

    # Ensure we have at least some valid ATR values
    if local_atr.isna().all():
        raise ValueError(f"All-NaN ATR values encountered for period={atr_period}. Cannot compute threshold.")

    if 'close' not in data.columns:
        raise ValueError("Data must contain 'close' column to compute threshold.")

    # Volatility scaling factor
    scaling_factor = np.sqrt(look_ahead)

    # Final threshold calculation
    threshold = (local_atr * atr_multiplier * scaling_factor) / data['close']

    return threshold


def process_symbol_data(symbol, timeframe, mode='inference', count=None):
    """
    Fetch, process, and optionally label data for the given symbol and timeframe.
    This function is the single pipeline for data processing in both live and backtest modes.

    Args:
        symbol (str): Trading symbol.
        timeframe (str): Timeframe string (e.g., "H1").
        mode (str): 'inference' or 'training'. If 'training', label data for model training.
        count (int): Number of bars to fetch. If None, defaults to config.bars_to_fetch.

    Returns:
        pd.DataFrame: Processed and (if training mode) labeled data.
    """
    
    # Centralized logic for determining the count
    if count is None:
        if mode == 'training':
            count = config.training_bars + config.bars_to_fetch  # Training requires both training and fetch bars
        elif mode == 'inference':
            count = config.bars_to_fetch
        else:
            raise ValueError(f"Unknown mode: {mode}")

    data = fetch_data(symbol, timeframe, count=count)
    if data.empty:
        logging.warning(f"No data fetched for {symbol} on {timeframe}.")
        return pd.DataFrame()
    
    logging.debug(f"Fetched {len(data)} rows for {symbol}.")

    # Segmentation: Adjust training and inference ranges
    if mode == 'training':
        logging.info(f"Processing data for {symbol} on {timeframe}. Mode: training. Count: {count}")
        # Exclude the most recent bars_to_fetch rows and use only the preceding training_bars rows
        data = data.iloc[:-config.bars_to_fetch]  # Exclude the most recent rows
        data = data.iloc[-config.training_bars:]  # Use the last `training_bars` rows from remaining data
    elif mode == 'inference':
        logging.info(f"Processing data for {symbol} on {timeframe}. Mode: inference. Count: {count}")
        # Use only the most recent bars_to_fetch rows
        data = data.iloc[-config.bars_to_fetch:]

    if mode == 'training':
        logging.info(f"Training data range: {data['time'].min()} to {data['time'].max()}")
    elif mode == 'inference':
        logging.info(f"Inference data range: {data['time'].min()} to {data['time'].max()}")

    # Standardize column names
    data = data.rename(columns=str.lower)

    # Calculate indicators as per features_selected
    if config.features_selected.get('rsi', False):
        data = calculate_rsi(data)
        logging.debug(f"RSI values for {symbol}: {data['rsi'].tail()}")
    if config.features_selected.get('bollinger_bands', False):
        data = calculate_bollinger_bands(data)
        logging.debug(f"Bollinger Bands for {symbol}: {data[['upper_band', 'lower_band']].tail()}")
    if config.features_selected.get('macd', False):
        data = calculate_macd(data)
        logging.debug(f"MACD for {symbol}: {data[['macd', 'macd_signal']].tail()}")
    if config.features_selected.get('adx', False):
        data = calculate_adx(data)
        logging.debug(f"ADX values for {symbol}: {data[['adx', '+di', '-di', '+dm', '-dm']].tail()}")
    if config.features_selected.get('stochastic', False):
        data = calculate_stochastic(data)
        logging.debug(f"Stochastic Oscillator for {symbol}: {data[['stochastic_k', 'stochastic_d']].tail()}")
    if config.features_selected.get('atr', False):
        data = calculate_atr(data, period=14)  # This assigns to 'atr'
        logging.debug(f"ATR for {symbol}: {data[['atr']].tail()}")
    if config.features_selected.get('long_term_ma', False):
        data = calculate_long_term_ma(data, config.long_term_ma_period)
        logging.debug(f"Long-Term Moving Average for {symbol}: {data[['long_term_ma']].tail()}")
    if config.features_selected.get('vwap', False):
        data = calculate_vwap(data)
        logging.debug(f"VWAP for {symbol}: {data[['vwap']].tail()}")
    if config.features_selected.get('pivot_points', False):
        data = calculate_pivot_points(data, method="classic")
        logging.debug(f"Pivot Points for {symbol}: {data[['pivot','r1','s1','r2','s2']].tail()}")
    if config.features_selected.get('vol_percentile', False):
        data = calculate_volatility_percentile(data, window=20)
        logging.debug(f"Vol Percentile for {symbol}: {data['vol_percentile'].tail()}")

    # Determine which Ichimoku components to calculate based on features_selected
    ichimoku_selected = {
        'tenkan_sen': config.features_selected.get('tenkan_sen', False),
        'kijun_sen': config.features_selected.get('kijun_sen', False),
        'senkou_span_a': config.features_selected.get('senkou_span_a', False),
        'senkou_span_b': config.features_selected.get('senkou_span_b', False),
        'chikou_span': config.features_selected.get('chikou_span', False),
    }
    # If any Ichimoku component is selected, calculate Ichimoku
    if any(ichimoku_selected.values()):
        data = calculate_ichimoku(data)  # Calculate all components
        # Drop Ichimoku components that are not selected
        components_to_drop = [key for key, selected in ichimoku_selected.items() if not selected]
        if components_to_drop:
            data.drop(columns=components_to_drop, inplace=True, errors='ignore')
        # Log only the selected components
        selected_components = [key for key, selected in ichimoku_selected.items() if selected]
        if selected_components:
            logging.debug(f"Ichimoku Cloud for {symbol}: {data[selected_components].tail()}")
    if config.features_selected.get('keltner_channels', False):
        data = calculate_keltner_channels(data)
        logging.debug(f"Keltner Channels for {symbol}: {data[['keltner_upper', 'keltner_lower']].tail()}")
    if config.features_selected.get('atr_multi', False):
        data = calculate_atr_multi(data, periods=[21, 28])  # This assigns to 'atr_21', 'atr_28'
        logging.debug(f"Multiple ATR values for {symbol}: {data[['atr_21', 'atr_28']].tail()}")

    # === Step 1: Add New Features ===

    # 1.1 On-Balance Volume (OBV)
    if config.features_selected.get('obv', False):
        data = calculate_obv(data)
        logging.debug(f"OBV for {symbol}: {data['obv'].tail()}")

    # 1.2 Chaikin Money Flow (CMF)
    if config.features_selected.get('cmf', False):
        data = calculate_cmf(data)
        logging.debug(f"CMF for {symbol}: {data['cmf'].tail()}")

    # 1.3 vwap_atr (VWAP × ATR)
    if config.features_selected.get('vwap_atr', False):
        if 'vwap' in data.columns and 'atr' in data.columns:
            data['vwap_atr'] = data['vwap'] * data['atr']
            logging.debug(f"vwap_atr for {symbol}: {data['vwap_atr'].tail()}")
        else:
            logging.warning(f"Cannot compute vwap_atr for {symbol}: 'vwap' or 'atr' not in data columns.")

    # 1.4 chikou_atr (Chikou Span × ATR)
    if config.features_selected.get('chikou_atr', False):
        if 'chikou_span' in data.columns and 'atr' in data.columns:
            data['chikou_atr'] = data['chikou_span'] * data['atr']
            logging.debug(f"chikou_atr for {symbol}: {data['chikou_atr'].tail()}")
        else:
            logging.warning(f"Cannot compute chikou_atr for {symbol}: 'chikou_span' or 'atr' not in data columns.")

    # 1.5 rsi_ma5 (5-Period Moving Average of RSI)
    if config.features_selected.get('rsi_ma5', False):
        if 'rsi' in data.columns:
            data['rsi_ma5'] = data['rsi'].rolling(window=5).mean()
            logging.debug(f"rsi_ma5 for {symbol}: {data['rsi_ma5'].tail()}")
        else:
            logging.warning(f"Cannot compute rsi_ma5 for {symbol}: 'rsi' not in data columns.")

    # 1.6 macd_std5 (5-Period Rolling Standard Deviation of MACD)
    if config.features_selected.get('macd_std5', False):
        if 'macd' in data.columns:
            data['macd_std5'] = data['macd'].rolling(window=5).std()
            logging.debug(f"macd_std5 for {symbol}: {data['macd_std5'].tail()}")
        else:
            logging.warning(f"Cannot compute macd_std5 for {symbol}: 'macd' not in data columns.")

    # === Step 2: Create Lagged Features if Enabled ===
    if config.features_selected.get('lagged_features', False):
        logging.debug(f"Calling create_lagged_features from data_processing.py for {symbol}")
        
        # Dynamically determine features to lag based on config.lagged_features_selected
        to_lag = []
        for feature_key, is_selected in config.lagged_features_selected.items():
            if is_selected:
                # Get the corresponding feature columns from base_feature_map
                feature_columns = config.base_feature_map.get(feature_key, [])
                to_lag.extend([col for col in feature_columns if col in data.columns])

        if to_lag:
            # Create lagged features
            data = create_lagged_features(data, features=to_lag, lags=config.max_lag)
            lagged_columns = [col for col in data.columns if 'lag_' in col]
            logging.debug(f"Lagged features created for {symbol}: {lagged_columns}")
        else:
            logging.warning(f"No features available to lag for {symbol}. Check config and data columns.")

    # === Step 3: Optional PCA ===
    if config.features_selected.get('enable_pca', False):
        # Identify numerical features (excluding 'label' if present)
        numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in numerical_features:
            numerical_features.remove('label')
        
        # Drop any NaN values that may have been introduced by feature engineering
        data = data.dropna(subset=numerical_features)
        
        # Standardize the features before applying PCA
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[numerical_features])
        logging.debug(f"Features scaled for PCA: {numerical_features}")

        # Apply PCA
        from sklearn.decomposition import PCA
        pca = PCA()
        principal_components = pca.fit_transform(data_scaled)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        logging.debug(f"PCA explained variance ratios: {explained_variance}")
        logging.debug(f"PCA cumulative variance: {cumulative_variance}")

        # Determine number of components to retain (e.g., 95% variance)
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(data_scaled)
        logging.info(f"PCA reduced feature set from {len(numerical_features)} to {n_components} components.")

        # Create DataFrame for principal components
        pca_columns = [f'pca_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(principal_components, columns=pca_columns, index=data.index)

        # Concatenate PCA components with the original data
        data = pd.concat([data.drop(columns=numerical_features), pca_df], axis=1)
        logging.debug(f"PCA features added for {symbol}: {pca_columns}")

    # === Step 4: Threshold Selection ===
    symbol_settings = config.symbol_configs.get(symbol, {})
    threshold_type = symbol_settings.get('threshold_type', 'normal')
    look_ahead = symbol_settings.get('look_ahead', config.look_ahead)
    
    if threshold_type == 'atr':
        atr_multiplier = symbol_settings.get('threshold_atr_multiplier', 1.0)
        atr_period = symbol_settings.get('threshold_atr_period', 14)  # or default from config

        # Now just call the function, passing the period
        threshold_series = determine_atr_based_threshold(
            data, 
            atr_multiplier=atr_multiplier,
            atr_period=atr_period,      # <— pass it in
            symbol=symbol
        )

        data['threshold'] = threshold_series
        logging.debug(f"Thresholds calculated (period={atr_period}): {data['threshold'].tail()}")

    else:
        threshold_value = symbol_settings.get('threshold', config.threshold)
        data['threshold'] = threshold_value
        logging.debug(f"Static threshold applied: {threshold_value}")

    # Labeling logic if training mode
    if mode == 'training':
        # Calculate 'future_close' and 'future_diff'
        future_close = data['close'].shift(-look_ahead)
        future_diff = (future_close - data['close']) / data['close']

        # Prepare label logic
        if threshold_type == 'atr':
            # Use per-bar thresholds
            labels = np.where(
                future_diff > data['threshold'], 1,
                np.where(future_diff < -data['threshold'], -1, 0)
            )
        else:
            # Use single threshold value
            labels = np.where(
                future_diff > threshold_value, 1,
                np.where(future_diff < -threshold_value, -1, 0)
            )

        # Add 'future_close', 'future_diff', and 'label' columns using pd.concat
        new_columns = pd.DataFrame({
            'future_close': future_close,
            'future_diff': future_diff,
            'label': labels
        }, index=data.index)

        data = pd.concat([data, new_columns], axis=1)

        # Drop unnecessary columns after labeling
        data.drop(['future_close', 'future_diff'], axis=1, inplace=True)

        # Remap labels from {-1,0,1} to {0,1,2}
        data['label'] = data['label'].map({-1:0, 0:1, 1:2})
        logging.debug(f"Label distribution for {symbol}: {data['label'].value_counts()}")
    else:
        # Inference mode: If labeling is needed by models, they do their own neutral labeling downstream.
        # We'll not add 'label' here.
        pass

    return data

# -----------------------------------------
# 1) FOREX FACTORY (Weekly XML) FETCH
# -----------------------------------------
import requests
import xml.etree.ElementTree as ET
import logging
from datetime import datetime

def scrape_forex_factory_weekly():
    """
    Fetch and parse the Forex Factory events from the XML feed.
    Returns a list[dict] with date_utc, country, impact, title, forecast, previous, and url.
    Logs an error if 429 or other HTTP errors occur, returning [].
    """
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    try:
        logging.info(f"Fetching ForexFactory data from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the XML content
        root = ET.fromstring(response.content)

        events = []
        for event in root.findall('event'):
            try:
                title = event.findtext('title', '').strip()
                country = event.findtext('country', '').strip()
                date = event.findtext('date', '').strip()
                time = event.findtext('time', '').strip()
                impact = event.findtext('impact', '').strip()
                forecast = event.findtext('forecast', '').strip()
                previous = event.findtext('previous', '').strip()
                url = event.findtext('url', '').strip()

                # Combine date and time into UTC datetime string
                if date and time:
                    date_time_str = f"{date} {time}"
                    dt_utc = datetime.strptime(date_time_str, "%m-%d-%Y %I:%M%p").strftime('%Y-%m-%d %H:%M:%S')
                else:
                    dt_utc = None

                # Append the event
                events.append({
                    "title": title,
                    "country": country,
                    "date_utc": dt_utc,
                    "impact": impact,
                    "forecast": forecast,
                    "previous": previous,
                    "url": url
                })
            except Exception as parse_err:
                logging.warning(f"Error parsing event: {parse_err}")

        logging.info(f"Successfully fetched {len(events)} events.")
        return events

    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 429:
            logging.error(f"Rate limit error (429). Error: {http_err}")
        else:
            logging.error(f"HTTP error fetching Forex Factory XML: {http_err}")
        return []
    except Exception as e:
        logging.error(f"Error fetching Forex Factory XML: {e}")
        return []

# -----------------------------------------
# 2) ALPHA VANTAGE NEWS SENTIMENT
# -----------------------------------------
def get_alpha_vantage_news_sentiment(
    symbol,
    time_from=None,
    time_to=None,
    topics=None,
    limit=50,
    sort="LATEST"
):
    """
    Fetch market news & sentiment data from Alpha Vantage's NEWS_SENTIMENT endpoint.
    """
    try:
        av_ticker = config.ALPHAVANTAGE_FOREX_MAP.get(symbol)
        if not av_ticker:
            logging.error(f"Symbol '{symbol}' not found in ALPHAVANTAGE_FOREX_MAP.")
            return {}

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": av_ticker,
            "apikey": config.alpha_key,
        }
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to
        if topics:
            params["topics"] = ",".join(topics)
        if limit:
            params["limit"] = str(limit)
        if sort:
            params["sort"] = sort

        logging.info(f"Fetching Alpha Vantage NEWS_SENTIMENT for {av_ticker} with params: {params}")
        resp = requests.get("https://www.alphavantage.co/query", params=params)
        resp.raise_for_status()

        data = resp.json()
        if "Error Message" in data:
            logging.error(f"Alpha Vantage API error: {data['Error Message']}")
            return {}

        logging.info(f"Alpha Vantage response for {av_ticker}: {len(data)} items received.")
        logging.debug(f"Raw Alpha Vantage response: {data}")
        return data
        

    except Exception as e:
        logging.error(f"Exception while fetching Alpha Vantage news for {symbol}: {e}")
        return {}


# -----------------------------------------
# 3) MARKETAUX SENTIMENT & NEWS
# -----------------------------------------
def get_marketaux_sentiment(
    keywords=["forex", "market"],
    language="en",
    published_after=None,
    sentiment_gte=None,
    sentiment_lte=None,
    must_have_entities=False,
    group_similar=True,
    limit=50,
):
    """
    Fetch aggregated market sentiment from MarketAux by searching for `keywords`.
    """
    try:
        base_url = "https://api.marketaux.com/v1/news/all"
        api_key = config.marketaux_key

        search_query = " OR ".join(keywords)
        params = {
            "api_token": api_key,
            "search": search_query,
            "language": language,
            "limit": limit,
            "group_similar": str(group_similar).lower(),
        }
        if published_after:
            params["published_after"] = published_after
        if sentiment_gte is not None:
            params["sentiment_gte"] = sentiment_gte
        if sentiment_lte is not None:
            params["sentiment_lte"] = sentiment_lte
        if must_have_entities:
            params["must_have_entities"] = "true"

        logging.info(f"Fetching MarketAux sentiment with params: {params}")
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()

        data = resp.json()
        articles = data.get("data", [])
        if not articles:
            logging.warning(f"No MarketAux articles for query: {search_query}.")
            return 0.0, []

        scores = [ent.get("sentiment_score") for art in articles for ent in art.get("entities", []) if ent.get("sentiment_score") is not None]
        avg_sent = sum(scores) / len(scores) if scores else 0.0
        logging.info(f"MarketAux sentiment fetched. Avg Sentiment: {avg_sent:.2f}, Articles: {len(articles)}.")
        return avg_sent, articles

    except Exception as e:
        logging.error(f"Error fetching MarketAux sentiment: {e}")
        return 0.0, []


def get_marketaux_financial_news(

    keywords=["forex", "interest rate"],
    language="en",
    published_after=None,
    high_impact_only=False,
    limit=3,  # Max articles per request
    max_pages=3  # Maximum number of pages to fetch
):
    """
    Fetch up to 9 articles using pagination (3 pages with 3 articles per request).
    """
    try:
        base_url = "https://api.marketaux.com/v1/news/all"
        api_key = config.marketaux_key

        search_query = " OR ".join(keywords)
        all_articles = []
        page = 1

        while page <= max_pages:
            params = {
                "api_token": api_key,
                "search": search_query,
                "language": language,
                "limit": limit,
                "page": page,
            }
            if published_after:
                params["published_after"] = published_after

            logging.info(f"Fetching MarketAux articles with params: {params}")
            resp = requests.get(base_url, params=params)
            resp.raise_for_status()

            data = resp.json()
            articles = data.get("data", [])

            if not articles:
                logging.info(f"No more articles found on page {page}. Ending pagination.")
                break

            logging.info(f"Fetched {len(articles)} articles on page {page}.")
            all_articles.extend(articles)

            page += 1

        # Apply high-impact filter if necessary
        if high_impact_only:
            logging.info("Filtering for high-impact articles (sentiment score > 0.75).")
            all_articles = [
                art for art in all_articles
                if any(abs(ent.get("sentiment_score", 0)) > 0.75 for ent in art.get("entities", []))
            ]

        logging.info(f"Total articles fetched after pagination: {len(all_articles)}.")
        return all_articles[:9]  # Ensure no more than 9 articles are returned

    except requests.exceptions.RequestException as req_err:
        logging.error(f"HTTP error while fetching MarketAux financial news: {req_err}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error during article pagination: {e}")
        return []
