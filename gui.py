# gui.py

import tkinter as tk
from tkinter import ttk, messagebox
import config
import logging
from tooltip import ToolTip
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from mt5_utils import fetch_historical_deals
import json
import importlib
from pathlib import Path
import threading
import logging
from data_processing import (fetch_data, scrape_forex_factory_weekly,
    get_alpha_vantage_news_sentiment,
    get_marketaux_sentiment,
    get_marketaux_financial_news
)
from llm import analyze_market_with_external_data
from indicators import (
    calculate_bollinger_bands, calculate_rsi, calculate_macd, calculate_vwap,
    calculate_ichimoku, calculate_keltner_channels, calculate_adx, calculate_stochastic, calculate_atr,
    calculate_long_term_ma
)

class TradingGUI(tk.Tk):
    def __init__(self, strategy, start_auto_callback, stop_auto_callback, execute_trades_callback):
        super().__init__()
        self.strategy = strategy
        self.start_auto_callback = start_auto_callback
        self.stop_auto_callback = stop_auto_callback
        self.execute_trades_callback = execute_trades_callback
        self.title("Trading Strategy Bot")
        self.geometry("2000x875")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both')

        self.trade_recommendations_tab = TradeRecommendationsTab(self.notebook, strategy, self.start_auto_callback, self.stop_auto_callback, self.execute_trades_callback)
        self.open_trades_tab = OpenTradesTab(self.notebook, strategy)
        self.config_tab = ConfigTab(self.notebook, strategy)
        self.risk_assessment_tab = RiskAssessmentTab(self.notebook, strategy)

        self.notebook.add(self.trade_recommendations_tab, text="Trade Recommendations")
        self.notebook.add(self.open_trades_tab, text="Open Trades")
        self.notebook.add(self.config_tab, text="Config")
        self.notebook.add(self.risk_assessment_tab, text="Risk Assessment")

        # New tab for Market Analysis
        self.market_analysis_tab = MarketAnalysisTab(self.notebook, self)
        self.notebook.add(self.market_analysis_tab, text="Market Analysis")

        # We can optionally do a deferred analysis with self.after(0, ...).
        # e.g.: self.after(0, lambda: self.update_market_analysis(config.symbols))
        
        self.update_gui()

        if not hasattr(strategy, 'models') or not strategy.models:
            logging.warning("No trained models available. Skipping feature importance display.")
        
        # # Display feature importances on launch (if available)
        # if hasattr(strategy, 'models') and strategy.models:
        #     feature_importances_df = self.strategy.get_feature_importances()
        #     logging.debug(f"Feature Importances DataFrame: {feature_importances_df}")
        #     if not feature_importances_df.empty:
        #         self.display_feature_importances(feature_importances_df)

    def update_gui(self):
        try:
            # ----------------------------------------------------------------------
            # NEW LOGIC: reflect concurrency lock state (is_refreshing) in the GUI
            # ----------------------------------------------------------------------
            if self.strategy.is_refreshing:
                # If refreshing is in progress:
                # Disable the refresh button and show the "Refreshing..." label
                self.trade_recommendations_tab.refresh_button.config(state='disabled')
                self.trade_recommendations_tab.loading_label.config(text="Refreshing recommendations...")
            else:
                # If NOT refreshing:
                # Re-enable the refresh button and clear the label
                self.trade_recommendations_tab.refresh_button.config(state='normal')
                self.trade_recommendations_tab.loading_label.config(text="")

            # ----------------------------------------------------------------------
            # Existing logic: auto_mode vs. manual mode updates
            # ----------------------------------------------------------------------
            if self.strategy.auto_mode:
                self.trade_recommendations_tab.refresh_recommendations()
                self.open_trades_tab.refresh_open_trades()
            else:
                self.trade_recommendations_tab.refresh_recommendations()
                self.open_trades_tab.refresh_open_trades()

            # If you have more manual mode GUI updates, do them here.

        except Exception as e:
            logging.error(f"Error updating GUI: {e}")
        finally:
            # Keep re-scheduling update_gui so it refreshes every scan_interval seconds
            self.after(config.scan_interval * 1000, self.update_gui)

    def display_feature_importances(self, feature_importances_df):
        if feature_importances_df.empty:
            logging.warning("No feature importances to display.")
            messagebox.showinfo("Feature Importances", "No feature importances available.")
            return
        
        # Deduplicate and aggregate feature importances
        aggregated_importances = (
            feature_importances_df.groupby("feature", as_index=False)
            .agg({"importance": "sum", "model": lambda x: ", ".join(set(x))})
            .sort_values(by="importance", ascending=False)
        )
    
        imp_window = tk.Toplevel(self)
        imp_window.title("Feature Importances")
        imp_window.geometry("600x400")

        frame = tk.Frame(imp_window)
        frame.pack(expand=True, fill='both', padx=10, pady=10)

        columns = ("Feature", "Importance")
        tree = ttk.Treeview(frame, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor='center', width=200)

        tree.pack(side='left', expand=True, fill='both')
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        for _, row in aggregated_importances.iterrows():
            tree.insert('', 'end', values=(row['feature'], f"{row['importance']:.4f}"))

    def refresh_all_tabs(self):
        self.trade_recommendations_tab.refresh_recommendations()
        self.open_trades_tab.refresh_open_trades()
        self.risk_assessment_tab.refresh_risk_metrics()

    def summarize_recent_data_for_llm(self, data):
        """
        Given a DataFrame with indicators, create a short string summary
        of the most recent bar's prices/indicators. E.g. RSI, MACD, Boll Bands,
        etc. to avoid sending all rows to the LLM.
        """
        if data.empty:
            return "No recent data available."
        # We'll use the last row
        last = data.iloc[-1]
        msg = (f"Recent Price: {last['close']:.5f}, "
            f"RSI: {last.get('rsi', float('nan')):.2f}, "
            f"MACD: {last.get('macd', float('nan')):.6f}, "
            f"UpperBand: {last.get('upper_band', float('nan')):.5f}, "
            f"LowerBand: {last.get('lower_band', float('nan')):.5f}, "
            f"ATR: {last.get('atr', float('nan')):.5f}, "
            f"LongTermMA: {last.get('long_term_ma', float('nan')):.5f}"
            )
        return msg

    def update_market_analysis(self, symbols):
        """
        1) Fetch weekly ForexFactory events once.
        2) For each symbol, gather short data summary, relevant events, 
        fundamentals & sentiment info, etc.
        3) Accumulate these pieces into a single "big_prompt".
        4) Pass the entire prompt to analyze_market_with_external_data() only once.
        5) Display the LLM's single combined response in the MarketAnalysisTab.
        """
        try:
            # 1) Fetch weekly events ONCE
            all_weekly_events = scrape_forex_factory_weekly()
            if not all_weekly_events:
                logging.warning("No weekly ForexFactory data (possibly rate-limited).")

            # 2) Build one combined prompt for all symbols
            big_prompt = []
            big_prompt.append(
                "You are an expert forex trading analyst. For each symbol below, "
                "I have a short technical summary, relevant economic data, and news sentiment. "
                "The news will relate to either side of the currency pair, or the country of that currency, please understand the "
                "respective impacts it will have on the pair. Explain how each side of the pair "
                "impacts the overall sentiment and trading decision."
                "Please provide:\n"
                "1) Market sentiment summary\n"
                "2) Key trends or potential reversals\n"
                "3) Upcoming economic events summary with directional forecast\n"
                "4) Actionable insights\n\n"
                "Note: It is crucial to critically evaluate the information provided in these responses, "
                "recognizing that some of it may be biased or incorrect. Your response should not simply "
                "replicate the given answers but should offer a refined, accurate, and comprehensive reply "
                "to the instruction. Ensure your response is well-structured, coherent, and adheres to the "
                "highest standards of accuracy and reliability.\n"
            )

            # Loop over symbols to build mini-blocks
            for symbol in symbols:
                logging.info(f"Gathering data for {symbol}...")

                # (a) Minimal data for indicators
                data = fetch_data(symbol, timeframe=config.timeframe, count=config.bars_to_fetch)
                if data.empty:
                    logging.warning(f"No historical data found for {symbol}. Skipping.")
                    # We'll still put a small note in the prompt
                    symbol_text = f"Symbol: {symbol}\nNo historical data.\n\n"
                    big_prompt.append(symbol_text)
                    continue

                # (b) Indicators (only need the final row summarized)
                data = calculate_bollinger_bands(data)
                data = calculate_rsi(data)
                data = calculate_macd(data)
                data = calculate_adx(data)
                data = calculate_stochastic(data)
                data = calculate_atr(data)
                data = calculate_long_term_ma(data)
                data = calculate_vwap(data)
                data = calculate_ichimoku(data)
                data = calculate_keltner_channels(data)

                short_summary = self.summarize_recent_data_for_llm(data)

                # (c) Filter out only relevant FF events for this symbol
                relevant_ff_events = self.filter_events_for_symbol(all_weekly_events, symbol)

                # (d) External sentiment & news
                av_data = get_alpha_vantage_news_sentiment(symbol)
                ma_sentiment, ma_articles = get_marketaux_sentiment(
                    config.MARKETAUX_FOREX_KEYWORDS.get(symbol, ["forex", symbol])
                )
                ma_news = get_marketaux_financial_news(
                    config.MARKETAUX_FOREX_KEYWORDS.get(symbol, ["forex", symbol]),
                    high_impact_only=False
                )

                # (e) Compose mini-block text for just this symbol
                symbol_block = (
                    f"Symbol: {symbol}\n"
                    f"Technical Summary:\n{short_summary}\n\n"
                    f"ForexFactory (Filtered): {relevant_ff_events}\n"
                    f"AlphaVantage: {av_data}\n"
                    f"MarketAux Sentiment: {ma_sentiment:.2f}, #Articles={len(ma_articles)}\n"
                    f"MarketAux News: {len(ma_news)} articles\n\n"
                )
                big_prompt.append(symbol_block)

            # 3) Combine everything into a single prompt string
            final_prompt = "\n".join(big_prompt)

            # 4) Make a SINGLE LLM request
            logging.debug(f"Sending one LLM request with prompt length={len(final_prompt)} chars.")
            llm_response = analyze_market_with_external_data(final_prompt)

            # 5) Display the single combined response
            self.market_analysis_tab.display_analysis(llm_response)

        except Exception as e:
            logging.error(f"Error updating market analysis: {e}")
            self.market_analysis_tab.display_analysis("Failed to update market analysis. Check logs for details.")


    def filter_events_for_symbol(self, all_events, symbol):
        """
        For symbol='EURUSD', parse to ['EUR','USD'] and filter.
        If 'country' in each event is in that set, we keep it.
        """
        currencies = self.parse_symbol_currencies(symbol)  # e.g. ['EUR','USD']
        filtered = [
            ev for ev in all_events
            if ev["country"].upper() in currencies
        ]
        return filtered

    def parse_symbol_currencies(self, symbol):
        """
        Given a symbol like 'EURUSD', return ['EUR', 'USD'].
        Strips off '', then splits the first 3 chars & the next 3 chars.
        """
        stripped = symbol
        c1 = stripped[:3]  # "EUR"
        c2 = stripped[3:]  # "USD"
        return [c1, c2]

class TradeRecommendationsTab(ttk.Frame):
    def __init__(self, parent, strategy, start_auto_callback, stop_auto_callback, execute_trades_callback):
        super().__init__(parent)
        self.strategy = strategy
        self.execute_trades_callback = execute_trades_callback
        self.start_auto_callback = start_auto_callback
        self.stop_auto_callback = stop_auto_callback
        self.last_refreshed_time = None  # Track last refreshed time
        self.create_widgets()
        self.load_initial_recommendations()

    def create_widgets(self):
        header = tk.Label(self, text="Trade Recommendations", font=("Arial", 14, "bold"))
        header.pack(pady=10)

        button_frame = tk.Frame(self)
        button_frame.pack(pady=5)

        self.refresh_button = tk.Button(button_frame, text="Refresh Recommendations", command=self.refresh_recommendations)
        self.refresh_button.pack(side='left', padx=5)

        self.loading_label = tk.Label(self, text="", font=("Arial", 10), fg="blue")
        self.loading_label.pack()

        columns = ("Symbol", "Action", "Strength", "Reason")
        self.tree = ttk.Treeview(self, columns=columns, show='headings', selectmode='extended')
        # Define custom widths for each column
        column_widths = {
            "Symbol": 50,    # Adjust based on the typical size of the content
            "Action": 50,     # For "buy", "sell", etc.
            "Strength": 50,  # For "Strong", "Normal", etc.
            "Reason": 300     # This column is longer and needs more space
        }
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor='center', width=column_widths.get(col, 100))  # Default width to 100 if not specified


        self.tree.pack(expand=True, fill='both', padx=20, pady=10)

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        self.approve_button = tk.Button(self, text="Approve Selected Trades", command=self.approve_trades)
        self.approve_button.pack(pady=10)
        self.approve_button.pack_forget()

        # Mode Dropdown
        self.mode_var = tk.StringVar(value="Manual" if not config.auto_mode else "Auto")
        mode_frame = tk.Frame(self)
        mode_frame.pack(pady=5)

        mode_label = tk.Label(mode_frame, text="Mode:")
        mode_label.pack(side="left")

        mode_dropdown = tk.OptionMenu(
            mode_frame,
            self.mode_var,
            "Manual",
            "Auto",
            command=self.change_mode
        )
        mode_dropdown.pack(side="left")

        self.last_refreshed_label = tk.Label(self, text="Last Refreshed: Never", font=("Arial", 10), fg="green")
        self.last_refreshed_label.pack(pady=5)

    def change_mode(self, selected_mode):
        config.auto_mode = (selected_mode == "Auto")
        logging.info(f"Trading mode updated to {'Auto' if config.auto_mode else 'Manual'}.")

        if config.auto_mode:
            self.refresh_button.config(state='disabled')
            self.approve_button.pack_forget()
            logging.info("Starting auto mode...")
            self.execute_all_recommendations()
            self.start_auto_callback()  # Ensure this is called
        else:
            self.refresh_button.config(state='normal')
            self.approve_button.pack(pady=10)
            self.stop_auto_callback()  # Stop auto trading when switching back


    def execute_all_recommendations(self):
        """
        Execute all trade recommendations when switching to Auto mode.
        """
        try:
            recommendations = self.strategy.trade_recommendations
            if not recommendations:
                logging.info("No trade recommendations available to execute.")
                return

            # Extract symbols from recommendations
            symbols_to_execute = list(recommendations.keys())
            logging.info(f"Executing all trade recommendations: {symbols_to_execute}")

            # Execute trades using the callback
            self.execute_trades_callback(symbols_to_execute)

            # Optionally, refresh recommendations after execution
            self.refresh_recommendations()

            logging.info("All recommended trades have been executed.")
        except Exception as e:
            logging.error(f"Error executing all recommendations: {e}")
            messagebox.showerror("Error", f"Failed to execute trades: {e}")

    
    def load_initial_recommendations(self):
        """
        Load initial recommendations on GUI startup.
        This ensures that the UI reflects any recommendations generated at launch.
        """
        if self.strategy.is_refreshing:
            # If already refreshing, show "Refreshing..." and disable the button
            self.refresh_button.config(state='disabled')
            self.loading_label.config(text="Refreshing recommendations...")
        else:
            self.refresh_recommendations()

    def refresh_recommendations(self):
        """
        Manually trigger a refresh of trade recommendations.
        Always disable the refresh button and show 'Refreshing...' so the user
        sees it's in progress, even if concurrency causes a skip.
        """
        try:
            if self.strategy.is_refreshing:
                logging.info("Refresh skipped: already in progress.")
                return

            # Immediately show the 'blue text' and disable the refresh button
            self.refresh_button.config(state='disabled')
            self.loading_label.config(text="Refreshing recommendations...")

            logging.info("Manual refresh of trade recommendations initiated.")

            def task():
                # Call generate_trade_recommendations; returns True if run, False if skip
                did_run = self.strategy.generate_trade_recommendations()
                # Jump back to main thread to finalize
                self.after(0, self._post_refresh, did_run)

            # Start background thread
            threading.Thread(target=task, daemon=True).start()

        except Exception as ex:
            logging.error(f"Error initiating manual refresh: {ex}")
            self.refresh_button.config(state='normal')
            self.loading_label.config(text="")
            messagebox.showerror("Error", f"Failed to initiate refresh: {ex}")

    def _post_refresh(self, did_run: bool):
        """
        Runs on the main Tk thread after generate_trade_recommendations() completes.
        did_run == True -> actually refreshed
        did_run == False -> concurrency skip
        """
        try:
            if did_run:
                # If we actually ran, update the TreeView
                self._update_ui_with_recommendations()
                self.last_refreshed_time = datetime.now()
                self._update_last_refreshed_label()
            else:
                # If concurrency skip, optionally log or show a small message
                logging.info("Refresh request was skipped (already refreshing).")

        finally:
            # Re-enable the button and clear label regardless
            self.refresh_button.config(state='normal')
            self.loading_label.config(text="")

    def _update_last_refreshed_label(self):
        if self.last_refreshed_time:
            timestamp = self.last_refreshed_time.strftime("%Y-%m-%d %H:%M:%S")
            self.last_refreshed_label.config(text=f"Last Refreshed: {timestamp}")

    def _handle_recommendation_error(self, ex):
        messagebox.showerror("Error", f"Failed to refresh recommendations: {ex}")

    def _update_ui_with_recommendations(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        recommendations = self.strategy.trade_recommendations
        if not recommendations:
            self.tree.insert('', 'end', values=("No recommendations at this time.", "", ""))
        else:
            for symbol, rec in recommendations.items():
                models_used = rec.get('models', [])
                confidence_details = ", ".join(
                    f"{model[0]}: {model[2]*100:.2f}%"
                    for model in models_used if model[2] >= config.model_confidence_thresholds.get(model[0], 0.5)
                )
                strength = rec.get('strength', 'Normal')  # Default to 'Normal'
                reason = f"{rec.get('reason', 'N/A')} ({confidence_details})"
                self.tree.insert('', 'end', values=(symbol, rec['action'].upper(), strength, reason))

        if config.auto_mode:
            self.approve_button.pack_forget()
        else:
            self.approve_button.pack(pady=10)

        self.loading_label.config(text="")

    def approve_trades(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showinfo("No Selection", "No trades selected.")
            return
        selected_symbols = []
        for item in selected_items:
            values = self.tree.item(item, 'values')
            symbol = values[0]
            if symbol != "No recommendations at this time.":
                selected_symbols.append(symbol)
        if not selected_symbols:
            messagebox.showinfo("No Valid Selection", "No valid trades selected.")
            return
        self.execute_trades_callback(selected_symbols)
        self.refresh_recommendations()
        messagebox.showinfo("Trades Approved", f"Approved trades for: {', '.join(selected_symbols)}")


class OpenTradesTab(ttk.Frame):
    def __init__(self, parent, strategy):
        super().__init__(parent)
        self.strategy = strategy
        self.create_widgets()

    def create_widgets(self):
        """
        Create widgets for the Open Trades tab.
        """
        # Header
        header = tk.Label(self, text="Open Trades", font=("Arial", 14, "bold"))
        header.pack(pady=10)

        # Treeview for open trades
        columns = ("Symbol", "Type", "Volume", "Entry Price", "Current Price", "Profit", "Recommendation")
        self.tree = ttk.Treeview(self, columns=columns, show='headings', selectmode='extended')
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor='center')

        self.tree.pack(expand=True, fill='both', padx=20, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        # Close Trade Button
        self.close_button = tk.Button(self, text="Close Selected Trades", command=self.close_trades)
        self.close_button.pack(pady=10)

    def refresh_open_trades(self):
        """
        Refresh the open trades displayed in the Treeview.
        """
        # Clear existing entries
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Fetch open trades
        open_trades = self.strategy.generate_open_trade_recommendations()

        if not open_trades:
            self.tree.insert('', 'end', values=("No open trades at this time.", "", "", "", "", "", ""))
            return

        for trade in open_trades:
            symbol = trade.get('symbol', 'N/A')
            trade_type = "BUY" if trade.get('type') == mt5.ORDER_TYPE_BUY else "SELL"
            volume = trade.get('volume', 0)
            entry_price = trade.get('entry_price', 0)
            current_price = trade.get('current_price', 0)
            profit = trade.get('profit', 0)
            recommendation = trade.get('recommendation', 'Keep Open')
            ticket = trade.get('ticket', 'N/A')  

            self.tree.insert('', 'end', values=(symbol, trade_type, volume, entry_price, current_price, profit, recommendation, ticket))

    def close_trades(self):
        """
        Close the selected open trades.
        """
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showinfo("No Selection", "No trades selected to close.")
            return

        selected_trades = []
        for item in selected_items:
            values = self.tree.item(item, 'values')
            symbol = values[0]
            volume = values[2]  # Fetch volume from TreeView
            recommendation = values[6]
            ticket = values[7]  
            trade_type = values[1]  # "BUY" or "SELL"

            if symbol != "No open trades at this time.":
                selected_trades.append({"ticket": ticket,"symbol": symbol, "volume": float(volume), "type": mt5.ORDER_TYPE_BUY if trade_type == "BUY" else mt5.ORDER_TYPE_SELL}) 

        if not selected_trades:
            messagebox.showinfo("No Valid Selection", "No valid trades selected to close.")
            return

        for trade in selected_trades:
            self.strategy.close_open_trade(trade)

        # Refresh open trades after closure
        self.refresh_open_trades()

        messagebox.showinfo("Trades Closed", f"Closed trades for symbols: {', '.join([t['symbol'] for t in selected_trades])}")

class ConfigTab(ttk.Frame):
    """
    ConfigTab allows users to modify global, feature selection, and symbol-specific configurations.
    """

    def __init__(self, parent, strategy):
        super().__init__(parent)
        self.strategy = strategy
        self.global_entries = {}
        self.feature_vars = {}
        self.symbol_edit_vars = {}
        self.model_entries = {}
        self.model_available_vars = {}
        self.model_threshold_vars = {}
        self.model_weight_vars = {}
        
        self.create_widgets()


    def create_widgets(self):
        """
        Create widgets for the Config tab, including Global Config, Feature Selection, and Symbol-Specific Config.
        """
        # Header
        header = tk.Label(self, text="Configuration Settings", font=("Arial", 14, "bold"))
        header.pack(pady=10)

        # Save Button moved above the notebook
        save_button = tk.Button(self, text="Save Configuration", command=self.save_configurations)
        save_button.pack(pady=10)  # Packed before the notebook

        # Container Frame for the Notebook
        container = tk.Frame(self)
        container.pack(expand=True, fill='both', padx=10, pady=10)

        # Notebook
        self.config_notebook = ttk.Notebook(container)
        self.config_notebook.pack(side='top', expand=True, fill='both')

        # Add tabs to the notebook
        self.global_config_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.global_config_frame, text="Global Config")

        self.feature_config_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.feature_config_frame, text="Feature Selection")

        self.symbol_config_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.symbol_config_frame, text="Symbol-Specific Config")

        self.model_config_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.model_config_frame, text="Model Config")

        # Initialize Config Sections
        self.create_global_config_widgets()
        self.create_feature_selection_widgets()
        self.create_symbol_config_widgets()
        self.create_model_config_widgets()

    def create_global_config_widgets(self):
        """
        Create widgets for the Global Config section with logical groups and two-column layout.
        """
        # Define groups of parameters
        grouped_params = {
            "Trading Parameters": [
                ("Lot Size", "lot_size", "float"),
                ("SL (%)", "sl_percent", "float"),
                ("TP (%)", "tp_percent", "float"),
                ("Dynamic Position Sizing", "dynamic_position_sizing", "bool"),
                ("Dynamic Signal Type", "dynamic_signal_type", "str"),
                ("Risk per Trade (%)", "risk_per_trade_percent", "float"),
            ],
            "Risk Management": [
                ("Max Trades per Day", "max_trades_per_day", "int"),
                ("Max Open Trades", "max_open_trades", "int"),
                ("Min Equity Threshold", "min_equity_threshold", "float"),
                ("Strong Threshold", "strong_threshold", "float"),
                ("Strong Min Models Agreement", "strong_min_models_agreement", "int"),
            ],
            "Trailing Stops & Advanced Settings": [
                ("Enable Trailing Stop", "enable_trailing_stop", "bool"),
                ("Trailing Stop Fraction", "trailing_stop_fraction", "float"),
                ("Break-Even Factor", "break_even_factor", "float"),
                ("ATR SL/TP Enabled", "atr_sl_tp", "bool"),
            ],
            "Indicators": [
                ("RSI Buy Limit", "rsi_buy_limit", "int"),
                ("RSI Sell Limit", "rsi_sell_limit", "int"),
                ("Post Prediction ATR", "post_pred_atr_val", "float"),
            ],
            "Miscellaneous": [
                ("Scan Interval (Seconds)", "scan_interval", "int"),
                ("Start Trading Hour (UTC)", "start_trading_hour", "int"),
                ("End Trading Hour (UTC)", "end_trading_hour", "int"),
                ("Feature Scaling", "feature_scaling", "str"),
                ("Rolling Window Size", "rolling_window_size", "int"),
            ]
        }

        self.global_entries = {}

        # Add each group
        for group_name, params in grouped_params.items():
            group_frame = ttk.LabelFrame(self.global_config_frame, text=group_name)
            group_frame.pack(fill='x', padx=5, pady=5)

            # Two-column layout
            left_column = tk.Frame(group_frame)
            left_column.pack(side='left', fill='y', padx=5, pady=5)
            right_column = tk.Frame(group_frame)
            right_column.pack(side='left', fill='y', padx=5, pady=5)

            for i, (label_text, var_name, var_type) in enumerate(params):
                column = left_column if i % 2 == 0 else right_column

                frame = tk.Frame(column)
                frame.pack(fill='x', pady=2)

                label = tk.Label(frame, text=f"{label_text}:", anchor='w', width=25)
                label.pack(side='left', padx=5)

                current_value = getattr(config, var_name, None)

                if var_type == "bool":
                    var = tk.BooleanVar(value=current_value if current_value is not None else False)
                    chk = tk.Checkbutton(frame, variable=var)
                    chk.pack(side='left')
                    self.global_entries[var_name] = var

                elif var_type in ("float", "int"):
                    var_cls = tk.DoubleVar if var_type == "float" else tk.IntVar
                    var = var_cls(value=current_value if current_value is not None else 0)
                    entry = tk.Entry(frame, textvariable=var, width=10)
                    entry.pack(side='left', padx=5)
                    self.global_entries[var_name] = var

                elif var_type == "str":
                    # Add dropdowns for specific string parameters
                    if var_name == "trailing_mode":
                        var = tk.StringVar(value=current_value if current_value else "normal")
                        cmb = ttk.Combobox(frame, textvariable=var, values=["breakeven", "normal", "breakeven_then_normal"], state='readonly')
                        cmb.pack(side='left', padx=5)
                        self.global_entries[var_name] = var
                    elif var_name == "feature_scaling":
                        var = tk.StringVar(value=current_value if current_value else "standard")
                        cmb = ttk.Combobox(frame, textvariable=var, values=["standard", "minmax"], state='readonly')
                        cmb.pack(side='left', padx=5)
                        self.global_entries[var_name] = var
                    elif var_name == "dynamic_signal_type":
                        var = tk.StringVar(value=current_value if current_value else "All")
                        cmb = ttk.Combobox(frame, textvariable=var, values=["All", "Strong", "Normal"], state='readonly')
                        cmb.pack(side='left', padx=5)
                        self.global_entries[var_name] = var
                    else:
                        # Default string entry
                        var = tk.StringVar(value=str(current_value) if current_value is not None else "")
                        entry = tk.Entry(frame, textvariable=var, width=15)
                        entry.pack(side='left', padx=5)
                        self.global_entries[var_name] = var


    def create_feature_selection_widgets(self):
        """
        Create widgets for the Feature Selection tab, including individual Ichimoku components.
        """
        feature_selection_label = tk.Label(self.feature_config_frame, text="Select Features for ML Model", font=("Arial", 12, "bold"))
        feature_selection_label.pack(pady=10)

        available_features = [
            ("RSI", "rsi"),
            ("Bollinger Bands", "bollinger_bands"),
            ("MACD", "macd"),
            ("ADX", "adx"),
            ("Stochastic Oscillator", "stochastic"),
            ("ATR", "atr"),
            ("Long-Term MA", "long_term_ma"),
            ("VWAP", "vwap"),
            ("Ichimoku Tenkan Sen", "tenkan_sen"),
            ("Ichimoku Kijun Sen", "kijun_sen"),
            ("Ichimoku Senkou Span A", "senkou_span_a"),
            ("Ichimoku Senkou Span B", "senkou_span_b"),
            ("Ichimoku Chikou Span", "chikou_span"),
            ("Keltner Channels", "keltner_channels"),
            ("ATR Multiple Periods", "atr_multi"),
            ("Lagged Features", "lagged_features")
        ]

        frame = tk.Frame(self.feature_config_frame)
        frame.pack(expand=True, fill='both', padx=20, pady=10)

        canvas = tk.Canvas(frame)
        canvas.pack(side='left', fill='both', expand=True)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side='right', fill='y')

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        checkbox_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=checkbox_frame, anchor='nw')

        for feature_label, feature_key in available_features:
            var = tk.BooleanVar(value=config.features_selected.get(feature_key, False))
            chk = tk.Checkbutton(checkbox_frame, text=feature_label, variable=var)
            chk.pack(anchor='w', pady=2)
            ToolTip(chk, f"Toggle inclusion of {feature_label} in ML model")
            self.feature_vars[feature_key] = var

    def create_symbol_config_widgets(self):
        """
        Create widgets for the Symbol-Specific Config section with symbols across the top and metrics down the side.
        """
        symbol_params = [
            ("Timeframe", "timeframe", "str", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]),
            ("SL ATR Multiplier", "atr_multiplier_sl", "float", None),
            ("TP ATR Multiplier", "atr_multiplier_tp", "float", None),
            ("Risk per Trade (%)", "risk_per_trade_percent", "float", None),
            ("Look Ahead (periods)", "look_ahead", "int", None),
            ("Threshold", "threshold", "float", None),
            ("Threshold Type", "threshold_type", "str", ["atr", "normal"]),
            ("Threshold ATR Period", "threshold_atr_period", "int", None),
            ("Threshold ATR Multi", "threshold_atr_multiplier", "float", None),
            ("Trailing Stop Fraction", "trailing_stop_fraction", "float", None),
            ("Break-Even Factor", "break_even_factor", "float", None),
            ("Trailing Mode", "trailing_mode", "str", ["breakeven", "normal", "breakeven_then_normal"]),
            ("Open Trade Limit", "max_open_trades", "int", None),
            ("Dynamic ATR", "dynamic_atr", "str", ["dynamic", "static"]),
            ("Dynamic ATR Filter", "dynamic_atr_filter_direction", "str", ["above", "below", "band"]),
            ("Dynamic ATR Band", "dynamic_atr_band", "float", None),
            ("ATR Threshold", "post_pred_atr_threshold", "float", None)
        ]

        frame = tk.Frame(self.symbol_config_frame)
        frame.pack(expand=True, fill='both', padx=10, pady=10)

        canvas = tk.Canvas(frame)
        canvas.pack(side='left', fill='both', expand=True)

        scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollbar_y.pack(side='right', fill='y')

        scrollbar_x = ttk.Scrollbar(self.symbol_config_frame, orient="horizontal", command=canvas.xview)
        scrollbar_x.pack(side='bottom', fill='x')

        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        grid_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=grid_frame, anchor='nw')

        # Create header row: Symbols
        for col_idx, symbol in enumerate(config.symbols):
            header = tk.Label(grid_frame, text=symbol, font=("Arial", 10, "bold"), borderwidth=1, relief="solid", padx=5, pady=5)
            header.grid(row=0, column=col_idx + 1, sticky='nsew')

        # Create first column: Metrics
        for row_idx, (label_text, var_name, var_type, options) in enumerate(symbol_params, start=1):
            metric_label = tk.Label(grid_frame, text=label_text + ":", font=("Arial", 10, "bold"), borderwidth=1, relief="solid", padx=5, pady=5)
            metric_label.grid(row=row_idx, column=0, sticky='nsew')

            for col_idx, symbol in enumerate(config.symbols):
                symbol_config = config.symbol_configs.get(symbol, {})
                current_value = symbol_config.get(var_name, getattr(config, var_name, ""))

                if var_type == "bool":
                    var = tk.BooleanVar(value=current_value)
                    chk = tk.Checkbutton(grid_frame, variable=var)
                    chk.grid(row=row_idx, column=col_idx + 1, sticky='nsew', padx=2, pady=2)
                    ToolTip(chk, f"Toggle {label_text} for {symbol}")
                    self.symbol_edit_vars[(symbol, var_name)] = var
                elif var_type == "str" and options:
                    var = tk.StringVar(value=current_value)
                    cmb = ttk.Combobox(grid_frame, textvariable=var, values=options, state='readonly')
                    cmb.grid(row=row_idx, column=col_idx + 1, sticky='nsew', padx=2, pady=2)
                    ToolTip(cmb, f"Select {label_text} for {symbol}")
                    self.symbol_edit_vars[(symbol, var_name)] = var
                elif var_type == "str":
                    var = tk.StringVar(value=str(current_value))
                    entry = tk.Entry(grid_frame, textvariable=var, width=10)
                    entry.grid(row=row_idx, column=col_idx + 1, sticky='nsew', padx=2, pady=2)
                    ToolTip(entry, f"Enter {label_text} for {symbol}")
                    self.symbol_edit_vars[(symbol, var_name)] = var
                elif var_type == "float":
                    var = tk.DoubleVar(value=float(current_value))
                    entry = tk.Entry(grid_frame, textvariable=var, width=10)
                    entry.grid(row=row_idx, column=col_idx + 1, sticky='nsew', padx=2, pady=2)
                    ToolTip(entry, f"Enter {label_text} for {symbol}")
                    self.symbol_edit_vars[(symbol, var_name)] = var
                elif var_type == "int":
                    var = tk.IntVar(value=int(current_value))
                    entry = tk.Entry(grid_frame, textvariable=var, width=10)
                    entry.grid(row=row_idx, column=col_idx + 1, sticky='nsew', padx=2, pady=2)
                    ToolTip(entry, f"Enter {label_text} for {symbol}")
                    self.symbol_edit_vars[(symbol, var_name)] = var
                elif var_type == "int":
                    var = tk.IntVar(value=int(current_value))
                    entry = tk.Entry(grid_frame, textvariable=var, width=10)
                    entry.grid(row=row_idx, column=col_idx + 1, sticky='nsew', padx=2, pady=2)
                    ToolTip(entry, f"Enter {label_text} for {symbol}")
                    self.symbol_edit_vars[(symbol, var_name)] = var
                    
    def create_model_config_widgets(self):
        """
        Create widgets for the Model Config tab.
        We'll handle:
        - models_available (dict of bool)
        - enable_model_tuning (bool)
        - rf_max_depth (int)
        - rf_n_estimators (int)
        - min_confidence_threshold (float)
        - model_confidence_thresholds (dict of float)
        - model_weighting_mode (str)
        - manual_model_weights (dict of float)
        - min_models_agreement (int)
        """

        # First, single-value model params
        # These are similar to global config items:
        model_params = [
            ("Enable Model Tuning", "enable_model_tuning", "bool"),
            ("RF Max Depth", "rf_max_depth", "int"),
            ("RF n Estimators", "rf_n_estimators", "int"),
            ("Min Confidence Threshold", "min_confidence_threshold", "float"),
            ("Model Weighting Mode", "model_weighting_mode", "str"),
            ("Min Models Agreement", "min_models_agreement", "int")
        ]

        # Frame for single-value params
        single_frame = ttk.LabelFrame(self.model_config_frame, text="Model Parameters")
        single_frame.pack(fill='x', padx=5, pady=5)

        for label_text, var_name, var_type in model_params:
            frame = tk.Frame(single_frame)
            frame.pack(fill='x', pady=2)
            label = tk.Label(frame, text=label_text + ":", width=25, anchor='w')
            label.pack(side='left', padx=5)

            current_value = getattr(config, var_name, None)
            if var_type == "bool":
                var = tk.BooleanVar(value=current_value if current_value is not None else False)
                chk = tk.Checkbutton(frame, variable=var)
                chk.pack(side='left')
                ToolTip(chk, f"Toggle {label_text}")
                self.model_entries[var_name] = var
            elif var_type == "str":
                if var_name == "model_weighting_mode":
                    var = tk.StringVar(value=current_value if current_value else "manual")
                    cmb = ttk.Combobox(frame, textvariable=var, values=["manual","performance_based"], state='readonly')
                    cmb.pack(side='left', padx=5)
                    ToolTip(cmb, f"Select {label_text}")
                    self.model_entries[var_name] = var
                else:
                    var = tk.StringVar(value=str(current_value) if current_value is not None else "")
                    entry = tk.Entry(frame, textvariable=var, width=20)
                    entry.pack(side='left', padx=5)
                    ToolTip(entry, f"Enter {label_text}")
                    self.model_entries[var_name] = var
            elif var_type == "int":
                var = tk.IntVar(value=int(current_value) if current_value is not None else 0)
                entry = tk.Entry(frame, textvariable=var, width=20)
                entry.pack(side='left', padx=5)
                ToolTip(entry, f"Enter {label_text}")
                self.model_entries[var_name] = var
            elif var_type == "float":
                var = tk.DoubleVar(value=float(current_value) if current_value is not None else 0.0)
                entry = tk.Entry(frame, textvariable=var, width=20)
                entry.pack(side='left', padx=5)
                ToolTip(entry, f"Enter {label_text}")
                self.model_entries[var_name] = var

        # models_available: dict of bool for each model
        models_available_frame = ttk.LabelFrame(self.model_config_frame, text="Models Available")
        models_available_frame.pack(fill='x', padx=5, pady=5)

        for model_name, active in config.models_available.items():
            var = tk.BooleanVar(value=active)
            chk = tk.Checkbutton(models_available_frame, text=model_name, variable=var)
            chk.pack(anchor='w', padx=5, pady=2)
            self.model_available_vars[model_name] = var

        # model_confidence_thresholds: dict of floats
        thresholds_frame = ttk.LabelFrame(self.model_config_frame, text="Model Confidence Thresholds")
        thresholds_frame.pack(fill='x', padx=5, pady=5)

        for model_name, thresh in config.model_confidence_thresholds.items():
            fr = tk.Frame(thresholds_frame)
            fr.pack(fill='x', pady=2)
            lbl = tk.Label(fr, text=model_name + ":", width=25, anchor='w')
            lbl.pack(side='left', padx=5)
            var = tk.DoubleVar(value=thresh)
            entry = tk.Entry(fr, textvariable=var, width=10)
            entry.pack(side='left', padx=5)
            ToolTip(entry, f"Set confidence threshold for {model_name}")
            self.model_threshold_vars[model_name] = var

        # manual_model_weights: dict of floats
        weights_frame = ttk.LabelFrame(self.model_config_frame, text="Manual Model Weights (Only used if manual weighting_mode)")
        weights_frame.pack(fill='x', padx=5, pady=5)

        for model_name, weight in config.manual_model_weights.items():
            wf = tk.Frame(weights_frame)
            wf.pack(fill='x', pady=2)
            wlbl = tk.Label(wf, text=model_name + ":", width=25, anchor='w')
            wlbl.pack(side='left', padx=5)
            wvar = tk.DoubleVar(value=weight)
            wentry = tk.Entry(wf, textvariable=wvar, width=10)
            wentry.pack(side='left', padx=5)
            ToolTip(wentry, f"Set manual weight for {model_name}")
            self.model_weight_vars[model_name] = wvar

    def save_configurations(self):
        """
        Save all global configurations, feature selections, and symbol-specific configurations.
        """
        try:
            # Save Global Config
            for var_name, var in self.global_entries.items():
                value = var.get()
                var_type = self.get_var_type(var_name)
                self.validate_config_value(var_name, value, var_type)
                self.update_global_config_py(var_name, value, var_type)

            # Save Feature Selections
            for feature_key, var in self.feature_vars.items():
                config.features_selected[feature_key] = var.get()

            # Update features_selected in config.py
            self.update_global_config_py('features_selected', config.features_selected, 'dict')

            # Save symbol-specific configurations
            self.save_symbol_configurations()
            
            # **New Step: Save Model Configurations**
            self.save_model_configurations()
            
            messagebox.showinfo("Success", "All configurations have been saved.")
            logging.info("All configurations saved via GUI.")
        except ValueError as ve:
            messagebox.showerror("Validation Error", str(ve))
            logging.error(f"Validation error: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configurations: {e}")
            logging.error(f"Error saving configurations: {e}")

    def get_var_type(self, var_name):
        """
        Determine the variable type based on the var_name.
        """
        var_types = {
            "look_ahead": "int",
            "threshold": "float",
            "threshold_type": "str",
            "threshold_atr_period": "int",
            "threshold_atr_multiplier": "float",
            "strong_threshold": "float",
            "strong_min_models_agreement": "int",
            "risk_per_trade_percent": "float",
            "sl_percent": "float",
            "tp_percent": "float",
            "dynamic_position_sizing": "bool",
            "lot_size": "float",            
            "min_lot_size": "float",
            "max_lot_size": "float",
            "volume_step": "float",
            "atr_sl_tp": "bool",
            "atr_multiplier_sl": "float",
            "atr_multiplier_tp": "float",
            "enable_trailing_stop": "bool",
            "trailing_stop_fraction": "float",
            "break_even_factor": "float",
            "trailing_mode": "str",
            "scan_interval": "int",
            "min_confidence_threshold": "float",
            "max_trades_per_day": "int",
            "max_open_trades": "int",
            "min_equity_threshold": "float",
            "start_trading_hour": "int",
            "end_trading_hour": "int",
            "feature_scaling": "str",
            "features_selected": "dict",
            "rolling_window_size": "int",
            "min_models_agreement": "int",
            "rsi_buy_limit": "int",
            "rsi_sell_limit": "int",
            "dynamic_atr": "str",
            "dynamic_atr_filter_direction": "str",
            "dynamic_atr_band": "float",
            "post_pred_atr_val": "float"
        }
        return var_types.get(var_name, "str")

    def validate_config_value(self, var_name, value, var_type):
        """
        Validate the configuration value based on its type.
        """
        if var_type == "int":
            if not isinstance(value, int):
                raise ValueError(f"Configuration '{var_name}' expects an integer value.")
        elif var_type == "float":
            if not isinstance(value, float) and not isinstance(value, int):
                raise ValueError(f"Configuration '{var_name}' expects a float value.")
        elif var_type == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"Configuration '{var_name}' expects a boolean value.")
        elif var_type == "str":
            if not isinstance(value, str):
                raise ValueError(f"Configuration '{var_name}' expects a string value.")
        elif var_type == "dict":
            if not isinstance(value, dict):
                raise ValueError(f"Configuration '{var_name}' expects a dictionary value.")

    def update_global_config_py(self, var_name, value, var_type):
        """
        Update the config.py file with the new global configurations.
        """
        try:
            config_path = Path(config.BASE_DIR) / "config.py"
            with open(config_path, 'r') as file:
                lines = file.readlines()

            new_lines = []
            features_selected_start_idx = None
            features_selected_end_idx = None

            # Identify the start and end indices of features_selected if needed
            if var_name == "features_selected":
                for idx, line in enumerate(lines):
                    if line.strip().startswith("features_selected ="):
                        features_selected_start_idx = idx
                        # Find the closing brace
                        for end_idx in range(idx, len(lines)):
                            if lines[end_idx].strip().endswith("}"):
                                features_selected_end_idx = end_idx
                                break
                        break

            if var_type == "dict" and var_name == "features_selected":
                # Update the features_selected dict
                if features_selected_start_idx is not None and features_selected_end_idx is not None:
                    from pprint import pformat
                    formatted_dict = pformat(value, indent=4)
                    new_dict_str = f"features_selected = {formatted_dict}\n"
                    new_lines = lines[:features_selected_start_idx] + [new_dict_str] + lines[features_selected_end_idx + 1:]
                else:
                    from pprint import pformat
                    formatted_dict = pformat(value, indent=4)
                    new_dict_str = f"\n# Feature Selection Configuration\nfeatures_selected = {formatted_dict}\n"
                    new_lines = lines + [new_dict_str]
            else:
                # Handle non-dict variables
                var_found = False
                for idx, line in enumerate(lines):
                    if line.strip().startswith(f"{var_name} ="):
                        var_found = True
                        if var_type == "bool":
                            new_line = f"{var_name} = {value}\n"
                        elif var_type == "str":
                            new_line = f'{var_name} = "{value}"\n'
                        elif var_type == "int":
                            new_line = f"{var_name} = {value}\n"
                        elif var_type == "float":
                            new_line = f"{var_name} = {value}\n"
                        elif var_type == "dict":
                            from pprint import pformat
                            formatted_dict = pformat(value, indent=4)
                            new_line = f"{var_name} = {formatted_dict}\n"
                        else:
                            # Default to string if var_type is unrecognized
                            new_line = f'{var_name} = "{value}"\n'
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)

                if not var_found:
                    # Append the new variable if not found
                    if var_type == "bool":
                        new_line = f"{var_name} = {value}\n"
                    elif var_type == "str":
                        new_line = f'{var_name} = "{value}"\n'
                    elif var_type == "int":
                        new_line = f"{var_name} = {value}\n"
                    elif var_type == "float":
                        new_line = f"{var_name} = {value}\n"
                    elif var_type == "dict":
                        from pprint import pformat
                        formatted_dict = pformat(value, indent=4)
                        new_line = f"{var_name} = {formatted_dict}\n"
                    else:
                        new_line = f'{var_name} = "{value}"\n'
                    new_lines.append(new_line)

            with open(config_path, 'w') as file:
                file.writelines(new_lines)

            logging.info(f"config.py updated for global config: {var_name} = {value} (Type: {var_type})")

            # Reload config to apply changes
            importlib.reload(config)
            self.strategy.reload_symbol_settings()

        except Exception as e:
            logging.error(f"Error updating config.py for global config {var_name}: {e}")
            raise

    def save_symbol_configurations(self):
        """
        Save all symbol-specific configurations.
        """
        try:
            for (symbol, var_name), var in self.symbol_edit_vars.items():
                value = var.get()
                # Update symbol_configs in config
                if symbol not in config.symbol_configs:
                    config.symbol_configs[symbol] = {}
                config.symbol_configs[symbol][var_name] = value

            # Write to symbol_configs.json
            symbol_config_path = Path(config.symbol_config_path)
            with open(symbol_config_path, 'w') as f:
                json.dump(config.symbol_configs, f, indent=4)

            importlib.reload(config)
            self.strategy.reload_symbol_settings()

            messagebox.showinfo("Success", "Symbol configurations have been saved.")
            logging.info("Symbol configurations saved via GUI.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save symbol configurations: {e}")
            logging.error(f"Error saving symbol configurations: {e}")
            
    def save_model_configurations(self):
        """
        Save model-related configurations:
        - Single-value model parameters in model_entries
        - models_available dictionary
        - model_confidence_thresholds dictionary
        - manual_model_weights dictionary
        """

        # First, save single-value model parameters
        # These are already stored in model_entries and are single values like global config
        single_model_params_types = {
            'enable_model_tuning': 'bool',
            'rf_max_depth': 'int',
            'rf_n_estimators': 'int',
            'min_confidence_threshold': 'float',
            'model_weighting_mode': 'str',
            'min_models_agreement': 'int'
        }

        for var_name, var in self.model_entries.items():
            value = var.get()
            var_type = single_model_params_types.get(var_name, 'str')
            self.validate_config_value(var_name, value, var_type)
            self.update_global_config_py(var_name, value, var_type)

        # Update models_available dictionary
        new_models_available = {}
        for model_name, var in self.model_available_vars.items():
            new_models_available[model_name] = var.get()
        self.update_global_config_dict('models_available', new_models_available)

        # Update model_confidence_thresholds dictionary
        new_model_conf_thresholds = {}
        for model_name, var in self.model_threshold_vars.items():
            new_model_conf_thresholds[model_name] = var.get()
        self.update_global_config_dict('model_confidence_thresholds', new_model_conf_thresholds)

        # Update manual_model_weights dictionary
        new_manual_model_weights = {}
        for model_name, var in self.model_weight_vars.items():
            new_manual_model_weights[model_name] = var.get()
        self.update_global_config_dict('manual_model_weights', new_manual_model_weights)

        # Reload config after all changes
        importlib.reload(config)
        self.strategy.reload_symbol_settings()

    def update_global_config_dict(self, var_name, dict_value):
        """
        Update a dictionary variable (like models_available, model_confidence_thresholds, 
        or manual_model_weights) in config.py, ensuring no duplicates or partial entries.
        """
        config_path = Path(config.BASE_DIR) / "config.py"
        with open(config_path, 'r') as file:
            lines = file.readlines()

        from pprint import pformat
        formatted_dict = pformat(dict_value, indent=4)
        new_dict_str = f"{var_name} = {formatted_dict}\n"

        var_found = False
        start_idx = None
        end_idx = None

        # Find the dictionary definition start line
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{var_name} ="):
                var_found = True
                start_idx = i
                # Determine if this line already contains a '{'
                # If not, dictionary might be defined on multiple lines or not defined properly
                # We'll look forward until we find '{' or end of file
                brace_count = 0
                dict_start_line = i

                # Check if current line has '{'
                if '{' in line:
                    brace_count += line.count('{')
                    brace_count -= line.count('}')
                else:
                    # If no '{' on this line, look ahead for the first '{'
                    j = i
                    while j < len(lines):
                        if '{' in lines[j]:
                            brace_count += lines[j].count('{')
                            brace_count -= lines[j].count('}')
                            dict_start_line = j
                            break
                        j += 1

                # Now find the matching closing '}' for this dictionary
                k = dict_start_line
                while k < len(lines):
                    # Count braces on each line
                    if k != dict_start_line:  # Already counted for dict_start_line above
                        brace_count += lines[k].count('{')
                        brace_count -= lines[k].count('}')
                    if brace_count == 0:
                        end_idx = k
                        break
                    k += 1
                if end_idx is None:
                    # Didn't find closing '}', consider whole file until end
                    end_idx = len(lines) - 1
                break

        if var_found and start_idx is not None and end_idx is not None:
            # Replace old dictionary definition from start_idx to end_idx
            new_lines = lines[:start_idx] + [new_dict_str] + lines[end_idx+1:]
        else:
            # Not found, just append at the end
            new_lines = lines + [f"\n{new_dict_str}"]

        with open(config_path, 'w') as file:
            file.writelines(new_lines)

        logging.info(f"config.py updated for dict config: {var_name} = {dict_value}")

        # Reload config after updating
        importlib.reload(config)
        self.strategy.reload_symbol_settings()
    
class RiskAssessmentTab(ttk.Frame):
    """
    RiskAssessmentTab displays risk metrics by symbol and trade type over various timeframes.
    Utilizes MT5's historical data for comprehensive assessment.
    """

    def __init__(self, parent, strategy):
        super().__init__(parent)
        self.strategy = strategy
        self.create_widgets()

    def create_widgets(self):
        """
        Create widgets for the Risk Assessment tab.
        """
        # Header
        header = tk.Label(self, text="Risk Assessment", font=("Arial", 14, "bold"))
        header.pack(pady=10)

        # Create a frame for Treeview and scrollbars
        frame = tk.Frame(self)
        frame.pack(expand=True, fill='both', padx=10, pady=10)

        # Treeview for risk metrics
        columns = ("Symbol", "Type", "Trades Today", "Trades This Week", "Trades This Month", "Trades All Time",
                   "Profit Today", "Profit This Week", "Profit This Month", "Profit All Time",
                   "Wins Today", "Wins This Week", "Wins This Month", "Wins All Time",
                   "Losses Today", "Losses This Week", "Losses This Month", "Losses All Time")
        self.tree = ttk.Treeview(frame, columns=columns, show='headings')
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor='center')  # Adjust widths as needed

        # Scrollbars
        scrollbar_y = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        scrollbar_y.pack(side='right', fill='y')

        scrollbar_x = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        scrollbar_x.pack(side='bottom', fill='x')

        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        self.tree.pack(expand=True, fill='both')

        # Refresh Button
        refresh_button = tk.Button(self, text="Refresh Risk Metrics", command=self.refresh_risk_metrics)
        refresh_button.pack(pady=10)

    def refresh_risk_metrics(self):
        """
        Refresh the risk metrics displayed in the Treeview.
        """
        try:
            # Clear existing entries
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Define timeframes
            now = datetime.utcnow()
            today_start = datetime(now.year, now.month, now.day)
            week_start = today_start - timedelta(days=now.weekday())  # Assuming week starts on Monday
            month_start = datetime(now.year, now.month, 1)

            # Initialize a list to store metrics
            metrics_list = []
            
            # Fetch all deals at once
            all_deals = fetch_historical_deals(from_time=datetime(2000, 1, 1), to_time=datetime.now())
         
            for symbol in config.symbols:
                for trade_type in ['BUY', 'SELL']:
                    # Map trade_type string to MT5 type
                    if trade_type.upper() == 'BUY':
                        mt5_type = mt5.ORDER_TYPE_BUY
                    elif trade_type.upper() == 'SELL':
                        mt5_type = mt5.ORDER_TYPE_SELL
                    else:
                        logging.warning(f"Unknown trade_type '{trade_type}' for {symbol}. Skipping.")
                        continue

                    # Filter deals for the current symbol and trade_type
                    filtered_deals = all_deals[
                        (all_deals['symbol'] == symbol) &
                        (all_deals['type'] == mt5_type) &
                        (all_deals['entry'] == 1)  # Include only closed trades
                    ]

                    # Initialize default metrics
                    metrics = {
                        "Symbol": symbol,
                        "Type": trade_type,
                        "Trades Today": 0,
                        "Trades This Week": 0,
                        "Trades This Month": 0,
                        "Trades All Time": 0,
                        "Profit Today": "0.00",
                        "Profit This Week": "0.00",
                        "Profit This Month": "0.00",
                        "Profit All Time": "0.00",
                        "Wins Today": 0,
                        "Wins This Week": 0,
                        "Wins This Month": 0,
                        "Wins All Time": 0,
                        "Losses Today": 0,
                        "Losses This Week": 0,
                        "Losses This Month": 0,
                        "Losses All Time": 0
                    }

                    if not filtered_deals.empty:
                        # Trades All Time
                        trades_all_time = len(filtered_deals)
                        profit_all_time = filtered_deals['profit'].sum()
                        wins_all_time = len(filtered_deals[filtered_deals['profit'] > 0])
                        losses_all_time = len(filtered_deals[filtered_deals['profit'] <= 0])

                        # Trades This Month
                        month_deals = filtered_deals[
                            (filtered_deals['time'] >= month_start) &
                            (filtered_deals['time'] <= now)
                        ]
                        trades_month = len(month_deals)
                        profit_month = month_deals['profit'].sum()
                        wins_month = len(month_deals[month_deals['profit'] > 0])
                        losses_month = len(month_deals[month_deals['profit'] <= 0])

                        # Trades This Week
                        week_deals = filtered_deals[
                            (filtered_deals['time'] >= week_start) &
                            (filtered_deals['time'] <= now)
                        ]
                        trades_week = len(week_deals)
                        profit_week = week_deals['profit'].sum()
                        wins_week = len(week_deals[week_deals['profit'] > 0])
                        losses_week = len(week_deals[week_deals['profit'] <= 0])

                        # Trades Today
                        today_deals = filtered_deals[
                            (filtered_deals['time'] >= today_start) &
                            (filtered_deals['time'] <= now)
                        ]
                        trades_today = len(today_deals)
                        profit_today = today_deals['profit'].sum()
                        wins_today = len(today_deals[today_deals['profit'] > 0])
                        losses_today = len(today_deals[today_deals['profit'] <= 0])

                        # Update metrics dictionary
                        metrics.update({
                            "Trades Today": trades_today,
                            "Trades This Week": trades_week,
                            "Trades This Month": trades_month,
                            "Trades All Time": trades_all_time,
                            "Profit Today": f"{profit_today:.2f}",
                            "Profit This Week": f"{profit_week:.2f}",
                            "Profit This Month": f"{profit_month:.2f}",
                            "Profit All Time": f"{profit_all_time:.2f}",
                            "Wins Today": wins_today,
                            "Wins This Week": wins_week,
                            "Wins This Month": wins_month,
                            "Wins All Time": wins_all_time,
                            "Losses Today": losses_today,
                            "Losses This Week": losses_week,
                            "Losses This Month": losses_month,
                            "Losses All Time": losses_all_time
                        })
                    else:
                        logging.debug(f"No historical deals found for {symbol} {trade_type}.")

                    metrics_list.append(metrics)

            # Insert metrics into the Treeview
            for metric in metrics_list:
                self.tree.insert('', 'end', values=tuple(metric[col] for col in self.tree['columns']))

            logging.info("Risk metrics refreshed successfully.")
        except KeyError as ke:
            logging.error(f"KeyError in refresh_risk_metrics: {ke}")
            messagebox.showerror("Error", f"An error occurred while refreshing risk metrics: {ke}")
        except Exception as e:
            logging.error(f"Error refreshing risk metrics: {e}")
            messagebox.showerror("Error", f"Failed to refresh risk metrics: {e}")

class MarketAnalysisTab(ttk.Frame):
    def __init__(self, parent, gui):
        super().__init__(parent)
        self.gui = gui
        self.create_widgets()

    def create_widgets(self):
        header = tk.Label(self, text="Market Analysis", font=("Arial", 14, "bold"))
        header.pack(pady=10)

        self.analysis_text = tk.Text(self, wrap="word", font=("Arial", 12))
        self.analysis_text.pack(expand=True, fill="both", padx=10, pady=10)

        refresh_button = tk.Button(self, text="Refresh Analysis",
                                   command=lambda: self.gui.update_market_analysis(config.symbols))
        refresh_button.pack(pady=10)

    def display_analysis(self, analysis):
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, analysis)