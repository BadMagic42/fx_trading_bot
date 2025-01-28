import warnings
import logging
import pandas as pd
import numpy as np
import config
import joblib

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting

from sklearn.utils.validation import check_X_y, check_array
from indicators import create_lagged_features
from lstm_model import train_lstm_model
from sarima_model import train_sarima_model
from interpretability import analyze_feature_importance, perform_correlation_analysis, evaluate_model
from collections import Counter


def get_final_feature_list(data):
    """
    Determine final features (excluding lagged features) based on features_selected in config and data columns.
    """

    # Start with essential features like 'close'
    final_features = ['close']

    # Add selected features if present in the data
    for key, feats in config.base_feature_map.items():
        if config.features_selected.get(key, False):
            final_features.extend([f for f in feats if f in data.columns])

    # Remove duplicates and ensure only columns in data are included
    final_features = list(dict.fromkeys([f for f in final_features if f in data.columns]))

    return final_features


def prepare_ml_dataset(data):
    """
    Prepare the dataset for machine learning by selecting features and labels.
    Ensures lagged features are checked (not created) based on configuration.
    """
    logging.info(f"Initial dataset shape: {data.shape}")
    
    if data.empty:
        logging.warning("Empty data received for ML dataset preparation.")
        return pd.DataFrame(), pd.Series(), []

    if 'label' not in data.columns:
        logging.error("No 'label' in data for ML preparation.")
        return pd.DataFrame(), pd.Series(), []

    # Step 1: Gather the final features
    final_features = get_final_feature_list(data)

    # Step 2: Ensure lagged features exist in the data (if configured)
    if config.features_selected.get('lagged_features', False):
        logging.debug(f"Checking for lagged features as per configuration.")
        lagged_features = []
        for feature_group, feature_names in config.base_feature_map.items():
            if config.lagged_features_selected.get(feature_group, False):
                lagged_features.extend(
                    [f"{feature}_lag_{lag}" for feature in feature_names for lag in range(1, config.max_lag + 1)]
                )
        
        # Check if lagged features exist in the data
        missing_lagged_features = [feat for feat in lagged_features if feat not in data.columns]
        if missing_lagged_features:
            logging.error(f"Missing lagged features in data: {missing_lagged_features}")
            return pd.DataFrame(), pd.Series(), []

    # Step 3: Ensure all required features and labels exist
    needed_cols = final_features + ['label']
    missing_columns = [col for col in needed_cols if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns in data: {missing_columns}")
        return pd.DataFrame(), pd.Series(), []

    # Step 4: Finalize X (features) and y (labels)
    X = data[final_features]
    y = data['label']

    # Step 5: Handle missing values
    X = X.ffill().bfill().fillna(0)
    y = y.fillna(0)

    logging.info(f"Dataset shape after NaN handling: {X.shape}, Labels shape: {y.shape}")
    feature_names = X.columns.tolist()
    logging.debug("ML dataset prepared with selected features and labels.")
    return X, y, feature_names



class CompatibleLGBMClassifier(LGBMClassifier):
    """
    A minor extension to LGBMClassifier for sklearn compatibility if needed.
    """
    def __sklearn_tags__(self):
        return {
            "multioutput": False,
            "multilabel": False,
            "no_validation": False,
        }


class CompatibleXGBClassifier(XGBClassifier):
    """
    A minor extension to XGBClassifier for better sklearn compatibility.
    """
    def __sklearn_tags__(self):
        return {
            "multioutput": False,
            "multilabel": False,
            "no_validation": False,
        }

    def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
        logging.debug("Validating data for CompatibleXGBClassifier")
        if validate_separately:
            X = check_array(X, **check_params)
            if y is not None:
                y = check_array(y, ensure_2d=False, **check_params)
        else:
            X, y = check_X_y(X, y, **check_params)
        return X, y


def train_ml_model(symbol_data):
    """
    Main pipeline for model training:
      1) Gather data from multiple symbols, combine.
      2) Split into train/calibrate/test.
      3) Optionally resample to handle imbalance.
      4) Scale features.
      5) Train & calibrate classical models (RF, LightGBM, XGB, etc.).
      6) Optionally train LSTM & SARIMA.
      7) Optionally train a Meta-Model that stacks everything.
      8) Save feature names, logging metrics.
    Returns: (models, feature_names, feature_importances_df)
    """
    # -------------------------------
    # 1) SUPPRESS SPECIFIC WARNINGS
    # -------------------------------
    warnings.filterwarnings(
        "ignore",
        message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.",
        category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*`BaseEstimator._validate_data` is deprecated.*",
        category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*`_get_tags` and `_more_tags`.*",
        category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Usage of np.ndarray subset.*",
        category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*invalid value encountered in cast.*",
        category=RuntimeWarning
    )

    # -------------------------------
    # 2) INITIALIZE AGGREGATORS
    # -------------------------------
    combined_X = []
    combined_y = []
    feature_names = []
    feature_importances = []
    metrics_collection = []

    # -------------------------------
    # 3) COLLECT & PREPARE DATA
    # -------------------------------
    for symbol, data in symbol_data.items():
        logging.debug(f"Symbol: {symbol}, Data Shape: {data.shape}")
        if data.empty or 'label' not in data.columns:
            logging.warning(f"No labeled data for {symbol}, skipping training.")
            continue

        X, y, feats = prepare_ml_dataset(data)
        if X.empty or y.empty:
            continue

        logging.debug(f"Adding {X.shape[0]} rows to combined_X for symbol: {symbol}")
        combined_X.append(X)
        combined_y.append(y)
        feature_names = feats  # We assume consistent features across symbols

    if not combined_X:
        logging.critical("No data for training.")
        raise ValueError("No training data available.")

    # Merge all symbol data
    combined_X = pd.concat(combined_X, axis=0)
    combined_y = pd.concat(combined_y, axis=0)
    logging.debug(f"Combined dataset shape: {combined_X.shape}, Target shape: {combined_y.shape}")

    # Basic fill/cleanup
    combined_X = combined_X.ffill().bfill().fillna(0)
    combined_y = combined_y.fillna(0)

    # Log some stats on the merged dataset
    logging.debug(
        f"Min/Max of combined_X (pre-split): {combined_X.min().min()} / {combined_X.max().max()}"
    )
    logging.debug(f"Any negative values (pre-split)? {(combined_X < 0).any().any()}")

    logging.info(f"Training models with {len(combined_X)} samples total.")
    logging.debug(f"Sample rows from combined dataset: {combined_X.head()}")

    # Convert boolean columns to int if any
    bool_cols = [col for col in combined_X.columns if combined_X[col].dtype == bool]
    if bool_cols:
        logging.debug(f"Converting boolean columns to integers: {bool_cols}")
        for c in bool_cols:
            combined_X[c] = combined_X[c].astype(int)

    # -------------------------------
    # 4) TRAIN/CALIBRATION/TEST SPLIT
    #    (Before resampling & scaling)
    # -------------------------------

    # 1) If you have a 'timestamp' or date column in combined_X, ensure we can sort by it.
    #    If 'combined_X' is a DataFrame, do it BEFORE converting to .values:
    #    e.g., merged_df = pd.concat(...) # from all symbols
    #    merged_df = merged_df.sort_values('timestamp')
    #    Then X_full = merged_df[feature_names].values, y_full = merged_df['label'].values

    # For demonstration, assume combined_X is still a DataFrame at this point:
    if 'time' in combined_X.columns:
        combined_X = combined_X.sort_values('time')
        
    # Convert to numpy arrays only after sorting
    X_full = combined_X[feature_names].values
    y_full = combined_y.values  # Or however you store labels

    total_samples = len(X_full)

    calibrate_size = int(config.calibrate_perc * total_samples)
    test_size = int(config.test_perc * total_samples)
    train_size = total_samples - calibrate_size - test_size

    if train_size <= 0 or calibrate_size <= 0 or test_size <= 0:
        logging.error("Invalid split sizes. Check your config calibrate_perc and test_perc.")
        raise ValueError("Train/Calibrate/Test sizes must be positive.")

    # Chronological slices
    X_train = X_full[:train_size]
    y_train = y_full[:train_size]

    X_calibrate = X_full[train_size: train_size + calibrate_size]
    y_calibrate = y_full[train_size: train_size + calibrate_size]

    X_test = X_full[train_size + calibrate_size:]
    y_test = y_full[train_size + calibrate_size:]

    logging.info(
        f"Data split => Training: {X_train.shape}, Calibration: {X_calibrate.shape}, Test: {X_test.shape}"
    )


    # -------------------------------
    # 5) OPTIONAL RESAMPLING
    # -------------------------------
    if config.RESAMPLING_METHOD in ['SMOTE', 'SMOTEEN', 'ADASYN']:
        try:
            logging.info(f"Applying resampling method: {config.RESAMPLING_METHOD}")
            from imblearn.over_sampling import SMOTE, ADASYN
            from imblearn.combine import SMOTEENN

            if config.RESAMPLING_METHOD == 'SMOTE':
                sampler = SMOTE(random_state=42)
            elif config.RESAMPLING_METHOD == 'SMOTEEN':
                sampler = SMOTEENN(random_state=42)
            else:  # 'ADASYN'
                sampler = ADASYN(random_state=42)

            logging.info(f"Original class distribution: {Counter(y_train)}")
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            logging.info(f"Resampled class distribution: {Counter(y_train_resampled)}")

            X_train, y_train = X_train_resampled, y_train_resampled
            logging.info(f"Resampling completed => X_train shape: {X_train.shape}")
        except Exception as e:
            logging.error(f"Error during resampling with {config.RESAMPLING_METHOD}: {e}")
            logging.info("Proceeding with original X_train without resampling.")
    else:
        logging.info("No resampling method selected. Using original training data.")

    # -------------------------------
    # 6) FEATURE SCALING
    # -------------------------------
    if config.feature_scaling == 'standard':
        scaler = StandardScaler()
    elif config.feature_scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_calibrate = scaler.transform(X_calibrate)
    X_test = scaler.transform(X_test)

    # Save the scaler
    scaler_path = config.BASE_DIR / "models/scaler.joblib"
    scaler_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved at {scaler_path}")

    # Check for any NaNs/inf after scaling
    if np.isnan(X_train).any() or np.isnan(X_calibrate).any() or np.isnan(X_test).any():
        logging.warning("Found NaN values in scaled data.")
    if np.isinf(X_train).any() or np.isinf(X_calibrate).any() or np.isinf(X_test).any():
        logging.warning("Found Inf values in scaled data.")

    # -------------------------------
    # 7) TRAIN & CALIBRATE BASE MODELS
    # -------------------------------
    models = {}
    feature_importances_df_list = []
    metrics_collection = []

    def tune_hyperparameters(model_name, clf, param_grid):
        """
        Hyperparameter tuning wrapper that can do grid_search, random_search, bayesian, or tpot.
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from skopt import BayesSearchCV
        from tpot import TPOTClassifier

        if config.TUNING_METHOD == 'grid_search':
            searcher = GridSearchCV(
                clf, param_grid=param_grid, cv=3, scoring="accuracy",
                verbose=2, n_jobs=-1, refit=True
            )
        elif config.TUNING_METHOD == 'random_search':
            searcher = RandomizedSearchCV(
                clf, param_distributions=param_grid, n_iter=50, cv=3,
                scoring="accuracy", verbose=2, random_state=42, n_jobs=-1, refit=True
            )
        elif config.TUNING_METHOD == 'bayesian_optimization':
            searcher = BayesSearchCV(
                clf, search_spaces=param_grid, n_iter=config.BAYESIAN_MAX_EVALS,
                cv=3, scoring="accuracy", verbose=2, random_state=42, n_jobs=-1, refit=True
            )
        elif config.TUNING_METHOD == 'tpot':
            tpot = TPOTClassifier(
                generations=config.AUTOML_GENERATIONS,
                population_size=config.AUTOML_POPULATION_SIZE,
                verbosity=3, n_jobs=1, max_eval_time_mins=10, random_state=42
            )
            tpot.fit(X_train, y_train)
            logging.info(f"{model_name} TPOT best pipeline: {tpot.fitted_pipeline_}")
            return tpot.fitted_pipeline_
        else:
            logging.error(f"Unsupported tuning method: {config.TUNING_METHOD}. Proceeding without HP tuning.")
            clf.fit(X_train, y_train)
            return clf

        searcher.fit(X_train, y_train)
        logging.info(f"{model_name} best params: {searcher.best_params_}")
        return searcher.best_estimator_

    # Train each configured model
    for model_name, active in config.models_available.items():
        if not active:
            continue

        # pick the classifier
        param_grid = config.TUNING_PARAMETERS.get(model_name, {})
        if model_name == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        elif model_name == 'XGBoost':
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        elif model_name == 'LightGBM':
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(
                objective='multiclass',
                random_state=42,
                n_jobs=-1,
                force_col_wise=True
            )
        elif model_name == 'Neural Network':
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(random_state=42, max_iter=2000)
        else:
            logging.warning(f"Unrecognized model: {model_name}. Skipping.")
            continue

        # Possibly do HP Tuning
        try:
            if config.ENABLE_HYPERPARAMETER_TUNING and param_grid:
                clf = tune_hyperparameters(model_name, clf, param_grid)
            else:
                clf.fit(X_train, y_train)
                logging.info(f"{model_name} trained (no HP tuning).")
        except Exception as e:
            logging.error(f"{model_name} training error: {e}")
            continue

        # Calibrate the model
        try:
            calibrated_clf = CalibratedClassifierCV(estimator=clf, method='sigmoid', cv='prefit')
            calibrated_clf.fit(X_calibrate, y_calibrate)

            # Save
            model_path = config.BASE_DIR / f"models/{model_name.lower().replace(' ', '_')}_calibrated_model.joblib"
            joblib.dump(calibrated_clf, model_path)
            logging.info(f"Calibrated {model_name} saved at {model_path}")

        except Exception as e:
            logging.error(f"Calibration error for {model_name}: {e}")
            continue

        # Evaluate the calibrated model
        try:
            metrics = evaluate_model(calibrated_clf, X_test, y_test, model_name, feature_names)
            metrics['model_name'] = model_name
            metrics_collection.append(metrics)
        except Exception as e:
            logging.error(f"Evaluation error for {model_name}: {e}")

        # Feature importances
        fi_df = analyze_feature_importance(clf, feature_names, model_name)
        if not fi_df.empty:
            feature_importances_df_list.append(fi_df)

        # Correlation analysis
        df_train_corr = pd.DataFrame(X_train, columns=feature_names)
        perform_correlation_analysis(df_train_corr, feature_names, model_name)

        # store the final calibrated model
        models[model_name] = calibrated_clf

    # Summarize feature importances
    if feature_importances_df_list:
        feature_importances_df = pd.concat(feature_importances_df_list, axis=0)
    else:
        feature_importances_df = pd.DataFrame()

    # -------------------------------
    # 8) LSTM TRAINING
    # -------------------------------
    if config.ENABLE_LSTM:
        try:
            logging.info("Starting LSTM training.")
            # the LSTM requires special code => we pass X_train, y_train, etc.
            lstm_model = train_lstm_model(X_train, y_train, X_calibrate, y_calibrate, feature_names)
            if lstm_model:
                models["LSTM"] = lstm_model
        except Exception as e:
            logging.error(f"LSTM training error: {e}")

    # -------------------------------
    # 9) SARIMA TRAINING
    # -------------------------------
    if config.ENABLE_SARIMA:
        try:
            logging.info("Starting SARIMA training.")
            for symbol, df_sym in symbol_data.items():
                if df_sym.empty or 'close' not in df_sym.columns:
                    logging.warning(f"No close for {symbol}, skipping SARIMA.")
                    continue
                # set 'time' as index
                df_sym = df_sym.set_index('time')
                sarima_m = train_sarima_model(
                    df_sym,
                    order=config.sarima_order,
                    seasonal_order=config.sarima_seasonal_order,
                    symbol=symbol
                )
                if sarima_m:
                    models[f"SARIMA_{symbol}"] = sarima_m
        except Exception as e:
            logging.error(f"SARIMA training error: {e}")

    # -------------------------------
    # 10) META MODEL TRAINING
    # -------------------------------
    if config.ENABLE_META_MODEL:
        try:
            logging.info("Starting meta-model stacking approach.")
            from meta_model import MetaModel
            from sklearn.linear_model import LogisticRegression

            # Ensure base models are properly passed to the MetaModel
            base_models = [
                (m_name, m_obj)
                for m_name, m_obj in models.items()
                if hasattr(m_obj, "predict_proba")  # Ensure model supports `predict_proba`
            ]

            if not base_models:
                logging.warning("No valid base models found for MetaModel. Skipping stacking.")
            else:
                # Build the MetaModel
                meta_learner = LogisticRegression(random_state=42, max_iter=1000)
                stacked_model = MetaModel(meta_learner, base_models)

                # Fit the stacked model on training data
                stacked_model.fit(X_train, y_train)

                # Evaluate MetaModel
                try:
                    y_pred = stacked_model.predict(X_test)
                    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                    meta_metrics = {
                        "accuracy":  accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
                        "recall":    recall_score(y_test, y_pred, average='macro', zero_division=0),
                        "f1_score":  f1_score(y_test, y_pred, average='macro', zero_division=0),
                        "model_name": "MetaModel"
                    }
                    metrics_collection.append(meta_metrics)
                    logging.info(f"MetaModel metrics: {meta_metrics}")
                except Exception as eval_error:
                    logging.error(f"Error evaluating MetaModel: {eval_error}")

                # Save the MetaModel
                models["MetaModel"] = stacked_model
                meta_path = config.BASE_DIR / "models/metamodel_calibrated_model.joblib"
                joblib.dump(stacked_model, meta_path)
                logging.info(f"MetaModel saved to {meta_path}")

        except Exception as meta_error:
            logging.error(f"Error in meta-model training block: {meta_error}")

    # -------------------------------
    # 11) SAVE FEATURE NAMES & LOG METRICS
    # -------------------------------
    feature_path = config.BASE_DIR / "models/trained_feature_names.joblib"
    joblib.dump(feature_names, feature_path)
    logging.info(f"{len(feature_names)} feature names saved at {feature_path}")

    # Log classes for each model that supports .classes_
    for m_name, m_obj in models.items():
        if hasattr(m_obj, 'classes_'):
            logging.info(f"{m_name} => classes_: {m_obj.classes_}")

    # Log final metrics
    for m in metrics_collection:
        logging.info(f"Metrics for {m['model_name']}: {m}")

    # Save to CSV
    if metrics_collection:
        df_metrics = pd.DataFrame(metrics_collection)
        metrics_csv = config.BASE_DIR / "models/analysis" / "model_metrics.csv"
        df_metrics.to_csv(metrics_csv, index=False)
        logging.info(f"Model metrics saved to {metrics_csv}")

    return models, feature_names, feature_importances_df
