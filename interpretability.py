# interpretability.py

import shap
import lime
import lime.lime_tabular
import logging
import config
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
    roc_curve, auc, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

def explain_with_shap(model, X, feature_names, model_name=""):
    try:
        # Ensure X is 2D
        X = np.atleast_2d(X)

        # Validate or create feature names
        if feature_names is None or len(feature_names) != X.shape[1]:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
            logging.warning(f"Feature names were missing or mismatched. Using default feature names.")

        # Check if the model has predict_proba
        if not hasattr(model, "predict_proba"):
            raise AttributeError(f"Model {model_name} does not support predict_proba.")

        # Create SHAP explainer
        explainer = shap.Explainer(model.predict_proba, X)
        shap_values = explainer(X)

        # Handle multi-class SHAP values
        if len(shap_values.shape) == 3:  # Check for multi-class
            logging.debug(f"Multi-class SHAP values detected: {shap_values.shape}")

            for class_idx in range(shap_values.shape[2]):
                logging.info(f"Generating SHAP summary plot for class {class_idx}")

                # Generate summary plot for the specific class
                shap.summary_plot(shap_values[:, :, class_idx], features=X, feature_names=feature_names, show=False)

                # Save the plot
                if config.save_plots:
                    file_path = config.BASE_DIR / "models/analysis" / f"shap_summary_{model_name}_class_{class_idx}.png"
                    file_path.unlink(missing_ok=True)
                    plt.savefig(file_path)
                    logging.info(f"SHAP summary plot for class {class_idx} saved at {file_path}")
                plt.close()

        else:  # Binary or single-class
            logging.debug(f"Binary or single-class SHAP values detected: {shap_values.shape}")
            shap.summary_plot(shap_values, features=X, feature_names=feature_names, show=False)

            # Save the plot
            if config.save_plots:
                file_path = config.BASE_DIR / "models/analysis" / f"shap_summary_{model_name}.png"
                file_path.unlink(missing_ok=True)
                plt.savefig(file_path)
                logging.info(f"SHAP summary plot saved at {file_path}")
            plt.close()

        return shap_values

    except Exception as e:
        logging.error(f"Error generating SHAP explanations: {e}", exc_info=True)
        return None


def explain_with_lime(model, X, feature_names):
    """
    Generate LIME explanations for the model's predictions.

    Args:
        model: Trained and calibrated model.
        X (np.ndarray): Feature matrix.
        feature_names (list): List of feature names.

    Returns:
        list: List of LIME explanations.
    """
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X,
            feature_names=feature_names,
            class_names=model.classes_,
            mode='classification'
        )
        explanations = []
        for i in range(min(5, X.shape[0])):  # Generate explanations for first 5 samples
            exp = explainer.explain_instance(X[i], model.predict_proba, num_features=10)
            exp_path = config.BASE_DIR / "models/analysis/lime_explanation_{}.html".format(i)
            exp.save_to_file(exp_path)
            logging.info(f"LIME explanation for sample {i} saved at {exp_path}")
            explanations.append(exp)
        return explanations
    except Exception as e:
        logging.error(f"Error generating LIME explanations: {e}")
        return []


def evaluate_model(model, X_test, y_test, model_name, feature_names):
    """
    Evaluate the model on test data and log the performance metrics.
    
    Args:
        model: Trained and calibrated model.
        X_test (np.ndarray): Test features.
        y_test (pd.Series or np.ndarray): True labels.
        model_name (str): Name of the model.
    
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    

    metrics = {}
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)
    logging.debug(f"Unique classes for {model_name}: {unique_classes}")

    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    except Exception as e:
        logging.error(f"Prediction failed for {model_name}: {e}", exc_info=True)
        return metrics  # Return empty metrics if prediction fails

    try:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
        metrics['f1_score'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
    except Exception as e:
        logging.error(f"Metric calculation failed for {model_name}: {e}", exc_info=True)
        return metrics

    try:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
    except Exception as e:
        logging.error(f"Confusion matrix calculation failed for {model_name}: {e}", exc_info=True)

    try:
        # ROC-AUC Score
        if n_classes > 2:
            # Binarize the output for multi-class ROC-AUC
            y_test_binarized = label_binarize(y_test, classes=unique_classes)
            auc_score = roc_auc_score(y_test_binarized, y_pred_proba, average='macro', multi_class='ovr')
        else:
            auc_score = roc_auc_score(y_test, y_pred_proba[:,1])
        metrics['roc_auc'] = auc_score
    except Exception as e:
        logging.error(f"ROC-AUC calculation failed for {model_name}: {e}", exc_info=True)
        metrics['roc_auc'] = None

    try:
        # Brier Score
        if n_classes > 2:
            # Binarize the output for multi-class Brier Score
            y_test_binarized = label_binarize(y_test, classes=unique_classes)
            brier_scores = []
            for i in range(y_pred_proba.shape[1]):
                brier = brier_score_loss(y_test_binarized[:, i], y_pred_proba[:, i])
                brier_scores.append(brier)
            metrics['brier_score'] = np.mean(brier_scores)
        else:
            brier = brier_score_loss(y_test, y_pred_proba[:,1])
            metrics['brier_score'] = brier
    except Exception as e:
        logging.error(f"Brier Score calculation failed for {model_name}: {e}", exc_info=True)
        metrics['brier_score'] = None

    # Log metrics
    logging.info(f"--- {model_name} Evaluation Metrics ---")
    logging.info(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    logging.info(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
    logging.info(f"Recall: {metrics.get('recall', 'N/A'):.4f}")
    logging.info(f"F1-Score: {metrics.get('f1_score', 'N/A'):.4f}")
    if metrics.get('roc_auc') is not None:
        logging.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    if metrics.get('brier_score') is not None:
        logging.info(f"Brier Score: {metrics['brier_score']:.4f}")

    # Classification Report
    try:
        report = classification_report(y_test, y_pred, zero_division=0)
        logging.debug(f"Classification Report for {model_name}:\n{report}")
    except Exception as e:
        logging.error(f"Classification report generation failed for {model_name}: {e}", exc_info=True)

    # Plot Confusion Matrix
    try:
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        if config.save_plots:
            file_path = config.BASE_DIR / "models/analysis" / f"confusion_matrix_{model_name}.png"
            file_path.unlink(missing_ok=True)
            save_plot(file_path, plt)
            logging.info(f"Confusion matrix plot saved at {file_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to plot/save confusion matrix for {model_name}: {e}", exc_info=True)

    # Plot ROC Curve
    try:
        logging.debug(f"Attempting to plot ROC curve for {model_name}")

        if n_classes > 2:
            # One-vs-Rest ROC Curves
            y_test_binarized = label_binarize(y_test, classes=unique_classes)
            plt.figure(figsize=(8,6))
            for i, class_label in enumerate(unique_classes):
                fpr, tpr, thresholds = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.2f})')
            # Micro-average ROC Curve
            fpr_micro, tpr_micro, _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', linewidth=4,
                     label=f'Micro-average ROC curve (AUC = {roc_auc_micro:.2f})')
            # Macro-average ROC Curve
            roc_auc_macro = roc_auc_score(y_test_binarized, y_pred_proba, average='macro', multi_class='ovr')
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves for {model_name} (One-vs-Rest)')
            plt.legend(loc='lower right')
            if config.save_plots:
                file_path = config.BASE_DIR / "models/analysis" / f"roc_curve_{model_name}.png"
                file_path.unlink(missing_ok=True)
                save_plot(file_path, plt)
                logging.info(f"ROC Curve plot saved at {file_path}")
            plt.close()
        
        else:
            # Binary Classification ROC Curve
            prob_pos = y_pred_proba[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, prob_pos)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            plt.title(f'ROC Curve for {model_name}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            if config.save_plots:
                file_path = config.BASE_DIR / "models/analysis" / f"roc_curve_{model_name}.png"
                file_path.unlink(missing_ok=True)
                save_plot(file_path, plt)
                logging.info(f"ROC Curve plot saved at {file_path}")
            plt.close()
    except Exception as e:
        logging.error(f"Failed to plot/save ROC curve for {model_name}: {e}", exc_info=True)


    # Plot Calibration Curve
    try:
        logging.debug(f"Attempting to plot calibration curve for {model_name}")
        
        if n_classes > 2:
            # One-vs-Rest Calibration Curves
            y_test_binarized = label_binarize(y_test, classes=unique_classes)
            plt.figure(figsize=(8,6))
            for i, class_label in enumerate(unique_classes):
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test_binarized[:, i], y_pred_proba[:, i], n_bins=10, strategy='uniform'
                )
                plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"Class {class_label}")
                logging.debug(f"Mean predicted probability for class {class_label}: {mean_predicted_value}")
                logging.debug(f"Fraction of positives for class {class_label}: {fraction_of_positives}")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title(f"Calibration Curve for {model_name} (One-vs-Rest)")
            plt.legend()
            if config.save_plots:
                file_path = config.BASE_DIR / "models/analysis" / f"calibration_curve_{model_name}.png"
                file_path.unlink(missing_ok=True)
                save_plot(file_path, plt)
                logging.info(f"Calibration Curve plot saved at {file_path}")
            plt.close()
        else:
            # Binary Calibration Curve
            prob_pos = y_pred_proba[:,1]
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10, strategy='uniform')
            logging.debug(f"Mean predicted probability: {mean_predicted_value}")
            logging.debug(f"Fraction of positives: {fraction_of_positives}")

            plt.figure(figsize=(8,6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name} Calibration")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title(f"Calibration Curve for {model_name}")
            plt.legend()
            if config.save_plots:
                file_path = config.BASE_DIR / "models/analysis" / f"calibration_curve_{model_name}.png"
                file_path.unlink(missing_ok=True)
                save_plot(file_path, plt)
                logging.info(f"Calibration Curve plot saved at {file_path}")
            plt.close()
    except Exception as e:
        logging.error(f"Failed to plot/save calibration curve for {model_name}: {e}", exc_info=True)

    # SHAP and LIME Explanations
    try:
        if config.ENABLE_SHAP:
            explain_with_shap(model, X_test, feature_names, model_name)
        if config.ENABLE_LIME:
            explain_with_lime(model, X_test, feature_names)
    except Exception as e:
        logging.error(f"Error during model explainability: {e}")


    return metrics



def analyze_feature_importance(model, feature_names, model_name):
    """
    Analyze and plot feature importances for various models.
    
    Args:
        model: Trained model with either 'feature_importances_' or 'coef_' attribute.
        feature_names (list): List of feature names.
        model_name (str): Name of the model.
    
    Returns:
        pd.DataFrame: DataFrame containing feature importances.
    """
    try:
        # Handle models wrapped in CalibratedClassifierCV
        if isinstance(model, CalibratedClassifierCV) and hasattr(model.base_estimator, 'feature_importances_'):
            model = model.base_estimator

        # Extract feature importances for tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # Handle LightGBM models specifically
        elif hasattr(model, 'booster_'):
            importances = model.booster_.feature_importance(importance_type='gain')
        # Handle XGBoost models specifically
        elif hasattr(model, 'get_booster'):
            booster = model.get_booster()
            importances = booster.get_score(importance_type='weight')
            importances = [importances.get(f'f{i}', 0) for i in range(len(feature_names))]
        # Handle linear models with coefficients
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            logging.warning(f"{model_name} does not have 'feature_importances_' or 'coef_' attributes.")
            return pd.DataFrame()

        # Validate importances length matches feature names
        if len(importances) != len(feature_names):
            logging.error(f"Feature importance mismatch for {model_name}: "
                          f"{len(importances)} importances vs. {len(feature_names)} features.")
            return pd.DataFrame()

        # Create a DataFrame of feature importances
        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        # Log feature importances
        logging.info(f"Feature importances for {model_name}:")
        logging.info(feature_importances)
        
        # Plot feature importances
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
        plt.title(f'Feature Importances for {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        if config.save_plots:
            file_path = config.BASE_DIR / "models/analysis" / f"feature_importances_{model_name}.png"
            file_path.unlink(missing_ok=True)
            plt.savefig(file_path)
            logging.info(f"Feature importances plot saved at {file_path}")
        plt.close()

        return feature_importances
    except Exception as e:
        logging.error(f"Error analyzing feature importances for {model_name}: {e}")
        return pd.DataFrame()



def perform_correlation_analysis(X, feature_names, model_name):
    """
    Perform correlation analysis on the features.
    
    Args:
        X (pd.DataFrame or np.ndarray): Feature set.
        feature_names (list): List of feature names.
        model_name (str): Name of the model.
    
    Returns:
        None
    """
    try:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=feature_names)
        
        correlation_matrix = X.corr()
        plt.figure(figsize=(12,10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
        plt.title(f'Feature Correlation Matrix for {model_name}')
        if config.save_plots:
            file_path = config.BASE_DIR / "models/analysis" / f"correlation_matrix_{model_name}.png"
            file_path.unlink(missing_ok=True)
            save_plot(file_path, plt)
            logging.info(f"Correlation Matrix plot saved at {file_path}")
        plt.close()
        
        # Optionally, identify highly correlated pairs
        threshold = 0.8
        correlated_features = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    correlated_features.add((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        
        if correlated_features:
            logging.info(f"Highly correlated feature pairs for {model_name} (|correlation| > {threshold}):")
            for pair in correlated_features:
                logging.info(pair)
        else:
            logging.info(f"No highly correlated features found for {model_name}.")
    
    except Exception as e:
        logging.error(f"Error performing correlation analysis for {model_name}: {e}")

def save_plot(file_path, plt):
    """
    Save the plot, overwriting if the file exists.
    Args:
        file_path (str or Path): Path to the file.
        plt (matplotlib.pyplot): The current plot.
    """
    try:
        file_path = Path(file_path)  # Ensure it's a Path object
        if file_path.exists():
            try:
                file_path.unlink()  # Delete the file if it exists
                logging.info(f"Existing plot at {file_path} removed.")
            except Exception as remove_err:
                logging.error(f"Failed to remove existing plot at {file_path}: {remove_err}", exc_info=True)
        # Save the new plot
        plt.savefig(file_path)
        logging.info(f"Plot saved at {file_path}")
    except Exception as e:
        logging.error(f"Failed to save plot at {file_path}: {e}", exc_info=True)
