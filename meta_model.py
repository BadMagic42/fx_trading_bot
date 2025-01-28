# meta_model.py

import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin

class MetaModel(BaseEstimator, ClassifierMixin):
    """
    MetaModel for stacking:
      - Expects each base model is already fitted.
      - On .fit(X, y), it calls base_model.predict_proba(X) for each model,
        stacks them horizontally, and then fits the meta_model on top.
      - On .predict(X), it again calls each base_model.predict_proba(X),
        stacks, and calls meta_model.predict.
    """

    def __init__(self, meta_model, base_models):
        """
        Args:
            meta_model: A scikit-learn estimator (e.g. LogisticRegression) 
                        that will be the meta-learner.
            base_models (list): A list of (name, model) tuples, 
                                where each model is already fitted.
        """
        self.meta_model = meta_model
        self.base_models = base_models
        self.classes_ = None  # For compatibility with sklearn pipelines

    def fit(self, X, y):
        """
        Build stacked features by calling each base model's predict_proba on X,
        then fit the meta_model on those stacked features.

        Args:
            X (np.ndarray): The training inputs (2D array).
            y (np.ndarray): The training labels (1D array).
        """
        logging.info("Generating stacked features via base_models (predict_proba).")
        base_outputs = []
        for name, model in self.base_models:
            if not hasattr(model, "predict_proba"):
                raise ValueError(f"Base model {name} does not support predict_proba.")
            proba = model.predict_proba(X)  # shape: (n_samples, n_classes)
            base_outputs.append(proba)

        # Stack each model's proba horizontally
        stacked_X = np.hstack(base_outputs)
        logging.info("Fitting meta_model on stacked base-model outputs.")
        self.meta_model.fit(stacked_X, y)

        # Save classes for compatibility
        if hasattr(self.meta_model, "classes_"):
            self.classes_ = self.meta_model.classes_
        else:
            self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """
        Predict final labels by stacking base model probabilities and passing them 
        to the meta_model's predict method.

        Args:
            X (np.ndarray): The test inputs (2D array).

        Returns:
            np.ndarray: predicted labels
        """
        stacked_X = self._generate_stacked_features(X)
        return self.meta_model.predict(stacked_X)

    def predict_proba(self, X):
        """
        Predict final probabilities by stacking base model probabilities.

        Args:
            X (np.ndarray): The test inputs (2D array).

        Returns:
            np.ndarray: predicted probabilities for each class (shape = (n_samples, n_classes)).
        """
        stacked_X = self._generate_stacked_features(X)
        if not hasattr(self.meta_model, "predict_proba"):
            raise ValueError("MetaModel's meta_model does not support predict_proba.")
        return self.meta_model.predict_proba(stacked_X)

    def _generate_stacked_features(self, X):
        """
        Generate stacked features from base models for a given input.

        Args:
            X (np.ndarray): The input data (2D array).

        Returns:
            np.ndarray: Stacked features (shape = (n_samples, sum(n_classes from base models))).
        """
        base_outputs = []
        for name, model in self.base_models:
            if not hasattr(model, "predict_proba"):
                raise ValueError(f"Base model {name} does not support predict_proba.")
            proba = model.predict_proba(X)
            base_outputs.append(proba)
        return np.hstack(base_outputs)
