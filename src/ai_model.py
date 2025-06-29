"""
Neural Pulse Scalper - AI/ML Model Module

This module contains the logic for the AI/ML decision-making component of the strategy.
It provides a framework for feature engineering, training, evaluating, and deploying
various types of machine learning models to predict profitable trading opportunities.

The key components are:
- FeatureEngineer: Creates features and target variables from raw market data.
- AIModel (Abstract Base Class): Defines the interface for all models.
- Concrete Model Implementations: DecisionTree, RandomForest, XGBoost, NeuralNetwork.
- ModelFactory: A helper to instantiate the correct model based on configuration.
"""

import logging
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional

# Scikit-learn imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# Other ML libraries
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles the creation of features and target variables for the ML model.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ai_config = config['ai_model']
        self.feature_list = self.ai_config['training']['features']

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates all necessary features from the OHLCV data.
        This assumes that the indicators have already been added to the dataframe.
        
        Args:
            df: DataFrame with OHLCV and technical indicators.
            
        Returns:
            DataFrame with the selected features.
        """
        features = pd.DataFrame(index=df.index)
        
        # Example of creating/selecting features based on config
        for feature_name in self.feature_list:
            if feature_name in df.columns:
                features[feature_name] = df[feature_name]
            else:
                # Placeholder for more complex feature generation
                # e.g., 'ema_diff' = df['ema_5'] - df['ema_20']
                logger.warning(f"Feature '{feature_name}' not found in DataFrame. Skipping.")
        
        return features

    def create_target(self, df: pd.DataFrame, future_periods: int = 3, profit_threshold: float = 0.005) -> pd.Series:
        """
        Creates the target variable for a classification task.
        The target is 1 if the price increases by `profit_threshold` within `future_periods`, otherwise 0.
        
        Args:
            df: DataFrame with OHLCV data.
            future_periods: How many periods into the future to look for a profit.
            profit_threshold: The percentage increase required to be considered a 'buy' signal.
            
        Returns:
            A pandas Series representing the target variable (1 for buy, 0 for hold/sell).
        """
        # Calculate future returns
        future_returns = df['close'].pct_change(periods=future_periods).shift(-future_periods)
        
        # Create target: 1 if future return > threshold, else 0
        target = (future_returns > profit_threshold).astype(int)
        target.name = 'target'
        
        return target

    def prepare_data_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares the full dataset for training by creating features and target.
        
        Args:
            df: The raw DataFrame with OHLCV and indicator data.
            
        Returns:
            A tuple of (features_df, target_series).
        """
        features = self.create_features(df)
        target = self.create_target(df)
        
        # Align features and target, dropping NaNs
        full_df = pd.concat([features, target], axis=1).dropna()
        
        X = full_df[self.feature_list]
        y = full_df['target']
        
        return X, y


class AIModel(ABC):
    """Abstract Base Class for all machine learning models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate the model's performance."""
        y_pred = self.predict(X_test)
        y_pred_class = (y_pred > 0.5).astype(int) # Assuming probability output

        report = classification_report(y_test, y_pred_class, output_dict=True, zero_division=0)
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred_class, zero_division=0)}")
        
        return {
            "accuracy": accuracy_score(y_test, y_pred_class),
            "precision": precision_score(y_test, y_pred_class, zero_division=0),
            "recall": recall_score(y_test, y_pred_class, zero_division=0),
            "f1_score": f1_score(y_test, y_pred_class, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred_class).tolist(),
            "classification_report": report
        }

    def save(self, path: str) -> None:
        """Save the model and the scaler to a file."""
        logger.info(f"Saving model and scaler to {path}...")
        try:
            joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
            logger.info("Model saved successfully.")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {e}", exc_info=True)

    def load(self, path: str) -> None:
        """Load the model and the scaler from a file."""
        logger.info(f"Loading model and scaler from {path}...")
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}", exc_info=True)


class DecisionTreeModel(AIModel):
    """Decision Tree Classifier model."""
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        X_scaled = self.scaler.fit_transform(X)
        self.model = DecisionTreeClassifier(random_state=42)
        
        # Placeholder for hyperparameter tuning
        logger.info("Training DecisionTreeModel...")
        self.model.fit(X_scaled, y)
        logger.info("Training complete.")
        
        return self.evaluate(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        # predict_proba returns [[prob_class_0, prob_class_1]]
        return self.model.predict_proba(X_scaled)[:, 1]


class RandomForestModel(AIModel):
    """Random Forest Classifier model."""

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
        
        # Placeholder for hyperparameter tuning
        logger.info("Training RandomForestModel...")
        self.model.fit(X_scaled, y)
        logger.info("Training complete.")
        
        return self.evaluate(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class XGBoostModel(AIModel):
    """XGBoost Classifier model."""

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        
        logger.info("Training XGBoostModel...")
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )
        logger.info("Training complete.")
        
        return self.evaluate(X_test, y_test)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class NeuralNetworkModel(AIModel):
    """Neural Network model using TensorFlow/Keras."""

    def _build_model(self, input_shape):
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = self._build_model(X_train_scaled.shape[1])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        logger.info("Training NeuralNetworkModel...")
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        logger.info("Training complete.")
        
        return self.evaluate(X_test, y_test)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).flatten()

    def save(self, path: str) -> None:
        """Save the Keras model and the scaler."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Cannot save.")
        
        # Keras models are saved in a directory format
        # We'll save the scaler separately.
        model_path = f"{path}_keras_model"
        scaler_path = f"{path}_scaler.joblib"
        
        logger.info(f"Saving Keras model to {model_path} and scaler to {scaler_path}")
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info("Keras model and scaler saved.")

    def load(self, path: str) -> None:
        """Load the Keras model and the scaler."""
        model_path = f"{path}_keras_model"
        scaler_path = f"{path}_scaler.joblib"

        logger.info(f"Loading Keras model from {model_path} and scaler from {scaler_path}")
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        logger.info("Keras model and scaler loaded.")


class ModelFactory:
    """Factory class to create AI models based on configuration."""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> AIModel:
        """
        Creates an instance of an AIModel subclass based on the config.
        
        Args:
            config: The main configuration dictionary.
            
        Returns:
            An instance of a class that inherits from AIModel.
        """
        model_type = config['ai_model']['model_type']
        logger.info(f"Creating model of type: {model_type}")

        if model_type == "decision_tree":
            return DecisionTreeModel(config)
        elif model_type == "random_forest":
            return RandomForestModel(config)
        elif model_type == "xgboost":
            return XGBoostModel(config)
        elif model_type == "neural_network":
            return NeuralNetworkModel(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


def run_hyperparameter_optimization(config: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
    """
    (Placeholder) Runs hyperparameter optimization using Optuna.
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna is not installed. Skipping hyperparameter optimization.")
        return None

    logger.info("Starting hyperparameter optimization...")
    
    def objective(trial):
        # This is a simplified example for RandomForest
        # A real implementation would have different spaces for each model type
        model_type = config['ai_model']['model_type']
        
        if model_type != "random_forest":
            logger.warning(f"Hyperparameter optimization not implemented for {model_type}. Skipping.")
            return 1.0 # Return a high value to indicate failure

        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            class_weight='balanced'
        )
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=config['ai_model']['training']['cross_validation_folds'])
        
        # Scale data for each fold to prevent data leakage
        scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            scores.append(f1_score(y_test, y_pred, zero_division=0))
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) # Number of trials can be configured
    
    logger.info(f"Best trial F1-score: {study.best_value}")
    logger.info(f"Best hyperparameters: {study.best_params}")
    
    return study.best_params

