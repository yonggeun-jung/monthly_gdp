"""
Multi-Layer Perceptron (MLP) Model
----------------------------------
Feedforward neural network with Bayesian hyperparameter optimization.
"""

import os
import numpy as np
from .base import BaseGDPModel


class MLPModel(BaseGDPModel):
    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(name="MLP", seed=seed)
        self.config = {
            "max_trials": kwargs.get("max_trials", 30),
            "max_layers": kwargs.get("max_layers", 2),
            "max_units": kwargs.get("max_units", 64),
            "epochs": kwargs.get("epochs", 500),
            "patience": kwargs.get("patience", 30),
            "batch_size": kwargs.get("batch_size", 4),
        }
        self.best_hp = None

    def _set_seeds(self):
        import random, tensorflow as tf
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _build_from_hp(self, hp_dict, n_features):
        """Build a keras model from a fixed hyperparameter dict."""
        from tensorflow import keras

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(n_features,)))
        for i in range(hp_dict["num_layers"]):
            units_key = f"units_{i}"
            dropout_key = f"dropout_{i}"
            model.add(keras.layers.Dense(
                units=hp_dict.get(units_key, 32),
                activation=hp_dict.get("activation", "relu"),
            ))
            model.add(keras.layers.Dropout(
                rate=hp_dict.get(dropout_key, 0.0)
            ))
        model.add(keras.layers.Dense(1, activation="linear"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp_dict.get("lr", 1e-3)
            ),
            loss="mse",
        )
        return model

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        import tensorflow as tf
        from tensorflow import keras
        import keras_tuner as kt

        tf.keras.backend.clear_session()
        self._set_seeds()
        n_features = X_train.shape[1]
        cfg = self.config

        def build_model(hp):
            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(n_features,)))
            for i in range(hp.Int("num_layers", 1, cfg["max_layers"])):
                model.add(keras.layers.Dense(
                    units=hp.Int(f"units_{i}", 8, cfg["max_units"], step=8),
                    activation=hp.Choice("activation", ["relu", "tanh", "elu", "selu", "swish"]),
                ))
                model.add(keras.layers.Dropout(
                    rate=hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.05)
                ))
            model.add(keras.layers.Dense(1, activation="linear"))
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=hp.Choice("lr", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
                ),
                loss="mse",
            )
            return model

        tuner = kt.BayesianOptimization(
            build_model,
            objective="val_loss",
            max_trials=cfg["max_trials"],
            overwrite=True,
            directory="tuner_dir",
            project_name=f"mlp_{self.seed}",
        )
        tuner.search(
            X_train, y_train,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=[keras.callbacks.EarlyStopping(
                patience=cfg["patience"], restore_best_weights=True
            )],
            verbose=0,
        )
        self.model = tuner.get_best_models(1)[0]
        self.best_hp = tuner.get_best_hyperparameters(1)[0].values
        self.is_fitted = True
        return self.best_hp

    def fit_fixed(self, X_train, y_train, X_val, y_val, hp_dict, **kwargs):
        """Train with a fixed architecture (no hyperparameter search)."""
        import tensorflow as tf
        from tensorflow import keras

        tf.keras.backend.clear_session()
        self._set_seeds()
        n_features = X_train.shape[1]
        cfg = self.config

        model = self._build_from_hp(hp_dict, n_features)
        model.fit(
            X_train, y_train,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=[keras.callbacks.EarlyStopping(
                patience=cfg["patience"], restore_best_weights=True
            )],
            verbose=0,
        )
        self.model = model
        self.best_hp = hp_dict
        self.is_fitted = True
        return hp_dict

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()