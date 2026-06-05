"""Função objetivo do Optuna como factory (fecha sobre datasets e config)."""

import gc

import keras
import optuna
from keras.callbacks import Callback, EarlyStopping

from src.models import create_optuna_cnn


class OptunaPruningCallback(Callback):
    """Callback que reporta val_loss a cada época e poda o trial se for ruim.

    Usar junto com um `pruner` no `optuna.create_study`. Tipicamente:
    `optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)`.
    """

    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return
        self.trial.report(val_loss, epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def make_objective(
    df_train_tuning,
    df_validation_tuning,
    num_classes: int,
    img_size: tuple[int, int] = (150, 150),
    epochs: int = 15,
    patience: int = 5,
):
    """Cria uma função objective fechada sobre os datasets de tuning e config.

    Retorna `min(val_loss)` (e o study deve usar `direction="minimize"`).
    Loss é mais suave que acurácia, então trials são avaliados de forma menos
    ruidosa — TPE consegue escolher direções de melhoria com mais confiança.

    Inclui também um `OptunaPruningCallback` que reporta val_loss a cada
    época e poda o trial se ficar abaixo da mediana das trials anteriores.
    """
    def objective(trial):
        keras.backend.clear_session()
        gc.collect()

        model = create_optuna_cnn(
            trial,
            num_classes=num_classes,
            img_size=img_size,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )

        pruning_cb = OptunaPruningCallback(trial)

        history = model.fit(
            df_train_tuning,
            validation_data=df_validation_tuning,
            epochs=epochs,
            callbacks=[early_stopping, pruning_cb],
            verbose=0,
        )

        return min(history.history["val_loss"])

    return objective
