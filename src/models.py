"""Definições de arquitetura: baseline, modelo parametrizado pelo Optuna, modelo do vencedor."""

import keras
from keras import Sequential
from keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling,
)
from keras.optimizers import Adam


def _build_cnn(
    num_filtros_1: int,
    num_filtros_2: int,
    dense_units: int,
    dropout_rate: float,
    learning_rate: float,
    num_classes: int,
    img_size: tuple[int, int],
) -> Sequential:
    """Constrói e compila a arquitetura padrão do projeto (4 blocos Conv + GAP).

    Inclui data augmentation (RandomFlip, RandomRotation, RandomZoom) antes
    do Rescaling. As camadas de augmentation só atuam em training=True.
    """
    model = Sequential([
        Input(shape=(img_size[0], img_size[1], 3)),

        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),

        Rescaling(1.0 / 255),

        Conv2D(num_filtros_1, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(num_filtros_2, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        GlobalAveragePooling2D(),
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_baseline(
    num_classes: int,
    img_size: tuple[int, int] = (150, 150),
) -> Sequential:
    """CNN baseline com hiperparâmetros fixos (mesmos do exemplo da Lauda)."""
    return _build_cnn(
        num_filtros_1=32,
        num_filtros_2=64,
        dense_units=128,
        dropout_rate=0.3,
        learning_rate=0.001,
        num_classes=num_classes,
        img_size=img_size,
    )


def create_optuna_cnn(
    trial,
    num_classes: int,
    img_size: tuple[int, int] = (150, 150),
) -> Sequential:
    """CNN parametrizada pelos suggest_* do trial Optuna.

    Espaço de busca restringido em relação à Lauda (Categoria A do STATUS.md):
    - dropout_rate: [0.2, 0.5] (removido 0.1 — instável com Conv profunda).
    - learning_rate: [3e-4, 5e-3] log (removidos extremos: 1e-4 não converge em 15
      épocas, 1e-2 oscila demais).
    """
    return _build_cnn(
        num_filtros_1=trial.suggest_int("num_filtros_1", 16, 64, step=16),
        num_filtros_2=trial.suggest_int("num_filtros_2", 32, 128, step=32),
        dense_units=trial.suggest_int("dense_units", 64, 256, step=64),
        dropout_rate=trial.suggest_float("dropout_rate", 0.2, 0.5, step=0.1),
        learning_rate=trial.suggest_float("learning_rate", 3e-4, 5e-3, log=True),
        num_classes=num_classes,
        img_size=img_size,
    )


def create_cnn_with_params(
    params: dict,
    num_classes: int,
    img_size: tuple[int, int] = (150, 150),
) -> Sequential:
    """CNN construída a partir de um dict de hiperparâmetros (ex: study.best_params)."""
    return _build_cnn(
        num_filtros_1=params["num_filtros_1"],
        num_filtros_2=params["num_filtros_2"],
        dense_units=params["dense_units"],
        dropout_rate=params["dropout_rate"],
        learning_rate=params["learning_rate"],
        num_classes=num_classes,
        img_size=img_size,
    )
