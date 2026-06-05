"""Funções de preparação de dados: listar imagens, splits estratificados, pipeline tf.data."""

from pathlib import Path

import polars as pl
import tensorflow as tf


def list_images(directory: str) -> pl.DataFrame:
    """Varre <diretorio>/<classe>/*.jpg e devolve DataFrame polars com path e classe."""
    paths = list(Path(directory).glob("*/*.jpg"))
    return pl.DataFrame({
        "path": [str(p) for p in paths],
        "class": [p.parent.name for p in paths],
    })


def stratified_split(
    df: pl.DataFrame,
    frac_val: float = 0.2,
    seed: int = 0,
    col_class: str = "class",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Separa df em treino/validação mantendo proporção de classes."""
    marked_df = df.with_columns(
        is_val=(
            pl.int_range(pl.len()).shuffle(seed=seed)
            < (pl.len() * frac_val).cast(pl.Int64)
        ).over(col_class)
    )
    validation_df = marked_df.filter(pl.col("is_val")).drop("is_val")
    train_df = marked_df.filter(~pl.col("is_val")).drop("is_val")
    return train_df, validation_df


def stratified_subsampling(
    df: pl.DataFrame,
    n_total: int,
    seed: int = 0,
    col_class: str = "class",
) -> pl.DataFrame:
    """Amostra estratificada de aproximadamente n_total linhas."""
    frac = n_total / len(df)
    return (
        df.with_columns(
            is_sample=(
                pl.int_range(pl.len()).shuffle(seed=seed)
                < (pl.len() * frac).cast(pl.Int64)
            ).over(col_class)
        )
        .filter(pl.col("is_sample"))
        .drop("is_sample")
    )


def _make_load_image(img_size: tuple[int, int]):
    """Cria a função de carregamento de imagem fechada sobre img_size."""
    def load_image(path, label):
        img_file = tf.io.read_file(path)
        img_decoded = tf.image.decode_jpeg(img_file, channels=3)
        img_resized = tf.image.resize(img_decoded, img_size)
        return img_resized, label
    return load_image


def create_dataset_for_tf(
    df: pl.DataFrame,
    class_names: list[str],
    shuffle: bool = True,
    img_size: tuple[int, int] = (150, 150),
    batch_size: int = 32,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Converte DataFrame polars (path, class) em tf.data.Dataset pronto para fit.

    Quando `shuffle=True`, **embaralha o DataFrame inteiro antes** de criar
    o `tf.data.Dataset`. Isso é necessário porque o DataFrame de origem fica
    em ordem alfabética por classe (resultado de `Path.glob` + `stratified_split`
    que preservam ordem), e o buffer do `tf.data.shuffle` é pequeno demais
    para misturar grupos contíguos de ~1800 imagens da mesma classe.
    Sem esse pré-shuffle, o modelo vê uma classe de cada vez e sofre
    *catastrophic forgetting* — train_acc sobe mas val_acc trava em ~1/N.
    """
    class_to_index = {name: i for i, name in enumerate(class_names)}

    df_labeled = df.with_columns(
        pl.col("class").replace_strict(class_to_index).alias("label")
    )

    if shuffle:
        df_labeled = df_labeled.sample(fraction=1.0, shuffle=True, seed=seed)

    paths = df_labeled["path"].to_list()
    labels = df_labeled["label"].to_list()

    load_image = _make_load_image(img_size)

    ds = (
        tf.data.Dataset.from_tensor_slices((paths, labels))
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    )

    if shuffle:
        ds = ds.shuffle(1000, reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
