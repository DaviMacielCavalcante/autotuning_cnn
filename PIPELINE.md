# Pipeline de autotuning — referência

Documento de referência sobre o estado atual da pipeline de autotuning e o que falta implementar. Atualizar a cada sessão.

---

## Contexto do projeto

**Disciplina:** Inteligência Computacional / Machine Learning (CESUPA)
**Lauda:** `Lauda CNN 2026.pdf` — define entregas obrigatórias.
**Referência metodológica:** `CNN_com_flowers.ipynb` — notebook da professora; serve como template do fluxo.
**Notebook de trabalho:** `dataset.ipynb`.
**Diário de aprendizado:** `APRENDIZADO.md`.

## Dataset

- **Intel Image Classification** (Kaggle).
- 6 classes: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.
- ~14034 imagens em `data/seg_train/seg_train/<classe>/*.jpg`.
- ~3000 imagens em `data/seg_test/seg_test/<classe>/*.jpg`.
- Imagens 150×150 RGB no pipeline (redimensionadas pelo `image_dataset_from_directory`).

## Ambiente

- Python 3.12, venv via `uv`.
- TF 2.21 com `[and-cuda]`, cuDNN 9.
- GPU: RTX 4060 Ti (compute 8.9, Ada Lovelace) — sem hacks de compatibilidade.
- Polars (não pandas) como DataFrame engine.
- Sklearn instalado, mas usado pouco — splits são em polars nativo.

## Fluxo da pipeline (4 etapas)

```
ETAPA A — Preparar dados                    (em andamento)
   1. DataFrame (path, class)               ✓
   2. Split estratificado treino/validação  ✓
   3. Sub-amostra estratificada (tuning)    ✓
   4. Função criar_dataset → tf.data        ⧖ pendente
   5. Validação de contagens                ⧖ pendente
                ↓
ETAPA B — Autotuning com Optuna             (pendente)
   • criar_cnn_optuna(trial)
   • objective(trial) com early stopping
   • study.optimize(n_trials=15)
                ↓
ETAPA C — Treino final + avaliação          (pendente)
   • Reconstruir modelo com best_params
   • Treinar no dataset COMPLETO (não amostra)
   • evaluate(df_teste) + matriz de confusão
                ↓
ETAPA D — Arquitetura clássica (item 6)     (pendente)
   • VGG/ResNet com transfer learning
```

## Estado atual das variáveis

Já criadas no notebook ao fim da última sessão:

| Variável | Tipo | Tamanho | O que é |
|---|---|---|---|
| `df_train`, `df_validation`, `df_teste` | `tf.data.Dataset` | 11228/2806/3000 | Datasets do baseline (image_dataset_from_directory) |
| `model` | `keras.Sequential` | 10.6M params | Modelo baseline já treinado |
| `history` | History | 20 épocas | Histórico do fit do baseline |
| `df_treino_completo` | `pl.DataFrame` | 14034 | Path + class do `seg_train` |
| `df_teste_completo` | `pl.DataFrame` | 3000 | Path + class do `seg_test` |
| `df_treino` | `pl.DataFrame` | ~11227 | 80% estratificado de `df_treino_completo` |
| `df_validacao` | `pl.DataFrame` | ~2807 | 20% estratificado de `df_treino_completo` |
| `df_treino_tuning` | `pl.DataFrame` | 2997 | Sub-amostra estratificada de `df_treino` |
| `df_validacao_tuning` | `pl.DataFrame` | 597 | Sub-amostra estratificada de `df_validacao` |

Funções já definidas:
- `plotar_curvas(history, titulo)` — gráfico duplo (acurácia + loss).
- `listar_imagens(diretorio) -> pl.DataFrame` — varre filesystem.
- `split_estratificado(df, frac_val, seed, col_classe) -> tuple[pl.DataFrame, pl.DataFrame]`.
- `subamostrar_estratificado(df, n_total, seed, col_classe) -> pl.DataFrame`.

Constantes:
- `DATA_DIR_TRAIN = "data/seg_train/seg_train"`
- `DATA_DIR_TEST = "data/seg_test/seg_test"`
- `IMG_SIZE = (150, 150)`
- `BATCH_SIZE = 32`
- `EPOCHS = 20`
- `SEED = 69`

## Baseline já estabelecido (referência para comparação)

- Arquitetura: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout(0.3) → Dense(6, softmax).
- Optimizer: Adam(lr=0.001), loss: sparse_categorical_crossentropy.
- 20 épocas, sem early stopping, sem augmentation.
- **Resultado**: train_acc 99%, val_acc ~76%, test_acc **76.5%**, val_loss subindo 2.0 → 2.8.
- Overfitting forte — Optuna precisa atacar com dropout maior, lr menor, dense menor.

## Decisões de implementação

- **Polars > pandas**: por consistência com o stack já presente e elegância das window functions para split estratificado. Para a pipeline atual (14k linhas) a diferença de performance é irrelevante; é decisão de DX.
- **Split por window function (`.over`)**: em vez de `sklearn.train_test_split`. Mais idiomático em polars e evita conversões.
- **Teste separado desde o início**: usa-se o `seg_test` que já vem separado; só estratifica treino/validação dentro do `seg_train`.
- **Amostra de 3000+597 pro tuning**: dentro do range 1000–5000 que a Lauda recomenda. Garante ciclo rápido por trial.
- **Modelo vencedor treinado no completo**: amostra é só pra escolher hiperparâmetros.

## Espaço de busca do Optuna (espelha a tabela da Lauda)

| Hiperparâmetro | Intervalo | Tipo |
|---|---|---|
| `num_filtros_1` | 16 a 64, step 16 | `suggest_int` |
| `num_filtros_2` | 32 a 128, step 32 | `suggest_int` |
| `dense_units` | 64 a 256, step 64 | `suggest_int` |
| `dropout_rate` | 0.1 a 0.5, step 0.1 | `suggest_float` |
| `learning_rate` | 1e-4 a 1e-2, log scale | `suggest_float(log=True)` |
| `batch_size` | 32 ou 64 | `suggest_categorical` |
| `epochs` por trial | 10 a 15 | constante |

Fixos: kernel 3×3, ativação ReLU, optimizer Adam, loss sparse_categorical_crossentropy.

Restrições obrigatórias:
- Mínimo **15 trials**.
- **Early stopping** obrigatório (`monitor="val_loss"`, `patience=3`, `restore_best_weights=True`).
- Sampler **TPE** (`optuna.samplers.TPESampler(seed=SEED)`).

## Próxima sessão — passo 4 (criar_dataset)

Esboço da função a implementar:

```python
def criar_dataset(df: pl.DataFrame, shuffle: bool = True) -> tf.data.Dataset:
    # 1. paths e labels (mapear "class" string → int via class_to_idx)
    # 2. ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # 3. ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # 4. if shuffle: ds = ds.shuffle(buffer_size=1000)
    # 5. ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # 6. return ds

def load_image(path, label):
    # tf.io.read_file → tf.image.decode_jpeg(channels=3) → tf.image.resize(IMG_SIZE)
    ...
```

Cuidados:
- `class_to_idx` precisa ser consistente entre `ds_treino_tuning` e `ds_validacao_tuning` (mesma ordem alfabética, igual ao `class_names` do baseline).
- Não aplicar `Rescaling(1./255)` aqui — fica na arquitetura, como no baseline.
- `shuffle ANTES de batch`, sempre.
- `prefetch(tf.data.AUTOTUNE)` no final para pipelining com a GPU.
