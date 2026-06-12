# autotuning_cnn 🌄

Pipeline de autotuning de uma CNN para classificação de paisagens (Intel Image Classification), com baseline próprio, busca de hiperparâmetros via **Optuna** e comparação contra **EfficientNetB0** (transfer learning).

## Visão geral

Trabalho da disciplina de Inteligência Computacional / Machine Learning (CESUPA, 2026). A pergunta de fundo é simples: **vale a pena afinar hiperparâmetros de uma CNN pequena, ou um extrator pré-treinado já resolve?** Para responder, o repositório implementa três modelos sobre o mesmo split:

1. **Baseline** — CNN com 4 blocos `Conv+Pool`, hiperparâmetros fixos.
2. **Vencedor do Optuna** — mesma arquitetura, hiperparâmetros tunados em 20 trials com `TPESampler` + `MedianPruner`, objetivo `min(val_loss)`.
3. **EfficientNetB0** — transfer learning como feature extractor (item 6 da lauda).

Os três passam pelo **mesmo pipeline de dados** (`tf.data` construído a partir de DataFrames Polars com split estratificado), o que garante comparação maçã-com-maçã. A variância dos modelos do zero foi medida em 3 runs cada.

## Resultados

| Modelo | `test_acc` | Observação |
| --- | --- | --- |
| Baseline (do zero) | **0.8248 ± 0.0363** | 3 runs |
| Vencedor Optuna | **0.8353 ± 0.0064** | 3 runs — ~6× mais estável |
| EfficientNetB0 (feature extraction) | **0.9143** | 1 run |

Leitura curta: o autotuning **não aumentou a acurácia de pico** da CNN pequena, mas reduziu drasticamente o desvio padrão (robustez à inicialização). O grande salto — ~9 pontos — veio do **transfer learning**, não da busca de hiperparâmetros.

Visualizações do estudo Optuna estão em [`figures/`](figures/):

- `optuna_optimization_history.png` — evolução do `val_loss` por trial.
- `optuna_param_importances.png` — importância relativa dos hiperparâmetros.

## Stack

Python 3.12 · TensorFlow 2.21 (+ CUDA) · Keras 3 · Optuna 4 · Polars · scikit-learn · Plotly · uv

## Dataset

[**Intel Image Classification**](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (Kaggle). 6 classes — `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`. ~14k imagens de treino + 3k de teste, 150×150 RGB.

Após baixar e descompactar, a estrutura esperada é:

```
data/
├── seg_train/seg_train/<classe>/*.jpg
└── seg_test/seg_test/<classe>/*.jpg
```

## Setup

### Pré-requisitos

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) para gerenciamento de venv e dependências
- (Opcional, mas recomendado) GPU NVIDIA com cuDNN 9 — testado em RTX 4060 Ti

### Instalação

```bash
git clone https://github.com/<seu-usuario>/autotuning_cnn.git
cd autotuning_cnn
uv sync
```

### Execução

O fluxo completo (baseline → tuning → vencedor → EfficientNetB0 → análise) vive no notebook principal:

```bash
uv run jupyter lab dataset.ipynb
```

Rodar as células em ordem reproduz toda a tabela de resultados acima.

## Arquitetura do código

O notebook orquestra; o código de verdade está modularizado em `src/`:

```
src/
├── data.py        # list_images, stratified_split, stratified_subsampling, create_dataset_for_tf
├── models.py      # build_baseline, create_optuna_cnn, create_cnn_with_params (compartilham _build_cnn)
├── plotting.py    # plotar_curvas
└── tuning.py      # make_objective (factory) + OptunaPruningCallback
```

Baseline, modelo do Optuna e modelo do vencedor são construídos pela **mesma função interna** (`_build_cnn`), o que elimina qualquer risco de drift entre arquiteturas. A CNN tem o seguinte formato:

```
Input(150,150,3)
  → RandomFlip + RandomRotation + RandomZoom   (só em treino)
  → Rescaling(1/255)
  → Conv2D(f1) → MaxPool                       [tunável]
  → Conv2D(f2) → MaxPool                       [tunável]
  → Conv2D(128) → MaxPool                      [fixo]
  → Conv2D(256) → MaxPool                      [fixo]
  → GlobalAveragePooling2D
  → Dense(units) → Dropout(rate) → Dense(6, softmax)
```

### Espaço de busca do Optuna

| Hiperparâmetro | Intervalo | Tipo |
| --- | --- | --- |
| `num_filtros_1` | 16–64, step 16 | `suggest_int` |
| `num_filtros_2` | 32–128, step 32 | `suggest_int` |
| `dense_units` | 64–256, step 64 | `suggest_int` |
| `dropout_rate` | 0.2–0.5, step 0.1 | `suggest_float` |
| `learning_rate` | 3e-4 a 5e-3 (log) | `suggest_float(log=True)` |

20 trials · `TPESampler(seed=69)` · `MedianPruner(n_startup_trials=5, n_warmup_steps=2)` · `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`.

## Documentação interna

- [`PIPELINE.md`](PIPELINE.md) — referência da pipeline de autotuning.
- [`STATUS.md`](STATUS.md) — status atual, histórico de resultados e pendências.
- [`APRENDIZADO.md`](APRENDIZADO.md) — diário de decisões metodológicas, incluindo o diagnóstico do bug de shuffle que colapsava o vencedor (§30i) e a comparação justa com variância N=3 (§31).
- [`USO_IA.md`](USO_IA.md) — declaração de uso de IA generativa no trabalho.
- [`Lauda CNN 2026.pdf`](Lauda%20CNN%202026.pdf) — especificação oficial da disciplina.

## Licença

Licença MIT — ver [LICENSE](LICENSE) para detalhes.

---

**Autor:** Davi Cavalcante — disciplina de IA Computacional, CESUPA, 2026.

*"In my experience, there's no such thing as luck." — Obi-Wan Kenobi. Por isso a gente mede variância em três runs antes de dizer que o tuning melhorou a acurácia.*
