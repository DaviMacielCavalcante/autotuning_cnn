# Status do projeto — autotuning_cnn

**Última atualização:** 2026-06-06 (análise dos trials fechada: viz 7/8 + tabela final 9; PNGs em `figures/`)
**Dataset:** Intel Image Classification — 6 classes (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`)
**Split:** 11230 treino / 2804 validação / 3000 teste

---

## Estado atual

### Arquitetura

```text
Input(150, 150, 3)
  ↓
RandomFlip("horizontal") → RandomRotation(0.1) → RandomZoom(0.1)  (só em treino)
  ↓
Rescaling(1./255)
  ↓
Conv2D(32,  padding="same", relu)  → MaxPool(2,2)   [75 × 75]
Conv2D(64,  padding="same", relu)  → MaxPool(2,2)   [37 × 37]
Conv2D(128, padding="same", relu)  → MaxPool(2,2)   [18 × 18]
Conv2D(256, padding="same", relu)  → MaxPool(2,2)   [ 9 ×  9]
  ↓
GlobalAveragePooling2D()  →  (256,)
  ↓
Dense(128, relu) → Dropout(0.3) → Dense(6, softmax)
```

- **Filtros 1 e 2 tunáveis pelo Optuna**, 3 e 4 fixos (`128`, `256`).
- **Total params:** ~420k.

### Organização do código

Notebook reorganizado em módulo `src/`:

```text
src/
├── __init__.py
├── data.py        # list_images, stratified_split, stratified_subsampling, create_dataset_for_tf
├── models.py      # build_baseline, create_optuna_cnn, create_cnn_with_params (compartilham _build_cnn)
├── plotting.py    # plotar_curvas
└── tuning.py      # make_objective (factory) + OptunaPruningCallback
```

Notebook agora só orquestra — não tem `def`s no corpo. Isso garante que baseline, Optuna e vencedor compartilham construtor por design (eliminando risco de drift entre arquiteturas).

### Mudanças metodológicas aplicadas (referência: APRENDIZADO.md)

| # | Mudança | Referência | Status |
| --- | --- | --- | --- |
| 1 | Removido `BatchNormalization` | §26 | ✅ |
| 2 | `Flatten` → `GlobalAveragePooling2D` | §30c | ✅ |
| 3 | 4 blocos `Conv+Pool` (feature pyramid 32→64→128→256) | §30d | ✅ |
| 4 | Data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`) | §30e | ✅ |
| 5 | `padding="same"` em todas as Conv2D | §30f | ✅ |
| 6 | Restrição do search space (`dropout` `[0.2, 0.5]`, `lr` `[3e-4, 5e-3]`) + epochs 15 + patience 5 | §30g | ✅ |
| 7 | Objetivo Optuna `min(val_loss)` com `direction="minimize"` + `MedianPruner` | §30h | ✅ |
| 8 | **Fix do shuffle do pipeline polars** (causa do colapso do vencedor) | §30i | ✅ |

### Configuração de tuning

- **Amostra de tuning:** 5000 treino / 1000 validação.
- **Trials Optuna:** 20.
- **Sampler:** `TPESampler(seed=69)`.
- **Pruner:** `MedianPruner(n_startup_trials=5, n_warmup_steps=2)`.
- **Objetivo:** `min(val_loss)` (`direction="minimize"`).
- **Early stopping:** `monitor="val_loss"`, `patience=5`, `restore_best_weights=True`.
- **Epochs por trial:** 15.
- **Espaço de busca restringido** (cf. §30g do APRENDIZADO):
  - `num_filtros_1`: `[16, 64]` step 16 (Lauda)
  - `num_filtros_2`: `[32, 128]` step 32 (Lauda)
  - `dense_units`: `[64, 256]` step 64 (Lauda)
  - `dropout_rate`: `[0.2, 0.5]` step 0.1 (**restringido**)
  - `learning_rate`: `[3e-4, 5e-3]` log (**restringido**)

### Histórico de resultados

| Versão | Baseline test | Optuna `best_value` | Vencedor test |
| --- | --- | --- | --- |
| Original (Flatten, sem BN) | 0.7650 | 0.5092 val_acc | 0.3667 |
| Com BN, batch 64 | 0.7520 | 0.6951 val_acc | OOM |
| GAP só (2 Conv) | 0.7393 | 0.4132 val_acc | 0.1750 (colapso) |
| GAP + 4 Conv | 0.74 | 0.41 val_acc | 0.3243 |
| + aug + `padding="same"` | 0.8227 | 0.3591 val_acc | 0.3050 |
| + Categoria A (search space) | 0.8267 | 0.3541 val_acc | 0.3117 |
| + Categoria B (`min(val_loss)` + pruner) | 0.8270 | 1.7008 val_loss | 0.3020 |
| + Fix do shuffle (§30i) | 0.8087¹ | 0.6206 val_loss | 0.8107 |
| **+ Comparação justa (§31) — rodada atual** | **0.8357²** | **0.6206 val_loss** | **0.8107** |

¹ Baseline no pipeline keras (`image_dataset_from_directory`), batch 64, 20 épocas sem early stopping — **não comparável** ao vencedor.
² Baseline no **mesmo pipeline polars** do vencedor, batch 32, mesmo `EPOCHS` + mesmo early stopping. Comparação maçã-com-maçã (§29).

O baseline subiu progressivamente conforme as mudanças arquiteturais (0.76 → 0.82). O vencedor estacionou em ~0.30 em todas as rodadas porque o bug do pipeline (catastrophic forgetting) anulava qualquer melhoria. Com o fix do shuffle aplicado, ambos passaram a aprender de verdade.

**Conclusão consolidada (§31, confirmada por variância N=3):**

| Modelo | test_acc (3 runs) | média ± dp |
| --- | --- | --- |
| Baseline (justo) | 0.8423 / 0.7830 / 0.8490 | **0.8248 ± 0.0363** |
| Vencedor Optuna | 0.8280 / 0.8380 / 0.8400 | **0.8353 ± 0.0064** |

- **Médias estatisticamente indistinguíveis** (gap −0.0106, bem menor que o dp do baseline). O autotuning **não aumentou a acurácia de pico**.
- **O modelo tunado é ~6× mais estável** (dp 0.0064 vs 0.0363). O baseline despencou pra 0.783 numa rodada; o vencedor ficou colado em 0.83–0.84.
- **Lição metodológica:** a comparação de 1 run (que sugeria baseline 0.8357 vs vencedor 0.8107) era enganosa nas duas pontas — pegou um baseline sortudo e um vencedor azarado. Comparar modelos estocásticos com 1 treino cada é não-confiável.
- **Conclusão pra análise crítica:** o ganho do tuning foi **robustez à inicialização**, não pico de acurácia. Resultado honesto e favorável ao processo.

Notas sobre a rodada atual:

- Diagnóstico controlado pós-fix: baseline na pipeline polars em 5 épocas alcança 0.7437 test_acc (vs 0.31 antes do fix).
- Sanity check da cell 27: labels `[2 4 0 0 1]` (mistura aleatória — era `[5 5 5 5 5]`).
- Pruner ativo: 14 de 20 trials podadas pelo `MedianPruner`, orçamento focado em 6 trials completas.
- Best trial: `num_filtros_1=48, num_filtros_2=128, dense_units=64, dropout_rate=0.3, learning_rate=0.0024`.

### Próximo passo

Análise dos trials fechada: visualizações (7/8) e tabela final (9) prontas, PNGs exportados em `figures/`. Faltam pra fechar o trabalho:

- **Item 6 — arquitetura clássica com transfer learning** (VGG/ResNet/EfficientNet). Sugestão: ResNet50 ou EfficientNetB0 em feature extraction (extrator congelado + cabeça nova). Atenção ao `preprocess_input` específico da família (não usar `Rescaling(1/255)`).
- **Item 11 — análise crítica redigida** — insumo forte (robustez §31 + matriz de confusão + importância de hiperparâmetros).
- Slides + declaração de uso de IA generativa.

---

## Possíveis melhorias (não aplicadas)

Estas só fazem sentido se a próxima rodada com o shuffle corrigido ainda mostrar gap entre baseline e vencedor.

### Categoria C — Refinamentos arquiteturais (impacto baixo-médio)

**C1. `SpatialDropout2D(0.1)` entre blocos Conv**
Diferente de `Dropout` (zera pixels), `SpatialDropout2D` zera **canais inteiros** — regularização mais apropriada pra CNN.

**C2. `kernel_regularizer=keras.regularizers.l2(1e-4)` nas Dense**
Penaliza pesos grandes. Reduz overfit no classificador.

### Categoria D — Imagem (impacto baixo)

**D1. Reduzir `IMG_SIZE` para `(128, 128)`**
~30% mais rápido, libera memória, costuma manter performance equivalente em paisagens.

### Categoria E — Tuning mais agressivo (impacto médio)

**E1. Aumentar `N_TRIALS` para 30+**
Com o pruner em vigor, trials ruins são abortadas cedo. Dá pra rodar mais trials no mesmo tempo, explorando o espaço com mais profundidade.

**E2. Usar `HyperbandPruner` em vez de `MedianPruner`**
Pruner mais sofisticado que aloca orçamento em "rounds" — mais agressivo cortando trials medianas.

---

## Pendências para fechar o trabalho (Lauda)

### Itens 5 e 6 da Lauda — análise final

| # | Item | Status |
| --- | --- | --- |
| 1 | Tabela com todos os trials (`study.trials_dataframe()`) | ✅ existe no notebook |
| 2 | Melhor configuração (`study.best_params`) | ✅ existe no notebook |
| 3 | Curvas de treino/validação do baseline e do vencedor | ✅ via `plotar_curvas` |
| 4 | Avaliação final no conjunto de teste (`evaluate`) | ✅ existe no notebook |
| 5 | **Matriz de confusão** | ✅ feito (sklearn `ConfusionMatrixDisplay`) |
| 6 | **`classification_report` por classe** | ✅ feito (sklearn) |
| 7 | **`optuna.visualization.plot_optimization_history(study)`** | ✅ feito (plotly + export PNG em `figures/`) |
| 8 | **`optuna.visualization.plot_param_importances(study)`** | ✅ feito (plotly + export PNG em `figures/`) |
| 9 | **Tabela comparativa final** (baseline / Optuna best / vencedor) | ✅ feito (polars, com média ± dp + best_value) |
| 10 | **Re-treinar baseline para comparação justa** (cf. §29/§31) | ✅ feito + variância N=3 |
| 11 | **Análise crítica** (overfitting, classes mais confundidas, limitações) | ⚠️ insumo forte (§31 robustez + matriz), falta redigir |
| 12 | **Item 6 da Lauda — arquitetura clássica** (VGG/ResNet/EfficientNet com transfer learning) | ❌ a fazer |

### Item 4 da Lauda — entrega

| # | Item | Status |
| --- | --- | --- |
| 1 | Notebook completo, executável em ordem, sem erros | ⚠️ verificar após próxima rodada |
| 2 | Textos markdown intercalados explicando cada etapa | ✅ notas metodológicas (7 itens, incluindo a `min(val_loss)` + pruner) |
| 3 | Seção inicial com nomes da equipe e declaração de uso de IA generativa | ❌ a fazer |
| 4 | Slides para apresentação oral (12 min, todos falam) | ❌ a fazer |

### Limpeza pendente

- O markdown da cell 13 ("o modelo ficou uma merda, hora de preparar a pipe pro tuning") está desatualizado — baseline atual em 0.82.
- O markdown da cell 26 ("testando") pode virar algo mais descritivo tipo "Sanity check do pipeline tf.data".
- A cell 0 importa `pandas` apenas porque `study.trials_dataframe()` retorna `pd.DataFrame`. Justificável.
- O `.env` na raiz precisa estar no `.gitignore` antes de qualquer commit (paths absolutos do venv local).
- A cell 37 foi **repurposada** de diagnóstico para a comparação justa (§31) — não é mais para remover; é a fonte do baseline justo.
- **Novas dependências adicionadas** nesta fase de análise: `scikit-learn` (matriz de confusão + `classification_report`), `plotly` + `kaleido` (visualizações Optuna + export PNG) e `nbformat` (render inline do plotly no Jupyter).
- **Nova pasta `figures/`** com os PNGs exportados (histórico de otimização, importância de hiperparâmetros). Conferir se está no `.gitignore` ou se deve ser versionada.

---

## Referências

- `Lauda CNN 2026.pdf` — especificação oficial do trabalho.
- `CNN_com_flowers.ipynb` — notebook da professora (estrutura de referência).
- `APRENDIZADO.md` — registro detalhado de cada decisão metodológica e lição aprendida (inclui §30c–§30i sobre o diagnóstico do pipeline e §31 sobre a comparação justa + variância).
- `src/` — código modularizado (data, models, plotting, tuning).
