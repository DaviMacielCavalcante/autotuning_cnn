# Aprendizado da Sessão

**Autor:** Davi Cavalcante
**Data:** 2026-05-27
**Projeto:** autotuning_cnn

---

## 1. Ambiente Python com `uv`

- Criação de venv com versão específica do Python: `uv venv --python 3.11`.
- O `uv` baixa a versão automaticamente caso não esteja instalada (a menos que `python-downloads = "never"` esteja configurado).
- Listar versões disponíveis: `uv python list`.
- Adicionar dependência: `uv add <pacote>`; remover: `uv remove <pacote>`.

## 2. Datasets do Kaggle

- O `kagglehub` permite carregar datasets diretamente (`kagglehub.dataset_load(...)`), mas faz cache em disco por trás.
- Para treino de CNN, "tudo em memória" raramente vale a pena — múltiplas épocas reusam os mesmos dados, então cache em disco é o ideal.
- Stream puro via API do Kaggle existe, mas é mais lento e raramente justificável.

## 3. Conceitos fundamentais sobre datasets de imagem

- Imagem virá tensor `(altura, largura, canais)` — modelo nunca vê o JPG/PNG, só números.
- Convenção padrão: **nome da pasta = rótulo da classe**.
- Imagens não cabem em RAM como CSV — usa-se pipeline com **lazy loading** (`tf.data.Dataset`, `DataLoader`).
- **Augmentation** acontece no pipeline em tempo de treino, não em disco.
- Preprocessing comum: resize, normalização (`/255` ou média/desvio ImageNet), batching.

## 4. Relação entre TensorFlow, Keras e sklearn

- **Keras é parte do TensorFlow** (desde TF 2.x): `import tensorflow as tf; tf.keras.Sequential(...)`.
- **sklearn** não treina CNN, mas é útil para:
  - `train_test_split` com `stratify` — split estratificado (mantém proporção das classes em treino/val/teste).
  - `classification_report` e `confusion_matrix` — métricas detalhadas por classe.
- O `image_dataset_from_directory` do TF **não suporta split estratificado**, por isso quando a metodologia exige balanceamento usa-se a abordagem DataFrame + sklearn (como no notebook da professora).

## 5. Instalação de drivers NVIDIA e CUDA

- Driver NVIDIA pelo Ubuntu: `ubuntu-drivers devices` → `sudo ubuntu-drivers autoinstall` → reboot.
- Verificar driver: `nvidia-smi`.
- O `nvidia-smi` mostra a **CUDA máxima** suportada pelo driver — instalar libs CUDA mais novas que o driver suporta não funciona.
- Driver 535 → max CUDA 12.2. Driver 575+ → CUDA 12.9.
- TensorFlow precisa das libs CUDA + cuDNN em runtime, **não basta o driver**.
- Instalar TF com CUDA: `uv add "tensorflow[and-cuda]"` (traz os pacotes `nvidia-*` como dependências pip).

## 6. Debug de incompatibilidades GPU/CUDA

- `tf.config.list_physical_devices('GPU')` retorna `[]` quando o TF não consegue carregar as libs CUDA.
- `tf.test.is_built_with_cuda()` apenas confirma que o **binário** tem suporte CUDA, não que a GPU está acessível.
- O parâmetro `soft_device_placement=True` (default) faz **fallback silencioso pra CPU** ao usar `tf.device('/GPU:0')`. Para testar de verdade: `tf.config.set_soft_device_placement(False)`.
- `LD_LIBRARY_PATH` é lido **uma vez** no início do processo Python. Setar via `os.environ["LD_LIBRARY_PATH"]` dentro do Python **não funciona** — o linker já carregou.
- A solução é setar a variável **antes** de iniciar o Python (via `~/.bashrc` ou `.env` do VS Code).
- Lançar o VS Code com `code .` a partir de um terminal que já tem `LD_LIBRARY_PATH` setado faz com que o kernel Jupyter herde a variável.

## 7. Configuração do shell

- `~/.bashrc` é arquivo de configuração, **não executável**. Editar com `nano ~/.bashrc`, não executar.
- `sudo` não deve ser usado para editar `.bashrc` do próprio usuário.
- Após editar: `source ~/.bashrc` aplica no terminal atual; novos terminais já carregam automaticamente.

## 8. Pipeline com `image_dataset_from_directory`

Método principal de carregamento. Parâmetros importantes:

- `directory`: caminho **direto** para a pasta-mãe que contém subpastas com as classes.
- `image_size`: tamanho de redimensionamento padronizado.
- `batch_size`: tamanho do lote (32 é padrão).
- `validation_split` + `subset` + `seed`: para cortar validação de dentro do treino.
- `shuffle=False`: usar para teste/predição (avaliação precisa ser determinística).
- `labels=None`: usar para conjuntos sem rótulo (predição).

**Crucial:** quando se usa `validation_split`, as duas chamadas (treino e validação) precisam da **mesma `seed`** para garantir splits disjuntos — senão há vazamento de dados.

## 9. Inspeção de `tf.data.Dataset`

- Dataset é **iterável**: cada elemento é uma tupla `(imagens, rótulos)`.
- `.take(n)` pega os primeiros `n` batches — útil para debug.
- Unpacking direto: `for imgs, lbls in ds.take(1):`.
- `.shape` mostra o formato do tensor; ex: `(16, 150, 150, 3)` = 16 imagens 150×150 RGB.
- `.numpy()` converte tensor TF em array NumPy para usar `.min()`, `.max()`, etc.
- `.class_names` lista as classes na ordem aprendida.
- `imshow` no matplotlib precisa de `uint8` (0-255) ou `float` (0-1) — converter com `.astype("uint8")` para exibir corretamente.

## 10. Construção de CNN com Keras

Padrão clássico de uma CNN:

```text
Input → Rescaling → [Conv2D → MaxPooling] × N → Flatten → Dense → Dropout → Dense(softmax)
```

- `Rescaling(1./255)` como primeira camada substitui normalização no pipeline.
- `Conv2D(filters, kernel_size, activation="relu")`: extrai features.
- `MaxPooling2D((2,2))`: reduz dimensão espacial pela metade.
- `Flatten`: achata o tensor 3D em 1D para entrar na camada Dense.
- `Dense(units, activation="relu")`: camada totalmente conectada.
- `Dropout(rate)`: zera aleatoriamente parte dos neurônios em treino (anti-overfit).
- `Dense(num_classes, activation="softmax")`: camada de saída com probabilidades.

## 11. Compilação do modelo

```python
model.compile(optimizer=..., loss=..., metrics=[...])
```

- **Optimizer**: `Adam(learning_rate=0.001)` é padrão razoável.
- **Loss**:
  - `sparse_categorical_crossentropy` → labels são inteiros (caso do `image_dataset_from_directory`).
  - `categorical_crossentropy` → labels one-hot.
  - `binary_crossentropy` → 2 classes.
- **Metrics**: `["accuracy"]` para monitoramento.
- Loss é **otimizada** (precisa ser diferenciável); metrics são apenas **monitoradas**.

## 12. Treinamento com `model.fit`

```python
history = model.fit(ds_treino, validation_data=ds_validacao, epochs=10)
```

- O `fit` itera por todo o dataset uma vez por época.
- `validation_data` é avaliada ao fim de cada época — não é usada para treinar, só para medir generalização.
- Retorna um objeto `history` com `history.history["accuracy"]`, `history.history["val_accuracy"]`, etc. — útil para plotar curvas.
- Sinais durante o treino:
  - `accuracy` e `val_accuracy` subindo juntos → aprendizado saudável.
  - `accuracy` sobe mas `val_accuracy` estagna ou cai → **overfitting**.

## 13. Compatibilidade GPU × cuDNN × arquitetura

Esta foi a maior fonte de problemas da sessão. Lições aprendidas:

- **GPUs Pascal (compute capability 6.1)** como GTX 1050 Ti, 1060, 1070 têm regressões conhecidas com **cuDNN 9.x** — alguns algoritmos de convolução backward foram removidos/quebrados.
- O erro típico é `Autotuning failed for HLO ... NOT_FOUND: No valid config found!` durante o `model.fit`.
- TensorFlow 2.18+ usa cuDNN 9; **TF 2.17 é a última versão com cuDNN 8.9**, que é compatível com Pascal.
- O log do TF mostra a versão da GPU e do cuDNN na inicialização — útil para identificar a causa:

  ```text
  StreamExecutor [0]: NVIDIA GeForce GTX 1050 Ti, Compute Capability 6.1
  Loaded cuDNN version 92200
  ```

### Tentativas que não resolveram este caso

- `tf.config.experimental.set_memory_growth(gpu, True)` — alocação de VRAM sob demanda; útil em geral, mas não muda algoritmos cuDNN.
- `TF_XLA_FLAGS=--tf_xla_auto_jit=0` — desativa autojit do XLA, mas o TF ainda usa XLA backend para ops cuDNN.
- `TF_CUDNN_USE_FRONTEND=0` e `TF_CUDNN_USE_AUTOTUNE=0` — força API legacy e desativa autotuner. Pode ajudar em alguns casos, mas não quando os algoritmos foram removidos do binário.

### O que resolve de fato

- **Downgrade do TF para 2.17** (último com cuDNN 8.9): `uv add "tensorflow[and-cuda]==2.17.*"`.
- Ou rodar em **CPU** (forçando com `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` antes do `import tensorflow`).
- Ou usar **hardware mais moderno** (RTX 20+, arquitetura Turing ou superior).

## 14. Wheels do TensorFlow e Python

- O `pip` distribui TF como **wheels binários** específicos por versão de Python (tags `cp39`, `cp310`, `cp311`, `cp312`, etc.).
- TF 2.15 e versões anteriores **não têm wheel para Python 3.12** — só até 3.11.
- Para usar TF antigo + Python novo, é preciso recriar o venv com Python compatível: `uv venv --python 3.11`.
- A versão do Python pinada no venv determina o universo de versões de TF disponíveis.

## 15. Loss vs Acurácia

Duas métricas diferentes para coisas diferentes:

- **Acurácia**: fração de acertos (`acertos / total`). Discreta — ou acertou ou errou.
- **Loss (cross-entropy no caso multiclasse)**: mede o quão errado o modelo está **com peso pela confiança da previsão**. Calcula `-log(p_classe_correta)` por imagem.

Tabela de intuição (probabilidade na classe correta → loss):

| p na classe correta | loss |
|---|---|
| 0.99 (certo e confiante) | 0.01 |
| 0.50 (na dúvida) | 0.69 |
| 0.10 (errou feio) | 2.30 |
| 0.01 (super confiante e errado) | 4.60 |

### Por que existem as duas

- A **loss é o que o otimizador minimiza** — gradient descent precisa de uma função contínua e diferenciável; acurácia é discreta e não tem gradiente útil.
- A **acurácia é só pra humano monitorar** — é a métrica intuitiva.
- Por isso `EarlyStopping` normalmente monitora `val_loss`, não `val_accuracy`: a loss capta degradação **antes** da acurácia (ver seção 16).

## 16. Interpretando curvas de treino

Quatro padrões clássicos ao plotar `accuracy`/`val_accuracy` e `loss`/`val_loss` por época:

| Padrão visual | Diagnóstico |
|---|---|
| Treino e val sobem juntos, gap pequeno | Aprendizado saudável |
| Treino sobe, val estagna baixa (gap grande) | **Overfitting** |
| Ambos baixos e estagnados | **Underfitting** |
| Val loss começa a **subir** enquanto train loss continua descendo | **Overfitting confirmado** |

### Sinal sutil: val_loss subindo com val_accuracy estável

Quando a `val_accuracy` mal mexe mas a `val_loss` sobe ao longo das épocas, o modelo continua acertando a mesma fração de exemplos **mas errando com mais confiança** (dando probabilidade alta para a classe errada).

A acurácia não capta isso porque "errar com 60% de confiança" e "errar com 99% de confiança" contam como 1 erro só. Já a cross-entropy diferencia bastante. Esse é exatamente o tipo de degradação que early stopping com `monitor="val_loss"` consegue interromper a tempo.

## 17. Avaliação no conjunto de teste

- `model.evaluate(df_teste)` retorna `(loss, accuracy)` no dataset informado.
- Não usa para treinar nem para tunar — só ao final, uma vez, pra reportar o desempenho real.
- O conjunto de teste deve ser separado **desde o início** (a Lauda enfatiza isso) — usar o teste pra qualquer decisão de modelo polui a métrica final.

## 18. RTX 4060 Ti e CUDA

- Compute capability 8.9 (arquitetura Ada Lovelace) — sem os problemas de Pascal com cuDNN 9.
- TF 2.18+ + cuDNN 9 roda sem nenhum hack: nem `TF_CUDNN_USE_FRONTEND`, nem `TF_CUDNN_USE_AUTOTUNE`, nem downgrade.
- Velocidade típica observada: ~11ms/step com batch 16 e imagens 150×150 — épocas de ~8s no dataset Intel (14k imagens).
- `tf.config.experimental.set_memory_growth(gpu, True)` continua útil pra evitar que o TF aloque toda a VRAM de cara.

## 19. DataFrame de imagens a partir do filesystem

- `pathlib.Path(diretorio).glob("*/*.jpg")` varre subpastas e devolve os paths das imagens.
- Convenção: o nome da pasta-mãe (`p.parent.name`) é o rótulo da classe.
- Converter `Path` em `str` antes de jogar no DataFrame — o `tf.data` mais à frente espera strings, não objetos `Path`.
- O `glob` é I/O sequencial do Python — polars não acelera essa parte. Pra 14k arquivos dá uns ~100ms, não compensa paralelizar.

## 20. Split estratificado nativo em polars com `.over()`

Alternativa elegante ao `sklearn.train_test_split` quando se está no ecossistema polars. A ideia central: usar **window functions** (`.over(col_classe)`) para aplicar uma expressão independentemente dentro de cada classe.

Padrão usado:

```python
df.with_columns(
    is_val=(
        pl.int_range(pl.len()).shuffle(seed=seed)
        < (pl.len() * frac_val).cast(pl.Int64)
    ).over("class")
)
```

Como funciona:
- `pl.int_range(pl.len())` gera `0..n-1` (tamanho do grupo).
- `.shuffle(seed=seed)` embaralha esses índices.
- `(pl.len() * frac_val).cast(pl.Int64)` calcula o ponto de corte por grupo.
- O `.over("class")` faz toda a expressão ser avaliada **por classe**, então cada classe tem seu próprio shuffle e corte independente.

Vantagens vs sklearn:
- Sem conversão pandas↔polars no meio do pipeline.
- Polars paraleliza naturalmente o cálculo por grupo.
- Lê-se como "shuffle per class, cut at threshold" — direto.

O mesmo padrão serve para sub-amostragem estratificada (passa o `n_total` desejado e calcula `frac = n_total / len(df)`).

## 21. Type hints e o Pylance no VS Code

- Sem type hints na assinatura, o Pylance marca o parâmetro como `Unknown` e **não autocompleta** os métodos do polars dentro da função.
- Conserto: anotar tudo (`df: pl.DataFrame`, `frac_val: float`, `-> tuple[pl.DataFrame, pl.DataFrame]`).
- O Pylance e o **kernel do Jupyter** são processos independentes. O kernel pode estar usando o `.venv` certo (e o código roda), enquanto o Pylance está apontado pra outro Python (sem autocomplete).
- Como diagnosticar: `print(pl.__file__)` numa célula confirma o kernel; barra de status do VS Code (canto inferior direito) mostra o interpretador do Pylance.
- Quando troca de interpretador ou adiciona type hints, às vezes é preciso "Python: Restart Language Server" para o Pylance reanalisar.

---

## Observações da metodologia da professora

- Usar DataFrame + `train_test_split` estratificado dá controle estatístico maior do que `image_dataset_from_directory` puro.
- Splits estratificados são essenciais para classes desbalanceadas.
- Sub-amostragem estratificada serve para acelerar tuning de hiperparâmetros.
- O resultado dela teve diferença grande entre validação (0.378) e teste (0.267) — sinal de overfitting do tuning numa amostra pequena. Possível ponto de investigação no projeto.

---

## Status atual da sessão

- Pipeline de dados (`image_dataset_from_directory`) funcionando.
- Modelo baseline montado, compilado e treinado: 76.5% no teste, com overfitting forte (train 99% vs val 76%).
- Curvas de treino + `evaluate` no teste prontos.
- **Etapa A da pipeline de autotuning iniciada** (preparação dos dados para Optuna):
  - Passo 1: DataFrame `(path, class)` em polars via `pathlib.glob`. ✓
  - Passo 2: split estratificado `df_treino` / `df_validacao` (~11227 / ~2807) com window function `.over("class")`. ✓
  - Passo 3: sub-amostra estratificada `df_treino_tuning` (2997) e `df_validacao_tuning` (597). ✓
  - Passo 4 (pendente): função `criar_dataset(df, shuffle=)` que converte DataFrame em `tf.data.Dataset`.
  - Passo 5 (pendente): validação final de contagens por classe.
- Depois da etapa A: definir `criar_cnn_optuna(trial)`, `objective(trial)` e rodar `study.optimize(n_trials=15)`.
