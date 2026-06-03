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
| --- | --- |
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
| --- | --- |
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

## 22. `tf.data.Dataset` é imutável

- Cada método (`.map`, `.shuffle`, `.batch`, `.prefetch`) **devolve um dataset novo**; não muta o original.
- Bug clássico: `if shuffle: ds.shuffle(1000)` é um no-op silencioso — o resultado é descartado. Tem que reatribuir: `ds = ds.shuffle(1000)`.
- O padrão idiomático é encadear tudo em uma expressão (`ds.map(...).shuffle(...).batch(...)`), mas isso conflita com chain condicional. Quando precisar de `if`, quebrar a chain e reatribuir é mais legível que ternários inline.
- Em fluent APIs imutáveis (tf.data, polars, pandas, Spark), ternário no meio de uma chain quase sempre quebra por precedência de operadores.

## 23. Função de carregamento de imagem (`tf.data`)

A função passada ao `.map(...)` roda em **modo grafo** do TF, não em Python normal. Isso significa:

- **Só usar APIs `tf.*`**: `tf.io.read_file`, `tf.image.decode_jpeg`, `tf.image.resize`.
- **Não usar** `PIL`, `open()`, `cv2`, `print` pra debug — quando muito, usar `tf.print`.
- A função recebe **tensores**, não strings Python. Por isso o `path` no `from_tensor_slices((paths, labels))` precisa ser passado como lista de strings (não objetos `pathlib.Path`).

Pipeline correto (`Conv2D` espera tensor 3D float32):

```text
tf.io.read_file(path) → tf.image.decode_jpeg(channels=3) → tf.image.resize(IMG_SIZE)
```

Erros comuns: pular o `decode_jpeg` e tentar fazer `resize` direto nos bytes brutos (não funciona — `resize` espera tensor de imagem, não buffer).

## 24. Batch Normalization

Camada que normaliza ativações intermediárias batch por batch: para cada feature (canal) na saída da camada anterior, calcula média/desvio dentro do batch, normaliza, e aplica dois parâmetros aprendidos `γ` e `β`.

### Problema que resolve

**Internal covariate shift**: durante o treino, conforme os pesos mudam, a distribuição das ativações que chegam em cada camada interna muda também. Cada camada está perseguindo um alvo móvel. BN estabiliza essa distribuição (média 0, desvio 1) — as camadas seguintes não precisam mais se readaptar a cada passo.

### Ganhos práticos

- Permite learning rate maior (sem explodir gradiente).
- Convergência mais rápida — frequentemente metade das épocas necessárias.
- Efeito regularizador leve (média/desvio do batch são amostras ruidosas, atuam como mini-dropout).
- Menos sensível à inicialização dos pesos.

### Posicionamento canônico (CNN)

```python
Conv2D(filtros, kernel)        # sem activation
BatchNormalization()
Activation("relu")
MaxPooling2D((2, 2))
```

**Por que separar a ativação do Conv2D**: BN deve receber a saída **linear** da convolução (antes da não-linearidade). Se `activation="relu"` está dentro do `Conv2D`, a ordem fica `Conv → ReLU → BN`, que é a versão antiga e funciona pior.

### Treino vs inferência

- **Treino**: usa média/desvio do batch atual.
- **Inferência**: usa média móvel das estatísticas vistas durante o treino. Keras cuida automaticamente — mas é o motivo pelo qual `model.predict(...)` pode dar resultado diferente se você reativar `training=True` manualmente.

## 25. BN exige batch suficiente

Como BN calcula estatísticas dentro do batch, **batch pequeno = estatísticas ruidosas = gradientes instáveis**.

| Batch size | BN |
| --- | --- |
| < 16 | Não usar BN puro — preferir LayerNorm ou GroupNorm |
| 16–32 | Funciona mas com ruído |
| ≥ 32 | Confortável |
| ≥ 64 | Ótimo |

Caso observado nesse projeto: baseline rodou com `batch_size=16` (herdado das células iniciais). Sem BN, train_acc chegava em 99% facilmente. Adicionando BN, train_acc travou em ~40% e val_acc em ~49% — bem pior que o baseline sem BN. Subindo `batch_size` pra 32 (valor da tabela da Lauda) destravou o aprendizado.

Lição: **mudar de arquitetura para uma com BN exige revisitar o batch size em paralelo**. As duas escolhas não são independentes.

## 26. Metodologia: comparar baseline e Optuna na mesma arquitetura

A comparação que a Lauda pede ("CNN baseline" vs "CNN otimizada com Optuna") só isola o efeito do tuning **se as duas arquiteturas forem idênticas**. Só os hiperparâmetros mudam.

| Opção | Baseline | Optuna | Compara o quê |
| --- | --- | --- | --- |
| A | Sem BN | Sem BN | Efeito do tuning numa rede simples |
| B | Com BN | Com BN | Efeito do tuning numa rede com BN |
| Inválida | Sem BN | Com BN | Confunde tuning e BN |

Escolha do projeto: **B** — adiciona BN no baseline e no espaço de busca do Optuna. Custo: re-treinar o baseline (~1–2min). Ganho: arquitetura mais sólida, Optuna pode explorar lr mais alto com segurança, e a comparação fica metodologicamente limpa. Vale documentar essa escolha na seção de análise do notebook.

## 27. `LD_LIBRARY_PATH` via `.env` do VS Code

A seção 6 já cobre o porquê de `LD_LIBRARY_PATH` precisar ser setado **antes** do Python iniciar. Solução sustentável quando o ambiente de desenvolvimento é VS Code:

**Criar `.env` na raiz do projeto** com a variável apontando para as libs CUDA do venv. O VS Code Python extension lê esse arquivo automaticamente antes de subir o kernel Jupyter — não precisa lembrar de exportar nada manualmente.

Comando que gera o `.env` correto:

```bash
echo "LD_LIBRARY_PATH=$(find $(pwd)/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib | tr '\n' ':' | sed 's/:$//')" > .env
```

Os pacotes nvidia trazem libs em pastas separadas (`cudnn/lib`, `cublas/lib`, `cuda_runtime/lib`, etc.) — todas precisam estar no path. O `find` resolve isso.

### Aplicar a mudança

`.env` é lido na criação do processo Python — **Reload Window** no VS Code (`Ctrl+Shift+P → Developer: Reload Window`), não basta restart kernel.

### Sintomas que indicam que essa é a causa

- `tf.config.list_physical_devices('GPU')` retorna `[]`.
- `nvidia-smi` mostra a GPU viva e livre.
- `!echo $LD_LIBRARY_PATH` dentro do notebook retorna vazio.
- `ls .venv/lib/python3.12/site-packages/nvidia/` lista as libs (sinal de que `tensorflow[and-cuda]` foi instalado corretamente).

Esse quadro apareceu nessa sessão após uma atualização do sistema (driver subiu pra 595, CUDA pra 13.2) — o venv estava intacto, mas o ambiente perdeu a variável.

### Lembrar de versionar

Adicionar `.env` no `.gitignore` se já não estiver — paths absolutos não fazem sentido pra outras máquinas, e arquivos `.env` normalmente carregam secrets.

## 28. `study.best_value` não é o resultado final

Erro de leitura comum depois de rodar `study.optimize`: olhar pra `study.best_value` e achar que é a métrica de qualidade do modelo otimizado. Não é.

`study.best_value` é o melhor `val_accuracy` atingido **na amostra de tuning** (ex: 3000 imagens), com no máximo as `epochs` configuradas no `objective` (ex: 10), e potencialmente com `EarlyStopping` interrompendo antes. É um número de **comparação relativa entre trials**, não a métrica final do modelo.

### O pipeline completo do tuning

```text
1. Optuna.optimize() na amostra de tuning   → study.best_params
2. Construir novo modelo com best_params
3. Treinar esse modelo no dataset CHEIO       (10–14k imagens, 20 épocas)
4. Avaliar no df_teste (separado desde o início) → métrica final
```

A Lauda exige explicitamente o passo 3:

> "Após o tuning, a melhor configuração encontrada deverá ser treinada novamente usando o maior conjunto de treino viável, mantendo o teste separado."

### Por que o número final tende a ser maior que `best_value`

- Mais dados (4× mais imagens) reduzem variância.
- Mais épocas permitem convergência mais completa.
- O `EarlyStopping` no tuning é agressivo (`patience=3`, max 10 épocas) pra acelerar — no treino final ele tolera mais.

Diferença típica: vencedor no teste fica 5–10 pontos acima de `best_value`.

Sintoma de problema: se `vencedor.evaluate(df_teste)` ≈ `study.best_value`, é sinal de que a arquitetura tem teto baixo (não escala com mais dados) — vale revisitar a estrutura, não os hiperparâmetros.

## 29. Comparação baseline vs Optuna precisa ser na mesma escala

Erro metodológico fácil de cometer: treinar o baseline no dataset cheio (20 épocas) e comparar diretamente com `study.best_value` (3k imagens, 10 épocas com early stopping). Os números **não são comparáveis** — o baseline teve 4× mais dados e o dobro de épocas.

### Duas formas válidas de comparar

**Forma A — Baseline na amostra de tuning (a que a professora usa):**

- Treina baseline em `ds_treino_tuning`/`ds_validacao_tuning` com mesmas épocas e early stopping.
- Compara `melhor_val_acc_baseline` vs `study.best_value`.
- Mostra: "o tuning melhorou esse setup específico?"

**Forma B — Vencedor no dataset cheio vs baseline no dataset cheio:**

- Treina o vencedor (com `best_params`) no dataset cheio.
- Compara `vencedor_test_acc` vs `baseline_test_acc`.
- Mostra: "o tuning produziu um modelo melhor pro problema real?"

Idealmente reporta as duas — uma valida o processo (tuning melhorou as métricas comparáveis), a outra mede o ganho prático.

### Anti-pattern

Comparar "baseline (cheio, 20 ep) vs Optuna best (3k, 10 ep com early stop)" mistura efeito de tuning com efeito de tamanho de dataset e número de épocas — não dá pra atribuir o gap a nada específico.

## 30. Estrutura completa do notebook (segundo a professora)

Roteiro mínimo pra um trabalho como esse, em ordem de execução:

1. **Imports + reprodutibilidade** (`SEED`).
2. **Carregar dataset** + DataFrame `(path, classe)`.
3. **Visualizar amostras** (sanity check).
4. **Split estratificado** treino / validação / teste (teste separado **desde aqui**).
5. **Amostra estratificada** pro tuning.
6. **Pipeline `tf.data`** (`carregar_imagem`, `criar_dataset`).
7. **Baseline** — treinar na amostra de tuning, mesmas épocas que o Optuna usará.
8. **Plot curvas baseline** + registrar `melhor_val_acc_baseline`.
9. **`criar_cnn_optuna(trial)`** — modelo parametrizado.
10. **`objective(trial)`** — fit na amostra + return `max(val_accuracy)`.
11. **`study.optimize(n_trials≥15)`** com `TPESampler`.
12. **Reportar** `study.best_value`, `study.best_params`, `study.trials_dataframe()`.
13. **Visualizar otimização**: `plot_optimization_history`, `plot_param_importances`.
14. **Bar chart** comparando baseline vs Optuna (na amostra).
15. **Construir vencedor** com `best_params`.
16. **Treinar vencedor** no dataset **cheio** (20 épocas).
17. **Plot curvas vencedor** + `evaluate` no `df_teste`.
18. **Matriz de confusão** (`sklearn.metrics.confusion_matrix` + `ConfusionMatrixDisplay`).
19. **Tabela final** comparando: baseline (tuning), Optuna best (tuning), vencedor (teste).
20. **Arquitetura clássica** (item 6 da Lauda) — VGG/ResNet/etc. com transfer learning.

Confirmar que **todas as células executam em ordem do zero** antes de entregar — a Lauda exige reprodutibilidade.

---

## Observações da metodologia da professora

- Usar DataFrame + `train_test_split` estratificado dá controle estatístico maior do que `image_dataset_from_directory` puro.
- Splits estratificados são essenciais para classes desbalanceadas.
- Sub-amostragem estratificada serve para acelerar tuning de hiperparâmetros.
- Baseline da professora é treinado **na mesma amostra de tuning** para comparação direta com Optuna best (ver §29).
- A professora usa BN **só no modelo vencedor** (não no baseline) — é a opção "inválida" da §26 do ponto de vista de isolar efeito do tuning, mas é uma escolha metodológica defensável se você documenta. Neste projeto escolhemos opção B (BN nas duas).
- O resultado dela teve diferença grande entre validação (0.378) e teste (0.267) — sinal de overfitting do tuning numa amostra pequena. Possível ponto de investigação no projeto.

---

## Status atual da sessão

- Pipeline de dados (`image_dataset_from_directory`) funcionando.
- Baseline **original** (sem BN, batch 16): 76.5% no teste, overfitting forte (train 99% vs val 76%, val_loss subindo).
- Baseline **revisado** (com BN, batch 64): travado em ~52% test_acc. Tentativa de retreinar com BN não destravou — o modelo simplesmente não chega na faixa de 99% train que conseguia sem BN. Suspeita: interação BN + XLA + arquitetura simples.
- **Pipeline de autotuning completo até o `study.optimize`**: split estratificado em polars (`.over("class")`), sub-amostra (3000/600), `create_dataset_for_tf`, `create_optuna_cnn(trial)`, `objective(trial)`, study TPE com 15 trials.
- **Resultado Optuna**: `best_value = 0.509` (trial 7), todos os trials entre 0.16 e 0.51. Mesmo sintoma do baseline com BN — arquitetura travada.
- **Problema de GPU resolvido** durante a sessão: depois de uma atualização de driver (595/CUDA 13.2), `tf.config.list_physical_devices('GPU')` voltou `[]` porque o `LD_LIBRARY_PATH` ficou vazio. Resolvido com `.env` na raiz do projeto + Reload Window.
- **Pendências para fechar o trabalho** (itens 12–20 da §30 acima):
  - Imprimir `study.best_params` e `study.trials_dataframe()`.
  - Construir e treinar **modelo vencedor** no dataset cheio.
  - Avaliar no teste + matriz de confusão.
  - Visualizações Optuna (`plot_optimization_history`, `plot_param_importances`).
  - Re-treinar baseline na amostra de tuning para comparação justa (§29).
  - Tabela final + arquitetura clássica (item 6 da Lauda).
- **Decisão pendente**: se o vencedor no teste continuar em ~0.50, voltar para opção A (sem BN nas duas arquiteturas) e refazer Optuna — o baseline sem BN tinha 76% test, então tem teto bem maior.
