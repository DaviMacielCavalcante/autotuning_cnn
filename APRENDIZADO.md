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

Escolha inicial do projeto: **B** — adicionamos BN ao baseline, ao espaço de busca do Optuna e ao modelo vencedor.

**Reversão para A.** Após rodar o pipeline completo com BN, observamos que a camada não estava interagindo bem com a arquitetura escolhida nesse dataset (Intel Image Classification). Voltamos para a opção A nas três funções (`Conv2D(filtros, (3,3), activation="relu")` direto, sem `BatchNormalization()` nem `Activation` separada). O requisito de "1 camada de normalização ou regularização" da Lauda continua atendido pelo `Dropout` no bloco denso.

A nota metodológica fica registrada como markdown no notebook, antes da célula do baseline, pra a professora ver o caminho percorrido.

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

## 30c. Diagnóstico arquitetural: `Flatten → Dense` carrega tudo

Padrão clássico de "CNN ruim" que apareceu nessa sessão. Quando o modelo é:

```text
Conv2D(32) → Pool → Conv2D(64) → Pool → Flatten → Dense(128) → Dense(num_classes)
```

Com input (150, 150, 3) e duas Pool 2×2, a saída do segundo MaxPool é `(36, 36, 64)`. `Flatten` cospe **82944 features** num vetor sem estrutura espacial. A `Dense(128)` em cima tem `82944 × 128 ≈ 10.6M` parâmetros — **99% do modelo inteiro**.

### Por que isso quebra

- **A Conv não precisa aprender features semânticas** — ela só precisa fazer "qualquer coisa", e a Dense gigante decora os 10M pesos pra mapear isso pras classes.
- **Overfitting é certo** quando train_acc passa de 0.90 mas val_acc trava bem abaixo do baseline.
- **OOM na GPU** com batch razoável + muitos trials Optuna: cada modelo segura ~40MB de pesos + ativações grandes. O `clear_session` libera mas a fragmentação acumula entre trials.

### O fix idiomático: `GlobalAveragePooling2D`

Substituir `Flatten` por `GlobalAveragePooling2D()` muda a entrada da Dense de `36 × 36 × 64 = 82944` para **só `64` features** (uma média escalar por feature map).

| Camada | Antes (Flatten) | Depois (GAP) |
| --- | --- | --- |
| Entrada da Dense | 82944 | 64–128 |
| Params da Dense(128) | 10.616.960 | 8.320 |
| Total do modelo | ~10.6M | ~28k |
| Razão | **1300× menor** | |

### Mecânica: o que GAP realmente faz

Considera o tensor que sai do último `MaxPool2D`: shape `(H, W, C)` — por exemplo `(36, 36, 64)`. Isso é um cubo de 64 "fatias" 36×36, onde cada fatia é um **feature map** (quão fortemente o filtro X foi ativado em cada posição).

- **`Flatten`** reorganiza a memória: `(36, 36, 64) → (82944,)`. Sem matemática. Preserva **toda a informação de posição** — você ainda sabe que o pixel (12,7) do filtro 23 está no índice 26591 do vetor.
- **`GlobalAveragePooling2D`** calcula a média espacial de cada feature map: `(36, 36, 64) → (64,)`. Joga fora **toda a informação de posição** — só sobra "no geral, quão presente foi cada filtro na imagem".

A Dense que vem depois recebe muito menos coisa, mas **mais densa em significado**: cada uma das 64 (ou 128) features representa "ativação média de um conceito" — não pixels nem posições.

### Prós

1. **Força a Conv a fazer o trabalho dela.** Sem a Dense gigante pra compensar preguiça, cada filtro precisa aprender uma feature que faça sentido sozinha. A profundidade da rede passa a importar.
2. **Translation invariance grátis.** Uma "árvore" no canto superior ativa o filtro "folhagem" igualzinho a uma árvore no canto inferior — a média não muda. Pra classificar paisagens (Intel: floresta, montanha, mar, rua, prédios, geleira) isso é exatamente o desejado: a categoria depende do *conteúdo*, não da *posição*.
3. **Muito menos parâmetros → menos overfitting.** 10M → 28k no modelo total. Com 11k imagens de treino, isso muda o regime do problema.
4. **Resolve OOM em runs longos.** Activations gigantes no Flatten são liberadas a cada trial mas fragmentam VRAM. GAP elimina o problema.
5. **A Dense vira o que ela é boa.** Dense não tem viés indutivo pra imagem; era inadequada pra raciocínio espacial e adequada pra raciocínio semântico em cima de features já abstraídas. Agora ela só faz a parte boa.
6. **É o padrão das CNNs modernas.** ResNet, Inception, EfficientNet, MobileNet — todas usam GAP. `Flatten + Dense gigante` em classificação é considerado legado.

### Contras / o que se perde

1. **Descarta toda a informação de posição.** Se a tarefa de classificação dependesse de **onde** algo está na imagem, GAP seria ruim. Exemplos:
   - "Tem ônibus à esquerda da imagem?" — GAP destruiria essa distinção.
   - Counting tasks ("quantas pessoas estão na imagem?") — número, não presença/ausência.
   - Tarefas de detecção (`object detection`, `segmentation`) — precisam de posição, e por isso usam arquiteturas diferentes (não terminam com GAP).
2. **Pode "deixar dinheiro na mesa" em datasets onde Dense gigante decora bem.** Em problemas com **muitos** dados (ImageNet de verdade, milhões de imagens), a Dense gigante consegue aprender combinações espaciais sem overfittar, e Flatten pode dar resultado bruto melhor. Em 11k imagens é o oposto.
3. **Mais difícil de aprender com poucos dados E pouca profundidade.** A conv tem que carregar todo o trabalho semântico. Com só 2 blocos Conv+Pool, pode ser que o modelo nem consiga representar conceitos suficientemente abstratos. Esse é o motivo de §30c geralmente vir junto com "adicionar mais blocos Conv" (3º e 4º bloco) — uma coisa habilita a outra.

### Quando escolher Flatten vs GAP

| Cenário | Use |
| --- | --- |
| Classificação onde categoria depende de conteúdo geral | **GAP** |
| Classificação com dataset pequeno (< 50k imagens) | **GAP** |
| Modelo profundo (4+ blocos Conv) com pouca regularização | **GAP** |
| Tarefa que depende de posição (esquerda/direita, contagem) | **Flatten** ou outra arquitetura |
| Pouquíssimas camadas Conv (1–2) **e** dataset enorme | Flatten pode ser melhor |

### Onde mais aparece esse problema

Sempre que você ver, no `model.summary()`, uma camada Dense logo após Flatten com **mais de 1M parâmetros**, é cheiro forte de capacidade ociosa. Vale considerar GAP, especialmente se o modelo está overfittando ou se você está perto do limite de VRAM.

### Decisão deste projeto

Substituir `Flatten` por `GAP` nas três funções (baseline, `create_optuna_cnn`, `create_cnn_with_params`). Justificativa para a análise da Lauda:

- Categoria depende do conteúdo geral da paisagem, não de posição → GAP é adequado.
- Dataset relativamente pequeno (~11k train) → menos params = menos overfit.
- `Flatten + Dense(82944→128)` causou `ResourceExhaustedError` em runs longos do Optuna e degeneração do modelo vencedor (cf. §28, §30b).
- A camada de pooling no fim mantém a estrutura "Conv → Pool" do baseline, então a comparação metodológica (baseline vs Optuna) continua dentro da mesma família arquitetural.

### Resultado observado isoladamente

Aplicar GAP **sem** outras mudanças levou a:

- **Baseline (sem tuning)**: subiu de val_acc ~0.78 (Flatten) para ~0.74 (GAP). Caiu um pouco, mas ainda funciona porque a Dense(128) padrão consegue extrair sinal das 64 features médias.
- **Optuna na amostra**: `best_value` desabou de ~0.73 pra ~0.41. Trials variando entre 0.18 e 0.41 — Optuna não conseguiu achar configuração que rende com tão pouca capacidade convolucional.
- **Vencedor no dataset cheio**: val_acc grudou em 0.179 (random pra 6 classes) ao longo das 20 épocas — colapso total.

**Diagnóstico:** sintoma do "Contras item 3" — GAP com só 2 blocos Conv é raso demais. Com Flatten, a Dense gigante compensava; com GAP, a profundidade da Conv precisa subir.

## 30d. Profundidade suficiente: 3º e 4º blocos Conv

GAP só funciona se a Conv aprender features semânticas. Isso requer **profundidade**: cada bloco `Conv+Pool` reduz a resolução espacial pela metade e dobra (idealmente) o número de filtros — esse é o padrão "feature pyramid" que aparece em qualquer CNN moderna.

### O padrão clássico de pyramid

```text
Bloco 1: Conv(32)  + Pool   →  150 × 150 → 74 × 74
Bloco 2: Conv(64)  + Pool   →   74 × 74  → 36 × 36
Bloco 3: Conv(128) + Pool   →   36 × 36  → 17 × 17
Bloco 4: Conv(256) + Pool   →   17 × 17  → 7 × 7
GlobalAveragePooling2D       →   7 × 7   → (256,)
Dense(128) → Dropout → Dense(num_classes, softmax)
```

Cada bloco aprende features mais abstratas que o anterior:

- Bloco 1: bordas, gradientes de cor, texturas locais.
- Bloco 2: combinações simples (cantos, contrastes, padrões repetitivos).
- Bloco 3: partes de objetos (folhagem, superfícies brilhantes, estruturas verticais).
- Bloco 4: conceitos semânticos (paisagem boscosa, paisagem urbana, paisagem aquosa).

GAP no fim só funciona se o Bloco 4 estiver carregando significado — caso contrário ele só agrega ruído.

### Por que dobrar filtros é regra geral

- Cada Pool reduz pela metade a quantidade de "posições" no mapa, ou seja, a quantidade de informação espacial cai 4×.
- Para compensar, dobramos o número de filtros — a "largura" da representação cresce enquanto a "altura/largura" diminui.
- Resultado: o volume de informação (filtros × spatial) fica relativamente constante por bloco, mas mais abstrato.

### Custo

- ~4× mais params na Conv (32 → 64 → 128 → 256 acumula 388k params na Conv).
- ~3× mais tempo por época (mais multiplicações).
- Ainda assim, modelo total fica em ~420k params — muito menos que os 10.6M do Flatten+Dense original.

### Decisão de manter blocos 3 e 4 fixos

- A Lauda só define ranges pros 2 primeiros (`num_filtros_1`, `num_filtros_2`).
- Adicionar `num_filtros_3` e `num_filtros_4` ao espaço de busca expandiria o problema (5 → 7 dimensões), e com 20 trials o TPE já está no orçamento mínimo.
- Fixando em 128 e 256, mantemos a coerência da pyramid e deixamos o Optuna focado nos hiperparâmetros que a Lauda especifica.
- Justificativa pra análise: "Conforme permitido pela Lauda, ampliamos a arquitetura com 2 blocos convolucionais adicionais (fixos em 128 e 256 filtros, seguindo o padrão feature pyramid). O Optuna continua tunando exatamente os 5 hiperparâmetros especificados pela Lauda."

## 30e. Data augmentation como camadas Keras

Augmentation moderna em Keras é feito com **camadas dentro do modelo**, não com `ImageDataGenerator` externo. Vantagens:

- A augmentation roda na **GPU** junto com o treino (não no CPU), sem overhead.
- O modelo carrega a augmentation como parte da sua definição — `model.save` preserva tudo.
- Comportamento "treino vs inferência" é automático: as camadas só transformam imagens quando `training=True` (durante `model.fit`). Em `evaluate` e `predict` viram identidade (no-op).

### As 3 escolhidas para esse projeto

```python
keras.layers.RandomFlip("horizontal"),
keras.layers.RandomRotation(0.1),
keras.layers.RandomZoom(0.1),
```

| Layer | Efeito | Por que vale para paisagens |
| --- | --- | --- |
| `RandomFlip("horizontal")` | Espelha lado-a-lado com prob 0.5 | Floresta/montanha/mar espelhados continuam sendo a mesma categoria |
| `RandomRotation(0.1)` | Rotaciona ±36° aleatoriamente | Simula pequenas variações de ângulo de captura |
| `RandomZoom(0.1)` | Zoom in/out até ±10% | Simula distância de captura diferente |

### O que foi deixado de fora e por quê

- **`RandomFlip("vertical")`**: paisagem invertida (céu embaixo) não existe no dataset.
- **`RandomBrightness` / `RandomContrast`**: o Intel já tem variação natural alta (manhã/tarde, ensolarado/nublado). Adicionar pode ser ruído sem ganho.
- **`RandomCrop`**: corta partes da imagem. Em paisagens, a categoria pode depender de elementos pequenos (uma trilha urbana = rua); cortar pode mudar o conteúdo significativamente.

### Posicionamento no pipeline

Augmentation entra **antes do `Rescaling`**, no início do Sequential:

```python
Sequential([
    Input(shape=(H, W, 3)),
    keras.layers.RandomFlip(...),
    keras.layers.RandomRotation(...),
    keras.layers.RandomZoom(...),
    Rescaling(1./255),
    Conv2D(...),
    ...
])
```

A ordem importa: aug em pixels uint8 [0, 255] preserva a semântica visual; aug em pixels normalizados pode introduzir artefatos numéricos sutis em alguns casos.

### Efeito esperado

- Train_acc cresce **mais devagar** por época (o modelo vê dados ligeiramente diferentes a cada vez).
- Val_acc cresce **mais rápido relativa ao train** (gap menor → menos overfit).
- Train_acc final fica abaixo do que seria sem aug (não passa de 90–95% tipicamente).
- Custo de tempo: ~5–15% mais lento por época.

### Por que ajuda especificamente o Optuna

Trials com hiperparâmetros instáveis (lr extremo, dropout muito baixo) tendem a se beneficiar de "decorar" amostras específicas. Com augmentation, **as amostras mudam de época pra época**, então decoração não é viável — o TPE acaba descartando essas regiões patológicas naturalmente.

## 30f. `padding="same"` vs `padding="valid"` em Conv2D

### O que mudam

Por default, `Conv2D` usa `padding="valid"`:

- Não adiciona zeros nas bordas.
- A saída é **menor** que a entrada: para kernel 3×3, perde 2 pixels (1 de cada lado).

Com `padding="same"`:

- Adiciona zeros nas bordas pra a saída ter o mesmo tamanho da entrada.
- Para kernel 3×3, adiciona 1 zero de cada lado.

### Impacto acumulado

Em uma rede com 4 blocos Conv 3×3 + Pool 2×2, input 150×150:

**Com `padding="valid"`:**

```text
150 → Conv(3): 148 → Pool: 74
 74 → Conv(3):  72 → Pool: 36
 36 → Conv(3):  34 → Pool: 17
 17 → Conv(3):  15 → Pool: 7
```

**Com `padding="same"`:**

```text
150 → Conv(3): 150 → Pool: 75
 75 → Conv(3):  75 → Pool: 37
 37 → Conv(3):  37 → Pool: 18
 18 → Conv(3):  18 → Pool: 9
```

8 pixels de informação perdidos no `valid`. Pra paisagens, isso retira informação da **borda**, onde tipicamente fica o céu, o horizonte, e o canto das construções — feature relevante.

### Custo do `same`

- Computacionalmente quase igual (zeros são baratos).
- Mantém a relação 1:1 entre entrada e saída de cada Conv, o que facilita raciocinar sobre shapes.

### Quando usar

- **`same`** é o padrão em CNNs modernas (ResNet, EfficientNet) — facilita stack profundo sem ter que ajustar cada camada.
- **`valid`** ainda tem uso quando você quer **forçar redução de dimensão** sem precisar de Pool — útil em alguns designs específicos, mas raro hoje.

Para esse projeto, `same` é a escolha óbvia: mais informação, menos surpresa nas dimensões, custo zero.

## 30g. Restringir o search space do Optuna quando ele converge para regiões patológicas

A Lauda permite "ampliar ou restringir os intervalos sugeridos, justificando a necessidade". Em vez de deixar o TPE explorar regiões que sabidamente quebram, o que é mais produtivo é **cortar fora** essas regiões — o orçamento de trials fica focado em configurações com chance real.

### Sintomas que indicam search space mal calibrado

1. **Configurações ganhadoras na amostra não generalizam para o full data** — sinal típico de overfit na amostra, geralmente associado a regularização baixa (dropout no extremo inferior).
2. **`best_value` fica abaixo do baseline com hiperparâmetros padrão** — Optuna não está explorando direito; o problema pode ser que os ranges incluam regiões onde nada converge bem.
3. **Trials concentrados em uma "região" do espaço** — TPE convergiu para um ponto, mas esse ponto é ruim. Restringir tira o ponto ruim do mapa.

### Cortes aplicados neste projeto

**`dropout_rate`: `[0.1, 0.5]` → `[0.2, 0.5]`**

Em todas as rodadas anteriores, o Optuna escolhia dropout=0.1 (mínimo do range) como o "melhor" na amostra. Mecanismo: dropout baixo permite o modelo memorizar 5000 imagens em poucas épocas, dando val_acc alto na amostra. Mas no dataset cheio (11k), o modelo overfitta e val despenca.

Removendo `0.1` do range, forçamos configurações com regularização mínima de `0.2`. Justificativa pra análise: "o extremo inferior do range mostrou-se patológico empiricamente — modelos com dropout 0.1 colapsavam no treino final, não generalizando da amostra para o full data".

**`learning_rate`: `[1e-4, 1e-2]` (log) → `[3e-4, 5e-3]` (log)**

Dois problemas opostos nos extremos:

- **Lr ≈ 1e-4**: convergência muito lenta. Com 15 épocas e early stopping `patience=5`, o modelo mal começa a aprender antes do TPE encerrar o trial. Reportava val_acc baixo não porque a configuração era ruim, mas porque não houve tempo de treinar.
- **Lr ≈ 1e-2**: oscilação. Modelo "salta" pelo loss landscape e early stopping aborta no ponto mais baixo do ruído, que não reflete a configuração de fato.

Restringir para `[3e-4, 5e-3]` mantém a faixa onde o baseline (lr=0.001) já demonstrou estabilidade — é uma janela ~1.5 décadas em log-scale, ainda significativa para o TPE explorar.

### Ajuste paralelo do regime de treino do trial

Mudanças complementares no `make_objective`:

- `epochs` por trial: `10 → 15` (limite da Lauda).
- `patience` do EarlyStopping: `3 → 5`.

Justificativa: com lrs menos extremos (cortados pelo passo anterior), o modelo precisa de mais épocas pra atingir o pico de val_acc. Patience maior tolera as flutuações naturais sem cortar antes do tempo.

### Como justificar a restrição na análise da Lauda

Não é "trapaça" — a Lauda explicita que a equipe pode ampliar ou restringir. O importante é:

1. **Documentar** que os ranges foram restringidos.
2. **Justificar** com base em observação empírica (não em palpite).
3. **Citar a regra da Lauda** que permite isso.

Texto sugerido para a análise:

> "Após observar que rodadas anteriores do Optuna convergiam consistentemente para `dropout_rate=0.1` (mínimo do range sugerido) com colapso do vencedor no full data, e que learning rates próximos aos extremos (1e-4 e 1e-2) não convergiam dentro do orçamento de épocas, restringimos os ranges para `dropout_rate ∈ [0.2, 0.5]` e `learning_rate ∈ [3e-4, 5e-3]`. Conforme permitido pela Lauda, justificamos a necessidade com base na instabilidade empírica observada nos extremos originais."

### Quando NÃO restringir

- Se o `best_value` está próximo do baseline e os trials estão dispersos pelo espaço, deixe o TPE explorar.
- Se apenas 1 ou 2 trials caíram em regiões ruins, é normal — o TPE aprende e descarta. Restringir prematuramente reduz a chance de descobertas inesperadas.

Restringir é solução para padrões **consistentes** em múltiplas rodadas, não para uma única observação ruim.

## 30h. Trocar o objetivo do Optuna para `min(val_loss)` + adicionar pruner

A escolha da métrica do `objective` afeta diretamente que tipo de configuração o TPE prefere. Em rodadas anteriores com `return max(history.history["val_accuracy"])` observamos que:

1. **Acurácia é discreta** (acertou/errou). Uma única época sortuda com val_acc inflado por ruído amostral inflava o `study.best_value` da trial inteira.
2. **TPE acumulava trials sortudos**, escolhendo regiões do espaço onde o pico isolado era alto mesmo sem o trial treinar bem em geral.
3. **Vencedor escalado pro full data colapsava** — as configurações "sortudas" na amostra não tinham fundamento real.

### A mudança: `min(val_loss)` com `direction="minimize"`

```python
# antes
return max(history.history["val_accuracy"])

# depois
return min(history.history["val_loss"])
```

E no `create_study`:

```python
# antes
study = create_study(direction="maximize", ...)

# depois
study = create_study(direction="minimize", ...)
```

### Por que loss é melhor para selecionar trials

- **Suave**: cada amostra contribui com `-log(p_correto)` que varia continuamente conforme as probabilidades mudam. Não há "saltos" de acurácia (de 60% pra 67%, por exemplo).
- **Reflete confiança da previsão**: um modelo certo com 60% de probabilidade e outro certo com 99% têm a mesma acurácia mas losses muito diferentes (`0.51` vs `0.01`). Selecionar pelo segundo é mais informativo.
- **Coerência com o regime de parada**: o `EarlyStopping` já monitora `val_loss`. Ter o `objective` retornando outra coisa criava incoerência — o trial parava por uma razão e era avaliado por outra.

### Adicionando o pruner

Pruners permitem abortar trials ruins no meio do treino, sem desperdiçar épocas. O padrão mais útil é o `MedianPruner`:

```python
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,   # primeiros 5 trials rodam até o fim, sem prune
    n_warmup_steps=2,     # cada trial roda pelo menos 2 épocas antes de poder ser podada
)
study = create_study(direction="minimize", sampler=sampler, pruner=pruner)
```

**Como funciona internamente:** depois de cada época, comparamos a val_loss atual com a mediana das losses das trials anteriores na mesma época. Se estiver acima da mediana (pior), o trial é abortado.

Para isso funcionar, o objective precisa **reportar a val_loss por época** e **checar should_prune**. Isso é feito via callback Keras:

```python
class OptunaPruningCallback(keras.callbacks.Callback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return
        self.trial.report(val_loss, epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
```

O `raise optuna.TrialPruned()` é a forma idiomática de sinalizar pro Optuna que o trial foi abortado (não falhou). O Optuna registra isso no histórico e aprende a evitar regiões similares no próximo trial.

### Trade-off do pruner

- **Ganho**: orçamento de tempo se concentra em trials promissores. Em uma run de 20 trials, normalmente 5–8 são podadas, economizando 30–50% do tempo total.
- **Risco**: trials podadas cedo podem ter sido "lentas pra começar mas boas no final". Os `n_warmup_steps=2` mitigam isso.
- **Sintoma de pruner agressivo demais**: muitas trials com `state="PRUNED"` na `study.trials_dataframe()`, especialmente nas primeiras 2 épocas. Aí relaxar `n_warmup_steps=3` ou usar `HyperbandPruner` (que tem schedule mais sofisticado).

### Interpretando `study.best_value` depois da mudança

Antes: `best_value` era val_acc — quanto **maior**, melhor (próximo de 1).
Depois: `best_value` é val_loss — quanto **menor**, melhor (próximo de 0).

Pra reportar de forma comparável na tabela final, vale converter para acurácia rodando uma avaliação no melhor modelo, ou ler a `val_accuracy` da época correspondente no `history`.

### Quando NÃO mudar para min(val_loss)

- Se sua loss não é proporcional ao que você quer otimizar (ex: tarefa onde a métrica é F1 e a loss é cross-entropy multilabel ruidosa).
- Se você tem motivos pra preferir o "pico de performance" sobre a "qualidade média" — competições onde só o melhor número conta, por exemplo.

Pra trabalhos acadêmicos com objetivo de generalização honesta, `min(val_loss)` é quase sempre a escolha melhor.

## 30i. Catastrophic forgetting por shuffle insuficiente em pipeline polars

**O bug que estava causando o colapso do vencedor em todas as iterações arquiteturais.** Foi identificado por experimento controlado (re-treinar o baseline na mesma arquitetura, mas com os dados polars em vez de `image_dataset_from_directory`). Baseline normalmente em test_acc 0.82 caiu para 0.31 — confirmando que **o problema é o pipeline polars, não a arquitetura nem o Optuna**.

### Estratificação ≠ ordem aleatória (a confusão central)

São duas propriedades **independentes** de um split:

| Propriedade | Definição | Estava correta? |
| --- | --- | --- |
| **Estratificação** | Cada classe está no conjunto na proporção que estava no dataset original. | ✅ sempre esteve. `stratified_split` garantiu. |
| **Ordem aleatória das linhas** | A ordem das linhas no DataFrame é aleatória, sem agrupar classes contíguas. | ❌ nunca esteve. `Path.glob` + `.over("class")` preservavam ordem alfabética por classe. |

Você pode ter um dataset perfeitamente estratificado **em ordem ordenada por classe** (foi o nosso caso) ou um perfeitamente estratificado **em ordem aleatória** (o que queríamos). Estratificação só garante proporção, não embaralhamento.

Confundir as duas é o que custou 7 iterações neste projeto: o split parecia "OK porque era estratificado", então procuramos o bug em todos os outros lugares. A ordem das linhas é tão básica que mal a auditamos.

### A cadeia que causa o bug

1. **`Path.glob("*/*.jpg")`** retorna paths em ordem alfabética por subdiretório: primeiro todos os arquivos de `buildings`, depois todos de `forest`, etc. Resultado: o DataFrame `df_train_full` fica em ordem `[buildings × 2191, forest × 2271, glacier × 2404, mountain × 2512, sea × 2274, street × 2382]` — **grupos contíguos por classe**.

2. **`stratified_split`** filtra com `.over("class")` mas **preserva a ordem original** das linhas. O train_df resultante mantém a estrutura por classe: todos os ~1750 buildings, depois ~1817 forest, etc.

3. **`create_dataset_for_tf`** fazia:

   ```python
   ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(load_image, ...)
   if shuffle:
       ds = ds.shuffle(1000)
   ```

   `from_tensor_slices` cria o dataset na ordem do DataFrame — grupos por classe. O `.shuffle(1000)` tem buffer de 1000 amostras, **menor que o tamanho de cada grupo de classe (~1800)**. Resultado: o shuffle só consegue misturar **dentro da classe atual**, não entre classes.

4. **Modelo treina vendo uma classe de cada vez**: ~1800 imagens de buildings em sequência, depois ~1800 de forest, etc. Cada bloco de batches é de uma classe só.

5. **Adam ajusta os pesos para a classe atual**. Como o gradiente vem todo da mesma classe por muitos batches consecutivos, o modelo "esquece" o que aprendeu sobre as classes anteriores — *catastrophic forgetting*.

6. **No fim da época, o modelo só sabe a última classe vista** (street, por ordem alfabética). Validação corre na mesma ordem alfabética; modelo prevê majoritariamente "street" para tudo → val_acc trava em ~0.17–0.33 (random ponderado pela proporção de street + classes próximas).

### Por que `image_dataset_from_directory` funciona

`image_dataset_from_directory(..., shuffle=True)` (default) usa um buffer **proporcional ao dataset** (10k+ pra 14k imagens). Isso é suficiente pra misturar entre classes. O baseline treinado nessa pipeline atinge 0.82 normalmente.

### O fix em `create_dataset_for_tf`

Embaralhar o **DataFrame polars** antes de criar o `tf.data.Dataset`:

```python
if shuffle:
    df_labeled = df_labeled.sample(fraction=1.0, shuffle=True, seed=seed)
```

`df.sample(fraction=1.0)` devolve todas as linhas em ordem aleatória. O `seed` deixa determinístico. Depois disso, manter o `.shuffle(1000)` no tf.data como shuffle entre épocas é OK — não é mais o único shuffle, é só uma camada extra de variação.

### Sintomas que indicam esse bug em outros projetos

- Train_acc sobe normalmente, val_acc trava em ~`1/num_classes` ou em uma fração fixa.
- Val_acc é igual desde a primeira época — não responde a treinamento.
- Val_loss alta (~`-log(1/num_classes) × 2-3`), indicando previsões confiantes mas erradas.
- Modelo treinado com `image_dataset_from_directory` funciona, mas com pipeline customizado falha — diferença está na qualidade do shuffle.
- O sintoma é **mais comum do que parece** em pipelines onde alguém pega dados ordenados por classe (típico de `Path.glob`, `os.listdir`, listagens de banco) e converte direto pra `tf.data` ou `DataLoader` sem shuffle full.

### A lição

`tf.data.Dataset.shuffle(buffer_size)` **só mistura dentro do buffer**. Se os dados estão ordenados por classe e cada classe tem mais elementos que o buffer, o shuffle não resolve. **Sempre embaralhe os dados de origem antes de construir o pipeline tf.data**, especialmente quando vier de fontes que naturalmente agrupam por classe (filesystem, grouping de DataFrame, etc.).

### Custo do diagnóstico

7 iterações arquiteturais (BN, GAP, 4 Conv, augmentation, padding="same", Categoria A, Categoria B) tentando consertar um sintoma que era **um bug no pipeline de dados**. A lição meta: quando uma intervenção repetidamente não funciona e o sintoma persiste idêntico, é sinal forte de que **a causa está em outro lugar** — vale parar de iterar e fazer um experimento controlado.

O experimento que resolveu (treinar baseline no pipeline suspeito) é o padrão "**isolar a variável**" da metodologia científica básica. Custou 1 minuto de execução e eliminou 6 hipóteses erradas em uma única medição.

## 31. Comparação justa: o baseline supera o vencedor do Optuna

**Descoberta da rodada atual, e provavelmente o resultado mais importante do trabalho.** Quando baseline e vencedor são treinados em condições idênticas, o **baseline ganha**: 0.8357 vs 0.8107 no teste. A "vitória" anterior (0.8107 vs 0.8087) era um artefato de comparação.

### O artefato: o titular antigo comparava maçã com laranja

O baseline original (cell 9–10, test_acc 0.8087) e o vencedor (cell 34, test_acc 0.8107) diferiam em **três fatores além dos hiperparâmetros**:

| Fator | Baseline original | Vencedor |
| --- | --- | --- |
| Pipeline de dados | `image_dataset_from_directory` (keras) | polars → tf.data |
| Batch size | 64 | 32 |
| Early stopping | não (20 épocas cheias) | sim (`patience=3`) |
| Test acc | 0.8087 | 0.8107 |

Com três variáveis confundidas, a diferença de 0.002 não dizia nada sobre o tuning. Poderia ser pipeline, batch ou protocolo de parada — não dava pra atribuir ao Optuna.

### A correção (§29 aplicado)

Re-treinar o baseline na **mesma arquitetura `build_baseline`**, no **mesmo pipeline polars**, com o **mesmo `EPOCHS` + mesmo `EarlyStopping(patience=3, restore_best_weights=True)`** do vencedor. Único fator que sobra variando entre os dois modelos: os hiperparâmetros escolhidos pelo Optuna.

Resultado: baseline justo **0.8357** (parou no epoch 15, melhor no 12 com val_loss 0.4482) vs vencedor **0.8107**.

### Por que o tuning não ajudou (a intuição central)

O Optuna foi rodado numa **amostra reduzida** (5000 treino / 1000 validação, decisão de velocidade). Ele otimizou `val_loss` *nesse regime pequeno*. O que escolheu:

| Hiperparâmetro | Baseline | Vencedor Optuna |
| --- | --- | --- |
| `dense_units` | 128 | **64** |
| filtros 1 / 2 | 32 / 64 | 48 / 128 |
| `dropout_rate` | 0.3 | 0.3 |
| `learning_rate` | 0.001 (default) | 0.0024 |

O vencedor tem um **classificador menor** (dense 64 vs 128). Faz sentido no regime de tuning: com 5k imagens, modelo menor generaliza melhor (menos overfit). Mas treinado no dataset cheio (11k), há dado suficiente pra alimentar o classificador maior — e o baseline de dense 128 leva vantagem.

**Lição:** os hiperparâmetros ótimos dependem do tamanho dos dados. Tunar numa subamostra encontra o modelo ótimo *para a subamostra*, que não necessariamente transfere pro regime de dados completo. Há um trade-off explícito entre custo do tuning (subamostra = rápido) e validade do resultado (subamostra = regime diferente do final).

### Caveat de variância — confirmado por N=3 (e reframe a conclusão)

A comparação de 1 treino acima (baseline 0.8357 vs vencedor 0.8107) era frágil: 1 run de cada, sujeito a ruído de init/augmentation/shuffle. Rodamos então o experimento de variância — baseline e vencedor **3 treinos cada**, semente explícita por rodada (`SEED + i`), design pareado (mesma semente pro par na mesma rodada), tudo o mais idêntico.

| | run 1 | run 2 | run 3 | média ± dp |
| --- | --- | --- | --- | --- |
| baseline | 0.8423 | 0.7830 | 0.8490 | **0.8248 ± 0.0363** |
| vencedor | 0.8280 | 0.8380 | 0.8400 | **0.8353 ± 0.0064** |

Gap médio (baseline − vencedor): **−0.0106** (vencedor ligeiramente à frente na média).

**Duas conclusões, ambas diferentes do que o §31-de-1-run sugeria:**

1. **Médias indistinguíveis.** O gap (−0.0106) é 3–4× menor que o desvio do baseline (0.0363). Em acurácia média os dois empatam em ~0.83 — nenhum "vence" o outro de forma confiável. Teste pareado informal (N=3): t ≈ −0.5, longe de significância.
2. **O modelo tunado é ~6× mais estável** (dp 0.0064 vs 0.0363). O baseline despencou pra 0.783 numa rodada (azar de init); o vencedor ficou colado em 0.83–0.84 nas três.

**A comparação de 1 run era enganosa nas duas pontas:** pegou um baseline sortudo (0.8357) *e* um vencedor azarado (0.8107). Daí a importância de medir variância antes de concluir — comparar modelos estocásticos com 1 treino cada é não-confiável. (Ponto metodológico forte pra análise crítica.)

### Conclusão revisada: o ganho do tuning foi robustez, não pico

O autotuning **não aumentou a acurácia de pico** (médias empatadas), mas entregou um modelo **substancialmente mais robusto à inicialização aleatória** — menor risco de um treino ruim. Benefício real e defensável: em produção, quer-se o modelo que entrega 0.83 *consistentemente*, não um que às vezes dá 0.85 e às vezes 0.78. Hipótese pro mecanismo (não comprovada): a combinação `lr=0.0024` + `dense=64` converge pra uma bacia mais consistente. O que está medido é a menor variância.

### Por que isso não invalida o trabalho

A Lauda pede o **processo de autotuning bem executado + análise crítica** — não exige que o tuning vença em acurácia. A história final é até favorável ao tuning (robustez) e rica metodologicamente (a armadilha do 1-run). Mais defensável numa banca que uma melhoria fabricada por comparação injusta.

## 30b. Treino do vencedor precisa replicar o regime de parada do `objective`

O `study.best_value` que o Optuna reporta é o **pico de `val_accuracy`** atingido durante o `model.fit` do trial, **com os pesos do pico restaurados** (porque o `objective` usa `EarlyStopping(restore_best_weights=True)` e retorna `max(history.history["val_accuracy"])`).

Quando o treino final do vencedor é feito **sem early stopping** e por **mais épocas** que o trial, ele segue treinando além do pico, perde os bons pesos, e o `evaluate` mede um modelo já degradado. Não é problema da arquitetura nem dos hiperparâmetros — é o regime de parada que mudou.

### Caso observado nessa sessão

Trial vencedor (lr=0.000388, dropout=0.1, dense=192) reportou `val_acc=0.6951` na amostra de tuning.

Vencedor treinado por 20 épocas **sem callback** no dataset cheio:

| Época | train_acc | val_acc | val_loss |
| --- | --- | --- | --- |
| 1 | 0.77 | ~0.65 (esperado) | ~1.0 |
| 4–6 | ~0.85 | pico ~0.65–0.70 | mínimo |
| 9–20 | 0.85→0.91 | colapsa para 0.33–0.35 | sobe pra 3–5 |

`evaluate` no teste: 0.3517 — bem abaixo do pico que o Optuna tinha medido.

### Regra prática

Qualquer parâmetro de regime de parada usado no `objective` (`epochs`, `callbacks`, `patience`, `restore_best_weights`) precisa ser **replicado ou estendido conservadoramente** no treino final:

- `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`: replicar.
- `epochs` pode ser **maior** no treino final (mais dados → mais tempo até overfit), mas o early stopping garante que não passe do ponto.
- `restore_best_weights=True` é especialmente importante — sem isso, mesmo com early stopping, você fica com os pesos do momento em que parou, não do pico.

### Como diagnosticar quando aparece

Sintomas típicos de "vencedor sem early stopping":

- Train_acc continua subindo até o fim das épocas.
- Val_acc estabiliza em valor bem abaixo do pico ou colapsa cedo.
- Val_loss sobe enquanto train_loss desce (cf. §16).
- `evaluate` no teste fica muito abaixo do `study.best_value`.

Quando todos os quatro aparecem juntos, **não é a arquitetura nem o `best_params` ruim** — é o regime de parada que ficou inconsistente entre `objective` e treino final.

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
- **Histórico do baseline em duas rodadas**:
  - Original (sem BN, batch 16): 76.5% no teste, overfitting forte (train 99% vs val 76%, val_loss subindo).
  - Tentativa com BN, batch 64: resultados inconsistentes em runs separadas (0.515 numa, 0.752 noutra). O melhor trial Optuna nessa arquitetura ficou em 0.631 (amostra), mas o vencedor escalou para 0.367 no dataset cheio — caso clássico de §28 (`best_value` na amostra não escalou).
- **Reversão para opção A confirmada**: BN removido das três funções (baseline, `create_optuna_cnn`, `create_cnn_with_params`). Nota metodológica adicionada como markdown no notebook.
- **Pipeline de autotuning completo até o `study.optimize`**: split estratificado em polars (`.over("class")`), sub-amostra (~3000/~600), `create_dataset_for_tf`, `create_optuna_cnn(trial)`, `objective(trial)` com `EarlyStopping`, study TPE com 15 trials.
- **Problema de GPU resolvido** durante a sessão: depois de uma atualização de driver (595/CUDA 13.2), `tf.config.list_physical_devices('GPU')` voltou `[]` porque o `LD_LIBRARY_PATH` ficou vazio. Resolvido com `.env` na raiz do projeto + Reload Window.
- **Run All com a arquitetura sem BN concluída**: baseline (sem BN, lr=0.001) chegou a `test_acc=0.7473`, similar à versão original. Optuna sem BN encontrou `best_value=0.6951` (trial 12) com lr=0.000388, dropout=0.1, dense=192.
- **Bug identificado no treino do vencedor**: a cell 42 do notebook treinava por 20 épocas **sem `EarlyStopping`**, então o modelo passava do pico medido pelo Optuna e degradava (cf. §30b). Sintoma: vencedor com train_acc 0.91 / val_acc 0.34 / test_acc 0.3517 — bem abaixo do `best_value` 0.6951.
- **Diagnóstico confirmou que o pipeline polars está correto**: o baseline avaliado em `df_validation` polars deu 0.94 (alto devido a sobreposição com seu próprio treino, mas confirma que labels e imagens batem).
- **Rodada com early stopping no vencedor**: test_acc subiu de 0.3517 → 0.4540. Confirmou §30b mas ainda bem abaixo do baseline (0.7740).
- **Amplificação do tuning (opção C)**: amostra 5000/1000, 30 trials. Optuna chegou a `best_value=0.7332` (trial 12), o melhor até então. Mas trial 16+ quebrou com OOM, e o `evaluate` final do vencedor também travou com OOM — sintoma da Dense gigante (10.6M params) saturando VRAM.
- **Diagnóstico arquitetural (§30c)**: identificada a Dense `82944 → 128` como causa raiz tanto do overfitting quanto do OOM. Aplicada mudança 1: `Flatten` → `GlobalAveragePooling2D` nas três funções.
- **Rodada com GAP isolado**: baseline manteve test_acc 0.7393 (sem OOM, sem overfit), mas Optuna travou em `best_value=0.413` e vencedor colapsou pra val_acc 0.179 (random). Confirmou §30c contras item 3.
- **Aplicada mudança 2**: 3º e 4º blocos Conv com 128 e 256 filtros (fixos), pyramid 32→64→128→256. Documentado na §30d.
- **Aplicadas mudanças 4 e 5**: data augmentation (3 layers) + `padding="same"` em todas as Conv. Documentado nas §30e e §30f.
- **Resultado após mudanças 4 e 5**:
  - Baseline test_acc: **0.8227** (vs 0.7393 — ganho de ~9 pontos, augmentation funcionou).
  - Optuna best_value: 0.3591 (caiu de 0.41 — augmentation tornou trials menos sortudos na amostra pequena).
  - Vencedor test_acc: 0.3050 (continua colapsando).
- **Diagnóstico**: arquitetura está sólida (baseline em 0.82); o problema agora é o **espaço de busca do Optuna**. Otimizador continua escolhendo configurações que ganham na amostra de 5000 mas falham no full data.
- **Refactor para módulo `src/`**: funções foram movidas para `src/data.py`, `src/models.py`, `src/plotting.py`, `src/tuning.py`. O notebook agora só orquestra (sem `def`s no corpo). Garante `_build_cnn` único — baseline, Optuna e vencedor compartilham construtor por design, eliminando risco de drift entre arquiteturas (cf. §26).
- **Aplicadas restrições do search space (Categoria A)**, documentadas na §30g:
  - `dropout_rate`: `[0.1, 0.5]` → `[0.2, 0.5]`.
  - `learning_rate`: `[1e-4, 1e-2]` → `[3e-4, 5e-3]`.
  - `epochs` por trial: `10 → 15`. `patience`: `3 → 5`.
- **Resultado da Categoria A**: baseline test_acc 0.8267 (estável), Optuna `best_value` 0.3541 (sem ganho real), vencedor test_acc 0.3117 (continua colapsando — val_acc travada em ~0.33 desde a primeira época).
- **Diagnóstico**: a métrica `max(val_accuracy)` no objective é volátil em amostras pequenas — TPE pode estar premiando trials sortudos com pico de acurácia em uma época específica. A val_loss é mais suave e refletiria melhor a qualidade real da configuração.
- **Aplicada Categoria B**: objetivo mudou para `min(val_loss)` com `direction="minimize"` + adicionado `MedianPruner`. Documentado na §30h.
- **Resultado da Categoria B**: vencedor test_acc 0.3020 — colapso persistente. Categoria B não resolveu o problema do vencedor.
- **Diagnóstico controlado** (re-treinar baseline na pipeline polars): baseline normal em 0.82 caiu para 0.31 → bug confirmado no pipeline polars.
- **Aplicado o fix (§30i)**: `df_labeled.sample(fraction=1.0, shuffle=True, seed=seed)` antes do `from_tensor_slices`.
- **Resultado final** (com fix):
  - Baseline test_acc: **0.8087** (inalterado, baseline sempre usou `image_dataset_from_directory`).
  - Optuna `best_value` (val_loss): **0.6206** (era 1.7+ antes — agora é loss de modelos que aprenderam de verdade).
  - **Vencedor test_acc: 0.8107** (ligeiramente acima do baseline — o Optuna conseguiu melhorar marginalmente).
  - Diagnóstico controlado pós-fix: baseline na pipeline polars em 5 épocas alcança 0.7437 test_acc.
  - Sanity check da cell 27: labels `[2 4 0 0 1]` (mistura aleatória — era `[5 5 5 5 5]` antes).
  - Pruner ativo: 14 de 20 trials podadas pelo `MedianPruner`, orçamento focado em 6 trials completas.
- **Análise final iniciada** (itens 5/6 da Lauda): matriz de confusão e `classification_report` por classe do vencedor gerados (via sklearn, `ConfusionMatrixDisplay` em cima do matplotlib — sem seaborn).
- **Comparação justa executada (§31 / item #10 da Lauda)**: a cell 37 (antes diagnóstico de 5 épocas) foi repurposada — baseline na mesma arquitetura, mesmo pipeline polars, mesmo `EPOCHS` + mesmo early stopping do vencedor. Resultado: **baseline justo 0.8357** (parou no epoch 15, melhor no 12) vs **vencedor 0.8107**. O baseline supera o vencedor numa comparação maçã-com-maçã. O titular antigo (0.8107 vs 0.8087) era artefato de pipeline/batch/protocolo diferentes. Causa provável: tuning na subamostra (5k/1k) escolheu classificador menor (dense 64) que não transfere pro dataset cheio (11k). Detalhe na §31.
- **Encaminhamento escolhido: confirmar variância (opção c).** Experimento N=3 baseline e vencedor (semente por rodada, design pareado). Resultado: **baseline 0.8248 ± 0.0363** vs **vencedor 0.8353 ± 0.0064**. Médias indistinguíveis (gap −0.0106 « dp do baseline), mas **vencedor ~6× mais estável**. A comparação de 1 run (§31) era enganosa nas duas pontas. **Conclusão revisada:** o tuning não ganhou em pico, mas entregou robustez à inicialização. Detalhe na §31.
- **Análise dos trials fechada (itens 7/8/9 da Lauda):** visualizações Optuna (`plot_optimization_history`, `plot_param_importances`) via plotly, exportadas como PNG em `figures/` (engine `kaleido`; `nbformat` pro render inline). Tabela comparativa final montada em polars com média ± dp + `best_value`. Dependências adicionadas pelo usuário: `scikit-learn`, `plotly`, `kaleido`, `nbformat`.
- **Pendências restantes:** item 6 (arquitetura clássica com transfer learning), item 11 (análise crítica redigida), slides + declaração de uso de IA.

### Quadro do colapso do vencedor (consistente desde a primeira rodada com GAP)

| Configuração | Baseline test | Optuna best (na amostra) | Vencedor test |
| --- | --- | --- | --- |
| Original (Flatten, sem BN) | 0.7650 | 0.5092 | 0.3667 |
| Flatten + BN | 0.5153 | 0.6951 | OOM |
| GAP só (2 Conv) | 0.7393 | 0.4132 | 0.1750 |
| GAP + 4 Conv | 0.74 | 0.41 | 0.3243 |
| + aug + `padding="same"` | 0.8227 | 0.3591 | 0.3050 |
| + Categoria A (search space) | 0.8267 | 0.3541 | 0.3117 |
| + Categoria B (min val_loss + pruner) | a medir | a medir | 0.3020 |

O padrão é claro: **baseline melhora com cada mudança arquitetural, vencedor estaciona em ~0.30**. Não é um problema do search space, do regime de parada, ou do pruner — é estrutural.

### Hipótese ativa

> **RESOLVIDA (cf. §30i).** A hipótese abaixo estava certa: a diferença entre os pipelines era o problema, especificamente o shuffle insuficiente no pipeline polars (catastrophic forgetting). Mantida como registro histórico do raciocínio que levou ao diagnóstico.

O baseline e o vencedor treinam com pipelines de dados **diferentes**:

- **Baseline**: `image_dataset_from_directory(DATA_DIR_TRAIN, validation_split=0.2)` — split interno do TF.
- **Vencedor**: `create_dataset_for_tf` em cima do split estratificado em polars.

Ambos deveriam ser funcionalmente equivalentes (mesmas imagens-fonte), mas o vencedor **nunca atinge val_acc decente desde a primeira época** — train sobe (0.71+), val grudou em 0.33 do início ao fim. Isso aponta para alguma diferença entre os dois pipelines que só se manifesta em conjunção com a arquitetura atual (aug + GAP).

### Próximo passo crítico: diagnóstico decisivo do pipeline

> **CONCLUÍDO.** O diagnóstico abaixo foi executado, confirmou o bug do shuffle (§30i) e levou ao fix. Registro histórico.

Antes de continuar adicionando complexidade, validar se o pipeline polars está OK **com a arquitetura atual**:

1. **Re-treinar o baseline na `df_train` polars** (mesma arquitetura, mesma config, mas dados polars em vez de `image_dataset_from_directory`). Se baseline cair pra 0.30 também, é o pipeline polars; se mantiver ~0.82, é específico do vencedor/Optuna.
2. **Avaliar o baseline em `df_validation` polars** (sanity check direto).

Sem esse diagnóstico, continuar mexendo no Optuna é tiro no escuro.

### Pendências para fechar o trabalho

- ~~Diagnóstico do pipeline polars~~ ✅ (§30i, bug do shuffle corrigido).
- ~~Matriz de confusão e `classification_report` por classe~~ ✅ (itens 5/6 da Lauda).
- ~~Re-treinar baseline para comparação justa~~ ✅ (§31 — baseline 0.8357 > vencedor 0.8107).
- **Decisão de encaminhamento** (usuário): reportar honestamente / re-rodar Optuna no full data / confirmar variância.
- Visualizações Optuna (`plot_optimization_history`, `plot_param_importances`) — itens 7/8.
- Tabela comparativa final (item #9): baseline justo / Optuna best_value / vencedor.
- Análise crítica (item #11): classes mais confundidas + lição do regime de dados (§31).
- Arquitetura clássica com transfer learning (item 6 da Lauda).
- Declaração de uso de IA + slides (item 4 da Lauda).
