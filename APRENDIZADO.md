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

---

## Observações da metodologia da professora

- Usar DataFrame + `train_test_split` estratificado dá controle estatístico maior do que `image_dataset_from_directory` puro.
- Splits estratificados são essenciais para classes desbalanceadas.
- Sub-amostragem estratificada serve para acelerar tuning de hiperparâmetros.
- O resultado dela teve diferença grande entre validação (0.378) e teste (0.267) — sinal de overfitting do tuning numa amostra pequena. Possível ponto de investigação no projeto.

---

## Status final da sessão

- Pipeline de dados (`image_dataset_from_directory`) **funcionando** no notebook.
- Modelo baseline montado e compilado.
- Treino **não rodou** no notebook (GTX 1050 Ti + cuDNN 9 incompatíveis).
- Decisão: migrar projeto para máquina desktop, onde a GPU é mais nova e o treino pode rodar sem essas limitações.
