# Uso da IA na Sessão

**Projeto:** autotuning_cnn
**Data:** 2026-05-27
**Modelo utilizado:** Claude Opus 4.7 (via Claude Code no VS Code)

---

## Resumo do papel da IA

A IA foi usada como **assistente didático e parceiro de debug**, não como gerador de código pronto. O usuário explicitamente pediu, em mais de um momento, que a IA explicasse os métodos das bibliotecas em vez de escrever scripts completos — para que o aprendizado fosse efetivo.

---

## Categorias de uso

### 1. Esclarecimento conceitual

A IA respondeu perguntas teóricas sobre:

- Diferença entre `uv venv` e outras formas de gerenciar ambientes Python.
- Como funciona carregamento de datasets do Kaggle (`kagglehub` vs API oficial vs stream).
- Convenções de organização de datasets de imagem (pastas como rótulos).
- Como funciona o pipeline `tf.data.Dataset` (lazy loading, augmentation in-place).
- Relação entre TensorFlow, Keras e sklearn — explicando que Keras está embutido no TF desde 2.x.
- Por que a professora usou sklearn no notebook (split estratificado, `classification_report`).
- Detalhes sobre como `LD_LIBRARY_PATH` é processado pelo linker dinâmico.

### 2. Guia de uso de APIs sem código pronto

A pedido do usuário, a IA forneceu **referências de métodos** com assinaturas e descrições — não scripts completos. Cobriu:

- `image_dataset_from_directory` e seus parâmetros (`validation_split`, `subset`, `seed`, `labels`, `shuffle`).
- Métodos de `tf.data.Dataset` (`.map`, `.cache`, `.shuffle`, `.prefetch`, `.take`).
- Camadas comuns de Keras (`Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `Rescaling`).
- Parâmetros de `model.compile()` (optimizer, loss, metrics).

### 3. Debug interativo

Maior parte da sessão foi diagnóstico de problemas de configuração:

- **Driver NVIDIA não detectado pelo TF** — a IA solicitou progressivamente diagnósticos (`nvidia-smi`, listagem de pacotes pip, output de `import tensorflow`) para isolar a causa.
- Identificação de incompatibilidade entre **driver 535 (max CUDA 12.2) e libs CUDA 12.9** instaladas via pip.
- Após upgrade do driver, identificação do problema secundário: `LD_LIBRARY_PATH` não sendo definido automaticamente.
- Identificação do erro do próprio assistente em sugerir setar `os.environ["LD_LIBRARY_PATH"]` **dentro** do Python (tarde demais — o linker já carregou).
- Diagnóstico de erro de uso do shell (usuário tentou executar `~/.bashrc` em vez de editá-lo).
- **Erro de `Autotuning failed` no `model.fit`** — a IA leu o log do TF, identificou a GPU (`GTX 1050 Ti, Compute Capability 6.1`) e a versão do cuDNN (`92200`), e correlacionou com a incompatibilidade conhecida entre cuDNN 9.x e arquitetura Pascal.
- A IA propôs uma sequência de fixes (memory growth → desativar XLA → forçar API legacy do cuDNN → downgrade do TF). Quando os primeiros falharam, escalou para o downgrade.
- Identificou que **TF 2.15 não tem wheel para Python 3.12**, sugerindo TF 2.17 como alternativa que mantém cuDNN 8.9.
- No final, reconheceu que migrar para hardware mais novo (desktop) era pragmaticamente mais fácil do que continuar lutando contra a incompatibilidade.

### 4. Análise de código existente

A IA leu o notebook da professora (`CNN_com_flowers.ipynb`) e explicou por que ela usou sklearn — destacando o uso de `train_test_split` com `stratify` e o fato de que `image_dataset_from_directory` não suporta split estratificado.

### 5. Correção de erros do usuário

- Apontou que o usuário tinha apontado `image_dataset_from_directory` para o diretório `data/` em vez de `data/seg_train/seg_train/`.
- Esclareceu confusão sobre o primeiro dataset carregado (era o treino completo, sem validação separada).
- Esclareceu equívoco sobre `seed` em datasets de teste (não é necessário quando não há `validation_split` nem `shuffle`).
- Pediu para o usuário não usar `sudo` ao editar `~/.bashrc`, e explicou a diferença entre executar e editar um arquivo de configuração de shell.

### 6. Reconhecimento de limitações e erros próprios

Em dois momentos a IA admitiu erro:

- Sugeriu inicialmente setar `LD_LIBRARY_PATH` via `os.environ` no notebook. Reconheceu o erro técnico quando o fix falhou.
- Falso positivo no teste de GPU: a operação `tf.matmul` rodou silenciosamente em CPU devido a `soft_device_placement=True`, mas a IA inicialmente afirmou que havia rodado na GPU. Corrigiu-se ao notar o `GPUs: []` contraditório.

---

## Estilo de interação

- Idioma: português (a pedido implícito do usuário).
- Tom: técnico, sem rodeios, mas com cuidado de explicar o "porquê" — não só "o quê".
- Ritmo: **um passo de cada vez**, depois de o usuário pedir explicitamente que a IA não despejasse roadmaps inteiros.
- Confirmação após cada etapa: a IA esperava o usuário rodar e reportar antes de avançar.

---

## Output prático gerado

- Comandos de terminal para diagnóstico (`nvidia-smi`, `uv pip list`, `find`).
- Snippets curtos de código TF/Keras com explicação dos parâmetros.
- Configuração de `~/.bashrc` para `LD_LIBRARY_PATH` persistente.
- Inspeção e correção de chamadas a `image_dataset_from_directory`.
- Arquitetura inicial da CNN baseline adaptada ao dataset Intel (6 classes, 150×150).
- Diagnóstico encadeado do erro de cuDNN/Pascal e recomendação de versionamento de TF compatível.

Nenhum arquivo de código foi escrito automaticamente pela IA no projeto — todo o código foi digitado/colado pelo usuário em seu notebook, com a IA atuando apenas como referência e revisor. Os únicos arquivos escritos pela IA foram `APRENDIZADO.md` e `USO_IA.md` (este).

---

## Status final da sessão

A IA conseguiu guiar o usuário até a montagem completa do pipeline e do modelo baseline. O treino (`model.fit`) não rodou devido à incompatibilidade da GPU GTX 1050 Ti com cuDNN 9.x. Após várias tentativas de fix sem sucesso, a decisão pragmática foi migrar a execução para uma máquina desktop com hardware mais moderno — decisão tomada pelo usuário, suportada pela IA.
