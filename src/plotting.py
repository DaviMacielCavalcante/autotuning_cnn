"""Funções de plotagem para acompanhamento de treino."""

from matplotlib import pyplot as plt


def plotar_curvas(history, titulo: str = "Treino vs Validação") -> None:
    """Plota curvas de acurácia e loss (treino + validação) lado a lado."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, acc, marker="o", label="treino")
    axes[0].plot(epochs_range, val_acc, marker="o", label="validação")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Acurácia")
    axes[0].set_title("Acurácia")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, loss, marker="o", label="treino")
    axes[1].plot(epochs_range, val_loss, marker="o", label="validação")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()
