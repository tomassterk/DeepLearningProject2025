import matplotlib.pyplot as plt
import os

def plot_train_val_loss(epochs, train, val, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train, label="Train Loss")
    plt.plot(epochs, val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create folder if it doesn't exist
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()