import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def load_all_npy_flat(epoch_path, prefix="grad_"):
    arrays = []
    for fname in os.listdir(epoch_path):
        if fname.startswith(prefix) and fname.endswith(".npy"):
            arr = np.load(os.path.join(epoch_path, fname))
            arrays.append(arr.ravel())
    return np.concatenate(arrays) if arrays else np.array([])

def plot_and_save_hist(data, label, epoch_num, save_dir):
    if data.size == 0:
        print(f"❌ No data found for {label} at epoch {epoch_num}")
        return
    
    bins = np.linspace(-0.1 , 0.1 ,200)
    plt.hist(data , bins=bins, alpha=0.7)
    # plt.hist(data, bins=200, alpha=0.7)
    plt.title(f"{label} Distribution (Epoch {epoch_num})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.yscale("log") # log scale
    plt.grid(True)
    plt.tight_layout()

    filename = f"{label.lower()}_epoch_{epoch_num}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def main():
    epoch_root = "epoch_data"
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)

    for epoch_num in range(1, 100):  # epoch 최대값 조절
        epoch_path = os.path.join(epoch_root, f"epoch_{epoch_num}")
        if not os.path.exists(epoch_path):
            continue

        grad_data = load_all_npy_flat(epoch_path, prefix="grad_")
        delta_data = load_all_npy_flat(epoch_path, prefix="delta_")

        plot_and_save_hist(grad_data, "Grad", epoch_num, image_dir)
        plot_and_save_hist(delta_data, "Delta", epoch_num, image_dir)

if __name__ == "__main__":
    main()
