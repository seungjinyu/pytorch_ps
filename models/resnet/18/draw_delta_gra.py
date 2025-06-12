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

def plot_and_save_hist(data, label, epoch_name, save_dir):
    if data.size == 0:
        print(f"❌ No data found for {label} at {epoch_name}")
        return
    
    bins = np.linspace(-0.1 , 0.1 , 200)
    plt.hist(data, bins=bins, alpha=0.7)
    plt.title(f"{label} Distribution ({epoch_name})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()

    filename = f"{label.lower()}_{epoch_name}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

def main():
    epoch_root = "epoch_data"
    image_dir = "images/epoch"
    os.makedirs(image_dir, exist_ok=True)

    for dirname in sorted(os.listdir(epoch_root)):
        if not dirname.startswith("epoch_"):
            continue

        epoch_path = os.path.join(epoch_root, dirname)
        if not os.path.isdir(epoch_path):
            continue

        grad_data = load_all_npy_flat(epoch_path, prefix="grad_")
        delta_data = load_all_npy_flat(epoch_path, prefix="delta_")

        plot_and_save_hist(grad_data, "Grad", dirname, image_dir)
        plot_and_save_hist(delta_data, "Delta", dirname, image_dir)

if __name__ == "__main__":
    main()
