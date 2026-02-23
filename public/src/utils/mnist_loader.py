"""Load MNIST from kagglehub path (IDX or CSV). Used by experiments and dashboard."""
from pathlib import Path
import numpy as np


def get_mnist_path():
    """Download MNIST via kagglehub; return path to dataset files."""
    import kagglehub
    return kagglehub.dataset_download("hojjatk/mnist-dataset")


def _read_idx_images(path, max_items=None):
    """Read MNIST IDX images (idx3-ubyte). Returns (N, H*W) float32 [0,1]."""
    with open(path, "rb") as f:
        f.read(4)  # magic
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        if max_items is not None:
            n = min(n, max_items)
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    data = data.reshape(n, rows * cols).astype(np.float32) / 255.0
    return np.clip(data, 0.0, 1.0)


def _read_idx_labels(path, max_items=None):
    """Read MNIST IDX labels (idx1-ubyte). Returns (N,) int32."""
    with open(path, "rb") as f:
        f.read(4)  # magic
        n = int.from_bytes(f.read(4), "big")
        if max_items is not None:
            n = min(n, max_items)
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data.astype(np.int32)


def load_mnist_from_path(data_path, max_train=1000, max_test=200):
    """Load MNIST from kagglehub path. Supports IDX or CSV. Returns X_train, y_train, X_test, y_test, n_pixels."""
    data_path = Path(data_path)
    train_img = data_path / "train-images.idx3-ubyte"
    train_lbl = data_path / "train-labels.idx1-ubyte"
    t10k_img = data_path / "t10k-images.idx3-ubyte"
    t10k_lbl = data_path / "t10k-labels.idx1-ubyte"
    if train_img.exists() and train_lbl.exists():
        X_train = _read_idx_images(str(train_img), max_items=max_train)
        y_train = _read_idx_labels(str(train_lbl), max_items=max_train)
        if t10k_img.exists() and t10k_lbl.exists():
            X_test = _read_idx_images(str(t10k_img), max_items=max_test)
            y_test = _read_idx_labels(str(t10k_lbl), max_items=max_test)
        else:
            X_test = X_train[-max_test:]
            y_test = y_train[-max_test:]
        return X_train, y_train, X_test, y_test, X_train.shape[1]
    import pandas as pd
    files = list(data_path.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No IDX or CSV files in {data_path}")
    train_files = [f for f in files if "train" in f.name.lower()]
    test_files = [f for f in files if "test" in f.name.lower()]
    train_csv = train_files[0] if train_files else files[0]
    test_csv = test_files[0] if test_files else (files[1] if len(files) > 1 else files[0])
    train_df = pd.read_csv(train_csv, nrows=max_train)
    test_df = pd.read_csv(test_csv, nrows=max_test) if train_csv != test_csv else train_df.tail(max_test)
    label_col = "label" if "label" in train_df.columns else train_df.columns[0]
    pixel_cols = [c for c in train_df.columns if c != label_col]
    if not pixel_cols:
        pixel_cols = list(train_df.columns[1:])
    n_pixels = len(pixel_cols)
    X_train = np.clip(train_df[pixel_cols].values.astype(np.float32) / 255.0, 0, 1)
    y_train = train_df[label_col].values.astype(np.int32)
    X_test = np.clip(test_df[pixel_cols].values.astype(np.float32) / 255.0, 0, 1)
    y_test = test_df[label_col].values.astype(np.int32)
    return X_train, y_train, X_test, y_test, n_pixels
