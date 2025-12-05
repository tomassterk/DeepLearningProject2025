import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch
import numpy as np
import os
from collections import Counter
import pyarrow.parquet as pq
import random
import joblib
import hashlib
import json
from pathlib import Path



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# 1) Load dataset
class ShipParquetDataset(Dataset):
    def __init__(self, files, seq_len=300, scaler=None, features=None, debug=False, mode="train"):
        self.files = files
        self.seq_len = seq_len
        self.scaler = scaler
        if features is None:
            raise ValueError("features must be provided")
        self.features = features
        self.debug = debug
        self.mode = mode

        # Ship-Type
        table = pq.read_table(self.files[0])
        df_tmp = table.to_pandas()
        self.ship_type_cols = [c for c in df_tmp.columns if c.startswith("Ship type_") and c != "Ship type_known"]

        # labels for ship types
        self.labels = []
        for file_path in self.files:
            try:
                table = pq.read_table(file_path, columns=self.ship_type_cols)
                df = table.to_pandas()
                label = df[self.ship_type_cols].sum().idxmax()
                self.labels.append(label)
            except:
                self.labels.append(None)

        # calculate how the ship types are distribited in the data set
        class_types = sorted(list(set(self.labels)))  # alle Schiffsarten
        self.type2idx = {t: i for i, t in enumerate(class_types)}
        self.labels = [self.type2idx[label] if label is not None else -1 for label in self.labels]

        label_series = pd.Series(self.labels)
        class_counts = label_series.value_counts()
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

        self.segment_weights = np.array([
            class_weights[label] if label is not None else 1.0
            for label in self.labels
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        table = pq.read_table(file_path)
        df = table.to_pandas()
        df = df.sort_values("Timestamp")
        orig_len = len(df)

        data = df[self.features].to_numpy(dtype=np.float32)

        if self.mode in ["val", "test"]:
            min_len = self.seq_len + 300  # predict 300 min
        else:
            min_len = self.seq_len + 1

        pad_len = max(0, min_len - orig_len)
        real = data[:orig_len].copy()

        if self.scaler is not None:
            scaler_cont, cont_features = self.scaler
            cont_idx = [self.features.index(f) for f in cont_features]
            real[:, cont_idx] = scaler_cont.transform(real[:, cont_idx])

        if pad_len > 0:
            pad = np.zeros((pad_len, data.shape[1]), dtype=np.float32)
            data = np.vstack([real, pad])
        else:
            data = real

        # mask: 1 = real, 0 = padded (length = min_len)
        mask_full = np.ones((min_len,), dtype=np.float32)
        if orig_len < min_len:
            mask_full[orig_len:] = 0.0

        # ensure known-flag columns are zero for padded rows (indices before scaling)
        known_idx = [i for i, f in enumerate(self.features) if f.endswith("_known")]
        if pad_len > 0 and known_idx:
            data[orig_len:, known_idx] = 0.0

        X_seq = data[:self.seq_len]

        # for the training we only need one step ahead
        Y_seq = data[1:(self.seq_len + 1)]

        mask_seq = mask_full[1:self.seq_len + 1].astype(np.float32)
        mask_future = mask_full[self.seq_len:self.seq_len + 300].astype(np.float32)



        if self.mode == "train":
            return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(Y_seq, dtype=torch.float32), torch.tensor(
                mask_seq, dtype=torch.float32)

        # for validation and testing we need the whole prediciton length, so wer start at datapoint seq:len +1 and store the next 300 predicitons
        Y_future = data[self.seq_len: self.seq_len + 300]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(Y_seq, dtype=torch.float32), torch.tensor(
            Y_future, dtype=torch.float32), torch.tensor(mask_seq, dtype=torch.float32), torch.tensor(mask_future,
                                                                                                      dtype=torch.float32)
# 2) collect all ship files & label them
def collect_files(path_parquet):
    all_files = []
    prefixes = ["Date=2024-05"]

    for dp, dirnames, filenames in os.walk(path_parquet, topdown=True):
        if dp == path_parquet:
            dirnames[:] = [d for d in dirnames if any(d.startswith(p) for p in prefixes)]

        if any(part.startswith(tuple(prefixes)) for part in dp.split(os.sep)):
            for f in filenames:
                if f.endswith("_cleaned.parquet"):
                    all_files.append(os.path.join(dp, f))

    return sorted(all_files)


def get_labels_for_files(files):
    labels = []
    ship_type_cols = None

    for fp in files:
        table = pq.read_table(fp)
        df = table.to_pandas()

        if ship_type_cols is None:
            ship_type_cols = [c for c in df.columns if c.startswith("Ship type_") and c != "Ship type_known"]

        label = df[ship_type_cols].sum().idxmax()
        labels.append(label)

    return labels


# 3) Stratified Split to keep the distributions of the classes
def stratified_split(files, labels):
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files, labels, test_size=0.3, stratify=labels, random_state=42
    )

    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


def fit_scaler_from_files(train_files, feature_list):

    cont_features = [f for f in feature_list if not f.endswith("_known")]
    all_data = []
    for f in train_files:
        df = pq.read_table(f).to_pandas()
        data = df[cont_features].to_numpy(dtype=np.float32)
        all_data.append(data)
    all_data = np.vstack(all_data)

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler().fit(all_data)


    return scaler, cont_features


# 4) data Loader for all three subsets
def create_dataloaders(train_files, val_files, test_files, batch_size=50, seq_len=30, scaler=None, features=None):
    if features is None:
        raise ValueError("features must be provided")
    train_dataset = ShipParquetDataset(train_files, seq_len, scaler, features, mode="train")
    val_dataset = ShipParquetDataset(val_files, seq_len, scaler, features, mode="val")
    test_dataset = ShipParquetDataset(test_files, seq_len, scaler, features, mode="test")

    g = torch.Generator()
    g.manual_seed(42)

    # Train Sampler
    train_sampler = WeightedRandomSampler(
        weights=train_dataset.segment_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=g

    )



    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=8,  # ← MAIN SPEEDUP
        pin_memory=True,  # ← GPU transfer faster
        persistent_workers=True,  # ← workers stay alive
        prefetch_factor=4,  # ← load batches ahead of time
        worker_init_fn = lambda worker_id: np.random.seed(42 + worker_id)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )


    return train_loader, val_loader, test_loader


# 5) Create batches
def build_all(path_parquet, batch_size=50, seq_len=300, FEATURES=None, val_files=None, test_files=None,  artifacts_dir="splits",
    save_artifacts=True,):
    if FEATURES is None:
        raise ValueError("features must be provided")
    
    files = collect_files(path_parquet)
    labels = get_labels_for_files(files)

    label_counts = Counter(labels)
    min_samples_per_class = 6  # adjust depending on dataset
    files_filtered = [f for f, l in zip(files, labels) if label_counts[l] >= min_samples_per_class]
    labels_filtered = [l for l in labels if label_counts[l] >= min_samples_per_class]

    if val_files is None or test_files is None:
        (train_files, _), (val_files, _), (test_files, _) = stratified_split(files_filtered, labels_filtered)

    else:
        # Keep only requested val/test that still exist after filtering
        val_set  = set(val_files)
        test_set = set(test_files)

        val_files  = [f for f in files_filtered if f in val_set]
        test_files = [f for f in files_filtered if f in test_set]

        # Everything else becomes train
        held_out = set(val_files) | set(test_files)
        train_files = [f for f in files_filtered if f not in held_out]

        # Optional sanity checks
        if len(val_files) == 0 or len(test_files) == 0:
            raise ValueError("Provided val_files/test_files resulted in empty split (maybe paths don't match filtering).")

    split_text = "\n".join(train_files) + "\n---\n" + "\n".join(val_files) + "\n---\n" + "\n".join(test_files)
    split_id = hashlib.sha1(split_text.encode("utf-8")).hexdigest()[:10]

    artifacts_path = Path(artifacts_dir)

    # scale data
    scaler,cont_features = fit_scaler_from_files(train_files, FEATURES)
    scaler_for_dataset = (scaler, cont_features)

    ################################## NEW ###################################
    
    # --- save artifacts (split + scaler) ---
    if save_artifacts:
        artifacts_path.mkdir(parents=True, exist_ok=True)

        (artifacts_path / f"train_files_{split_id}.txt").write_text("\n".join(train_files) + "\n")
        (artifacts_path / f"val_files_{split_id}.txt").write_text("\n".join(val_files) + "\n")
        (artifacts_path / f"test_files_{split_id}.txt").write_text("\n".join(test_files) + "\n")

        joblib.dump(scaler_for_dataset, artifacts_path / f"scaler_{split_id}.joblib")

        with open(artifacts_path / f"manifest_{split_id}.json", "w") as f:
            json.dump({
                "split_id": split_id,
                "path_parquet": path_parquet,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "features_for_loader": FEATURES,
                "train_files_txt": f"train_files_{split_id}.txt",
                "val_files_txt": f"val_files_{split_id}.txt",
                "test_files_txt": f"test_files_{split_id}.txt",
                "scaler_joblib": f"scaler_{split_id}.joblib",
            }, f, indent=2)

    train_loader, val_loader, test_loader = create_dataloaders(
    train_files, val_files, test_files,
    batch_size=batch_size,
    seq_len=seq_len,
    scaler=scaler_for_dataset,
    features=FEATURES
)

    return train_loader, val_loader, test_loader, val_files, test_files, split_id