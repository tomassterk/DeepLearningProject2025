# anomaly_detection/anomaly_core.py

import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt 
from lstm_model import *
from sklearn.preprocessing import RobustScaler
from .config import *
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import preprocess_data
import torch.nn as nn

def reimport(scaler_joblib = "scaler_with_mask.joblib", valfiles_txt = "val_files.txt", testfiles_txt = "test_files.txt" ):
    loaded = joblib.load(scaler_joblib)
    if isinstance(loaded, (tuple, list)) and len(loaded) == 2:
        scaler, cont_features = loaded
    else:
        scaler = loaded
        cont_features = [f for f in FEATURES if not f.endswith("_known")]

    scaler_for_dataset = (scaler, cont_features)

    with open(valfiles_txt) as f:
        val_files = [line.strip() for line in f]

    with open(testfiles_txt) as f:
        test_files = [line.strip() for line in f]

    # Reinitialize your dataset and DataLoader using the same parameters as before!
    val_dataset = preprocess_data.ShipParquetDataset(val_files, seq_len=30, scaler=scaler_for_dataset, features=FEATURES, mode="val")
    val_loader_new= DataLoader(val_dataset, batch_size=50, shuffle=False)

    test_dataset = preprocess_data.ShipParquetDataset(test_files, seq_len=30, scaler=scaler_for_dataset, features=FEATURES, mode="val")
    test_loader_new= DataLoader(test_dataset, batch_size=50, shuffle=False)

 
    return val_loader_new, test_loader_new

val_loader, test_loader = reimport(scaler_joblib = "scaler_with_mask.joblib", valfiles_txt = "val_files.txt", testfiles_txt = "test_files.txt" ) 

def load_base_model():
    model = LSTMModel(INPUT_SIZE_BASE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE_BASE, DROPOUT).to(DEVICE)
    state_dict = torch.load(MODEL_BASE_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

base_model = load_base_model()


def load_model_8():
    model = LSTMModel(INPUT_SIZE_8, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE_8, DROPOUT).to(DEVICE)
    state_dict = torch.load(MODEL_8_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model_8_features = load_model_8()

def load_scaler():
    loaded = joblib.load(SCALER_PATH)
    if isinstance(loaded, (tuple, list)) and len(loaded) == 2:
        scaler, _ = loaded
    else:
        scaler = loaded
    return scaler


def inverse_seq_partial(seq_scaled, scaler, scaled_idx=SCALED_IDX):
    """
    seq_scaled: (T, F) tensor or array
    scaler: RobustScaler
    scaled_idx: feature indices that were scaled
    """
    if isinstance(seq_scaled, torch.Tensor):
        arr = seq_scaled.detach().cpu().numpy()
    else:
        arr = np.asarray(seq_scaled)

    sub = arr[:, scaled_idx]                # (T, n_scaled)
    sub_unscaled = scaler.inverse_transform(sub)
    out = arr.copy()
    out[:, scaled_idx] = sub_unscaled
    return out

def rolling_forecast_lstm_batch_with_mask(model, batch_X, steps=300, device=DEVICE):
    model.eval()
    batch_X = batch_X.to(device)
    preds = []

    # global FEATURES has length 8, but the model output might not
    with torch.no_grad():
        input_seq = batch_X.clone()
        for _ in range(steps):
            out = model(input_seq)          # (B, seq_len, F)
            next_point = out[:, -1, :]      # (B, F)
            num_features = next_point.shape[-1]

            # only keep known-flag indices that are < num_features
            known_idx = [
                i for i, f in enumerate(FEATURES)
                if f.endswith("_known") and i < num_features
            ]
            if known_idx:
                next_point[:, known_idx] = 0.0

            preds.append(next_point.cpu().numpy())
            next_point_expanded = next_point.unsqueeze(1)  # (B,1,F)
            input_seq = torch.cat(
                [input_seq[:, 1:, :], next_point_expanded], dim=1
            )
    return np.stack(preds, axis=1)  # (B, steps, F)

def calculate_threshold(val_loader, model2, steps):
    mse_lat_sum = []
    mse_lon_sum = []

    for batch in val_loader:
        batch_X_val, batch_y_val, batch_future_val,mask_seq_val, mask_future = batch   
        y_pred = rolling_forecast_lstm_batch_with_mask(model2, batch_X_val, steps=steps, device=DEVICE)  # (B, steps, F)

        y_true = batch_future_val[:, :steps, :2].numpy()        # (B, steps, 2)
        mask_f = mask_future[:, :steps].numpy()             # (B, steps)  (1=valid)

        # Differenzen
        diff_lat = (y_pred[:,:,0] - y_true[:,:,0])
        diff_lon = (y_pred[:,:,1] - y_true[:,:,1])

        # MSE pro timestep, nur gültige Einträge einbeziehen
        valid = (mask_f == 1).astype(float)                 # (B, steps)
        mse_lat = (diff_lat**2 * valid).sum(axis=0) / np.clip(valid.sum(axis=0), 1.0, None)
        mse_lon = (diff_lon**2 * valid).sum(axis=0) / np.clip(valid.sum(axis=0), 1.0, None)

        mse_lat_sum.append(mse_lat)
        mse_lon_sum.append(mse_lon)

    all_mse_lat = np.mean(np.stack(mse_lat_sum), axis=0)
    all_mse_lon = np.mean(np.stack(mse_lon_sum), axis=0)
    
    return all_mse_lat, all_mse_lon

def find_anomalies(val_loader, test_loader, model, steps, k_sigma=None):
    """
    val_loader: used to estimate mean/std of squared error per timestep
    test_loader: where we actually detect anomalies
    model: LSTM model (base or 8-feature)
    steps: forecast horizon (e.g. 30)
    k_sigma: if None -> use mean only, else mean + k_sigma * std as threshold
    """
    # 1) Estimate error distribution on validation set
    mean_lat, std_lat, mean_lon, std_lon, _ = prediction_error_timestep(
        val_loader, model, steps
    )

    if k_sigma is None:
        thr_lat = mean_lat
        thr_lon = mean_lon
    else:
        thr_lat = mean_lat + k_sigma * std_lat
        thr_lon = mean_lon + k_sigma * std_lon

    results = []

    # 2) Scan test set for anomalies
    for batch in test_loader:
        batch_X, _, batch_future, _, mask_future = batch
        B = batch_X.shape[0]

        y_pred = rolling_forecast_lstm_batch_with_mask(
            model, batch_X, steps=steps, device=DEVICE
        )  # (B,steps,F)

        y_true = batch_future[:, :steps, :2].detach().numpy()         # (B,steps,2)
        mask_f = mask_future[:, :steps].detach().numpy().astype(bool)  # (B,steps)

        mse_lat = (y_pred[:, :, 0] - y_true[:, :, 0]) ** 2            # (B,steps)
        mse_lon = (y_pred[:, :, 1] - y_true[:, :, 1]) ** 2

        for b in range(B):
            valid_mask = mask_f[b]  # (steps,)

            anom_lat = (mse_lat[b] > thr_lat) & valid_mask
            anom_lon = (mse_lon[b] > thr_lon) & valid_mask
            anom_any = anom_lat | anom_lon

            results.append({
                "anomaly_mask_lat": anom_lat,
                "anomaly_mask_lon": anom_lon,
                "anomaly_mask_any": anom_any,
            })

    return results


####### TEST / VALIDATION ?? CHATGPT SAYS VALIDATION AND NOT TEST
def prediction_error_timestep(loader, model, steps):
    # We treat "X" as the squared error at timestep t (so X >= 0).
    sum_lat   = np.zeros(steps, dtype=np.float64)
    sum_lon   = np.zeros(steps, dtype=np.float64)
    sumsq_lat = np.zeros(steps, dtype=np.float64)  # sum of X^2 (i.e., SE^2)
    sumsq_lon = np.zeros(steps, dtype=np.float64)
    count     = np.zeros(steps, dtype=np.float64) #Count is the number of ships with a valid datapoint at timestep t

    for batch in loader:
        batch_X, _, batch_future, _, mask_future = batch
        if model.lstm.input_size == 2:
            batch_X_in = batch_X[:, :, :2]
        else:
            # 8‑feature model: use all features
            batch_X_in = batch_X

        y_pred = rolling_forecast_lstm_batch_with_mask(
            model, batch_X_in, steps=steps
        )  # (B, steps, F) numpy

        y_true = batch_future[:, :steps, :2].detach().numpy()          # (B,steps,2)
        valid  = mask_future[:, :steps].detach().numpy().astype(bool) # (B,steps)

        se_lat = (y_pred[:, :, 0] - y_true[:, :, 0]) ** 2  # (B,steps)
        se_lon = (y_pred[:, :, 1] - y_true[:, :, 1]) ** 2

        v = valid.astype(np.float64)
        sum_lat   += (se_lat * v).sum(axis=0)
        sum_lon   += (se_lon * v).sum(axis=0)
        sumsq_lat += ((se_lat ** 2) * v).sum(axis=0)
        sumsq_lon += ((se_lon ** 2) * v).sum(axis=0)
        count     += v.sum(axis=0)

    # mean and std per timestep
    count_safe = np.clip(count, 1.0, None)

    mean_lat = sum_lat / count_safe
    mean_lon = sum_lon / count_safe

    var_lat = (sumsq_lat / count_safe) - mean_lat**2
    var_lon = (sumsq_lon / count_safe) - mean_lon**2

    std_lat = np.sqrt(np.clip(var_lat, 0.0, None))
    std_lon = np.sqrt(np.clip(var_lon, 0.0, None))

    return mean_lat, std_lat, mean_lon, std_lon, count

# SAVE VALUES FROME PREDICTION # 

def plot_route_example(example, scaler, out_path, title_prefix=""):
    """
    example dict keys:
      - hist_scaled: (T_hist, F)
      - future_true_scaled: (T_future, F)
      - future_pred_scaled: (T_future, F)
      - mask_future: (T_future,)
    """

    hist_scaled = example["hist_scaled"]
    fut_true_scaled = example["future_true_scaled"]
    fut_pred_scaled = example["future_pred_scaled"]
    mask_future = example["mask_future"].astype(bool)

    # Inverse scaling
    hist_unscaled = inverse_seq_partial(hist_scaled, scaler, scaled_idx=SCALED_IDX)
    fut_true_unscaled = inverse_seq_partial(fut_true_scaled, scaler, scaled_idx=SCALED_IDX)
    fut_pred_unscaled = inverse_seq_partial(fut_pred_scaled, scaler, scaled_idx=SCALED_IDX)

    # Extract lat/lon
    lat_hist = hist_unscaled[:, LAT_IDX]
    lon_hist = hist_unscaled[:, LON_IDX]

    lat_true = fut_true_unscaled[mask_future, LAT_IDX]
    lon_true = fut_true_unscaled[mask_future, LON_IDX]

    lat_pred = fut_pred_unscaled[mask_future, LAT_IDX]
    lon_pred = fut_pred_unscaled[mask_future, LON_IDX]

    # Plot
    plt.figure(figsize=(6, 5))
    # Use default colors; just give labels
    plt.plot(lon_hist, lat_hist, marker="o", linestyle="-", label="History")
    plt.plot(lon_true, lat_true, marker="o", linestyle="--", label="Future (true)")
    plt.plot(lon_pred, lat_pred, marker="o", linestyle=":", label="Future (pred)")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    title = f"{title_prefix}  batch={example['batch_idx']} sample={example['sample_idx']}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
