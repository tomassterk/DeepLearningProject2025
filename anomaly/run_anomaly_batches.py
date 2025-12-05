# anomaly_detection/run_anomaly_batches.py

import numpy as np
import pandas as pd
from pathlib import Path

from .config import *
from .anomaly_core import *
import torch

STEPS = SEQ_LEN  # 30
K_SIGMA = 3.0    # number of std devs above mean for anomaly threshold

# How many examples to plot
N_ANOM_EXAMPLES = 10
N_NORMAL_EXAMPLES = 10

scaler = load_scaler()

# -------------------------
#  BASE MODEL (2 features)
# -------------------------
OUT_DIR_base = PROJECT_ROOT / "model_results_base" / "anomaly"
OUT_DIR_base.mkdir(parents=True, exist_ok=True)
OUT_TABLE_PATH_base = OUT_DIR_base / f"anomalies_steps_{STEPS}.parquet"
OUT_PLOTS_DIR_base = OUT_DIR_base / f"plots_steps_{STEPS}"
OUT_PLOTS_DIR_base.mkdir(parents=True, exist_ok=True)


def run_anomaly_detection_base():
    # 1) Estimate error distribution on validation set
    print("Computing base-model thresholds from validation set...")
    mean_lat, std_lat, mean_lon, std_lon, _ = prediction_error_timestep(
        val_loader, base_model, steps=STEPS
    )

    mse_lat_thr = mean_lat + K_SIGMA * std_lat
    mse_lon_thr = mean_lon + K_SIGMA * std_lon
    np.savez(
    OUT_DIR_base / "thresholds_base.npz",
    mse_lat_thr=mse_lat_thr,
    mse_lon_thr=mse_lon_thr,
    mean_lat=mean_lat,
    mean_lon=mean_lon,
    std_lat=std_lat,
    std_lon=std_lon,
    K_SIGMA=K_SIGMA,
    STEPS=STEPS,
    )
    print(f"Saved base-model thresholds to {OUT_DIR_base / 'thresholds_base.npz'}")


    all_rows = []
    anomaly_examples = []
    normal_examples = []

    print("Scanning test set for anomalies (base model)...")
    for batch_idx, batch in enumerate(test_loader):
        batch_X_test, batch_y_test, batch_future_test, mask_seq_test, mask_future_test = batch
        B = batch_X_test.shape[0]

        # Base model sees only first 2 features (Lat, Lon)
        batch_X_base = batch_X_test[:, :, :2]

        # 2) Predict future steps
        y_pred = rolling_forecast_lstm_batch_with_mask(
            base_model, batch_X_base, steps=STEPS, device=DEVICE
        )  # (B, STEPS, 2)

        # 3) Ground truth and masks
        y_true = batch_future_test[:, :STEPS, :2].detach().numpy()           # (B, STEPS, 2)
        mask_f = mask_future_test[:, :STEPS].detach().numpy().astype(bool)   # (B, STEPS)

        mse_lat = (y_pred[:, :, 0] - y_true[:, :, 0]) ** 2                   # (B, STEPS)
        mse_lon = (y_pred[:, :, 1] - y_true[:, :, 1]) ** 2

        for b in range(B):
            anom_lat_t = mse_lat[b] > mse_lat_thr
            anom_lon_t = mse_lon[b] > mse_lon_thr
            anom_any_t = (anom_lat_t | anom_lon_t) & mask_f[b]

            # Fill big table
            for t in range(STEPS):
                if not mask_f[b, t]:
                    continue

                row = {
                    "batch_idx": batch_idx,
                    "sample_idx": b,
                    "step_ahead": t + 1,
                    "mse_lat": float(mse_lat[b, t]),
                    "mse_lon": float(mse_lon[b, t]),
                    "anom_lat": bool(anom_lat_t[t]),
                    "anom_lon": bool(anom_lon_t[t]),
                    "anom_any": bool(anom_any_t[t]),
                }
                all_rows.append(row)

            # Decide if this sample is anomalous
            sample_has_anomaly = bool(anom_any_t.any())

            # Collect a few examples for plotting
            need_more_anom = sample_has_anomaly and len(anomaly_examples) < N_ANOM_EXAMPLES
            need_more_normal = (not sample_has_anomaly) and len(normal_examples) < N_NORMAL_EXAMPLES

            if need_more_anom or need_more_normal:
                example = {
                    "batch_idx": batch_idx,
                    "sample_idx": b,
                    # history and future are still 8D, but base model only sees first 2 features;
                    # that's fine for plotting as long as scaler & indices are consistent.
                    "hist_scaled": batch_X_test[b].detach().cpu().numpy(),                       # (SEQ_LEN, F)
                    "future_true_scaled": batch_future_test[b, :STEPS].detach().cpu().numpy(),   # (STEPS, F)
                    "future_pred_scaled": np.concatenate(
                        [
                            y_pred[b, :STEPS],                             # (STEPS, 2) predicted Lat/Lon
                            np.zeros((STEPS, batch_X_test.shape[-1] - 2))  # zeros for rest (SOG etc.)
                        ],
                        axis=1,
                    ),
                    "mask_future": mask_f[b],
                }
                if sample_has_anomaly:
                    anomaly_examples.append(example)
                else:
                    normal_examples.append(example)

    # 5) Save anomaly table
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_TABLE_PATH_base, index=False)
    print(f"Saved base-model anomaly table to {OUT_TABLE_PATH_base}")

    # 6) Save example plots
    print("Saving base-model example plots...")
    for i, ex in enumerate(anomaly_examples):
        out_path = OUT_PLOTS_DIR_base / f"anomaly_route_{i+1}.png"
        plot_route_example(ex, scaler, out_path, title_prefix="Base Anomaly")

    for i, ex in enumerate(normal_examples):
        out_path = OUT_PLOTS_DIR_base / f"normal_route_{i+1}.png"
        plot_route_example(ex, scaler, out_path, title_prefix="Base Normal")

    print(
        f"Saved {len(anomaly_examples)} anomaly plots and "
        f"{len(normal_examples)} normal plots to {OUT_PLOTS_DIR_base}"
    )


# -------------------------
#  8-FEATURE MODEL
# -------------------------
OUT_DIR_8features = PROJECT_ROOT / "model_results_8features" / "anomaly"
OUT_DIR_8features.mkdir(parents=True, exist_ok=True)
OUT_TABLE_PATH_8features = OUT_DIR_8features / f"anomalies_steps_{STEPS}.parquet"
OUT_PLOTS_DIR_8features = OUT_DIR_8features / f"plots_steps_{STEPS}"
OUT_PLOTS_DIR_8features.mkdir(parents=True, exist_ok=True)


def run_anomaly_detection_8features():
    # 1) Estimate error distribution on validation set
    print("Computing 8-feature-model thresholds from validation set...")
    mean_lat, std_lat, mean_lon, std_lon, _ = prediction_error_timestep(
        val_loader, model_8_features, steps=STEPS
    )

    mse_lat_thr = mean_lat + K_SIGMA * std_lat
    mse_lon_thr = mean_lon + K_SIGMA * std_lon

    np.savez(
    OUT_DIR_8features / "thresholds_8features.npz",
    mse_lat_thr=mse_lat_thr,
    mse_lon_thr=mse_lon_thr,
    mean_lat=mean_lat,
    mean_lon=mean_lon,
    std_lat=std_lat,
    std_lon=std_lon,
    K_SIGMA=K_SIGMA,
    STEPS=STEPS,
    )
    print(f"Saved base-model thresholds to {OUT_DIR_8features / 'thresholds_8features.npz'}")


    all_rows = []
    anomaly_examples = []
    normal_examples = []

    print("Scanning test set for anomalies (8-feature model)...")
    for batch_idx, batch in enumerate(test_loader):
        batch_X_test, batch_y_test, batch_future_test, mask_seq_test, mask_future_test = batch
        B = batch_X_test.shape[0]

        # 2) Predict future steps (full 8D)
        y_pred = rolling_forecast_lstm_batch_with_mask(
            model_8_features, batch_X_test, steps=STEPS, device=DEVICE
        )  # (B, STEPS, 8)

        # 3) Ground truth and masks
        y_true = batch_future_test[:, :STEPS, :2].detach().numpy()           # (B, STEPS, 2)
        mask_f = mask_future_test[:, :STEPS].detach().numpy().astype(bool)   # (B, STEPS)

        mse_lat = (y_pred[:, :, 0] - y_true[:, :, 0]) ** 2                   # (B, STEPS)
        mse_lon = (y_pred[:, :, 1] - y_true[:, :, 1]) ** 2

        for b in range(B):
            anom_lat_t = mse_lat[b] > mse_lat_thr
            anom_lon_t = mse_lon[b] > mse_lon_thr
            anom_any_t = (anom_lat_t | anom_lon_t) & mask_f[b]

            # Fill big table
            for t in range(STEPS):
                if not mask_f[b, t]:
                    continue

                row = {
                    "batch_idx": batch_idx,
                    "sample_idx": b,
                    "step_ahead": t + 1,
                    "mse_lat": float(mse_lat[b, t]),
                    "mse_lon": float(mse_lon[b, t]),
                    "anom_lat": bool(anom_lat_t[t]),
                    "anom_lon": bool(anom_lon_t[t]),
                    "anom_any": bool(anom_any_t[t]),
                }
                all_rows.append(row)

            # Decide if this sample is anomalous
            sample_has_anomaly = bool(anom_any_t.any())

            # Collect a few examples for plotting
            need_more_anom = sample_has_anomaly and len(anomaly_examples) < N_ANOM_EXAMPLES
            need_more_normal = (not sample_has_anomaly) and len(normal_examples) < N_NORMAL_EXAMPLES

            if need_more_anom or need_more_normal:
                example = {
                    "batch_idx": batch_idx,
                    "sample_idx": b,
                    "hist_scaled": batch_X_test[b].detach().cpu().numpy(),                     # (SEQ_LEN, 8)
                    "future_true_scaled": batch_future_test[b, :STEPS].detach().cpu().numpy(), # (STEPS, 8)
                    "future_pred_scaled": y_pred[b, :STEPS],                                   # (STEPS, 8)
                    "mask_future": mask_f[b],
                }
                if sample_has_anomaly:
                    anomaly_examples.append(example)
                else:
                    normal_examples.append(example)

    # 5) Save anomaly table
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_TABLE_PATH_8features, index=False)
    print(f"Saved 8-feature-model anomaly table to {OUT_TABLE_PATH_8features}")

    # 6) Save example plots
    print("Saving 8-feature-model example plots...")
    for i, ex in enumerate(anomaly_examples):
        out_path = OUT_PLOTS_DIR_8features / f"anomaly_route_{i+1}.png"
        plot_route_example(ex, scaler, out_path, title_prefix="8feat Anomaly")

    for i, ex in enumerate(normal_examples):
        out_path = OUT_PLOTS_DIR_8features / f"normal_route_{i+1}.png"
        plot_route_example(ex, scaler, out_path, title_prefix="8feat Normal")

    print(
        f"Saved {len(anomaly_examples)} anomaly plots and "
        f"{len(normal_examples)} normal plots to {OUT_PLOTS_DIR_8features}"
    )


# optional: run immediately
run_anomaly_detection_8features()
run_anomaly_detection_base()
