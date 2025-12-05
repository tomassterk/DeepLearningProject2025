
# VESSEL TRAJECTORY PREDICTION & ANOMALY DETECTION IN DANISH WATERS

This project downloads AIS ship tracking data, converts it to Parquet, preprocesses it into time series, and trains LSTM models to predict vessel trajectories and detects anomalies.

## Project Structure

- `Main results and plots.ipnyb`
  - Plotted example trajectories from the sample batch for both the 2-feature base model and the 8-feature advanced model.
  - Visualized the results of the anomaly detection, which are also summarized in the report.

- `download_files.py`  
  Script for downloading AIS ZIP archives from `http://aisdata.ais.dk` for specified years and months/days.  
  - Uses HTTP HEAD to check if a monthly ZIP exists; if not, falls back to daily ZIPs.  
  - Streams downloads with a progress bar and extracts ZIPs into `/dtu/blackhole/08/223112/<target_folder>`.  
  - Intended to run on the DTU HPC environment with access to the Blackhole storage.

- `ais_to_parquet.py`  
  Utilities for converting raw AIS text/CSV files into cleaned, per-ship Parquet files.

- `preprocess_data.py`  
  Data preparation pipeline for model training:  
  - Collects Parquet files from a given root directory.  
  - Builds train/validation/test splits, optionally stratified and/or based on predefined file lists (`train_files.txt`, `val_files.txt`, `test_files.txt`).  
  - Fits a scaler on continuous features and applies it when building PyTorch datasets.  
  - Creates `DataLoader`s for train/val/test and optionally saves:
    - Split files: `train_files_<split_id>.txt`, `val_files_<split_id>.txt`, `test_files_<split_id>.txt`  
    - Scaler: `scaler_<split_id>.joblib`  
    - Manifest: `manifest_<split_id>.json` with configuration metadata.

- `train_model.py`  
  Main training script for an LSTM sequence model:  
  - Parses common training hyperparameters (epochs, hidden size, num layers, dropout, learning rate).  
  - Calls `preprocess_data.build_all(...)` to obtain `DataLoader`s and a unique `split_id`.  
  - Selects device automatically (CUDA, MPS, or CPU).  
  - Trains `LSTMModel` on one or more feature sets, using a masked MSE loss to handle missing timesteps.  
  - Implements:
    - Periodic checkpointing (time-based) to `models/may/`  
    - Validation and early stopping with patience  
    - Saving the best model per feature configuration  
    - Logging of train/val loss histories.

- `lstm_model.py`  
  Defines `LSTMModel`, a multi-layer LSTM network for sequence-to-sequence regression on AIS trajectories.  
  - Input size and output size depend on the chosen feature subset.  
  - Supports configurable hidden size, number of layers, and dropout.

- `evaluation_plots.py`  
  Plotting training and validation loss curves and saving them as PNGs in `plots/`.

- `requirements.txt`  
  List of Python dependencies needed to run downloading, preprocessing, training, and plotting.

- `train_files.txt`, `val_files.txt`, `test_files.txt`  
  Optional text files listing Parquet paths for each split. These can override or document the automatic splitting logic in `preprocess_data.py`.
- `plots/`: Stores the training and validation loss plots generated during model training. Each run saves one or more PNG files here, typically timestamped and tagged by feature configuration.

- `anomaly/`: Contains the anomaly detection pipeline built on top of the trained LSTM models. This folder has its own README that explains how anomalies are scored and how to run the detection scripts or notebooks.

- `interpolation/`: Contains a custom interpolation library and its own README. The main entry point is the function `write_interpolated_day_to_parquet`, which converts irregular AIS measurements into minutely-resampled time series stored as Parquet. 

- `model_results_8features/`: Results of anomaly detection for the model trained with 8 input features. Includes a notebook with quantitative results and plots, plus example visualizations of 10 “normal” routes and 10 routes with detected anomalies.

- `model_results_2features/`: Results of anomaly detection for the model trained with a reduced 2-feature input. Similar to the 8-feature folder, it contains a notebook with results and plots, and 10 normal-route plots alongside 10 anomaly-route plots.

- `models/`: Stores serialized LSTM model checkpoints (`.pth` files), including intermediate checkpoints and best-model weights for each feature configuration and experiment.



## Setup

1. Create and activate a virtual environment:

   ```
   python -m venv .venv
   source .venv/bin/activate      # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Download and extract AIS 

   ```
   python download_files.py
   ```

   This will populate `/dtu/blackhole/08/223112/ship_data` (or any configu folder) with extracted raw AIS files.

4. Convert AIS to Parquet (if not already done):

   ```
   python ais_to_parquet.py
   ```

   Configure input/output paths inside `ais_to_parquet.py` as needed.

## Data Preprocessing

Preprocessing is typically invoked indirectly through `train_model.py`, via a helper such as `preprocess_data.build_all(...)`:

- Scans a Parquet directory (e.g. `/dtu/blackhole/08/223112/ship_data_interpolated`).
- Builds train/val/test splits (optionally using existing `*_files.txt`).
- Fits a scaler on continuous features and prepares batched sequences.
- Saves split artifacts and a JSON manifest under `splits/` using a short `split_id`.

You can also call the relevant builder functions from `preprocess_data.py` directly in your own scripts to:
- Generate new splits.
- Reuse an existing `split_id`.
- Inspect or modify the scaling and feature configuration.

## Training

To train LSTM models using the default configuration defined in `train_model.py`:

```
python train_model.py \
  --epochs 100 \
  --hidden_size 128 \
  --num_layers 3 \
  --dropout 0.2 \
  --learning_rate 0.001
```

`train_model.py` will:

- Use a Parquet root such as `/dtu/blackhole/08/223112/ship_data_interpolated` (adjust in the script if needed).
- Train over predefined feature sets (e.g. full feature list and a reduced `[Latitude, Longitude]` subset).
- Save:
  - Intermediate checkpoints in `models/may/`  
  - The best model per feature set  
  - Train/val loss plots under `plots/may/`  
  - A sample batch tensor for inspection.

## Reproducibility

- Random seeds for Python, NumPy, and PyTorch (including CUDA where applicable) are set in the preprocessing code to improve reproducibility.
- Each data split and scaler configuration is tied to a `split_id`, with accompanying text files and a JSON manifest under `splits/`, enabling exact replay of a previous experiment.

## Notes

- Paths in `download_files.py`, `ais_to_parquet.py`, and `train_model.py` are currently tailored to a DTU HPC setup (`/dtu/blackhole/08/223112/...`). Update these paths if you run the project in a different environment.
- For long-running training on HPC, consider wrapping `train_model.py` in an LSF/SLURM job script and monitoring the checkpoint directory for progress.

```