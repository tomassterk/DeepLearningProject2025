from pathlib import Path
import torch
import preprocess_data

# - ```val_loader```
# - ```test_loader```
# - ```val_files.txt``` 
# - ```test_files.txt``` 
# - ```scaler_with_masking``` 

# Paths
#############################
###### UPDATE FROM HPC ######
PROJECT_ROOT = Path(__file__).resolve().parents[1]


######### UPDATE PATH  + MAYBE ADD DIFFERENT SCALER FROM THE 2 MODELS ################
SCALER_PATH = PROJECT_ROOT / "scaler_with_mask.joblib" 
#################################

VAL_FILES = PROJECT_ROOT / "val_files.txt"
TEST_FILES = PROJECT_ROOT / "test_files.txt"

####################################
####################################
####################################
# UPDATE LTSMS PATHS 
MODEL_BASE_PATH = PROJECT_ROOT / "models/best_lstm_model_Latitude_Longitude.pth"
MODEL_8_PATH   = PROJECT_ROOT / "models/best_lstm_model_Latitude_Longitude_SOG_COG_SOG_known_COG_known_Latitude_known_Longitude_known.pth"

#PLOTS_ROOT     = PROJECT_ROOT / "plots" / "anomaly"
#RESULTS_ROOT   = PROJECT_ROOT / "results" / "anomaly"
FEATURES = [
    'Latitude', 'Longitude', 'SOG', 'COG', 'Latitude_known', 'Longitude_known', 'SOG_known', 'COG_known'
    
]


# LSTM hyperparams (must match training!)
INPUT_SIZE_8  = 8
INPUT_SIZE_BASE = 2
HIDDEN_SIZE = 128
NUM_LAYERS  = 3
OUTPUT_SIZE_8 = 8
OUTPUT_SIZE_BASE = 2
DROPOUT     = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature layout (match what you used in training!)
LAT_IDX = 0
LON_IDX = 1
SCALED_IDX = [0, 1, 2, 3]  # Lat, Lon, SOG, COG

# Sequence / batching
SEQ_LEN  = 30   # timesteps
BATCH_SIZE = 50  #

# Thresholding (3σ on some reference set)
#N_SIGMA = 3.0
