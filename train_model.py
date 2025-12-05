import torch, numpy as np, random
import preprocess_data
from lstm_model import LSTMModel
import torch.nn as nn
import argparse
import torch.optim as optim
from evaluation_plots import plot_train_val_loss
from datetime import datetime
import os
import time


print("LSF job script started", flush=True)


def main(path_parquet='parquet_output_with_modification', FEATURES_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--hidden_size", type=int, default=128, help="Number of units in hidden layers")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for regularization")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    args = parser.parse_args()
    hyperparams = (args.epochs, args.hidden_size, args.num_layers, args.dropout, args.learning_rate)

    preprocess_data.set_seed(42)

    FULL_FEATURE_LIST = ['Latitude', 'Longitude', 'SOG', 'COG',
                         'SOG_known', 'COG_known',
                         'Latitude_known', 'Longitude_known']

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print("\nUsing device:", device, flush=True)

    train_loader, val_loader, test_loader, val_files, test_files, split_id = preprocess_data.build_all(
        path_parquet,
        batch_size=50,
        seq_len=300, FEATURES=FULL_FEATURE_LIST,
        artifacts_dir="splits",
        save_artifacts=True)
    print(f"Saved/using split_id={split_id}", flush=True)

    print("Loaded data!", flush=True)

    # Start of for loop
    for FEATURES in FEATURES_list:
        print("Using features:", FEATURES, flush=True)
        feature_indices = [FULL_FEATURE_LIST.index(f) for f in FEATURES]

        input_size = len(FEATURES)
        output_size = len(FEATURES)
        num_epochs, hidden_size, num_layers, dropout, learning_rate = hyperparams

        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)

        loss_fn = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_hist = []
        val_hist = []

        #Ensure folders exist
        os.makedirs("plots/may", exist_ok=True)
        os.makedirs("models/may", exist_ok=True)

        #Initiate checkpoint directories
        feature_tag = "_".join(FEATURES)
        checkpoint_path = os.path.join("models/may", f"lstm_model_checkpoint_{feature_tag}.pth")
        save_interval = 20*60 # Save every 20 minutes
        last_save_time = time.time()

        patience = 10
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_path = os.path.join("models/may", f"best_lstm_model_{feature_tag}.pth")

        print("Started training:")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                batch_X, batch_y, mask_seq = batch

                batch_X = batch_X[:, :, feature_indices]
                batch_y = batch_y[:, :, feature_indices]
                # Not needed for training data but for val and test
                if epoch == 0 and batch_idx == 0:
                    print("Saving sample batch...")
                    torch.save(
                        {
                            "X": batch_X.cpu(),
                            "y": batch_y.cpu()
                        },
                        f"sample_batch_{feature_tag}.pt"
                    )

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                mask = mask_seq.to(device).float()

                if mask.ndim == 3 and mask.shape[-1] == 1:  ################################this line
                    mask = mask.squeeze(-1)

                optimizer.zero_grad()
                preds = model(batch_X)

                per_timestep_mse = ((preds - batch_y) ** 2).mean(dim=2)
                loss = (per_timestep_mse * mask).sum() / mask.sum().clamp(min=1.0)

                #loss = loss_fn(preds, batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                now = time.time()
                if now - last_save_time >= save_interval:
                    checkpoint  = {
                        'model_tag': feature_tag,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_hist': train_hist,
                        'val_hist': val_hist,
                        'features': FEATURES,}
                    tmp_path = checkpoint_path + ".tmp"
                    torch.save(checkpoint, tmp_path)
                    os.replace(tmp_path, checkpoint_path)  # atomic-ish replace
                    last_save_time = now

                    print(f"([Checkpoint] Saved model at epoch {epoch + 1} "
                            f"to {checkpoint_path}",
                            flush=True)

            avg_train_loss = total_loss / len(train_loader)
            train_hist.append(avg_train_loss)

            # Validation
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    ################################ start
                    batch_X_val, batch_y_val, batch_future_val, mask_seq_val, mask_future = batch
                    batch_X_val = batch_X_val[:, :, feature_indices]
                    batch_y_val = batch_y_val[:, :, feature_indices]

                    batch_X_val = batch_X_val.to(device)
                    batch_y_val = batch_y_val.to(device)
                    mask_seq_val = mask_seq_val.to(device).float()  # should be (B, seq_len)
                    mask_future = mask_future.to(device).float()

                    preds_val = model(batch_X_val)

                    ################################ new start

                    if mask_seq_val.ndim == 1:
                        mask_seq_val = mask_seq_val.unsqueeze(0)
                    if mask_seq_val.ndim == 3 and mask_seq_val.shape[-1] == 1:
                        mask_seq_val = mask_seq_val.squeeze(-1)

                    # align preds/predicted timesteps with batch_y_val shape
                    per_timestep_mse_val = ((preds_val - batch_y_val) ** 2).mean(dim=2)

                    if mask_seq_val.shape != per_timestep_mse_val.shape:
                        raise RuntimeError(f"Mask shape {mask_seq_val.shape} != mse shape {per_timestep_mse_val.shape}")

                    masked_sum = (per_timestep_mse_val * mask_seq_val).sum()
                    denom = mask_seq_val.sum().clamp(min=1.0)
                    val_loss = masked_sum / denom
                    ################################ end

                    # val_loss  = loss_fn(preds_val, batch_y_val) ################################
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_hist.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0

                # save best model
                torch.save(model.state_dict(), best_model_path)
                print(f"[EarlyStopping] Validation improved → best model saved (epoch {epoch + 1})", flush=True)

            else:
                epochs_no_improve += 1
                print(f"[EarlyStopping] No improvement for {epochs_no_improve} epochs", flush=True)

            # Stop training if patience exceeded
            if epochs_no_improve >= patience:
                print(f"\n[EarlyStopping] Training stopped early at epoch {epoch + 1}. "
                      f"Best val loss: {best_val_loss:.6f}", flush=True)
                break


            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] - "
                      f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}", flush=True)

        epochs = np.arange(1, len(train_hist) + 1)
        train = np.array(train_hist)
        val = np.array(val_hist)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


        plot_train_val_loss(epochs, train, val, save_path=f"plots/may/train_val_loss_{feature_tag}_{timestamp}.png")
        torch.save(model.state_dict(), f"models/may/lstm_model_{feature_tag}_{timestamp}.pth")
        print("Saved plot and model with timestamp:", timestamp, " and feature tag:", feature_tag, flush=True)


if __name__ == "__main__":
    main(path_parquet='/dtu/blackhole/08/223112/ship_data_interpolated', FEATURES_list=[['Latitude', 'Longitude', 'SOG', 'COG', 'SOG_known', 'COG_known', 'Latitude_known', 'Longitude_known'], ['Latitude', 'Longitude']])
