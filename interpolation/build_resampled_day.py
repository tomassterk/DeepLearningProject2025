import os
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from resample_interpolation import resample_interpolation
def build_resampled_day(
    root_dir: str, 
    date_str: str, 
    *, 
    freq='1min',
    sample_mmsi=None, # to only make a subset of MMSIs or list of MMSIs 
    min_len=30 # Min resampled points for (mmsi,segment)
):
    
    dataset = ds.dataset(root_dir, format='parquet', partitioning='hive')

    # Get MMSIs for that day -> date folder -> MMSI 
    date_path = os.path.join(root_dir, f"Date={date_str}")
    if not os.path.isdir(date_path):
        raise RuntimeError(f"No such date folder: {date_path}")

    mmsi_dirs = [d for d in os.listdir(date_path) if d.startswith("MMSI=")]
    if not mmsi_dirs:
        raise RuntimeError(f"No MMSI=... partitions found in {date_path}.")
    
    mmsis = [d.split("=",1)[1] for d in mmsi_dirs]

    # Sample a subset if true 
    if isinstance(sample_mmsi, int): 
        import random 
        k = min(sample_mmsi, len(mmsis))
        sampled_mmsis = random.sample(mmsis, k=k)
    elif isinstance(sample_mmsi, (list, tuple, set)):
        sampled_mmsis = list(set(sample_mmsi) & set(mmsis))
        if not sampled_mmsis:
            raise ValueError("Provided MMSIs list not found under the partition")
    else: 
        sampled_mmsis = mmsis # All 

    cols = ["Timestamp", "Latitude", "Longitude", "SOG", "COG","ETA","ETA_known",
            "Type of mobile_known", "Ship type_known", "SOG_known", "COG_known",
            "Latitude_known", "Longitude_known", "Type of mobile_Class A","Type of mobile_Class B",
            "Ship type_","Ship type_Anti-pollution","Ship type_Cargo","Ship type_Dredging","Ship type_Fishing",
            "Ship type_HSC","Ship type_Law enforcement","Ship type_Medical","Ship type_Military",
            "Ship type_Not party to conflict","Ship type_Passenger","Ship type_Pilot","Ship type_Pleasure",
            "Ship type_Port tender","Ship type_SAR","Ship type_Sailing","Ship type_Tanker","Ship type_Towing",
            "Ship type_Tug","Cargo type_Category OS","Cargo type_Category X","Cargo type_Category Y","Cargo type_Category Z",
            "Cargo type_No additional information","Cargo type_Reserved for future use",
            "day_of_week_sin",
            "day_of_week_cos",
            "Date",
            "MMSI",
            "Segment"
    ]
    
    filt = (ds.field("Date") == date_str) & (ds.field("MMSI").isin(sampled_mmsis))
    table = dataset.to_table(filter=filt, columns=[c for c in cols if c in dataset.schema.names])
    df = table.to_pandas()
    if df.empty:
        raise RuntimeError("No rows loaded. Check filters/columns.")

    # dtypes
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for c in ["Latitude","Longitude","SOG","COG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Timestamp","Latitude","Longitude"])
    df = df.sort_values(["MMSI","Segment","Timestamp"])

    # resample per (MMSI, Segment)
    out = []
    for (m, s), g in df.groupby(["MMSI","Segment"], sort=False):
        rg = resample_interpolation(g, freq=freq)
        #print(rg)
        if rg is None or len(rg) < min_len:
            continue
        out.append(rg)

    if not out:
        raise RuntimeError("All segments dropped by min_len filter; try lowering min_len or check data.")

    resampled = pd.concat(out, ignore_index=True)
    return resampled