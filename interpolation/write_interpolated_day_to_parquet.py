import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from resample_interpolation import resample_interpolation
from build_resampled_day import build_resampled_day
def write_interpolated_day_to_parquet(
    input_root: str, 
    output_root: str, 
    date_str: str,
    *,
    freq: str = "1min",
    sample_mmsi=None,
    min_len: int = 30
):
    """
    Make a 1-min resampled AIS data for one day and write it as a parquet dataset:
    output_root/Date=YYYY-MM-DD/MMSI=.../Segment=.../part-*.parquet

    Interpolates one segment at a time.
    """

    dataset = ds.dataset(input_root, format="parquet", partitioning="hive")
    mmsi_type = dataset.schema.field("MMSI").type


    # Find MMSIs for that date 
    date_path = os.path.join(input_root, f"Date={date_str}")
    if not os.path.isdir(date_path):
        raise RuntimeError(f"No such date folder: {date_path}")
    
    mmsi_dirs = [d for d in os.listdir(date_path) if d.startswith("MMSI=")]
    if not mmsi_dirs:
        raise RuntimeError(f"No MMSI=... partitions found under {date_path}")
    all_mmsis = [d.split("=", 1)[1] for d in mmsi_dirs]

    if isinstance(sample_mmsi, int):
        import random
        k = min(sample_mmsi, len(all_mmsis))
        mmsis = random.sample(all_mmsis, k=k)
    elif isinstance(sample_mmsi, (list, tuple, set)):
        mmsis = list(set(sample_mmsi) & set(all_mmsis))
        if not mmsis:
            raise ValueError("Provided MMSIs list not found under this date")
    else:
        mmsis = all_mmsis  # all of them

        # columns to load
    cols = [
        "Timestamp","Latitude", "Longitude", "SOG", "COG","ETA", "ETA_known",
        "Type of mobile_known", "Ship type_known", "SOG_known", "COG_known",
        "Latitude_known", "Longitude_known",
        "Type of mobile_Class A",
        "Type of mobile_Class B",
        "Ship type_","Ship type_Anti-pollution","Ship type_Cargo","Ship type_Dredging",
        "Ship type_Fishing","Ship type_HSC","Ship type_Law enforcement","Ship type_Medical",
        "Ship type_Military","Ship type_Not party to conflict","Ship type_Passenger","Ship type_Pilot",
        "Ship type_Pleasure","Ship type_Port tender","Ship type_SAR","Ship type_Sailing","Ship type_Tanker",
        "Ship type_Towing","Ship type_Tug","Cargo type_Category OS","Cargo type_Category X","Cargo type_Category Y",
        "Cargo type_Category Z","Cargo type_No additional information","Cargo type_Reserved for future use",
        "day_of_week_sin","day_of_week_cos","Date",
        "MMSI",
        "Segment",
    ]
    cols_in_schema = [c for c in cols if c in dataset.schema.names]

    os.makedirs(output_root, exist_ok=True)
    total_rows = 0
    total_segments = 0
    total_ships = 0

    for m in mmsis:
        # Cast MMSI value to match Arrow schema type
        if pa.types.is_integer(mmsi_type):
            m_val = int(m)
        else:
            m_val = str(m)

        # filter for one MMSI + date
        filt = (ds.field("Date") == date_str) & (ds.field("MMSI") == m_val)
        table = dataset.to_table(filter=filt, columns=cols_in_schema)
        df = table.to_pandas()

        if df.empty:
            continue

        # dtypes & cleaning
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        for c in ["Latitude", "Longitude", "SOG", "COG"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["Timestamp", "Latitude", "Longitude"])
        df = df.sort_values(["MMSI", "Segment", "Timestamp"])

        # resample per (MMSI, Segment) for THIS MMSI only
        out_segments = []
        for (m2, s), g in df.groupby(["MMSI", "Segment"], sort=False):
            rg = resample_interpolation(g, freq=freq)
            if rg is None or len(rg) < min_len:
                continue
            out_segments.append(rg)

        if not out_segments:
            continue

        resampled_mmsi = pd.concat(out_segments, ignore_index=True)

        # ensure Timestamp + Date columns
        if "Timestamp" not in resampled_mmsi.columns and resampled_mmsi.index.name == "Timestamp":
            resampled_mmsi = resampled_mmsi.reset_index()
        if "Date" not in resampled_mmsi.columns:
            resampled_mmsi["Date"] = date_str

        resampled_mmsi = resampled_mmsi.sort_values(
            ["MMSI", "Segment", "Timestamp"]
        ).reset_index(drop=True)

        # write this MMSI's resampled segments to hive-partitioned dataset
        table_out = pa.Table.from_pandas(resampled_mmsi, preserve_index=False)
        pq.write_to_dataset(
            table_out,
            root_path=output_root,
            partition_cols=["Date", "MMSI", "Segment"],
        )

        total_rows += len(resampled_mmsi)
        total_ships += 1
        total_segments += resampled_mmsi[["MMSI", "Segment"]].drop_duplicates().shape[0]


    print(
        f"Date {date_str}: wrote {total_rows:,} rows, "
        f"{total_ships:,} ships, {total_segments:,} segments"
    )

if __name__ == "__main__": ## Only run from HPC
    INPUT_FOLDER = "/dtu/blackhole/08/223112/ship_data_parquet/"
    OUTPUT_FOLDER = "/dtu/blackhole/08/223112/ship_data_interpolated/"

    # Get all dates in folder
    date_dirs = sorted(
        d for d in os.listdir(INPUT_FOLDER)
        if d.startswith("Date=")
    )

    print(f"Found {len(date_dirs)} dates.")

    for d in date_dirs:
        date_str = d.split("=", 1)[1]
        print(f"Interpolating: {date_str}")

        try:
            write_interpolated_day_to_parquet(
                INPUT_FOLDER,
                OUTPUT_FOLDER,
                date_str
            )
        except Exception as e:
            print(f"ERROR on {date_str}: {e}")