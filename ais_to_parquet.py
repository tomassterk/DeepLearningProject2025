import pandas 
import pyarrow
import pyarrow.parquet
import os 
import numpy


def fn(file_path, out_path):
    dtypes = {
        "MMSI": "object",
        "SOG": float,
        "COG": float,
        "Longitude": float,
        "Latitude": float,
        "# Timestamp": "object",
        "Type of mobile": "object",
        "Ship type": "object", 
        "Cargo type": "object",
        "ETA": "object"
    }

    usecols = list(dtypes.keys())
    df = pandas.read_csv(file_path, usecols=usecols, dtype=dtypes)

    # Remove errors
    bbox = [60, 0, 50, 20]
    north, west, south, east = bbox
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (
            df["Longitude"] <= east)]

    #df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
    df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Adhere to MID standard

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    df["Timestamp"] = pandas.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    
    df["ETA"] = pandas.to_datetime(df["ETA"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df['ETA_known'] = df['ETA'].notna().astype(int)
    df['ETA'] = (df['ETA'] - df['ETA'].min()).dt.total_seconds().fillna(0)

    df = df.drop_duplicates(["Timestamp", "MMSI", ], keep="first")

    def track_filter(g):
        len_filt = len(g) > 256  # Min required length of track/segment
        sog_filt = 1 <= g["SOG"].max() <= 50  # Remove stationary tracks/segments
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60  # Min required timespan
        return len_filt and sog_filt and time_filt

    # Track filtering
    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(['MMSI', 'Timestamp'])

    # Divide track into segments based on timegap
    df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
        lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum())  # Max allowed timegap

    # Segment filtering
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    df = df.reset_index(drop=True)

    #
    knots_to_ms = 0.514444
    df["SOG"] = knots_to_ms * df["SOG"]

    #Clustering
    # kmeans = KMeans(n_clusters=48, random_state=0)
    # kmeans.fit(df[["Latitude", "Longitude"]])
    # df["Geocell"] = kmeans.labels_
    # center = kmeans.cluster_centers_
    #"Latitude": center[0]
    #"Longitude": center[1]
    #centers_df = pandas.DataFrame(center, columns=["Latitude", "Longitude"])
    # df["Center_Latitude"] = df["Geocell"].apply(lambda x: center[x][0])
    # df["Center_Longitude"] = df["Geocell"].apply(lambda x: center[x][1])
    
    # Date to partion later
    df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")

    # Definiere Spalten, für die du eine "known"-Indikator-Spalte willst
    indicator_cols = ["Type of mobile", "Ship type", "SOG", "COG", "Latitude", "Longitude"]

    for col in indicator_cols:
        is_known = (
        ~df[col].isna() &
        ~df[col].isin(["Unknown", "unknown", "Undefined", "undefine", "", " ", "NA", "N/A", "null", "Other", "_"])
    )
        df[f"{col}_known"] = is_known.astype(int)
    
        # Fill missing / unknown values in original column
        if df[col].dtype == "object":
            df[col] = df[col].where(is_known, numpy.nan)  # leeren String als Ersatz
        else:
            df[col] = df[col].where(is_known, 0)   # 0 für numerische Spalten
    
    cols_one_hot = ["Type of mobile", "Ship type", "Cargo type"]
    for col in cols_one_hot:
        one_hot = pandas.get_dummies(df[col], prefix=col, dummy_na=False)
        df = pandas.concat([df, one_hot], axis=1)
        df = df.drop(columns=[col])

    
    df['day_of_week'] = df['Timestamp'].dt.dayofweek

    # 2. Zyklische Kodierung: Sinus & Cosinus
    df['day_of_week_sin'] = numpy.sin(2 * numpy.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = numpy.cos(2 * numpy.pi * df['day_of_week'] / 7)   
    df = df.drop(columns=['day_of_week'])

    # 3. Write to file
    os.makedirs(out_path, exist_ok=True)
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(
        table,
        root_path=out_path,
        partition_cols=[
            "Date",
            "MMSI",  # "MMSI",
            "Segment",  # "Segment"
            ]
    )




if __name__ == "__main__": # run this only on HPC
    INPUT_FOLDER = "/dtu/blackhole/08/223112/ship_data/"
    OUTPUT_FOLDER = "/dtu/blackhole/08/223112/ship_data_parquet/"
    all_files = sorted([
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(".csv")
    ])

    print(f"Found {len(all_files)} CSV files")

    for filename in all_files:
        file_path = os.path.join(INPUT_FOLDER, filename)
        print(f"Processing: {file_path}")

        try:
            fn(file_path=file_path, out_path=OUTPUT_FOLDER)
            print(f"Finished: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

