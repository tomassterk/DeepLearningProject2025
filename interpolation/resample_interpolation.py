import numpy as np
import pandas as pd
from interpolation_helpers import meters_per_degree, wrap_angle_diff_deg

def resample_interpolation(g: pd.DataFrame, freq='1min'):
    
    g = g.sort_values('Timestamp').copy()
    # Treat unknown SOG/COG (known==0) as missing so they get interpolated
    for col in ["SOG", "COG"]:
        known_col = f"{col}_known"
        if known_col in g.columns:
            g.loc[g[known_col] == 0, col] = np.nan


    # Keep original index to mark observed rows after resample (1 = real, 0 interpolated)
    g["is_observed"] = 1
    g = g.set_index('Timestamp')

    # Columns we will keep and interpolate on 
    keep_cols = ['Latitude', 'Longitude', 'SOG', 'COG']
    existing = [c for c in keep_cols if c in g.columns]

    # Resample the grid 
    rg = g.resample(freq).first()

    # Mark observed rows after resample: Keep 1 on those that landed on timestamp 
    rg['is_observed'] = rg['is_observed'].fillna(0).astype(int)

    # Forward/back-fill static / categorical features (ETA, *_known, one-hots, Date, etc.)
    static_cols = [
        c for c in g.columns
        if c not in keep_cols + ['Timestamp', 'MMSI', 'Segment', 'is_observed']
    ]
    for c in static_cols:
        if c in rg.columns:
            # Avoid FutureWarning: ensure consistent dtype before filling
            if rg[c].dtype == "object":
                rg[c] = rg[c].astype("string").ffill().bfill()
            else:
                rg[c] = rg[c].ffill().bfill()



    # Interpolate with pandas 
    for c in existing:
        rg[c] = rg[c].interpolate(limit_direction='both')

   # COG -> sin/cos (radians) - (cyclical)
    if 'COG' in rg.columns:
        rad = np.deg2rad(rg['COG'] % 360)
        rg['cog_sin'] = np.sin(rad)
        rg['cog_cos'] = np.cos(rad)
    else: # If no COG -> 0 
        rg['cog_sin'] = 0.0 
        rg['cog_cos'] = 0.0

    # Turn-rate (deg/s) and acceleration (m/s^2)
    # d_t in seconds 
    if len(rg) >= 2: # need atleast 2 data points to compute. 
        dt_sec = rg.index.to_series().diff().dt.total_seconds().fillna(0).to_numpy()
        dt_sec[dt_sec == 0] = np.nan # avoid div by zero in first row 

        # Delta COG 
        if 'COG' in rg.columns: 
            cog = rg['COG'].to_numpy()
            d_cog = wrap_angle_diff_deg(np.roll(cog, -1), cog) # (next - now) absolute, wrapped around 360
            d_cog[-1] = np.nan # Last value does not work 
            rg['turn_rate_deg_s'] = d_cog / dt_sec 

        # Delta SOG 
        if 'SOG' in rg.columns: 
            sog = rg['SOG'].to_numpy()
            d_sog = np.roll(sog, -1) - sog
            d_sog[-1] = np.nan 
            rg['accel_mps2'] = d_sog / dt_sec 

    else: 
        rg['turn_rate_deg_s'] = np.nan 
        rg['accel_mps2'] = np.nan 

    # Delta_x, delta_y in meters from lat/lon differences 
    if {'Latitude', 'Longitude'} <= set(rg.columns):
        lat = rg['Latitude'].to_numpy()
        lon = rg['Longitude'].to_numpy()
        lat_mean = np.nanmean(lat) if len(lat) else 56.0
        m_lat, m_lon = meters_per_degree(lat_mean)
        dlat = np.roll(lat, -1) - lat 
        dlon = np.roll(lon, -1) - lon 
        dlat[-1] = np.nan
        dlon[-1] = np.nan
        rg['dx_m'] = dlon * m_lon 
        rg['dy_m'] = dlat * m_lat 
    
    # Time of day (cyclical)
    idx = rg.index 
    hour_day = idx.hour + idx.minute/60.0 + idx.second/3600.0 
    rg['hour_sin'] = np.sin(2*np.pi*hour_day/24.0)
    rg['hour_cos'] = np.cos(2*np.pi*hour_day/24.0)

    # Interpolated mask - opposite of observed 
    rg['is_interpolated'] = (1 - rg['is_observed']).astype(int)

    # Bring back constants: MMSI & Segment 
    mmsi = g['MMSI'].iloc[0]
    seg = g['Segment'].iloc[0]
    rg['MMSI'] = mmsi 
    rg['Segment'] = seg 
    rg = rg.reset_index().rename(columns={'index': 'Timestamp'})

    return rg