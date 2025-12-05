from datetime import date, timedelta
import os

from write_interpolated_day_to_parquet import write_interpolated_day_to_parquet

def daterange_month(year: int, month: int):
    """Yield all dates in a given year-month as 'YYYY-MM-DD' strings."""
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)

    cur = start
    while cur < end:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


def resample_month(
    input_root,
    output_root,
    year: int,
    month: int,
    *,
    freq: str = "1min",
    sample_mmsi=None,
    min_len: int = 30,
    skip_if_done: bool = True,
):
    """
    Loop over all days in a month and call write_interpolated_day_to_parquet
    for each existing Date=YYYY-MM-DD partition under input_root.

    - input_root: cleaned parquet dataset (with Date=... partitions)
    - output_root: resampled parquet dataset root
    """

    for d in daterange_month(year, month):
        # check if that date even exists in the input
        date_path = os.path.join(input_root, f"Date={d}")
        if not os.path.isdir(date_path):
            print(f"[{d}] no input folder, skipping.")
            continue

        # optional: skip if we've already written this day in the resampled output
        out_date_path = os.path.join(output_root, f"Date={d}")
        if skip_if_done and os.path.isdir(out_date_path):
            print(f"[{d}] already resampled (found {out_date_path}), skipping.")
            continue

        print(f"[{d}] resampling...")
        try:
            write_interpolated_day_to_parquet(
                input_root=input_root,
                output_root=output_root,
                date_str=d,
                freq=freq,
                sample_mmsi=sample_mmsi,
                min_len=min_len,
            )
        except Exception as e:
            print(f"[{d}] ERROR during resampling: {e}")
