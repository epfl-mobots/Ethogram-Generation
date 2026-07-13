import numpy as np
import motionmapperpy as mmpy
import matplotlib.pyplot as plt
import pandas as pd
from VisualActivity.RHCVisualisation.RHCThermalPlots.InfluxDBInterface.libdb import download_tmp_DB, download_data_DB, download_co2_DB
from VisualActivity.RHCVisualisation.RHCThermalPlots.thermalutil import extractAmbientTemp


def download_dataset(hive_nb:int, ihl:str, resolution:int, start_ts:pd.Timestamp, end_ts:pd.Timestamp, only_amb_T:bool=False, visualActivity:bool=False)-> pd.DataFrame:
    """
    Download the dataset for a given hive number and in-hive location.
    """
    assert ihl in ["upper", "lower"], "inhive_loc must be either 'upper' or 'lower'"
    assert hive_nb in [1, 2], "hive_num must be either 1 or 2"
    filters = {
        "hive_num" : hive_nb,
        "inhive_loc" : ihl,
    }
    df = download_tmp_DB('ObsHiveABC', start_ts, end_ts, resolution=resolution, filters=filters, aggr="last")
    if only_amb_T:
        df = extractAmbientTemp(df)
        # Convert pd.Series to pd.Dataframe
        df = df.to_frame(name="Tamb")
    
    # remove "inhive_loc" key from filters
    filters.pop("inhive_loc")
    co2_data = download_co2_DB('ObsHiveABC', start_ts, end_ts, resolution = resolution, filters = filters)
    # Keep only the columns that start with the same letter as the first letter of ihl (i.e. "u" for "upper" and "l" for "lower")
    co2_data = co2_data.loc[:, co2_data.columns.str.startswith(ihl[0])]
    for col in co2_data.columns:
        col_name = f"co2_{col[-1].upper()}"
        df[col_name] = co2_data.loc[:,col] # This should add 2 columns: (ul and ur) OR (ll and lr)
    
    # Add a _field tag and _measurement tag to filters
    filters["field"] = ["rel_humid"]
    filters["measurement"] = ["co2", "rht"]
    filters["inhive_loc"] = ihl
    humid_data = download_data_DB('ObsHiveABC', start_ts, end_ts, resolution=resolution, filters=filters)
    
    # For every ts in df, there are several ts in humid_data. We want to take the average of the values in humid_data for each ts in df_resampled and store it in the "rel_humid" column of df_resampled.
    df["rel_humid"] = df.index.to_series().apply(lambda ts: humid_data.loc[humid_data.index == ts, "_value"].mean())
    
    # Filter out timestamps not allowed by HiveOpenings
    from VisualActivity.RHCVisualisation.RHCThermalPlots.RHCImaging.HiveOpenings.libOpenings import filter_timestamps
    print("Before filtering with HiveOpenings:", len(df), "lines")
    filtered_ts = filter_timestamps(df.index.to_list(), hive_nb=hive_nb, recovery_time=240)
    df_resampled = df[df.index.isin(filtered_ts)]
    print("After  :", len(df_resampled), "lines")
    
    return df_resampled

def build_ethogram(w):
    """Build ethogram matrix from watershed region vector `w`.

    Treat values <= 0 or NaN as 'cannot classify' and add a final row for that label.
    Returns (E, nregions) where E shape is (nregions + 1, len(w)) and the last row
    corresponds to 'cannot classify'. nregions is the number of real regions (not
    counting the cannot-classify row).
    """
    w_arr = np.asarray(w)
    # debug: expose dtype and small sample to help diagnose indexing problems
    try:
        sample_preview = w_arr[:10]
    except Exception:
        sample_preview = None
    print(f"build_ethogram: dtype={w_arr.dtype}, shape={w_arr.shape}, sample={sample_preview}")
    if w_arr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8), 0

    # consider non-positive or NaN as unclassified
    invalid_mask = (~np.isfinite(w_arr)) | (w_arr <= 0)
    # determine number of labeled regions
    max_region = int(np.nanmax(w_arr)) if np.any(np.isfinite(w_arr)) else 0
    nregions = max_region

    # allocate with an extra row for 'cannot classify'
    E = np.zeros((nregions + 1, w_arr.size), dtype=np.uint8)

    # fill per-sample to avoid any dtype/indexing edge cases
    for i in range(w_arr.size):
        v = w_arr[i]
        if not np.isfinite(v) or v <= 0:
            E[-1, i] = 1
            continue
        rid = int(round(v))
        if 1 <= rid <= nregions:
            E[rid - 1, i] = 1
        else:
            # outside expected region ids -> mark cannot classify
            E[-1, i] = 1

    return E, nregions

def plot_ethogram(E, nregions, day_names, day_boundaries, nb_points_per_day, title, timestamps=None):
    """Plot ethogram with linear time x-axis.

    If `timestamps` (an array-like of datetime-like) is provided, the x-axis will
    be linear in time and day tick positions will be computed from these timestamps.
    The last row of `E` is expected to be the 'cannot classify' row and will be
    labeled accordingly.
    """
    import matplotlib.dates as mdates

    nrows, ncols = E.shape
    fig, ax = plt.subplots(figsize=(20, 6))

    cmap = (mmpy.gencmap() if 'mmpy' in globals() else 'viridis')

    if timestamps is None:
        # fall back to original imshow behavior with implicit linear x
        ax.imshow(E, aspect='auto', cmap=cmap, origin='lower', interpolation='nearest')
        x_coords = None
    else:
        # convert timestamps to matplotlib numeric dates
        times = pd.to_datetime(timestamps)
        time_nums = mdates.date2num(times)
        if len(time_nums) < 2:
            # degenerate case
            x_edges = np.array([time_nums[0] - 0.5, time_nums[0] + 0.5])
        else:
            # build edge coordinates between time points
            dt = np.diff(time_nums)
            med = np.median(dt)
            left = time_nums[0] - med / 2.0
            right = time_nums[-1] + med / 2.0
            # edges length = ncols + 1
            x_edges = np.concatenate(([left], (time_nums[:-1] + time_nums[1:]) / 2.0, [right]))

        # y edges
        y_edges = np.arange(nrows + 1)
        # pcolormesh expects shape (nrows, ncols) data with x_edges len ncols+1
        ax.pcolormesh(x_edges, y_edges, E, cmap=cmap, shading='auto')
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        x_coords = time_nums

    # y ticks: include cannot-classify label as last row
    n_yticks = min(12, max(nrows, 1))
    if nrows > 1:
        y_pos = np.linspace(0, nrows - 1, n_yticks, dtype=int)
    else:
        y_pos = np.array([0], dtype=int)
    y_labels = []
    for i in y_pos:
        if i < nregions:
            y_labels.append(f"Region {i+1}")
        else:
            y_labels.append("cannot classify")
    ax.set_yticks(y_pos + 0.5)
    ax.set_yticklabels(y_labels)

    # day tick labels: compute positions as midpoint of each day's timestamps, if timestamps provided
    if timestamps is not None:
        times = pd.to_datetime(timestamps)
        start = 0
        ticks_x, labels_x = [], []
        for name, n in zip(day_names, nb_points_per_day):
            if n <= 0:
                start += n
                continue
            segment = times[start:start + n]
            if len(segment) == 0:
                pos = np.nan
            else:
                # segment may be a DatetimeIndex or Series; index by integer position
                mid_ts = segment[len(segment) // 2]
                pos = mdates.date2num(mid_ts)
            ticks_x.append(pos)
            labels_x.append(name)
            start += n
        # filter NaNs
        ticks_filt = [t for t in ticks_x if not np.isnan(t)]
        ax.set_xticks(ticks_filt)
        ax.set_xticklabels(labels_x[:len(ticks_filt)], rotation=45, ha='right')
    else:
        ticks_x, labels_x, start = [], [], 0
        for name, n in zip(day_names, nb_points_per_day):
            ticks_x.append(start + n // 2)
            labels_x.append(name)
            start += n
        ax.set_xticks(ticks_x)
        ax.set_xticklabels(labels_x, rotation=45, ha='right')

    ax.set_xlabel('Time' if timestamps is not None else 'Days')
    ax.set_ylabel('Regions')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def get_source_vector(df):
    """Retourne un vecteur numpy des source_id aligné aux lignes de df,
       que source_id soit dans l'index ou en colonne."""
    if isinstance(df.index, pd.MultiIndex) and 'source_id' in df.index.names:
        return df.index.get_level_values('source_id').to_numpy()
    elif 'source_id' in df.columns:
        return df['source_id'].to_numpy()
    else:
        raise KeyError("Aucun 'source_id' trouvé (ni niveau d'index ni colonne).")
    
def get_time_index(df):
    """Retourne un DatetimeIndex aligné aux lignes de df, quel que soit l'index."""
    if isinstance(df.index, pd.MultiIndex):
        # cherche un niveau datetime
        for name in df.index.names:
            vals = df.index.get_level_values(name)
            if pd.api.types.is_datetime64_any_dtype(vals):
                return pd.to_datetime(vals)
        raise TypeError("Aucun niveau datetime trouvé dans le MultiIndex.")
    else:
        return pd.to_datetime(df.index)

def day_boundaries_from_df(df_sub):
    ts = get_time_index(df_sub)
    dates = ts.date
    change = np.flatnonzero(np.diff(dates.astype('datetime64[D]'))) + 1
    boundaries = list(change) + [len(df_sub)]
    names, counts, start = [], [], 0
    for b in boundaries:
        names.append(str(pd.to_datetime(dates[start]).date()))
        counts.append(b - start)
        start = b
    return names, np.cumsum(counts).tolist(), counts