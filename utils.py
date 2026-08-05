import numpy as np
import motionmapperpy as mmpy
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from Metabolism.VisualActivity.RHCVisualisation.RHCThermalPlots.InfluxDBInterface.libdb import download_tmp_DB, download_data_DB, download_co2_DB
from Metabolism.VisualActivity.RHCVisualisation.RHCThermalPlots.thermalutil import extractAmbientTemp
from Metabolism.VisualActivity.RHCVisualisation.RHCImaging.libimage import fetchImagesPaths
from Metabolism.VisualActivity.libActivity import computeRpiActivities


def download_dataset(hive_nb:int, resolution:int, start_ts:pd.Timestamp, end_ts:pd.Timestamp, ihl:str=None, only_amb_T:bool=False, visualActivityPath:str=None, bucket:str="ObsHiveABC", whole_hive:bool=False, verbose:bool=False)-> pd.DataFrame:
    """
    Download the dataset for a given hive number, either for a single in-hive location (ihl) or
    for the whole hive (both ihls combined).

    Parameters:
    - hive_nb (int): Hive number.
    - resolution (int): Resolution in seconds.
    - start_ts (pd.Timestamp): Start timestamp for the data.
    - end_ts (pd.Timestamp): End timestamp for the data.
    - ihl (str): In-hive location, either "upper" or "lower". Required when whole_hive is False;
      must be left as None when whole_hive is True.
    - only_amb_T (bool): If True, only return ambient temperature data (single "Tamb" column).
    - visualActivityPath (str): Path to visual activity data. Will not be used if None.
    - bucket (str): InfluxDB bucket to download the data from (e.g. "ObsHiveABC", "winter_exp", "a_sensing"). See MetabolicExp.buckets_by_type.
    - whole_hive (bool): If True, combine both ihls into a single hive-level dataset: ambient
      temperature is the lower of both ihls' ambient temperature, relative humidity is averaged
      across both ihls, activity uses the whole-hive aggregate instead of a per-ihl one, and all
      4 CO2 sensors (co2_UL/co2_UR/co2_LL/co2_LR) are kept instead of 2. When only_amb_T is False,
      the two ihls' 64 temperature sensors are concatenated into 128 columns (t000-t063 = upper,
      t064-t127 = lower).
    - verbose (bool): If True, print verbose output.
    """
    assert hive_nb in [1, 2], "hive_num must be either 1 or 2"
    if whole_hive:
        assert ihl is None, "ihl must not be specified when whole_hive=True"
    else:
        assert ihl in ["upper", "lower"], "inhive_loc must be either 'upper' or 'lower'"

    filters = {
        "hive_num" : hive_nb,
        "inhive_loc" : "none" if whole_hive else ihl,
    }
    tmp_result = download_tmp_DB(bucket, start_ts, end_ts, resolution=resolution, filters=filters, aggr="last")

    if whole_hive:
        upper_tmp, lower_tmp = tmp_result
        if only_amb_T:
            amb_upper = extractAmbientTemp(upper_tmp)
            amb_lower = extractAmbientTemp(lower_tmp)
            # Whole-hive ambient temperature is the lower of the two ihls' estimates.
            df = pd.concat([amb_upper, amb_lower], axis=1).min(axis=1).to_frame(name="Tamb")
        else:
            upper_tmp = upper_tmp.rename(columns={c: f"t{i:03d}" for i, c in enumerate(upper_tmp.columns)})
            lower_tmp = lower_tmp.rename(columns={c: f"t{i + 64:03d}" for i, c in enumerate(lower_tmp.columns)})
            df = pd.concat([upper_tmp, lower_tmp], axis=1)
    else:
        df = tmp_result
        if only_amb_T:
            df = extractAmbientTemp(df)
            # Convert pd.Series to pd.Dataframe
            df = df.to_frame(name="Tamb")

    # remove "inhive_loc" key from filters
    filters.pop("inhive_loc")
    co2_data = download_co2_DB(bucket, start_ts, end_ts, resolution = resolution, filters = filters)
    if whole_hive:
        # Keep all 4 sensors (ul, ur, ll, lr) instead of subsetting to one ihl.
        for col in co2_data.columns:
            df[f"co2_{col.upper()}"] = co2_data.loc[:, col]  # co2_UL, co2_UR, co2_LL, co2_LR
    else:
        # Keep only the columns that start with the same letter as the first letter of ihl (i.e. "u" for "upper" and "l" for "lower")
        co2_data = co2_data.loc[:, co2_data.columns.str.startswith(ihl[0])]
        for col in co2_data.columns:
            col_name = f"co2_{col[-1].upper()}"
            df[col_name] = co2_data.loc[:,col] # This should add 2 columns: (ul and ur) OR (ll and lr)

    # Add a _field tag and _measurement tag to filters
    filters["field"] = ["rel_humid"]
    filters["measurement"] = ["co2", "rht"]
    if not whole_hive:
        filters["inhive_loc"] = ihl
    # When whole_hive, "inhive_loc" is left out of filters entirely so both ihls' readings come
    # back, and the per-timestamp .mean() below averages across both automatically.
    humid_data = download_data_DB(bucket, start_ts, end_ts, resolution=resolution, filters=filters)

    # For every ts in df, there are several ts in humid_data. We want to take the average of the values in humid_data for each ts in df_resampled and store it in the "rel_humid" column of df_resampled.
    df["rel_humid"] = df.index.to_series().apply(lambda ts: humid_data.loc[humid_data.index == ts, "_value"].mean())

    recovery_time = 240 # minutes
    if visualActivityPath is not None:
        assert os.path.isdir(visualActivityPath), f"Visual activity path {visualActivityPath} is not a directory."
        assert visualActivityPath.endswith("Images/"), f"Visual activity path {visualActivityPath} must end with 'Images/'."
        
        # Convert resolution from seconds to a Timedelta for easier handling of time intervals
        t_res = pd.to_timedelta(resolution, unit='s')

        # Get the target dt (for which we need an image, data, etc.)
        datetimes = pd.date_range(start_ts, end_ts, freq=t_res)
        datetimes = datetimes.to_list()
        if verbose:
            print(f"Number of timestamps considered: {len(datetimes)}")

        imgs_paths = fetchImagesPaths(visualActivityPath, datetimes, hive_nb, recovery_time, verbose=verbose)

        # Checking if any ts is invalid:
        invalid_ts = imgs_paths[imgs_paths['valid'] == False]
        if not invalid_ts.empty:
            print(f"{len(invalid_ts)} invalid timestamp(s) found out of {len(imgs_paths)} total timestamps.")

        # computeRpiActivities() only compares consecutive rows and doesn't care about validity
        # itself, so dropping ALL invalid timestamps would end up pairing up rows that aren't
        # actually temporally adjacent. Instead, we keep invalid timestamps that immediately
        # precede a valid one, since they are needed as the "before" reference to compute the
        # activity of the first valid timestamp following an invalid stretch. Other invalid
        # timestamps (whose own resulting activity would itself be invalid) are dropped.
        valid = imgs_paths['valid'].to_numpy()
        keep = valid.copy()
        keep[:-1] |= (~valid[:-1]) & valid[1:]
        imgs_paths = imgs_paths[keep].drop(columns=['valid'])
        if imgs_paths.empty:
            raise ValueError("No valid images found for the specified timestamps.")

        if imgs_paths.isnull().values.any():
            # Remove rows with None values
            imgs_paths = imgs_paths.dropna()
        
        # Split the remaining rows into chunks that are contiguous in time (i.e., with no
        # dropped timestamp in between) and run computeRpiActivities() separately on each chunk,
        # then concatenate the results. This is needed because computeRpiActivities() pairs up
        # consecutive rows of the DataFrame and assumes they are also temporally consecutive.
        chunk_ids = (imgs_paths.index.to_series().diff() != t_res).cumsum()
        chunks = [chunk for _, chunk in imgs_paths.groupby(chunk_ids)]

        # Progress is tracked in image-pairs (rather than chunks) since chunk sizes vary a lot,
        # which makes the tqdm ETA meaningful instead of jumping around with every chunk.
        total_pairs = sum(max(len(chunk) - 1, 0) for chunk in chunks)

        RpiActivities = []
        with tqdm(total=total_pairs, desc="Computing visual activity", unit="pair") as pbar:
            for chunk in chunks:
                if len(chunk) < 2:
                    continue  # Need at least 2 rows to compute an activity
                # pbar is advanced per image-pair inside computeRpiActivities (via the dask
                # distributed client) rather than once per chunk here.
                chunk_activities, _ = computeRpiActivities(chunk, pbar=pbar)
                RpiActivities.extend(chunk_activities)

        for _act in RpiActivities:
            df.loc[_act.ts,"activity"] = _act.hive_activity if whole_hive else _act.ihl_activity[ihl]

    
    # Filter out timestamps not allowed by HiveOpenings
    from Metabolism.VisualActivity.RHCVisualisation.RHCThermalPlots.RHCImaging.HiveOpenings.libOpenings import filter_timestamps
    print("Before filtering with HiveOpenings:", len(df), "lines")
    filtered_ts = filter_timestamps(df.index.to_list(), hive_nb=hive_nb, recovery_time=recovery_time)
    df_resampled = df[df.index.isin(filtered_ts)]
    print("After  :", len(df_resampled), "lines")

    return df_resampled

def reconstruct_whole_hive(upper_df:pd.DataFrame, lower_df:pd.DataFrame, only_amb_T:bool=False)-> pd.DataFrame:
    """
    Reconstruct a whole-hive dataset (as produced by download_dataset(..., whole_hive=True)) from
    two already-downloaded/cached single-ihl datasets, without re-hitting InfluxDB. Meant to let
    a whole_hive=True run reuse per-ihl pickles that were already cached from an earlier
    whole_hive=False run instead of re-downloading everything.

    Temperature, CO2, activity, and relative humidity are all reconstructed exactly -- they're
    arithmetic combinations of the same raw per-sensor/per-rpi values download_dataset(whole_hive=True)
    itself computes (see RpisActivity in libActivity.py for why averaging the two ihls' activity
    columns exactly reconstructs hive_activity). Averaging the two ihls' already per-ihl-averaged
    "rel_humid" columns exactly reconstructs the whole-hive average too, since both ihls have the
    same number of underlying humidity sensor readings per timestamp.

    :param upper_df: DataFrame for ihl="upper", as returned/cached by download_dataset.
    :param lower_df: DataFrame for ihl="lower", as returned/cached by download_dataset.
    :param only_amb_T: Must match the only_amb_T the two input DataFrames were downloaded with.
    :return: DataFrame with whole-hive columns (see download_dataset's whole_hive docstring).
    """
    if only_amb_T:
        assert "Tamb" in upper_df.columns and "Tamb" in lower_df.columns, \
            "only_amb_T=True requires both dataframes to have a 'Tamb' column."
        # Whole-hive ambient temperature is the lower of the two ihls' estimates (see download_dataset).
        temp_part = pd.concat([upper_df["Tamb"], lower_df["Tamb"]], axis=1).min(axis=1).to_frame(name="Tamb")
    else:
        upper_temp_cols = sorted(c for c in upper_df.columns if c.startswith("t") and c[1:].isdigit())
        lower_temp_cols = sorted(c for c in lower_df.columns if c.startswith("t") and c[1:].isdigit())
        assert len(upper_temp_cols) == 64 and len(lower_temp_cols) == 64, \
            "only_amb_T=False requires both dataframes to have all 64 temperature sensor columns."
        upper_temp = upper_df[upper_temp_cols].rename(columns={c: f"t{i:03d}" for i, c in enumerate(upper_temp_cols)})
        lower_temp = lower_df[lower_temp_cols].rename(columns={c: f"t{i + 64:03d}" for i, c in enumerate(lower_temp_cols)})
        temp_part = pd.concat([upper_temp, lower_temp], axis=1)

    assert {"co2_L", "co2_R"}.issubset(upper_df.columns) and {"co2_L", "co2_R"}.issubset(lower_df.columns), \
        "Both dataframes must have 'co2_L'/'co2_R' columns."
    # upper's co2_L/co2_R are the raw "ul"/"ur" sensors, lower's are "ll"/"lr" (see download_dataset).
    co2_part = pd.DataFrame({
        "co2_UL": upper_df["co2_L"],
        "co2_UR": upper_df["co2_R"],
        "co2_LL": lower_df["co2_L"],
        "co2_LR": lower_df["co2_R"],
    })

    assert "rel_humid" in upper_df.columns and "rel_humid" in lower_df.columns, \
        "Both dataframes must have a 'rel_humid' column."
    # Both ihls have the same number of humidity sensor readings per timestamp, so averaging
    # their already per-ihl-averaged "rel_humid" columns exactly reconstructs the whole-hive
    # average (same equal-group-size argument as hive_activity below).
    humid_part = ((upper_df["rel_humid"] + lower_df["rel_humid"]) / 2).to_frame(name="rel_humid")

    df = pd.concat([temp_part, co2_part, humid_part], axis=1)

    if "activity" in upper_df.columns and "activity" in lower_df.columns:
        # hive_activity is the unweighted mean of all 4 rpis' raw activity values, and
        # ihl_activity[ihl] is the mean of exactly 2 of those 4 (see RpisActivity._aggregateActivity
        # / _aggregateHiveActivity in libActivity.py). Since both groups have equal size, averaging
        # the two ihls' activity columns reconstructs hive_activity exactly, NaNs included.
        df["activity"] = (upper_df["activity"] + lower_df["activity"]) / 2

    return df

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

    # y ticks: one label per row (region), plus the cannot-classify row. Previously capped at
    # 12 ticks via np.linspace subsampling, which skipped whichever region numbers didn't land
    # on a sampled row (e.g. regions 4, 8, 12, 15 out of 16 rows) -- not because those regions
    # were empty, but purely as an artifact of the even sampling. Region counts here are small
    # enough (tens, not hundreds) that labeling every row is fine.
    y_pos = np.arange(nrows)
    y_labels = [f"Region {i+1}" if i < nregions else "cannot classify" for i in y_pos]
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