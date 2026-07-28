"""Reusable building blocks for interpreting a UMAP/watershed project: given a point in UMAP
space, find its nearest neighbors (pooled across every dataset in the project) and recover the
raw sensor data underlying them.

Kept separate from umap_video_utils.py because that module is scoped to the video-rendering
pipeline (DATASET_CONFIGS, frame composition); this one is scoped to point-level interpretation
and has no dependency on video rendering. Both can be imported side by side.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import hdf5storage
import numpy as np
import pandas as pd

from umap_video_utils import load_watershed_artifacts

PathLike = Union[str, Path]


def discover_projects(results_dir: PathLike = "Results") -> List[Path]:
    """Return project directories under results_dir that look complete: each must contain both
    UMAP/zVals_wShed_groups.mat and a Projections/ subdirectory."""
    results_path = Path(results_dir)
    if not results_path.is_dir():
        return []
    projects = []
    for candidate in sorted(results_path.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "UMAP" / "zVals_wShed_groups.mat").exists() and (candidate / "Projections").is_dir():
            projects.append(candidate)
    return projects


@dataclass
class ProjectIndex:
    """Everything needed to draw the density map and search for nearest neighbors across every
    dataset pooled into one project's watershed embedding."""

    project_dir: Path
    density: np.ndarray
    extent: Tuple[float, float, float, float]
    wbounds: List[np.ndarray]
    pooled_xy: np.ndarray                # (N, 2) float
    pooled_source_id: np.ndarray         # (N,) dtype=object, str
    pooled_timestamp: np.ndarray         # (N,) dtype=object, pd.Timestamp (or pd.NaT)
    pooled_local_index: np.ndarray       # (N,) int -- position within that dataset's own kept/
                                          # aligned block, i.e. the row to index into that
                                          # dataset's waveletAmplitudes/keptIdx-ordered arrays
    dataset_status: Dict[str, str] = field(default_factory=dict)  # source_id -> "ok" or a skip reason


def _strip_pcamodes_suffix(name: str) -> str:
    return name[: -len("_pcaModes")] if name.endswith("_pcaModes") else name


def load_project_index(project_dir: PathLike) -> ProjectIndex:
    """Load a project's watershed artifacts and build pooled (xy, source_id, timestamp) arrays
    covering every point in the concatenated embedding.

    Uses zValNames/zValLens from the watershed file (not a re-sorted file listing) to recover the
    true per-dataset (offset, length) blocks within the pooled zValues/watershedRegions arrays --
    the same technique used in Ethogram_Generation_Metabolism.ipynb's region-loading cell, needed
    because mmpy.findWatershedRegions concatenates datasets in whatever order glob.glob() returns
    them, not alphabetical or insertion order.
    """
    project_dir = Path(project_dir)
    wshed_path = project_dir / "UMAP" / "zVals_wShed_groups.mat"
    projections_dir = project_dir / "Projections"

    artifacts = load_watershed_artifacts(str(wshed_path))
    wshedfile = hdf5storage.loadmat(str(wshed_path))
    pooled_xy = np.asarray(wshedfile["zValues"], dtype=float)

    zval_names = [str(np.asarray(n).ravel()[0]) for n in wshedfile["zValNames"].flatten()]
    zval_lens = wshedfile["zValLens"].flatten().astype(int)
    zval_offsets = np.cumsum([0] + zval_lens[:-1].tolist())

    n_total = pooled_xy.shape[0]
    pooled_source_id = np.empty(n_total, dtype=object)
    pooled_timestamp = np.full(n_total, pd.NaT, dtype=object)
    pooled_local_index = np.full(n_total, -1, dtype=int)
    dataset_status: Dict[str, str] = {}

    for name, length, offset in zip(zval_names, zval_lens, zval_offsets):
        source_id = _strip_pcamodes_suffix(name)
        pooled_source_id[offset : offset + length] = source_id
        pooled_local_index[offset : offset + length] = np.arange(length)

        stats_path = projections_dir / f"{source_id}_pcaModes_uVals_outputStatistics.pkl"
        pca_path = projections_dir / f"{source_id}_pcaModes.mat"
        if not stats_path.exists() or not pca_path.exists():
            dataset_status[source_id] = f"missing projection files ({stats_path.name} / {pca_path.name})"
            continue

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        kept_idx = np.asarray(stats["keptIdx"]).astype(int).ravel()

        pca_data = hdf5storage.loadmat(str(pca_path))
        real_ts = pd.to_datetime(pca_data["real_timestamps"], utc=True)
        real_ts = real_ts[kept_idx]

        if len(real_ts) != length:
            raise ValueError(
                f"keptIdx length ({len(real_ts)}) doesn't match the watershed block length "
                f"({length}) for {source_id}; the wshed file may be stale relative to "
                f"{projections_dir}."
            )

        pooled_timestamp[offset : offset + length] = np.asarray(real_ts)
        dataset_status[source_id] = "ok"

    return ProjectIndex(
        project_dir=project_dir,
        density=artifacts["density"],
        extent=artifacts["extent"],
        wbounds=artifacts["wbounds"],
        pooled_xy=pooled_xy,
        pooled_source_id=pooled_source_id,
        pooled_timestamp=pooled_timestamp,
        pooled_local_index=pooled_local_index,
        dataset_status=dataset_status,
    )


def find_nearest_neighbors(index: ProjectIndex, click_xy: Sequence[float], k: int = 20) -> pd.DataFrame:
    """Return the k pooled points nearest to click_xy (Euclidean, in UMAP space), as a DataFrame
    with columns source_id/timestamp/distance/umap_x/umap_y/local_index, sorted by ascending
    distance. local_index is each point's row position within its own dataset's kept/aligned
    block (i.e. the row to use against that dataset's waveletAmplitudes array).

    Plain argsort over all pooled points -- fine at this scale (tens of thousands of points);
    swap in scipy.spatial.cKDTree if a project ever grows large enough for this to matter.
    """
    click_xy = np.asarray(click_xy, dtype=float).reshape(2)
    dists = np.linalg.norm(index.pooled_xy - click_xy[None, :], axis=1)
    order = np.argsort(dists)[:k]
    return pd.DataFrame(
        {
            "source_id": index.pooled_source_id[order],
            "timestamp": index.pooled_timestamp[order],
            "distance": dists[order],
            "umap_x": index.pooled_xy[order, 0],
            "umap_y": index.pooled_xy[order, 1],
            "local_index": index.pooled_local_index[order],
        }
    )


_raw_df_cache: Dict[str, Optional[pd.DataFrame]] = {}


def load_raw_sensor_df(source_id: str, intermediate_dir: PathLike = "Results/Intermediate_Results") -> Optional[pd.DataFrame]:
    """Load (and cache) the raw per-dataset sensor DataFrame saved at
    {intermediate_dir}/{source_id}.pkl. Returns None if it doesn't exist -- e.g. Observation_*/
    Idle_* datasets, which haven't been split out this way (yet)."""
    cache_key = f"{Path(intermediate_dir)}::{source_id}"
    if cache_key in _raw_df_cache:
        return _raw_df_cache[cache_key]
    path = Path(intermediate_dir) / f"{source_id}.pkl"
    if not path.exists():
        _raw_df_cache[cache_key] = None
        return None
    with open(path, "rb") as f:
        df = pickle.load(f)
    _raw_df_cache[cache_key] = df
    return df


_wavelet_data_cache: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}


def _load_wavelet_data(source_id: str, projections_dir: PathLike) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load (waveletAmplitudes, waveletFrequencies) for a dataset from its
    outputStatistics.pkl, if that field was saved -- requires findEmbeddings(...,
    saveWaveletAmps=True) to have been (re-)run for this dataset. Returns (None, None) if either
    the file or the field is missing."""
    cache_key = f"{Path(projections_dir)}::{source_id}"
    if cache_key in _wavelet_data_cache:
        return _wavelet_data_cache[cache_key]

    result = (None, None)
    stats_path = Path(projections_dir) / f"{source_id}_pcaModes_uVals_outputStatistics.pkl"
    if stats_path.exists():
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        amps = stats.get("waveletAmplitudes") if hasattr(stats, "get") else None
        freqs = stats.get("waveletFrequencies") if hasattr(stats, "get") else None
        if amps is not None and freqs is not None:
            result = (np.asarray(amps), np.asarray(freqs))

    _wavelet_data_cache[cache_key] = result
    return result


def _characteristic_period_minutes(amplitude_row: np.ndarray, frequencies: np.ndarray) -> float:
    """Amplitude-weighted average period (in minutes) for one point's wavelet amplitude row,
    summed across PCA modes first. amplitude_row is the flat (pcaModes*numPeriods,) row for this
    point; frequencies is the (numPeriods,) frequency axis (Hz) it was computed against."""
    num_periods = frequencies.shape[0]
    amps_by_mode = amplitude_row.reshape(-1, num_periods)  # (pcaModes, numPeriods), mode-major
    amp_per_period = amps_by_mode.sum(axis=0)              # total amplitude at each period
    total = amp_per_period.sum()
    if total <= 0:
        return float("nan")
    periods_minutes = (1.0 / frequencies) / 60.0
    return float(np.sum(periods_minutes * amp_per_period) / total)


def derive_neighbor_windows(
    neighbors: pd.DataFrame,
    project_dir: PathLike,
    default_window_minutes: float = 60,
) -> Tuple[float, pd.DataFrame]:
    """For each neighbor, derive its own characteristic window half-width (minutes) from its
    wavelet amplitude spectrum: the amplitude-weighted average period, amplitudes summed across
    PCA modes first. This is the pipeline's own notion of "how much time this point's
    representation actually spans", used instead of an arbitrary fixed window.

    Falls back to default_window_minutes for neighbors whose dataset doesn't have
    waveletAmplitudes saved yet (needs findEmbeddings(..., saveWaveletAmps=True) re-run for that
    dataset) or whose spectrum has zero total amplitude.

    Returns (shared_window_minutes, detail_df): shared_window_minutes is the *average* of the
    per-neighbor windows -- so every neighbor gets pulled with that one shared window rather than
    each contributing a different, ragged span -- and detail_df has one row per neighbor with its
    own derived minutes and how it was obtained (columns: source_id, window_minutes, source
    ["wavelet" or "fallback"], reason).
    """
    projections_dir = Path(project_dir) / "Projections"
    rows = []
    for _, row in neighbors.iterrows():
        source_id = row["source_id"]
        local_index = row.get("local_index", -1)
        amps, freqs = _load_wavelet_data(source_id, projections_dir)

        if amps is None or freqs is None or local_index is None or local_index < 0:
            rows.append({
                "source_id": source_id, "window_minutes": default_window_minutes,
                "source": "fallback", "reason": "no waveletAmplitudes saved for this dataset yet",
            })
            continue

        minutes = _characteristic_period_minutes(amps[int(local_index)], freqs)
        if not np.isfinite(minutes):
            rows.append({
                "source_id": source_id, "window_minutes": default_window_minutes,
                "source": "fallback", "reason": "zero-amplitude wavelet spectrum",
            })
            continue

        rows.append({"source_id": source_id, "window_minutes": minutes, "source": "wavelet", "reason": ""})

    detail_df = pd.DataFrame(rows)
    shared_window_minutes = float(detail_df["window_minutes"].mean()) if not detail_df.empty else float(default_window_minutes)
    return shared_window_minutes, detail_df


def average_neighbor_traces(
    neighbors: pd.DataFrame,
    window_minutes: int = 60,
    intermediate_dir: PathLike = "Results/Intermediate_Results",
) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """For each neighbor row (source_id, timestamp), pull a +/-window_minutes slice of its raw
    sensor data around its own timestamp, re-index it onto a relative-time axis (offset from that
    neighbor's timestamp), and average across neighbors per relative-time bin.

    Neighbors are skipped (not silently dropped -- recorded in the returned report) when their
    dataset has no cached raw sensor pickle, or has no samples in the requested window.

    Returns (averaged_df, report) where averaged_df is indexed by relative offset (pd.Timedelta,
    rounded to the nearest minute so datasets/resolutions still line up) with one column per
    sensor field common to all included neighbors, and report = {"included": [...], "skipped":
    [(source_id, reason), ...]}.
    """
    window = pd.Timedelta(minutes=window_minutes)
    windows = []
    included: List[str] = []
    skipped: List[Tuple[str, str]] = []

    for _, row in neighbors.iterrows():
        source_id = row["source_id"]
        ts = row["timestamp"]
        if pd.isna(ts):
            skipped.append((source_id, "missing timestamp"))
            continue

        raw_df = load_raw_sensor_df(source_id, intermediate_dir)
        if raw_df is None:
            skipped.append((source_id, "no cached raw sensor data"))
            continue

        sub = raw_df.loc[(raw_df.index >= ts - window) & (raw_df.index <= ts + window)]
        if sub.empty:
            skipped.append((source_id, "no raw samples in window"))
            continue

        sub = sub.select_dtypes(include="number").copy()
        sub.index = sub.index - ts  # relative offset from this neighbor's own timestamp
        windows.append(sub)
        included.append(source_id)

    report = {"included": included, "skipped": skipped}
    if not windows:
        return pd.DataFrame(), report

    common_cols = set(windows[0].columns)
    for w in windows[1:]:
        common_cols &= set(w.columns)
    common_cols = sorted(common_cols)

    aligned = []
    for w in windows:
        w = w[common_cols].copy()
        # Round relative offsets to the nearest minute so neighbors from datasets at slightly
        # different resolutions still land on a shared grid for averaging.
        w.index = pd.to_timedelta(np.round(w.index.total_seconds() / 60.0) * 60, unit="s")
        aligned.append(w)

    combined = pd.concat(aligned, axis=0)
    averaged = combined.groupby(combined.index).mean(numeric_only=True).sort_index()

    return averaged, report
