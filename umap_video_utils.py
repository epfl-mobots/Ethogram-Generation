"""Helpers for generating videos that pair a hive dataset view with a UMAP density map."""

import os, pickle, glob, subprocess, hdf5storage
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

_RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


DEFAULT_EXCLUDE_COLUMNS = {
    "region",
    "source_id",
    "real_timestamp",
    "timestamp",
    "UMAP",
    "UMAP_0",
    "UMAP_1",
}

def _normalize_optional_vector(vector):
    if vector is None:
        return None
    array = np.asarray(vector).squeeze()
    if array.ndim == 0:
        array = array.reshape(1)
    return array

def advance_umap_state(timeline: pd.DataFrame, cursor: int, current_point, current_timestamp, video_timestamp):
    timestamp_column = "real_timestamps"
    while cursor < len(timeline) and pd.Timestamp(timeline.loc[cursor, timestamp_column]) <= pd.Timestamp(video_timestamp):
        current_row = timeline.iloc[cursor]
        current_point = np.array([current_row["umap_x"], current_row["umap_y"]], dtype=float)
        current_timestamp = current_row[timestamp_column]
        cursor += 1
    return cursor, current_point, current_timestamp

def load_projection_timeline(dataset_cfg: Dict, projections_dir: Path, dataset_name: str) -> pd.DataFrame:
    '''
    Loads the projection timeline for a given dataset name.
    The timestamps correspond to UMAP data datetimes and include the datetimes that are not in the coordinates. 
    '''
    dataset_config = dataset_cfg[dataset_name]
    projection_source_id = dataset_config.get("projection_source_id")
    projection_file = projections_dir / "observation_OH_pcaModes.mat"
    if not projection_file.exists():
        raise FileNotFoundError(f"No projection metadata file found at {projection_file}")
    projection_payload = hdf5storage.loadmat(str(projection_file))
    timestamps = _normalize_optional_vector(projection_payload.get("real_timestamps"))
    if timestamps is None:
        raise ValueError("Projection metadata must contain 'real_timestamp' for canonical timeline")

    source_ids = _normalize_optional_vector(projection_payload.get("source_id"))
    if timestamps is None:
        raise ValueError("Projection metadata does not contain timestamps")
    timeline = pd.DataFrame({
        "real_timestamps": pd.to_datetime(np.asarray(timestamps).squeeze(), utc=True),
    })
    if source_ids is not None:
        timeline["projection_source_id"] = pd.to_numeric(np.asarray(source_ids).squeeze(), errors="coerce").astype("Int64")
    if projection_source_id is not None and "projection_source_id" in timeline.columns:
        timeline = timeline[timeline["projection_source_id"] == projection_source_id].copy()
    timeline = timeline.sort_values("real_timestamps").reset_index(drop=True)
    return timeline

def load_umap_data(dataset_cfg: Dict, projections_dir: Path, dataset_name: str) -> pd.DataFrame:
    projection_source_id = dataset_cfg[dataset_name].get("projection_source_id")
    print(f"Loading UMAP data for dataset '{dataset_name}' with projection_source_id={projection_source_id} from {projections_dir}")
    metadata = load_umap_projection_metadata(projections_dir, projection_source_id)

    umap_coords = metadata.get("coordinates")
    all_data_ts = metadata.get("real_timestamps")
    # Print out the unique value counts of all_data_source_ids
    kept_idx = metadata.get("keptIdx")
    umap_df = pd.DataFrame({"umap_x": umap_coords[:, 0], "umap_y": umap_coords[:, 1]})
    umap_df["real_timestamps"] = all_data_ts[kept_idx]

    umap_df.sort_values("real_timestamps", inplace=True)
    umap_df.reset_index(drop=True, inplace=True)

    return umap_df


def _normalize_wbounds(raw_bounds) -> List[np.ndarray]:
    if raw_bounds is None:
        return []
    if isinstance(raw_bounds, np.ndarray) and raw_bounds.dtype == object:
        out = []
        for item in raw_bounds.ravel():
            out.extend(_normalize_wbounds(item))
        return out
    if isinstance(raw_bounds, (list, tuple)):
        out = []
        for item in raw_bounds:
            out.extend(_normalize_wbounds(item))
        return out
    raw_array = np.asarray(raw_bounds)
    if raw_array.size == 0:
        return []
    if raw_array.ndim == 2 and raw_array.shape[1] == 2:
        return [raw_array]
    if raw_array.ndim == 3 and raw_array.shape[-1] == 2:
        return [raw_array[i] for i in range(raw_array.shape[0])]
    if raw_array.ndim == 2 and raw_array.shape == (1, 2):
        return [raw_array]
    if raw_array.dtype == object:
        return _normalize_wbounds(raw_array.tolist())
    return []


def _extent_from_xx(xx: np.ndarray, density: np.ndarray) -> Tuple[float, float, float, float]:
    xx_array = np.asarray(xx).squeeze()
    if xx_array.ndim == 1 and xx_array.size >= 2:
        return float(xx_array.min()), float(xx_array.max()), float(xx_array.min()), float(xx_array.max())
    if xx_array.ndim == 2 and xx_array.shape[1] >= 2:
        return float(xx_array[:, 0].min()), float(xx_array[:, 0].max()), float(xx_array[:, 1].min()), float(xx_array[:, 1].max())
    height, width = density.shape[:2]
    return 0.0, float(width - 1), 0.0, float(height - 1)


def load_watershed_artifacts(wshed_path: str) -> Dict[str, object]:
    """Load the saved watershed outputs used for the UMAP background."""
    wshed = hdf5storage.loadmat(wshed_path)
    density = np.asarray(wshed["density"])
    xx = np.asarray(wshed["xx"])
    wbounds = _normalize_wbounds(wshed.get("wbounds"))
    return {
        "density": density,
        "xx": xx,
        "extent": _extent_from_xx(xx, density),
        "wbounds": wbounds,
        "watershed_regions": np.asarray(wshed.get("watershedRegions", [])).squeeze(),
    }


def fig_to_rgb_array(fig, rgb: bool = True) -> np.ndarray:
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    if rgb:
        return buffer
    return buffer[..., ::-1]


def cropFrameToContent(frame: np.ndarray, padding: int = 0) -> np.ndarray:
    assert frame.ndim == 3 and frame.shape[2] == 3, "frame must be RGB/BGR"
    assert padding >= 0, "padding must be non-negative"

    content = np.any(frame < 250, axis=2)
    y, x = np.where(content)
    if y.size == 0 or x.size == 0:
        return frame

    miny, maxy = max(0, y.min() - padding), min(frame.shape[0], y.max() + padding)
    minx, maxx = max(0, x.min() - padding), min(frame.shape[1], x.max() + padding)
    return frame[miny:maxy, minx:maxx]


def generateVideoFromList(imgs: list, dest, name: str = "video", fps: int = 10, grayscale: bool = True):
    if not os.path.isdir(dest):
        os.makedirs(dest)
    if len(imgs) == 0:
        raise ValueError("imgs must be a non-empty list")

    name = str(dest) + "/" + name + ".mp4"
    base_name, ext = os.path.splitext(name)
    counter = 1
    while os.path.isfile(name):
        name = f"{base_name}_{counter}{ext}"
        counter += 1

    with imageio.get_writer(name, fps=fps) as writer:
        for frame in imgs:
            frame_array = np.asarray(frame)
            if frame_array.ndim == 2:
                frame_array = np.repeat(frame_array[:, :, np.newaxis], 3, axis=2)
            writer.append_data(frame_array.astype(np.uint8))


def generateVideoFromFrames(frames, dest, name: str = "video", fps: int = 10):
    if not os.path.isdir(dest):
        os.makedirs(dest)

    name = str(dest) + "/" + name + ".mp4"
    base_name, ext = os.path.splitext(name)
    counter = 1
    while os.path.isfile(name):
        name = f"{base_name}_{counter}{ext}"
        counter += 1

    with imageio.get_writer(name, fps=fps) as writer:
        for frame in frames:
            frame_array = np.asarray(frame)
            if frame_array.ndim == 2:
                frame_array = np.repeat(frame_array[:, :, np.newaxis], 3, axis=2)
            writer.append_data(frame_array.astype(np.uint8))

    return name


def infer_value_columns(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> List[str]:
    excluded = set(DEFAULT_EXCLUDE_COLUMNS)
    if exclude is not None:
        excluded.update(exclude)
    value_columns = []
    for column in df.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            value_columns.append(column)
    return value_columns


def extract_umap_points(df: pd.DataFrame, umap_col: str = "UMAP") -> np.ndarray:
    if umap_col in df.columns:
        values = df[umap_col].apply(lambda value: np.asarray(value, dtype=float).reshape(-1))
        return np.vstack(values.to_list())
    candidate_columns = [column for column in df.columns if column.startswith(f"{umap_col}_")]
    if len(candidate_columns) >= 2:
        return df[candidate_columns[:2]].to_numpy(dtype=float)
    raise ValueError("No UMAP coordinates found in the dataframe")


def load_umap_points_from_projection_file(projection_file: str) -> np.ndarray:
    projection = hdf5storage.loadmat(projection_file)
    if "zValues" in projection:
        return np.asarray(projection["zValues"], dtype=float)
    if "uVals" in projection:
        return np.asarray(projection["uVals"], dtype=float)
    raise ValueError("Projection file does not contain zValues or uVals")


def load_umap_points(watershed_path: Path) -> np.ndarray:
    wshedfile = hdf5storage.loadmat(str(watershed_path))
    return wshedfile['zValues']


def load_umap_projection_metadata(projections_dir: Path, dataset_id: int) -> Dict[str, object]:
    candidate = sorted(glob.glob(str(projections_dir / f"*_{dataset_id}_pcaModes_uVals.mat")))[0]
    pcamodes_candidate = sorted(glob.glob(str(projections_dir / f"*_{dataset_id}_pcaModes.mat")))[0]
    output_statistics_candidate = sorted(glob.glob(str(projections_dir / f"*_{dataset_id}_pcaModes_uVals_outputStatistics.pkl")))
    if not output_statistics_candidate:
        output_statistics_candidate = sorted(glob.glob(str(projections_dir / f"*_{dataset_id}_pcaModes_zVals_outputStatistics.pkl")))
    output_statistics_candidate = output_statistics_candidate[0]

    projection = hdf5storage.loadmat(candidate)
    if "zValues" in projection:
        coordinates = np.asarray(projection["zValues"], dtype=float)
    elif "uVals" in projection:
        coordinates = np.asarray(projection["uVals"], dtype=float)
    else:
        raise ValueError("Projection file does not contain zValues or uVals")

    metadata: Dict[str, object] = {"coordinates": coordinates}

    with open(output_statistics_candidate, "rb") as hfile:
        output_statistics = pickle.load(hfile)
    metadata["keptIdx"] = output_statistics["keptIdx"]

    pcamode = hdf5storage.loadmat(pcamodes_candidate)
    metadata["real_timestamps"] = np.asarray(pcamode["real_timestamps"]).squeeze()

    # Change dtype of real_timestamps to datetime64[ns] with UTC timezone
    metadata["real_timestamps"] = pd.to_datetime(metadata["real_timestamps"], utc=True)

    return metadata


def build_barycenters(df: pd.DataFrame, region_col: str = "region", umap_col: str = "UMAP") -> Dict[int, np.ndarray]:
    if region_col not in df.columns:
        return {}
    points = extract_umap_points(df, umap_col=umap_col)
    regions = pd.to_numeric(df[region_col], errors="coerce").to_numpy()
    barycenters: Dict[int, np.ndarray] = {}
    for region_id in sorted({int(value) for value in regions if not np.isnan(value)}):
        if region_id == 0:
            continue
        mask = regions == region_id
        if mask.any():
            barycenters[region_id] = points[mask].mean(axis=0)
    return barycenters


def render_snapshot_panel(
    row: pd.Series,
    value_columns: Sequence[str],
    *,
    title: str,
    timestamp=None,
    region=None,
    cmap: str = "viridis",
) -> np.ndarray:
    values = pd.to_numeric(row[list(value_columns)], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 3))
    image = ax.imshow(values[np.newaxis, :], aspect="auto", cmap=cmap)
    ax.set_yticks([])
    if len(value_columns) <= 24:
        step = max(1, len(value_columns) // 12)
        tick_positions = np.arange(0, len(value_columns), step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([value_columns[index] for index in tick_positions], rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
    title_parts = [title]
    if timestamp is not None:
        title_parts.append(str(timestamp))
    if region is not None and not pd.isna(region):
        title_parts.append(f"region {int(region)}")
    ax.set_title(" | ".join(title_parts))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    frame = fig_to_rgb_array(fig)
    plt.close(fig)
    return frame


def render_umap_panel(
    point: Sequence[float],
    density: np.ndarray,
    *,
    extent: Tuple[float, float, float, float],
    wbounds: Optional[Sequence[np.ndarray]] = None,
    barycenters: Optional[Dict[int, np.ndarray]] = None,
    title: str = "UMAP density map",
    timestamp=None,
    region=None,
    background_cmap: str = "magma",
    figsize: Tuple[float, float] = (8, 8),
    dpi: int = 180,
    title_fontsize: int = 16,
    axis_labelsize: int = 14,
    tick_labelsize: int = 12,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(density, origin="lower", cmap=background_cmap, extent=extent, aspect="auto")

    if wbounds:
        for boundary in wbounds:
            boundary = np.asarray(boundary, dtype=float)
            if boundary.ndim != 2 or boundary.shape[1] != 2:
                continue
            ax.plot(boundary[:, 0], boundary[:, 1], color="white", linewidth=0.8, alpha=0.8)

    if barycenters:
        bary_array = np.asarray(list(barycenters.values()), dtype=float)
        if bary_array.size:
            ax.scatter(bary_array[:, 0], bary_array[:, 1], marker="x", s=70, c="cyan", linewidths=2, label="Barycenters")

    if point is not None:
        point_array = np.asarray(point, dtype=float).reshape(-1)
        ax.scatter(point_array[0], point_array[1], c="red", s=90, edgecolors="black", linewidths=0.8, zorder=5)

    title_parts = [title]
    if timestamp is not None:
        title_parts.append(str(timestamp))
    if region is not None and not pd.isna(region):
        title_parts.append(f"region {int(region)}")
    ax.set_title(" | ".join(title_parts), fontsize=title_fontsize)
    ax.set_xlabel("UMAP 1", fontsize=axis_labelsize)
    ax.set_ylabel("UMAP 2", fontsize=axis_labelsize)
    ax.tick_params(axis="both", labelsize=tick_labelsize)
    if barycenters:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    frame = fig_to_rgb_array(fig)
    plt.close(fig)
    return frame


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = float(height) / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    return np.asarray(Image.fromarray(image).resize((width, height), _RESAMPLE_LANCZOS))


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    scale = float(width) / float(image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return np.asarray(Image.fromarray(image).resize((width, height), _RESAMPLE_LANCZOS))


def compose_panels(left: np.ndarray, right: np.ndarray, layout: str = "side_by_side") -> np.ndarray:
    if layout not in {"side_by_side", "top_bottom"}:
        raise ValueError("layout must be 'side_by_side' or 'top_bottom'")
    if layout == "side_by_side":
        target_height = max(left.shape[0], right.shape[0])
        left_resized = _resize_to_height(left, target_height)
        right_resized = _resize_to_height(right, target_height)
        return np.concatenate([left_resized, right_resized], axis=1)
    target_width = max(left.shape[1], right.shape[1])
    left_resized = _resize_to_width(left, target_width)
    right_resized = _resize_to_width(right, target_width)
    return np.concatenate([left_resized, right_resized], axis=0)


def generate_dataset_umap_video(
    df: pd.DataFrame,
    *,
    wshed_path: str,
    video_name: str,
    dest: str,
    layout: str = "side_by_side",
    source_label=None,
    source_col: str = "source_id",
    timestamp_col: str = "real_timestamp",
    region_col: str = "region",
    umap_col: str = "UMAP",
    value_columns: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    fps: int = 10,
    background_cmap: str = "magma",
    snapshot_cmap: str = "viridis",
) -> str:
    """Generate a video for one dataset/source."""
    frame_df = df.copy()
    if source_label is not None and source_col in frame_df.columns:
        frame_df = frame_df[frame_df[source_col] == source_label].copy()
    if frame_df.empty:
        raise ValueError("No rows available for the requested source")

    artifacts = load_watershed_artifacts(wshed_path)
    if umap_col in frame_df.columns or any(column.startswith(f"{umap_col}_") for column in frame_df.columns):
        points = extract_umap_points(frame_df, umap_col=umap_col)
    else:
        if source_label is None:
            if source_col in frame_df.columns:
                source_label = str(frame_df[source_col].iloc[0])
            else:
                source_label = str(video_name)
        points = load_umap_points(Path(wshed_path))
    if value_columns is None:
        value_columns = infer_value_columns(frame_df)
    if not value_columns:
        raise ValueError("No numeric value columns found to render the dataset panel")

    barycenters = build_barycenters(frame_df, region_col=region_col, umap_col=umap_col)

    frames = []
    for index, (_, row) in enumerate(frame_df.iterrows()):
        timestamp = row[timestamp_col]
        region = row[region_col] if region_col in frame_df.columns else None
        snapshot_title = title or (str(source_label) if source_label is not None else "Dataset snapshot")
        left_panel = render_snapshot_panel(
            row,
            value_columns,
            title=snapshot_title,
            timestamp=timestamp,
            region=region,
            cmap=snapshot_cmap,
        )
        right_panel = render_umap_panel(
            points[index],
            artifacts["density"],
            extent=artifacts["extent"],
            wbounds=artifacts["wbounds"],
            barycenters=barycenters,
            title="UMAP density map",
            timestamp=timestamp,
            region=region,
            background_cmap=background_cmap,
        )
        frames.append(compose_panels(left_panel, right_panel, layout=layout))

    generateVideoFromList(frames, dest=dest, name=video_name, fps=fps, grayscale=False)
    return os.path.join(dest, video_name + ".mp4")


def generate_videos_for_sources(
    df: pd.DataFrame,
    *,
    wshed_path: str,
    dest: str,
    video_prefix: str = "umap_video",
    source_col: str = "source_id",
    layout: str = "side_by_side",
    fps: int = 10,
    **kwargs,
) -> Dict[str, str]:
    """Generate one video per source value."""
    if source_col not in df.columns:
        raise ValueError("source_col is missing from the dataframe")
    outputs: Dict[str, str] = {}
    for source_label in sorted(df[source_col].dropna().unique().tolist()):
        source_name = str(source_label)
        video_name = f"{video_prefix}_{source_name}"
        outputs[source_name] = generate_dataset_umap_video(
            df,
            wshed_path=wshed_path,
            video_name=video_name,
            dest=dest,
            layout=layout,
            source_label=source_label,
            source_col=source_col,
            fps=fps,
            **kwargs,
        )
    return outputs

def resolve_macos_alias(path: str) -> Path:
    source_path = Path(path)
    if source_path.is_symlink():
        return source_path.resolve()
    apple_script = (
        'set a to POSIX file "{}" as alias\n'
        'tell application "Finder" to set b to original item of a\n'
        'return POSIX path of (b as alias)'
    ).format(str(source_path).replace('"', '\\"'))
    try:
        resolved = subprocess.check_output(["osascript", "-e", apple_script], universal_newlines=True).strip()
        if resolved:
            return Path(resolved)
    except Exception:
        pass
    return source_path

def blackout_unused_hive_half(dataset_frame: np.ndarray, ihl: str) -> np.ndarray:
    frame = np.asarray(dataset_frame).copy()
    if frame.ndim != 3 or frame.shape[0] < 2 or frame.shape[1] < 2:
        return frame

    original = frame.copy()
    height = frame.shape[0]
    width = frame.shape[1]
    midpoint = height // 2

    if ihl == "upper":
        frame[midpoint:, :, :] = 0
        ambient_x0 = int(width * (2700 / 3840))
        ambient_y0 = int(height * (2050 / 2160))
        frame[ambient_y0:height, ambient_x0:width, :] = original[ambient_y0:height, ambient_x0:width, :]
    elif ihl == "lower":
        frame[:midpoint, :, :] = 0
        timestamp_x0 = int(width * (1500 / 3840))
        timestamp_y0 = int(height * (980 / 2160))
        timestamp_x1 = int(width * (2350 / 3840))
        timestamp_y1 = int(height * (1165 / 2160))
        frame[timestamp_y0:timestamp_y1, timestamp_x0:timestamp_x1, :] = original[timestamp_y0:timestamp_y1, timestamp_x0:timestamp_x1, :]
    return frame