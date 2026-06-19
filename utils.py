import numpy as np
import motionmapperpy as mmpy
import matplotlib.pyplot as plt
import pandas as pd

def build_ethogram(w):
    nregions = int(w.max()) if len(w) else 0
    E = np.zeros((nregions, len(w)), dtype=np.uint8)
    if nregions > 0 and len(w) > 0:
        E[w - 1, np.arange(len(w))] = 1
    return E, nregions

def plot_ethogram(E, nregions, day_names, day_boundaries, nb_points_per_day, title):
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.imshow(
        E, aspect='auto',
        cmap=(mmpy.gencmap() if 'mmpy' in globals() else 'viridis'),
        origin='lower', interpolation='nearest'
    )
    n_yticks = min(12, max(nregions, 1))
    y_pos = np.linspace(0, max(nregions - 1, 0), n_yticks, dtype=int)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Region {i+1}" for i in y_pos])

    for b in day_boundaries[:-1]:
        ax.axvline(b, color='white', linestyle='--', linewidth=1)

    ticks_x, labels_x, start = [], [], 0
    for name, n in zip(day_names, nb_points_per_day):
        ticks_x.append(start + n // 2)
        labels_x.append(name)
        start += n
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(labels_x, rotation=45, ha='right')

    ax.set_xlabel('Days')
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