import numpy as np
from scipy.fftpack import dctn

def extract_thermal_dct_features(thermal_field: np.ndarray) -> np.ndarray:
    """
    Extract 8 spatial features from a thermal field using 2D DCT.
    
    Features:
    1. DC component (mean temperature)
    2. Low-frequency energy (top-left 2x2 block excluding DC)
    3. Horizontal gradient energy
    4. Vertical gradient energy
    5. Diagonal energy
    6. High-frequency energy
    7. Spectral centroid (frequency-weighted energy)
    8. Spatial variance proxy
    """

    x = np.asarray(thermal_field, dtype=float)

    # --- Normalize (important for stability) ---
    x = x - np.mean(x)

    # --- 2D DCT ---
    F = dctn(x, norm='ortho')

    H, W = F.shape

    # --- Feature 1: DC bias (mean temperature) ---
    dc = np.mean(thermal_field)

    # --- Feature 2: low-frequency energy (excluding DC) ---
    low_freq = F[:2, :2].copy()
    low_freq[0, 0] = 0
    low_energy = np.sum(low_freq**2)

    # --- Feature 3-5: directional energies ---
    horiz_energy = np.sum(F[0, 1:]**2)          # horizontal variation
    vert_energy = np.sum(F[1:, 0]**2)          # vertical variation
    diag_energy = np.sum(np.diag(F[1:, 1:])**2) if min(H, W) > 1 else 0

    # --- Feature 6: high frequency energy ---
    hf_mask = np.ones_like(F, dtype=bool)
    hf_mask[:H//4, :W//4] = False
    high_freq_energy = np.sum(F[hf_mask]**2)

    # --- Feature 7: spectral centroid ---
    u = np.arange(H)[:, None]
    v = np.arange(W)[None, :]
    freq_dist = np.sqrt(u**2 + v**2)

    energy = F**2
    spectral_centroid = np.sum(freq_dist * energy) / (np.sum(energy) + 1e-8)

    # --- Feature 8: spatial variance proxy ---
    spatial_variance = np.var(thermal_field)

    return np.array([
        dc,
        low_energy,
        horiz_energy,
        vert_energy,
        diag_energy,
        high_freq_energy,
        spectral_centroid,
        spatial_variance
    ])