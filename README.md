# Honeybee Collective Behaviour – Ethogram Generation

*A data-driven framework to generate and interpret collective ethograms of honeybee colonies from environmental sensor data.*

---

## Overview

This project implements a complete analysis pipeline for the **generation of collective ethograms in honeybee colonies**, inspired by MotionMapper-based behavioural mapping methods originally developed for individual animals.

Unlike classical ethogram approaches relying on video recordings of single organisms, this framework focuses on **colony-level behaviour** and leverages **non-invasive environmental sensors** embedded within beehives.

The pipeline processes multi-sensor time-series data to:

- construct a low-dimensional behavioural space,
- identify recurrent collective states using clustering,
- generate ethograms describing colony dynamics over time,
- and interpret behavioural regions in terms of biologically meaningful states (e.g. brood presence, spatial clustering, homogeneous distributions).

---

## Key Features

- Supports **multiple datasets** (different hives, locations, periods, experimental conditions)
- Works with **temperature, CO₂, and relative humidity sensors**
- Fully **global workflow** (global normalization, global embedding)
- MotionMapper-compatible (PCA, wavelets, embedding, density estimation)
- Robust region interpretation using **barycenters and nearest-point profiles**
- Includes **robustness testing** via synthetic (“fake”) sensor profiles

---

## Installation

### 1. Create the environment

MotionMapperPy requires **Python 3.7.x**.

```bash
    conda create -n motionmapper python=3.7.0
    conda activate motionmapper
```
---

### 2. Install dependencies

```bash
    pip install -r requirements.txt
```
⚠️ Some legacy dependencies (e.g. `mkl-fft`, older `numpy`) may require minor version adjustments depending on your OS.

---

### 3. Install MotionMapperPy

Follow the official repository instructions:

👉 https://github.com/bermanlabemory/motionmapperpy

---

## Expected Input Data

Each dataset must contain time-series measurements recorded inside a honeybee hive:

- **64 temperature sensors** (`t00`–`t63`)
- **CO₂ concentration** (ppm)
- **Relative humidity** (%)

### Requirements

- Timestamps must be convertible to a valid `DatetimeIndex`
- Datasets may have different sampling rates (handled via resampling)
- Multiple datasets are concatenated into a **MultiIndex**:
  (timestamp, source_id)

---

## Pipeline Summary

### 1. Data loading & cleaning

- Merge temperature, CO₂ and humidity data
- Synchronise sensors with `merge_asof`
- Remove invalid or unphysical measurements
- Optional removal of hive-unoccupied periods
- Resample to a common time resolution (e.g. 10 minutes)

---

### 2. Dataset concatenation

- Each dataset receives a unique `source_id`
- All datasets are concatenated **without reordering**
- Original timestamps can optionally be preserved (`real_timestamp`)

---

### 3. Dataset-wise Normalisation

Sensor values are normalised **independently for each dataset** prior to concatenation. This step compensates for systematic differences between hives, sensor calibrations, and environmental baselines, while preserving the internal structure and relative sensor patterns within each dataset.

Temperature sensors are normalised individually, and the same procedure is applied separately to CO₂ and humidity measurements. This dataset-wise normalisation ensures that the subsequent dimensionality reduction and embedding stages are driven by **relative spatial and temporal patterns**, rather than absolute offsets specific to a given hive or recording period.


---

### 4. Dimensionality reduction

- **Global PCA** (retain ≥99% explained variance)
- **Morlet wavelet transform** to capture temporal dynamics
- Construction of a **balanced training set** (equal contribution per dataset)
- **UMAP** (preferred) or **t-SNE** embedding
- Projection of the full dataset into the learned behavioural space

UMAP is favoured for:
- better scalability,
- improved global structure preservation,
- stable embeddings,
- and support for embedding unseen data.

---

### 5. Clustering

- Density estimation on the 2D manifold
- **Watershed segmentation** to identify behavioural regions

The number of regions is intentionally over-segmented to allow later biological interpretation and grouping.

---

### 6. Ethogram generation

For each dataset (`source_id`):

- Assign a behavioural region to each timestamp
- Aggregate points by day
- Display a colour-coded ethogram across days
- Analyse day/night structure and temporal transitions

---

## Interpretation Strategy

- Compute **barycenters** for each watershed region
- Analyse the **10 and 100 closest points** to each barycenter
- Regions are primarily organised by **relative spatial temperature profiles**, not absolute values
- Enables identification of:
  - brood presence regions,
  - spatial clustering states,
  - homogeneous distributions,
  - transitional behaviours

---

## Robustness Analysis

To evaluate generalisation:

- Synthetic (“fake”) datasets are generated from real region profiles
- Profiles are deliberately modified (e.g. inverted spatial structure)
- Fake data is passed through the **entire pipeline**
- Projection and region assignment are analysed

This highlights:
- which behavioural patterns are well represented,
- and where model limitations arise due to data sparsity.

---

## Outputs

The pipeline produces:

### ✔ Behavioural maps
UMAP/t-SNE projections of collective colony states

### ✔ Region labels
Watershed cluster assignment for each time point

### ✔ Ethograms
Daily behavioural state timelines per dataset

### ✔ Diagnostic & interpretation plots
- Temperature profiles per region
- CO₂ and humidity distributions
- Density maps
- Region occupancy histograms

All outputs are saved in MotionMapper-compatible formats.

---

## Notes & Best Practices

- Data quality is critical: remove unstable sensors early
- Wavelets improve temporal sensitivity but increase dimensionality
- Interpret clusters via **relative profiles**, not absolute values
- Robustness tests require representative training data

---

## Contact

**Pierre Mailler**  
MOBOTS Group – EPFL  
pierre.mailler@epfl.ch
