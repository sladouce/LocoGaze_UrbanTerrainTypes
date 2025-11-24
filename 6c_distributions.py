#!/usr/bin/env python3
"""
generate_distributions.py  (updated)

Now writes TWO files per variable:
  - <var>_distribution_floor.csv   -> only depth_looking_floor == True
  - <var>_distribution_allgaze.csv -> all samples (floor + non-floor)

Depth handling:
- For *allgaze*: use recorded values for all rows; cap any value > DEPTH_CAP to DEPTH_FILL (10_000).
- For *floor*:   keep only rows with depth_looking_floor == True; cap > DEPTH_CAP to DEPTH_FILL.
"""

import os
import pandas as pd
import numpy as np

# ---- USE META FILE TO RECONSTRUCT PATHS ----
META_CSV   = r'C:/LocoGaze/data/metadata.csv'
meta       = pd.read_csv(META_CSV, nrows=1)
reldir     = meta.at[0, 'reldir']
BASE_DIR   = r'C:/LocoGaze/data/'
input_dir  = os.path.join(BASE_DIR, reldir)
input_dir2 = os.path.join(input_dir, 'output')
OUTPUT_DIR = os.path.join(input_dir2, 'distributions')

INPUT_CSV = os.path.join(input_dir2, 'visual_events.csv')

# Define bin edges for each variable (you can tune these)
bins_dict = {
    'gaze_x':                np.arange(0, 1920 + 20, 20),
    'gaze_y':                np.arange(0, 1080 + 20, 20),
    'depth_d_s':             np.arange(0, 10050 + 100, 100),
    'depth_d_vergence':      np.arange(0, 10050 + 100, 100),
    'depth_calib_mm':        np.arange(0, 10050 + 100, 100),
    'depth_head_pitch_deg':  np.arange(-90, 20 + 1, 1),
    'spatial_entropy':       np.arange(0, 10 + 0.5, 0.5),
    'radius_of_gyration':    np.arange(0, 20 + 0.5, 0.5),
    'stride_duration_LEFT':  np.arange(0.4, 2.0 + 0.01, 0.01),
    'stride_length_LEFT':    np.arange(0, 5 + 0.01, 0.05),
    'pace_LEFT':             np.arange(0, 10 + 0.1, 0.1),
    'cadence_LEFT':          np.arange(60, 180 + 2, 1)
}

DEPTH_VARS = {'depth_d_s', 'depth_d_vergence', 'depth_calib_mm'}
DEPTH_CAP  = 10000
DEPTH_FILL = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load & filter walking ----
df = pd.read_csv(INPUT_CSV)
for col in ['is_walking', 'area_label', 'depth_looking_floor']:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in input CSV.")

walking_df = df[df['is_walking'] == True].copy()

def to_bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)) and not pd.isna(v): return bool(int(v))
    if isinstance(v, str): return v.strip().lower() in {'true','1','yes','y','t'}
    return False

walking_df['depth_looking_floor_bool'] = walking_df['depth_looking_floor'].apply(to_bool)

def hist_to_percent(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Return percent per bin (summing to ~100) from numeric vector (NaNs dropped)."""
    sub = values[~np.isnan(values)]
    counts, _ = np.histogram(sub, bins=bin_edges)
    total = counts.sum()
    return (counts / total * 100.0) if total > 0 else np.zeros_like(counts, dtype=float)

# ---- For each variable, compute two variants: FLOOR and ALLGAZE ----
for var, bin_edges in bins_dict.items():
    if var not in walking_df.columns:
        print(f"Skipping '{var}': column not found in {INPUT_CSV}")
        continue

    bin_labels = bin_edges[:-1]
    dist_floor   = pd.DataFrame({var + '_bin': bin_labels})
    dist_allgaze = pd.DataFrame({var + '_bin': bin_labels})

    # Iterate areas
    for area in walking_df['area_label'].dropna().unique():
        mask_area = (walking_df['area_label'] == area)

        # Base series (as float)
        series_all = walking_df.loc[mask_area, var].astype(float).copy()

        if var in DEPTH_VARS:
            # ---- ALLGAZE: keep actual values for non-floor; cap extreme values ----
            series_all = series_all.copy()
            series_all[series_all > DEPTH_CAP] = DEPTH_FILL

            # ---- FLOOR: keep only rows where looking_floor == True; cap extremes ----
            floor_mask = walking_df.loc[mask_area, 'depth_looking_floor_bool'].values
            series_floor = series_all[floor_mask].copy()  # reuse capped values
        else:
            # Non-depth variables: floor filter irrelevant
            series_floor = series_all.copy()

        # Percent per bin
        dist_allgaze[str(area)] = hist_to_percent(series_all.to_numpy(),  bin_edges)
        dist_floor[str(area)]   = hist_to_percent(series_floor.to_numpy(), bin_edges)

    # Write outputs
    out_floor   = os.path.join(OUTPUT_DIR, f"{var}_distribution_floor.csv")
    out_allgaze = os.path.join(OUTPUT_DIR, f"{var}_distribution_allgaze.csv")
    dist_floor.to_csv(out_floor, index=False)
    dist_allgaze.to_csv(out_allgaze, index=False)
    print(f"Saved: {out_floor}")
    print(f"Saved: {out_allgaze}")

print("All distributions (floor + allgaze) generated.")
