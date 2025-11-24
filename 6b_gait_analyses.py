#!/usr/bin/env python3
"""
compute_gait_stats_social.py (refined, total-people bins)

1) Part 1: mean (and variance) gait metrics by area_label × social
   where social = any person present (pedestrians + cyclists),
   saved to: stats/gait_stats_social.csv

2) Part 2: mean (and variance) gait metrics by area_label × people_bin
   where people_bin ∈ {"0","1","2plus"} for TOTAL people around
   (pedestrians + cyclists), saved to: stats/gait_stats_peoplebins.csv
"""

import os
import warnings
import pandas as pd
import numpy as np

# ---- PATHS (adapt these if needed) ----
META_CSV   = r'C:/LocoGaze/data/metadata.csv'
meta_df    = pd.read_csv(META_CSV, nrows=1)
reldir     = meta_df.at[0, 'reldir']

BASE_DIR   = r'C:/LocoGaze/data/'
INPUT_DIR  = os.path.join(BASE_DIR, reldir, 'output')
OUTPUT_DIR = os.path.join(INPUT_DIR, 'stats')
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_CSV          = os.path.join(INPUT_DIR, 'visual_events.csv')
OUTPUT_CSV_SOCIAL  = os.path.join(OUTPUT_DIR, 'gait_stats_social.csv')
OUTPUT_CSV_BINS    = os.path.join(OUTPUT_DIR, 'gait_stats_peoplebins.csv')

GAIT_COLS = [
    'mean_rms_LEFT',
    'stride_duration_LEFT',
    'stride_length_LEFT',
    'cadence_LEFT',
    'pace_LEFT'
]

VAR_BASE_COLS = ['stride_length_LEFT', 'stride_duration_LEFT']  # for variance outputs

# ---- LOAD ----
df = pd.read_csv(INPUT_CSV)

# ---- FILTER TO WALKING ----
if 'is_walking' not in df.columns:
    raise KeyError("visual_events.csv missing required column 'is_walking'")
df_walk = df[df['is_walking'] == True].copy()

# ---- SOCIAL FLAG (any person present: pedestrians + cyclists) ----
# Prefer explicit people_present if available (created as number_people_total > 0 upstream)
if 'people_present' in df_walk.columns:
    df_walk['social'] = df_walk['people_present'].astype(bool)
else:
    # Fallback: infer from person counts
    has_np  = 'number_people' in df_walk.columns
    has_npn = 'number_people_nocyclist' in df_walk.columns
    has_nc  = 'number_cyclist' in df_walk.columns

    if has_np:
        # number_people is already "all persons" (pedestrians + cyclists)
        df_walk['social'] = pd.to_numeric(df_walk['number_people'], errors='coerce').fillna(0) > 0
        warnings.warn("people_present missing: using number_people > 0 for social flag (total persons).")
    elif has_npn and has_nc:
        total_people = (
            pd.to_numeric(df_walk['number_people_nocyclist'], errors='coerce').fillna(0) +
            pd.to_numeric(df_walk['number_cyclist'],        errors='coerce').fillna(0)
        )
        df_walk['social'] = total_people > 0
        warnings.warn("people_present & number_people missing: reconstructed total people from nocyclist + cyclist.")
    else:
        df_walk['social'] = False
        warnings.warn("No people_present/number_people columns found; social set to False for all rows.")

# ---- Ensure GAIT_COLS exist (drop missing gracefully with warning) ----
existing_gait_cols = [c for c in GAIT_COLS if c in df_walk.columns]
missing_gait_cols  = [c for c in GAIT_COLS if c not in df_walk.columns]
if missing_gait_cols:
    warnings.warn(f"Missing gait columns (will be skipped): {missing_gait_cols}")
GAIT_COLS = existing_gait_cols

# ---- === PART 1: area_label × social (means + variances) === ----
grouped_by_area_mean = (
    df_walk
    .groupby(['area_label', 'social'])[GAIT_COLS]
    .mean()
)

var_cols = [c for c in VAR_BASE_COLS if c in df_walk.columns]
if var_cols:
    grouped_by_area_var = (
        df_walk
        .groupby(['area_label', 'social'])[var_cols]
        .var(ddof=1)  # sample variance
        .rename(columns={
            'stride_length_LEFT': 'stride_length_LEFT_var',
            'stride_duration_LEFT': 'stride_duration_LEFT_var'
        })
    )
    grouped_by_area = grouped_by_area_mean.join(grouped_by_area_var, how='left')
else:
    grouped_by_area = grouped_by_area_mean

grouped_by_area = grouped_by_area.reset_index()

# overall (across areas)
grouped_all_mean = (
    df_walk
    .groupby(['social'])[GAIT_COLS]
    .mean()
)

if var_cols:
    grouped_all_var = (
        df_walk
        .groupby(['social'])[var_cols]
        .var(ddof=1)
        .rename(columns={
            'stride_length_LEFT': 'stride_length_LEFT_var',
            'stride_duration_LEFT': 'stride_duration_LEFT_var'
        })
    )
    grouped_all = grouped_all_mean.join(grouped_all_var, how='left').reset_index()
else:
    grouped_all = grouped_all_mean.reset_index()

grouped_all['area_label'] = 'all'

result_social = pd.concat([grouped_by_area, grouped_all], ignore_index=True, sort=False)

out_cols = ['area_label', 'social'] + GAIT_COLS
for extra in ['stride_length_LEFT_var', 'stride_duration_LEFT_var']:
    if extra in result_social.columns:
        out_cols.append(extra)
result_social = result_social[out_cols]

# ---- SAVE PART 1 ----
result_social.to_csv(OUTPUT_CSV_SOCIAL, index=False)
print(f"Saved gait stats by social context (including overall) to {OUTPUT_CSV_SOCIAL}")

# ---- === PART 2: area_label × people_bin (0,1,2plus) using TOTAL people (pedestrians + cyclists) === ----

# Build total_people_count column (pedestrians + cyclists)
if 'number_people' in df_walk.columns:
    # number_people: all persons (already includes cyclists) from process_visual_events.py
    total_people_count = pd.to_numeric(df_walk['number_people'], errors='coerce').fillna(0).clip(lower=0)
elif ('number_people_nocyclist' in df_walk.columns) and ('number_cyclist' in df_walk.columns):
    total_people_count = (
        pd.to_numeric(df_walk['number_people_nocyclist'], errors='coerce').fillna(0) +
        pd.to_numeric(df_walk['number_cyclist'],        errors='coerce').fillna(0)
    ).clip(lower=0)
    warnings.warn("number_people missing; reconstructed total people as nocyclist + cyclist.")
elif 'number_people_nocyclist' in df_walk.columns:
    total_people_count = pd.to_numeric(df_walk['number_people_nocyclist'], errors='coerce').fillna(0).clip(lower=0)
    warnings.warn("Only number_people_nocyclist present; using this as total people (cyclists may be undercounted).")
elif 'number_cyclist' in df_walk.columns:
    total_people_count = pd.to_numeric(df_walk['number_cyclist'], errors='coerce').fillna(0).clip(lower=0)
    warnings.warn("Only number_cyclist present; using cyclists as total people (pedestrians may be undercounted).")
else:
    total_people_count = pd.Series(0, index=df_walk.index, dtype=float)
    warnings.warn("No people count columns found; assuming 0 total people for binning.")

# Create bins: 0, 1, 2plus (based on TOTAL people)
total_people_int = total_people_count.astype(int).clip(lower=0)
bins = np.where(
    total_people_int == 0, '0',
    np.where(total_people_int == 1, '1', '2plus')
)
df_walk['people_bin'] = pd.Categorical(bins, categories=['0', '1', '2plus'], ordered=True)

# Group means by area_label × people_bin
grouped_pb_area_mean = (
    df_walk
    .groupby(['area_label', 'people_bin'])[GAIT_COLS]
    .mean()
)

# Variances by area_label × people_bin
if var_cols:
    grouped_pb_area_var = (
        df_walk
        .groupby(['area_label', 'people_bin'])[var_cols]
        .var(ddof=1)
        .rename(columns={
            'stride_length_LEFT': 'stride_length_LEFT_var',
            'stride_duration_LEFT': 'stride_duration_LEFT_var'
        })
    )
    grouped_pb_area = grouped_pb_area_mean.join(grouped_pb_area_var, how='left')
else:
    grouped_pb_area = grouped_pb_area_mean

grouped_pb_area = grouped_pb_area.reset_index()

# Overall (across areas) per people_bin
grouped_pb_all_mean = (
    df_walk
    .groupby(['people_bin'])[GAIT_COLS]
    .mean()
)

if var_cols:
    grouped_pb_all_var = (
        df_walk
        .groupby(['people_bin'])[var_cols]
        .var(ddof=1)
        .rename(columns={
            'stride_length_LEFT': 'stride_length_LEFT_var',
            'stride_duration_LEFT': 'stride_duration_LEFT_var'
        })
    )
    grouped_pb_all = grouped_pb_all_mean.join(grouped_pb_all_var, how='left').reset_index()
else:
    grouped_pb_all = grouped_pb_all_mean.reset_index()

grouped_pb_all['area_label'] = 'all'

# Combine area + overall
result_bins = pd.concat([grouped_pb_area, grouped_pb_all], ignore_index=True, sort=False)

# Reorder columns
out_cols_bins = ['area_label', 'people_bin'] + GAIT_COLS
for extra in ['stride_length_LEFT_var', 'stride_duration_LEFT_var']:
    if extra in result_bins.columns:
        out_cols_bins.append(extra)
result_bins = result_bins[out_cols_bins]

# SAVE PART 2
result_bins.to_csv(OUTPUT_CSV_BINS, index=False)
print(f"Saved gait stats by TOTAL-people bins (0,1,2plus, including overall) to {OUTPUT_CSV_BINS}")
