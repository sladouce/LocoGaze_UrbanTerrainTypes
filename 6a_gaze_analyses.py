#!/usr/bin/env python3
"""
load_visual_events.py — environment object proportions (people now INCLUDE cyclists)

Outputs include:
- 'people' row: presence = (people_present OR cyclist_present),
                fixations = (fixated == 'person'),  # includes cyclist persons, not bikes
                avg_number_when_present = total people (non-cyclists + cyclists)
- 'cyclist', 'bicycle', 'car', 'floor', 'background' unchanged
- 'background' behaves like 'floor' (presence=1.0) but counts 'other' not-floor.

Assumes process_visual_events.py produced:
  people_present, cyclist_present, bicycle_present, car_present,
  looking_cyclist, depth_looking_floor, fixated,
  number_people (total), number_people_nocyclist, number_cyclist, number_bicycle_standalone (optional)
"""

import os
import warnings
import numpy as np
import pandas as pd

# ---- Load metadata ----
META_CSV = r'C:/LocoGaze/data/metadata.csv'
meta_df  = pd.read_csv(META_CSV, nrows=1)
reldir   = meta_df.at[0, 'reldir']

# ---- Paths ----
BASE_DIR       = r'C:/LocoGaze/data/'
INPUT_DIR      = os.path.join(BASE_DIR, reldir, 'output')
OUTPUT_DIR     = os.path.join(INPUT_DIR, 'stats')
VISUAL_CSV     = os.path.join(INPUT_DIR, 'visual_events.csv')
PROPORTION_CSV = os.path.join(OUTPUT_DIR, 'environment_objects_proportion.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load & pre-filter ----
vis_df = pd.read_csv(VISUAL_CSV)

# Keep only the first row of each continuous event (where event_duration changes)
vis_df = vis_df.loc[vis_df['event_duration'].ne(vis_df['event_duration'].shift())].reset_index(drop=True)

# Only walking + fixation samples
event_df = vis_df[(vis_df['is_walking'] == True) & (vis_df['event_label'] == 'FIXA')].copy()

# Duration for weighting
event_df['duration'] = pd.to_numeric(event_df['event_duration'], errors='coerce').fillna(0.0)
if event_df['duration'].sum() <= 0:
    warnings.warn("Total duration is zero after filtering; outputs will be empty/zero.")

# ---- Required columns (presence + labels) ----
required_bool = [
    'people_present',      #
    'cyclist_present',
    'bicycle_present',     # standalone bikes
    'car_present',
    'depth_looking_floor',
    'looking_cyclist',
]
required_any = required_bool + ['fixated', 'area_label', 'duration']
missing = [c for c in required_any if c not in event_df.columns]
if missing:
    raise KeyError(f"Missing columns in visual_events.csv: {missing}")

# Optional count columns (for averages)
has_people_total      = 'number_people' in event_df.columns
has_people_nocyc_cnt  = 'number_people_nocyclist' in event_df.columns
has_cyclist_cnt       = 'number_cyclist' in event_df.columns
has_bike_standalone   = 'number_bicycle_standalone' in event_df.columns

# sanitize dtypes
for c in required_bool:
    event_df[c] = event_df[c].astype(bool)

# ---- People-inclusive masks/counters ----
# Presence of *any* people = non-cyclists OR cyclists
event_df['people_any_present'] = event_df['people_present'] | event_df['cyclist_present']

# Fixations counted as "people" = gaze inside a person bbox (includes cyclists' persons).
# (Bike-only looks remain with 'bicycle' / 'cyclist'.)
fix_people_any   = (event_df['fixated'] == 'person')
fix_cyclist      = event_df['looking_cyclist']
fix_bicycle_st   = (event_df['fixated'] == 'bicycle') & (event_df['bicycle_present'])
fix_car          = (event_df['fixated'] == 'car')
fix_floor        = (event_df['fixated'] == 'other') & (event_df['depth_looking_floor'])
fix_bg           = (event_df['fixated'] == 'other') & (~event_df['depth_looking_floor'])

# Total people count (prefer explicit total; else sum components if available)
num_people_total = None
if has_people_total:
    num_people_total = pd.to_numeric(event_df['number_people'], errors='coerce').fillna(0.0)
elif has_people_nocyc_cnt and has_cyclist_cnt:
    num_people_total = (
        pd.to_numeric(event_df['number_people_nocyclist'], errors='coerce').fillna(0.0) +
        pd.to_numeric(event_df['number_cyclist'], errors='coerce').fillna(0.0)
    )

# ---- Helpers ----
def wmean(mask: pd.Series, dur: pd.Series) -> float:
    denom = float(dur.sum())
    if denom <= 0:
        return 0.0
    return float(dur[mask].sum()) / denom

def conditional_wmean(mask_present: pd.Series, mask_fix: pd.Series, dur: pd.Series) -> float:
    """Weighted mean of fixations given presence."""
    if not mask_present.any():
        return 0.0
    dsel = dur[mask_present]
    if dsel.sum() <= 0:
        return 0.0
    fix_sel = mask_fix & mask_present
    return float(dur[fix_sel].sum()) / float(dsel.sum())

def duration_weighted_average_count(count_series: pd.Series | None,
                                    present_mask: pd.Series,
                                    dur: pd.Series):
    """Average number when present, duration-weighted."""
    if (count_series is None) or (not present_mask.any()):
        return None
    dsel = dur[present_mask]
    if dsel.sum() <= 0:
        return None
    cnt = pd.to_numeric(count_series, errors='coerce').fillna(0.0)
    cnt_sel = cnt[present_mask]
    return float((cnt_sel * dsel).sum() / dsel.sum())

# ---- Core computation ----
def compute_environment_object_proportions(df: pd.DataFrame) -> pd.DataFrame:
    out = []

    def push(obj, area, pres, fix, cond_fix, avg_num):
        out.append({
            'object': obj,
            'area_label': area,
            'proportion_present': pres,
            'proportion_fixated': fix,
            'proportion_fixated_when_present': cond_fix,
            'avg_number_when_present': avg_num
        })

    dur = df['duration']

    # ---------- Overall (area = 'all') ----------
    pres_people  = wmean(df['people_any_present'], dur)
    pres_cyc     = wmean(df['cyclist_present'],    dur)
    pres_bike    = wmean(df['bicycle_present'],    dur)
    pres_car     = wmean(df['car_present'],        dur)
    pres_floor   = 1.0
    pres_bg      = 1.0

    fix_people_o = wmean(fix_people_any,  dur)
    fix_cyc_o    = wmean(fix_cyclist,     dur)
    fix_bike_o   = wmean(fix_bicycle_st,  dur)
    fix_car_o    = wmean(fix_car,         dur)
    fix_floor_o  = wmean(fix_floor,       dur)
    fix_bg_o     = wmean(fix_bg,          dur)

    cond_fix_people = conditional_wmean(df['people_any_present'], fix_people_any, dur)
    cond_fix_cyc    = conditional_wmean(df['cyclist_present'],    fix_cyclist,   dur)
    cond_fix_bike   = conditional_wmean(df['bicycle_present'],    fix_bicycle_st, dur)
    cond_fix_car    = conditional_wmean(df['car_present'],        fix_car,       dur)

    avg_people = duration_weighted_average_count(num_people_total, df['people_any_present'], dur)
    avg_bike   = duration_weighted_average_count(
        df['number_bicycle_standalone'] if has_bike_standalone else None,
        df['bicycle_present'], dur
    )

    push('people',   'all', pres_people, fix_people_o, cond_fix_people, avg_people)
    push('cyclist',  'all', pres_cyc,    fix_cyc_o,    cond_fix_cyc,    None)
    push('bicycle',  'all', pres_bike,   fix_bike_o,   cond_fix_bike,   avg_bike)
    push('car',      'all', pres_car,    fix_car_o,    cond_fix_car,    None)
    push('floor',    'all', pres_floor,  fix_floor_o,  None,            None)
    push('background','all', pres_bg,    fix_bg_o,     None,            None)

    # ---------- Per area_label ----------
    for area, g in df.groupby('area_label'):
        d = g['duration']

        pres_people_a = wmean(g['people_any_present'], d)
        pres_cyc_a    = wmean(g['cyclist_present'],    d)
        pres_bike_a   = wmean(g['bicycle_present'],    d)
        pres_car_a    = wmean(g['car_present'],        d)

        idx           = g.index
        fix_people_a  = wmean(fix_people_any.loc[idx],  d)
        fix_cyc_a     = wmean(fix_cyclist.loc[idx],     d)
        fix_bike_a    = wmean(fix_bicycle_st.loc[idx],  d)
        fix_car_a     = wmean(fix_car.loc[idx],         d)
        fix_floor_a   = wmean(((g['fixated'] == 'other') & g['depth_looking_floor']), d)
        fix_bg_a      = wmean(((g['fixated'] == 'other') & (~g['depth_looking_floor'])), d)

        cond_fix_people_a = conditional_wmean(g['people_any_present'], fix_people_any.loc[idx], d)
        cond_fix_cyc_a    = conditional_wmean(g['cyclist_present'],    fix_cyclist.loc[idx],   d)
        cond_fix_bike_a   = conditional_wmean(g['bicycle_present'],    fix_bicycle_st.loc[idx], d)
        cond_fix_car_a    = conditional_wmean(g['car_present'],        fix_car.loc[idx],       d)

        # average number of people (total) when present
        avg_people_a = None
        if num_people_total is not None:
            avg_people_a = duration_weighted_average_count(num_people_total.loc[idx], g['people_any_present'], d)

        avg_bike_a = duration_weighted_average_count(
            g['number_bicycle_standalone'] if has_bike_standalone else None,
            g['bicycle_present'], d
        )

        push('people',  area, pres_people_a, fix_people_a, cond_fix_people_a, avg_people_a)
        push('cyclist', area, pres_cyc_a,    fix_cyc_a,    cond_fix_cyc_a,    None)
        push('bicycle', area, pres_bike_a,   fix_bike_a,   cond_fix_bike_a,   avg_bike_a)
        push('car',     area, pres_car_a,    fix_car_a,    cond_fix_car_a,    None)
        push('floor',   area, 1.0,           fix_floor_a,  None,              None)
        push('background', area, 1.0,        fix_bg_a,     None,              None)

    return pd.DataFrame(out)

# ---- Run ----
if __name__ == '__main__':
    prop_df = compute_environment_object_proportions(event_df)
    prop_df.to_csv(PROPORTION_CSV, index=False)
    print(f"Saved environment object proportions to {PROPORTION_CSV}")

#!/usr/bin/env python3
"""
compute_rog_entropy_social.py

Social context now defined as:
    social = people_present OR cyclist_present
(i.e., any people around, including cyclists)
"""

import os
import pandas as pd

# ---- Load metadata ----
META_CSV = r'C:/LocoGaze/data/metadata.csv'
meta_df  = pd.read_csv(META_CSV, nrows=1)
reldir   = meta_df.at[0, 'reldir']

# ---- Paths ----
BASE_DIR   = r'C:/LocoGaze/data/'
INPUT_DIR  = os.path.join(BASE_DIR, reldir, 'output')
OUTPUT_DIR = os.path.join(INPUT_DIR, 'stats')
INPUT_CSV  = os.path.join(INPUT_DIR, 'visual_events.csv')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'rog_entropy_stats_social.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_COLS = ['radius_of_gyration', 'spatial_entropy']

df = pd.read_csv(INPUT_CSV)

required = ['is_walking', 'people_present', 'cyclist_present', 'area_label'] + METRIC_COLS
missing  = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"visual_events.csv missing required columns: {missing}")

df_walk = df[df['is_walking'] == True].copy()

# SOCIAL = any people (non-cyclists OR cyclists)
df_walk['social'] = df_walk['people_present'].astype(bool) | df_walk['cyclist_present'].astype(bool)

grouped_by_area = (
    df_walk.groupby(['area_label', 'social'])[METRIC_COLS]
           .mean()
           .reset_index()
)

grouped_area_both = (
    df_walk.groupby(['area_label'])[METRIC_COLS]
           .mean()
           .reset_index()
)
grouped_area_both['social'] = 'both'

grouped_all = (
    df_walk.groupby(['social'])[METRIC_COLS]
           .mean()
           .reset_index()
)
grouped_all['area_label'] = 'all'

grouped_all_both = (
    df_walk[METRIC_COLS].mean().to_frame().T
)
grouped_all_both['area_label'] = 'all'
grouped_all_both['social'] = 'both'

per_area = pd.concat([grouped_by_area, grouped_area_both], ignore_index=True)
result   = pd.concat([per_area, grouped_all, grouped_all_both], ignore_index=True)

cols = ['area_label', 'social'] + METRIC_COLS
result = result[cols]
result.to_csv(OUTPUT_CSV, index=False)
print(f"Saved RoG & entropy stats (social=people OR cyclists) to {OUTPUT_CSV}")


#!/usr/bin/env python3
"""
compute_depth_stats_social.py

Social context now defined as:
    social = people_present OR cyclist_present
(i.e., any people around, including cyclists)

Filters to walking AND depth_looking_floor == True.
"""

import os
import pandas as pd

# ---- Load metadata ----
META_CSV = r'C:/LocoGaze/data/metadata.csv'
meta_df  = pd.read_csv(META_CSV, nrows=1)
reldir   = meta_df.at[0, 'reldir']

# ---- Paths ----
BASE_DIR   = r'C:/LocoGaze/data/'
INPUT_DIR  = os.path.join(BASE_DIR, reldir, 'output')
OUTPUT_DIR = os.path.join(INPUT_DIR, 'stats')
INPUT_CSV  = os.path.join(INPUT_DIR, 'visual_events.csv')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'gazedepth_stats_social.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_COLS = [
    'depth_d_s',
    'depth_head_pitch_deg',
    'depth_d_vergence',
    'depth_calib_mm',
]

df = pd.read_csv(INPUT_CSV)

required = ['is_walking', 'depth_looking_floor', 'people_present', 'cyclist_present', 'area_label'] + METRIC_COLS
missing  = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"visual_events.csv missing required columns: {missing}")

# Filter: walking + floor-looking
df_walk = df[(df['is_walking'] == True) & (df['depth_looking_floor'] == True)].copy()

# SOCIAL = any people (non-cyclists OR cyclists)
df_walk['social'] = df_walk['people_present'].astype(bool) | df_walk['cyclist_present'].astype(bool)

grouped_by_area = (
    df_walk.groupby(['area_label', 'social'])[METRIC_COLS]
           .mean()
           .reset_index()
)

grouped_area_both = (
    df_walk.groupby(['area_label'])[METRIC_COLS]
           .mean()
           .reset_index()
)
grouped_area_both['social'] = 'both'

grouped_all = (
    df_walk.groupby(['social'])[METRIC_COLS]
           .mean()
           .reset_index()
)
grouped_all['area_label'] = 'all'

grouped_all_both = (
    df_walk[METRIC_COLS].mean().to_frame().T
)
grouped_all_both['area_label'] = 'all'
grouped_all_both['social'] = 'both'

per_area = pd.concat([grouped_by_area, grouped_area_both], ignore_index=True)
result   = pd.concat([per_area, grouped_all, grouped_all_both], ignore_index=True)

cols = ['area_label', 'social'] + METRIC_COLS
result = result[cols]
result.to_csv(OUTPUT_CSV, index=False)
print(f"Saved gaze depth & head pitch stats (social=people OR cyclists) to {OUTPUT_CSV}")
