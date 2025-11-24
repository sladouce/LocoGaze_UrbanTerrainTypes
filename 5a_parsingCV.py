#!/usr/bin/env python3
"""
process_visual_events.py  (updated to distinguish cyclists vs. standalone bikes vs. non-cyclist people)

- Loads DeepSort gaze+tracking JSON
- Integrates REMoDNaV events
- Computes fixation target from gaze–bbox inclusion
- Detects cyclist pairs (person+bicycle) via greedy matching
- Counts:
    * number_people               (all persons)
    * number_cyclist              (matched person+bicycle pairs)
    * number_people_nocyclist     (people minus cyclists)
    * number_bicycle_total        (all bicycles)
    * number_bicycle_standalone   (bikes minus cyclists)
- Flags presence:
    * people_present      (non-cyclist people present)
    * cyclist_present
    * bicycle_present     (standalone bikes present)
    * car_present
    * looking_cyclist     (gaze on the person or bike of any cyclist pair)
- Merges downsampled gaze depth (25 Hz), gait metadata, and computes rolling spatial metrics
- Writes visual_events.csv with all fields required by downstream scripts
"""

import os
import json
import pandas as pd
import numpy as np

# ---- HELPER FUNCTIONS ----
def spatial_entropy(x, y, n_bins=12, log_base=2):
    H2d, _, _ = np.histogram2d(x, y, bins=n_bins, density=True)
    p = H2d.flatten()
    p = p[p > 0]
    return -np.sum(p * np.log(p) / np.log(log_base))

# degrees-per-pixel scales
SX = 95.0 / 1920.0
SY = 63.0 / 1080.0

def radius_of_gyration_deg(x, y, sx=SX, sy=SY):
    cx, cy = np.mean(x), np.mean(y)
    dx = (x - cx) * sx
    dy = (y - cy) * sy
    return np.sqrt(np.mean(dx**2 + dy**2))

def _to_float_or_nan(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def _parse_gaze(g):
    """Accepts [x,y], (x,y), or dict with keys x/y (also gx/gy)."""
    if g is None:
        return np.nan, np.nan
    if isinstance(g, (list, tuple)) and len(g) >= 2:
        return _to_float_or_nan(g[0]), _to_float_or_nan(g[1])
    if isinstance(g, dict):
        x = g.get('x', g.get('gx', g.get(0)))
        y = g.get('y', g.get('gy', g.get(1)))
        return _to_float_or_nan(x), _to_float_or_nan(y)
    return np.nan, np.nan

def _parse_bbox(bbox):
    """Accepts [x1,y1,w,h] or dict with keys x1,y1,w,h (or left/top/width/height)."""
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, w, h = map(_to_float_or_nan, bbox[:4])
    elif isinstance(bbox, dict):
        x1 = _to_float_or_nan(bbox.get('x1', bbox.get('left', bbox.get(0))))
        y1 = _to_float_or_nan(bbox.get('y1', bbox.get('top', bbox.get(1))))
        w  = _to_float_or_nan(bbox.get('w',  bbox.get('width', bbox.get(2))))
        h  = _to_float_or_nan(bbox.get('h',  bbox.get('height', bbox.get(3))))
    else:
        return None
    if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0):
        return None
    return float(x1), float(y1), float(w), float(h)

def _parse_objects(objs):
    """Normalize JSON objects -> [{'class': <str>, 'bbox': (x1,y1,w,h)}]."""
    out = []
    if objs is None:
        return out
    if isinstance(objs, dict):
        objs = [objs]
    if not isinstance(objs, (list, tuple)):
        return out
    for o in objs:
        try:
            cls_raw = o.get('class', o.get('label', None)) if isinstance(o, dict) else None
            cls_str = str(cls_raw).strip().lower() if cls_raw is not None else None
            bbox = _parse_bbox(o.get('bbox') if isinstance(o, dict) else None)
            if cls_str and bbox is not None:
                out.append({'class': cls_str, 'bbox': bbox})
        except Exception:
            continue
    return out

def _bbox_center(b):
    x1, y1, w, h = b
    return (x1 + w/2.0, y1 + h/2.0)

def _point_in_bbox(px, py, b):
    x1, y1, w, h = b
    return (x1 <= px <= x1 + w) and (y1 <= py <= y1 + h)

def _match_cyclists(persons, bikes):
    """
    Greedy one-to-one matching between persons and bicycles to infer cyclist pairs.

    Heuristics:
      - horizontal proximity: |pcx - bcx| <= 0.6 * max(pw, bw)
      - vertical relation: bike center below person center: (bcy - pcy) >= 0.15 * max(ph, bh)
    Returns:
      matches: list of (p_idx, b_idx)
    """
    if not persons or not bikes:
        return []

    candidates = []
    for pi, p in enumerate(persons):
        px1, py1, pw, ph = p['bbox']
        pcx, pcy = _bbox_center(p['bbox'])
        for bi, b in enumerate(bikes):
            bx1, by1, bw, bh = b['bbox']
            bcx, bcy = _bbox_center(b['bbox'])
            if not all(map(np.isfinite, [pcx, pcy, bcx, bcy, pw, ph, bw, bh])):
                continue
            horiz_ok = abs(pcx - bcx) <= 0.6 * max(pw, bw)
            vert_ok  = (bcy - pcy) >= 0.15 * max(ph, bh)
            if horiz_ok and vert_ok:
                dist = abs(pcx - bcx) + max(0.0, (pcy - bcy))  # prefer closer, and bike not above person
                candidates.append((dist, pi, bi))

    # Greedy matching by smallest distance first
    candidates.sort(key=lambda t: t[0])
    used_p, used_b, matches = set(), set(), []
    for _, pi, bi in candidates:
        if pi in used_p or bi in used_b:
            continue
        matches.append((pi, bi))
        used_p.add(pi)
        used_b.add(bi)
    return matches

# ---- PATHS FROM META ----
META_CSV   = r'C:/LocoGaze/data/metadata.csv'
meta       = pd.read_csv(META_CSV, nrows=1)
reldir     = meta.at[0, 'reldir']
BASE_DIR   = r'C:/LocoGaze/data/'
input_dir  = os.path.join(BASE_DIR, reldir)
output_dir = os.path.join(input_dir, 'output')

# ---- I/O ----
JSON_FILE         = os.path.join(output_dir, 'tracking_gaze_deepsort.json')
REMODNAV_TSV      = os.path.join(output_dir, 'events_remodnav.tsv')
GAZE_DEPTH_CSV    = os.path.join(output_dir, 'gaze_depth.csv')
GAIT_METRICS_CSV  = os.path.join(output_dir, 'gait_metrics.csv')
OUTPUT_CSV        = os.path.join(output_dir, 'visual_events.csv')

# ---- LOAD JSON LOGS ----
with open(JSON_FILE, 'r') as f:
    frame_logs = json.load(f)
first_time = frame_logs[0].get('time_s', 0.0) if frame_logs else 0.0

# ---- LOAD REMoDNaV ----
events_df = pd.read_csv(REMODNAV_TSV, sep='\t')
events = []
for _, row in events_df.iterrows():
    onset_abs = float(row['onset']) + first_time
    dur       = float(row['duration'])
    events.append({
        'onset_abs': onset_abs,
        'offset_abs': onset_abs + dur,
        'event_type': str(row['label']).upper(),
        'duration': dur
    })
events.sort(key=lambda e: e['onset_abs'])

# ---- LOAD & DOWNSAMPLE GAZE DEPTH (100 Hz -> 25 Hz) ----
depth_df = pd.read_csv(GAZE_DEPTH_CSV)
depth_df = depth_df[depth_df['Timestamp'] >= first_time].reset_index(drop=True)

depth_df['d_s']            = pd.to_numeric(depth_df.get('d_s', np.nan), errors='coerce')
depth_df['head_pitch_deg'] = pd.to_numeric(depth_df.get('head_pitch_deg', np.nan), errors='coerce')
if 'looking_floor' in depth_df.columns:
    if depth_df['looking_floor'].dtype == bool:
        depth_df['looking_floor'] = depth_df['looking_floor'].astype(bool)
    else:
        depth_df['looking_floor'] = depth_df['looking_floor'].map(lambda v: str(v).strip().lower() in ('true','1','t','yes','y'))
else:
    depth_df['looking_floor'] = False
depth_df['d_vergence']     = pd.to_numeric(depth_df.get('d_vergence', np.nan), errors='coerce')
depth_df['depth_calib_mm'] = pd.to_numeric(depth_df.get('depth_calib_mm', np.nan), errors='coerce')

total_rows  = len(depth_df)
group_count = total_rows // 4
trimmed     = depth_df.iloc[:group_count * 4]

ar_d   = trimmed['d_s'            ].values.reshape(group_count, 4) if group_count > 0 else np.empty((0,4))
ar_hp  = trimmed['head_pitch_deg' ].values.reshape(group_count, 4) if group_count > 0 else np.empty((0,4))
ar_lf  = trimmed['looking_floor'  ].astype(bool).values.reshape(group_count, 4) if group_count > 0 else np.empty((0,4), dtype=bool)
ar_dv  = trimmed['d_vergence'     ].values.reshape(group_count, 4) if group_count > 0 else np.empty((0,4))
ar_dc  = trimmed['depth_calib_mm' ].values.reshape(group_count, 4) if group_count > 0 else np.empty((0,4))

depth_summary = pd.DataFrame({
    'depth_d_s':             np.nanmean(ar_d,  axis=1) if group_count > 0 else [],
    'depth_head_pitch_deg':  np.nanmean(ar_hp, axis=1) if group_count > 0 else [],
    'depth_looking_floor':   ar_lf.any(axis=1)          if group_count > 0 else [],
    'depth_d_vergence':      np.nanmean(ar_dv, axis=1)  if group_count > 0 else [],
    'depth_calib_mm':        np.nanmean(ar_dc, axis=1)  if group_count > 0 else [],
})

# ---- LOAD GAIT ----
gait_df = pd.read_csv(
    GAIT_METRICS_CSV,
    usecols=['Timestamp','is_walking','latitude','longitude','area_label',
             'mean_rms_LEFT','stride_duration_LEFT','stride_length_LEFT',
             'cadence_LEFT','pace_LEFT']
).sort_values('Timestamp')

# ---- MAIN LOOP ----
rows = []
current_event_idx   = 0
last_event_type     = 'NONE'
last_event_duration = 0.0
group_i             = 0

for entry in frame_logs:
    time_s = entry.get('time_s', np.nan)
    gaze_x, gaze_y = _parse_gaze(entry.get('gaze'))
    objects = _parse_objects(entry.get('objects', []))

    # --- assign current REMoDNaV event ---
    while (current_event_idx < len(events) and time_s >= events[current_event_idx]['offset_abs']):
        current_event_idx += 1
    if (current_event_idx < len(events) and
        events[current_event_idx]['onset_abs'] <= time_s < events[current_event_idx]['offset_abs']):
        ev = events[current_event_idx]
        last_event_type     = ev['event_type']
        last_event_duration = ev['duration']
    else:
        last_event_type     = 'NONE'
        last_event_duration = 0.0

    # --- objects by class ---
    persons = [o for o in objects if o['class'] == 'person']
    bikes   = [o for o in objects if o['class'] == 'bicycle']
    cars    = [o for o in objects if o['class'] == 'car']

    # --- cyclist matching (person+bicycle) ---
    matches = _match_cyclists(persons, bikes)  # list of (pi, bi)
    number_people_total      = len(persons)
    number_bicycle_total     = len(bikes)
    number_cyclist           = len(matches)
    number_people_nocyclist  = max(number_people_total  - number_cyclist, 0)
    number_bicycle_standalone= max(number_bicycle_total - number_cyclist, 0)

    cyclist_present  = number_cyclist > 0
    bicycle_present  = number_bicycle_standalone > 0  # standalone bikes only
    people_present   = number_people_total > 0     # cyclists included
    car_present      = len(cars) > 0

    # --- gaze-based fixation target (first bbox containing gaze) ---
    fixated = 'other'
    if np.isfinite(gaze_x) and np.isfinite(gaze_y):
        for obj in objects:
            if _point_in_bbox(gaze_x, gaze_y, obj['bbox']):
                fixated = obj['class']
                break

    # --- looking at cyclist (either the person or the matched bike) ---
    looking_cyclist = False
    if np.isfinite(gaze_x) and np.isfinite(gaze_y) and cyclist_present:
        for (pi, bi) in matches:
            if _point_in_bbox(gaze_x, gaze_y, persons[pi]['bbox']) or _point_in_bbox(gaze_x, gaze_y, bikes[bi]['bbox']):
                looking_cyclist = True
                # keep 'fixated' as-is; downstream will exclude cyclists from "people" using this flag
                break

    # --- merge downsampled depth by index ---
    if group_i < len(depth_summary):
        ds   = depth_summary.at[group_i, 'depth_d_s']
        dv   = depth_summary.at[group_i, 'depth_d_vergence']
        dcal = depth_summary.at[group_i, 'depth_calib_mm']
        hp   = depth_summary.at[group_i, 'depth_head_pitch_deg']
        lf   = depth_summary.at[group_i, 'depth_looking_floor']
    else:
        ds, dv, dcal, hp, lf = np.nan, np.nan, np.nan, np.nan, False
    group_i += 1

    rows.append({
        'time_s':                   time_s,
        'gaze_x':                   gaze_x,
        'gaze_y':                   gaze_y,
        'fixated':                  fixated,
        'number_people':            number_people_total,
        'number_cyclist':           number_cyclist,
        'number_people_nocyclist':  number_people_nocyclist,
        'number_bicycle_total':     number_bicycle_total,
        'number_bicycle_standalone':number_bicycle_standalone,
        'people_present':           people_present,      # cyclists included
        'cyclist_present':          cyclist_present,
        'bicycle_present':          bicycle_present,     # standalone bikes
        'car_present':              car_present,
        'looking_cyclist':          looking_cyclist,
        'event_label':              last_event_type,
        'event_duration':           last_event_duration,
        'depth_d_s':                ds,      # smoothed fused depth (mm)
        'depth_d_vergence':         dv,      # raw vergence depth (mm)
        'depth_calib_mm':           dcal,    # calibrated (pre-smoothing) depth (mm)
        'depth_head_pitch_deg':     hp,
        'depth_looking_floor':      bool(lf)
    })

# ---- BUILD DF & MERGE GAIT ----
df = pd.DataFrame(rows)

# Suppress floor-looking when fixated on something
df.loc[df['fixated'] != 'other', 'depth_looking_floor'] = False

df = df.sort_values('time_s')
df = pd.merge_asof(
    df,
    gait_df,
    left_on='time_s',
    right_on='Timestamp',
    direction='backward'
).drop(columns=['Timestamp'])

# ---- ROLLING FEATURES ----
window_s = 5.0
times = df['time_s'].values
xs    = df['gaze_x'].values
ys    = df['gaze_y'].values

ent = np.full(len(df), np.nan)
rog = np.full(len(df), np.nan)
for i, t in enumerate(times):
    mask = (times > t - window_s) & (times <= t)
    xw = xs[mask]; yw = ys[mask]
    valid = np.isfinite(xw) & np.isfinite(yw)
    xw = xw[valid]; yw = yw[valid]
    if len(xw) >= 2:
        ent[i] = spatial_entropy(xw, yw, n_bins=200)
        rog[i] = radius_of_gyration_deg(xw, yw)

df['spatial_entropy']    = ent
df['radius_of_gyration'] = rog

# ---- SAVE ----
os.makedirs(output_dir, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved visual events to {OUTPUT_CSV} with cyclist/bicycle/person separation")
