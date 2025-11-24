import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
import warnings

#=== Load metadata ===
meta_file_path = r'C:\LocoGaze\data\metadata.csv'
meta_df        = pd.read_csv(meta_file_path, nrows=1)
reldirectory   = meta_df.at[0, 'reldir']
base_dir       = r'C:/LocoGaze/data/'
input_dir      = os.path.join(base_dir, reldirectory)
output_dir     = os.path.join(input_dir, 'output')

# File paths
ORIGIN_PATH    = os.path.join(output_dir, 'realigned_gaze_origin3d.csv')
DIRECTION_PATH = os.path.join(output_dir, 'realigned_gaze_direction3d.csv')
D2D_PATH       = os.path.join(output_dir, 'realigned_gaze2d.csv')  # normalized (0-1) video coords
HEAD_PATH      = os.path.join(output_dir, 'realigned_head.csv')
OUTPUT_PATH    = os.path.join(output_dir, 'gaze_depth.csv')

# Parameters
stature_mm    = meta_df.at[0, 'height'] * 10  # cm→mm
EYE_RATIO     = 0.934
EYE_HEIGHT_MM = stature_mm * EYE_RATIO
IPD_MM        = 70.0
TS_TOL        = 0.01

# Time windows in seconds (will convert if ms)
windows = {
    'depth1':  (meta_df.at[0, 'depth_1_start'], meta_df.at[0, 'depth_1_stop']),
    'depth2':  (meta_df.at[0, 'depth_2_start'], meta_df.at[0, 'depth_2_stop']),
    'depth3':  (meta_df.at[0, 'depth_3_start'], meta_df.at[0, 'depth_3_stop']),
    'depth4':  (meta_df.at[0, 'depth_4_start'], meta_df.at[0, 'depth_4_stop']),
    'central': (meta_df.at[0, 'calib_central_start'], meta_df.at[0, 'calib_central_stop']),
}
# Use only 1m, 2m, and 4m for calibration (exclude 3m)
truth_m = {'depth1': 1, 'depth4': 4}

# Thresholds
CONF_THR = 0.1
VERG_THR = 0.1
MAX_FAR_DEPTH = 7000  # mm
G2D_Y_THRESH  = 0.25   # used to tighten geometric floor hits
G2D_Y_THRESH2 = 0.60   # bottom part of frame in normalized [0,1]

# --- Helper functions ---
def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = np.nan
    return v / n

def sg_filter_safe(x, window=201, poly=3):
    """Savitzky–Golay with guards: odd window <= len(x), >= poly+2."""
    n = len(x)
    if n < 5:
        return x  # too short, skip smoothing
    w = min(window, n - (1 - n % 2))  # make it odd and <= n
    if w < poly + 2:
        w = poly + 2 + (1 - (poly + 2) % 2)  # smallest odd >= poly+2
        if w > n:
            return x
    return savgol_filter(x, w, poly, mode='interp')

def compute_metrics(df):
    """Compute vergence depth, floor-plane ground-projected depth, gaze pitch, etc."""
    dimension = 1  # y-axis index
    O_L = df[['OLx','OLy','OLz']].values
    O_R = df[['ORx','ORy','ORz']].values
    O_mid = (O_L + O_R) / 2.0

    D_L = normalize(df[['DLx','DLy','DLz']].values)
    D_R = normalize(df[['DRx','DRy','DRz']].values)
    D_mid = normalize((D_L + D_R) / 2.0)

    # Vergence depth
    dots  = np.einsum('ij,ij->i', D_L, D_R).clip(-1, 1)
    valid = dots > VERG_THR
    theta = np.arccos(dots)
    tan_h = np.tan(theta / 2.0)
    tan_h[tan_h == 0] = np.nan
    d_verg = IPD_MM / (2.0 * tan_h)
    d_verg[~valid] = np.nan

    # Floor-plane intersection depth along ground (XZ) direction
    O_y = O_mid[:, dimension]
    D_y = D_mid[:, dimension]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (EYE_HEIGHT_MM - O_y) / D_y

    # ✅ Correct horizontal magnitude in XZ plane (bug fix)
    horiz = np.sqrt(D_mid[:, 0]**2 + D_mid[:, 2]**2)

    hit_floor = (D_y > 0) & (t > 0)
    d_plane   = np.full_like(t, np.nan)
    d_plane[hit_floor] = t[hit_floor] * horiz[hit_floor]

    pitch = np.degrees(np.arcsin(D_mid[:, 1]))
    depth_use = np.where(~np.isnan(d_verg), d_verg, t * horiz)
    gaze_point_y = O_y + D_y * depth_use

    out = df.copy()
    out['d_vergence']   = d_verg
    out['d_plane_mm']   = d_plane
    out['hit_floor']    = hit_floor
    out['pitch_deg']    = pitch
    out['gaze_point_y'] = gaze_point_y
    return out

def calibrate_robust(df):
    mask = df.d_vergence.notna() & df.true_depth_mm.notna()
    X = df.loc[mask, 'd_vergence'].values.reshape(-1, 1)
    y = df.loc[mask, 'true_depth_mm'].values
    r = RANSACRegressor(LinearRegression(), min_samples=0.5, residual_threshold=200)
    r.fit(X, y)
    return r.estimator_.coef_[0], r.estimator_.intercept_

def calibrate_pitch_threshold(df):
    m_c = df.loc[df.Timestamp.between(*windows['central']), 'head_pitch_deg'].mean()
    m_f = df.loc[df.Timestamp.between(*windows['depth4']),   'head_pitch_deg'].mean()
    return m_c - 0.65 * (m_c - m_f)

def calibrate_gaze_threshold(df):
    m_c = df.loc[df.Timestamp.between(*windows['central']), 'pitch_deg'].mean()
    m_f = df.loc[df.Timestamp.between(*windows['depth4']),   'pitch_deg'].mean()
    return m_c - 0.50 * (m_c - m_f)

def _window_median_pairs(df, truth_map, use_floor_only=True):
    """Return (x_mm, y_mm) pairs using median raw vergence in each GT window."""
    xs, ys = [], []
    for name, meters in truth_map.items():
        t0, t1 = windows[name]
        sel = df['Timestamp'].between(t0, t1) & df['d_vergence'].notna()
        if use_floor_only and 'looking_floor' in df:
            sel &= df['looking_floor'] == True
        if not sel.any():
            continue
        # trim outliers within the window (robust median)
        v = df.loc[sel, 'd_vergence'].astype(float)
        lo, hi = np.nanpercentile(v, [10, 90])  # tighten to central mass
        v = v[(v >= lo) & (v <= hi)]
        if len(v) < 10:
            continue
        xs.append(np.nanmedian(v))            # raw vergence median (mm)
        ys.append(float(meters) * 1000.0)     # true depth (mm)
    return np.array(xs), np.array(ys)

def _fit_constrained_linear(x, y):
    """
    Fit y ≈ m*x + b with basic constraints / fallbacks.
    Returns m, b and a note about the path taken.
    """
    if x.size == 0:
        return 1.0, 0.0, "fallback_identity(no_pairs)"
    x = x.astype(float); y = y.astype(float)

    # If we have >=3 points, do robust LS on medians
    if x.size >= 3:
        m, b = np.polyfit(x, y, 1)
        path = "polyfit3+"
    elif x.size == 2:
        # 2-point exact fit
        dx = (x[1] - x[0])
        if abs(dx) < 1e-6:
            return 1.0, 0.0, "fallback_identity(degenerate_2pt)"
        m = (y[1] - y[0]) / dx
        b = y[0] - m * x[0]
        path = "2point"
    else:
        # Only one window available: gain‑only fit (no intercept)
        if x[0] <= 0:
            return 1.0, 0.0, "fallback_identity(one_nonpos)"
        m = y[0] / x[0]
        b = 0.0
        path = "gain_only_1pt"

    # Enforce sane constraints
    if not np.isfinite(m) or not np.isfinite(b):
        return 1.0, 0.0, f"fallback_identity(nan_fit_{path})"
    # physical sanity: positive slope; intercept not huge
    if m <= 0:
        # try gain‑only fit using the farthest two points if we can
        idx = np.argsort(x)
        if idx.size >= 2:
            xa, xb = x[idx[0]], x[idx[-1]]
            ya, yb = y[idx[0]], y[idx[-1]]
            if abs(xb - xa) > 1e-6:
                m = (yb - ya) / (xb - xa)
                b = ya - m * xa
                path += "=>gain_from_extremes"
    if m <= 0:
        # final guard: identity
        return 1.0, 0.0, f"fallback_identity(nonpos_slope_{path})"

    # clamp ridiculous intercepts that cause flatlining
    if abs(b) > 1500 and x.size >= 2:
        # re-center by forcing line to pass through the median pair
        xm, ym = np.median(x), np.median(y)
        b = ym - m * xm
        path += "=>recentred"

    return float(m), float(b), path


def run():
    global IPD_MM, EYE_HEIGHT_MM

    print("Calibrating using true-depth windows: 1 m, 2 m, 3 m, 4 m.")

    # 1) Load & merge 3D gaze origin + direction
    orig = pd.read_csv(ORIGIN_PATH)
    dirs = pd.read_csv(DIRECTION_PATH)
    for d in (orig, dirs):
        d['Timestamp'] = pd.to_numeric(d['Timestamp'], errors='coerce')
    orig.rename(columns={'X_L':'OLx','Y_L':'OLy','Z_L':'OLz',
                         'X_R':'ORx','Y_R':'ORy','Z_R':'ORz'}, inplace=True)
    dirs.rename(columns={'X_L':'DLx','Y_L':'DLy','Z_L':'DLz',
                         'X_R':'DRx','Y_R':'DRy','Z_R':'DRz'}, inplace=True)
    orig.sort_values('Timestamp', inplace=True)
    dirs.sort_values('Timestamp', inplace=True)
    df = pd.merge_asof(orig, dirs, on='Timestamp', tolerance=TS_TOL, direction='nearest')

    # 1b) Load & merge normalized 2D gaze (detect Y-column automatically)
    try:
        d2d = pd.read_csv(D2D_PATH)
        d2d['Timestamp'] = pd.to_numeric(d2d['Timestamp'], errors='coerce')
        d2d.sort_values('Timestamp', inplace=True)
        y_cols = [c for c in d2d.columns if c.lower().endswith('y') and 'timestamp' not in c.lower()]
        if not y_cols:
            raise KeyError('No Y-coordinate column found')
        gaze2d_col = y_cols[0]
        df = pd.merge_asof(df, d2d[['Timestamp', gaze2d_col]],
                           on='Timestamp', tolerance=TS_TOL, direction='nearest')
        df.rename(columns={gaze2d_col: 'gaze2d_y'}, inplace=True)
    except Exception as e:
        df['gaze2d_y'] = np.nan
        warnings.warn(f"2D gaze Y merge skipped: {e}")

    # 2) Unit conversion if needed
    if df.Timestamp.mean() > 1000:
        df['Timestamp'] /= 1000.0
        for k in windows:
            s, e = windows[k]
            windows[k] = (s / 1000.0, e / 1000.0)

    # 3) Confidence filter (if two conf columns exist)
    confs = [c for c in df.columns if 'conf' in c.lower()]
    if len(confs) >= 2:
        df = df[(df[confs[0]] >= CONF_THR) & (df[confs[1]] >= CONF_THR)]

    # 4) IPD & eye height from central window
    sel = df.Timestamp.between(*windows['central'])
    if sel.any():
        OL = df.loc[sel, ['OLx','OLy','OLz']].values
        OR = df.loc[sel, ['ORx','ORy','ORz']].values
        ipd = np.mean(np.linalg.norm(OL - OR, axis=1))
        eh  = np.mean((OL[:, 0] + OR[:, 0]) / 2)
        if 50 < ipd < 80 and 0 < eh < stature_mm:
            IPD_MM, EYE_HEIGHT_MM = ipd, eh

    # 5) Compute raw metrics & assign true depths
    df_raw = compute_metrics(df)
    df_raw['true_depth_mm'] = np.nan
    for name, mult in truth_m.items():
        t0, t1 = windows[name]
        df_raw.loc[df_raw.Timestamp.between(t0, t1), 'true_depth_mm'] = mult * 1000

        # 6) Calibrate vergence with robust, constrained mapping on window medians
    x_med, y_med = _window_median_pairs(df_raw, truth_m, use_floor_only=True)
    m, b, path = _fit_constrained_linear(x_med, y_med)
    print(f"Calibration fit: m={m:.4f}, b={b:.1f} mm  [{path}], "
          f"pairs_used={len(x_med)} (x_med={np.round(x_med,1)}, y_med={np.round(y_med).astype(int) if len(y_med) else y_med})")

    # apply to vergence; keep plane as fallback when vergence is NaN
    df_raw['depth_calib_mm'] = (df_raw['d_vergence'] * m + b).astype(float)
    df_raw.loc[~np.isfinite(df_raw['depth_calib_mm']), 'depth_calib_mm'] = np.nan
    df_raw['depth_calib_mm'].fillna(df_raw['d_plane_mm'], inplace=True)


    # 7) Load & merge head-pose
    head = pd.read_csv(HEAD_PATH)
    head['Timestamp'] = pd.to_numeric(head['Timestamp'], errors='coerce')
    head.sort_values('Timestamp', inplace=True)
    head['head_pitch_deg'] = np.degrees(np.arctan2(head.ACC_Z, head.ACC_Y))

    df_all = pd.merge_asof(
        df_raw.sort_values('Timestamp'),
        head[['Timestamp', 'head_pitch_deg']],
        on='Timestamp', tolerance=TS_TOL, direction='nearest'
    )

    # 8) Dynamic thresholds
    head_thresh = calibrate_pitch_threshold(df_all)
    gaze_thresh = calibrate_gaze_threshold(df_all)
    print(f"Head threshold={head_thresh:.1f}°, Gaze threshold={gaze_thresh:.1f}°")

    # 9) Smooth signals (safe Savitzky–Golay)
    df_all['hp_s'] = sg_filter_safe(df_all.head_pitch_deg.values, window=201, poly=3)
    df_all['gp_s'] = sg_filter_safe(df_all.pitch_deg.values,      window=201, poly=3)
    df_all['d_s']  = sg_filter_safe(df_all.depth_calib_mm.values, window=201, poly=1)

    # 10) Floor detection: combine cues; exclude far looks
    cond_hp       = df_all.hp_s < head_thresh
    cond_gp       = df_all.gp_s < gaze_thresh
    cond_gaze2d   = df_all.gaze2d_y > G2D_Y_THRESH
    cond_gaze2d2  = df_all.gaze2d_y > G2D_Y_THRESH2
    cond_not_far  = df_all.d_s <= MAX_FAR_DEPTH

    # Tighten geometric floor hit to also require gaze2d threshold
    df_raw['hit_floor'] = (df_raw['hit_floor'] & cond_gaze2d)

    # Final floor-looking rule
    df_all['looking_floor'] = (((df_raw['hit_floor'] & cond_hp) | cond_hp | cond_gaze2d2) & cond_not_far)

    # ** Print how many samples were floor-looks **
    n_floor = int(df_all['looking_floor'].sum())
    total   = len(df_all)
    print(f"Detected {n_floor} floor-look samples out of {total} total frames "
          f"({(n_floor/total*100 if total else 0):.1f}%)")

    # 10b) Summarize calibrated gaze depth by true depth (meters)
    depth_summary = (
        df_all[['true_depth_mm', 'd_s']]
        .dropna()
        .assign(true_depth_m = lambda x: x['true_depth_mm'] / 1000.0,
                gaze_depth_m = lambda x: x['d_s'] / 1000.0)
        .query('true_depth_m in [1, 2, 3, 4]')  # ensure 3 m excluded
        .groupby('true_depth_m', as_index=False)['gaze_depth_m']
        .mean()
        .rename(columns={'gaze_depth_m': 'mean_gaze_depth_m'})
    )

    print("\nAverage calibrated gaze depth by true depth (meters):")
    if len(depth_summary):
        for _, row in depth_summary.iterrows():
            print(f"  True {row.true_depth_m:.0f} m -> mean gaze depth = {row.mean_gaze_depth_m:.3f} m")
    else:
        print("  (No samples found in 1/2/4 m windows.)")

    # Persist summary
    summary_path = os.path.join(output_dir, 'depth_calibration_summary.csv')
    depth_summary.to_csv(summary_path, index=False)
    print(f"Saved depth calibration summary to {summary_path}")

    

    # 11) Plot results (head pitch with floor-look markers) — labels fixed
    plt.figure(figsize=(12,5))
    plt.plot(df_all.Timestamp, df_all.head_pitch_deg, label='Head pitch (deg)')
    mask = df_all.looking_floor
    plt.scatter(df_all.loc[mask, 'Timestamp'],
                df_all.loc[mask, 'head_pitch_deg'],
                s=5, label='Floor looks')
    plt.title('Floor look detection (head pitch)')
    plt.xlabel('Time (s)')
    plt.ylabel('Head pitch (deg)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    # 12) Save results — expanded columns including depth variants & source
    cols = [
        'Timestamp',
        'd_vergence',        # raw vergence depth (mm)
        'd_plane_mm',        # plane-intersection depth (mm)
        'depth_calib_mm',    # calibrated/fused depth (mm)
        'd_s',               # smoothed fused depth (mm)
        'depth_source',      # 'vergence' | 'plane' | 'unknown'
        'pitch_deg',         # gaze pitch (deg)
        'head_pitch_deg',    # head pitch (deg)
        'looking_floor',     # boolean
        'gaze2d_y',          # 2D gaze Y (norm)
        'hit_floor',         # geometric floor hit (tightened)
        'true_depth_mm'      # ground truth labels (only for 1,2,4 m windows)
    ]
    df_all.to_csv(OUTPUT_PATH, index=False, columns=[c for c in cols if c in df_all.columns])
    print(f"Saved output to {OUTPUT_PATH}")

if __name__ == '__main__':
    run()
