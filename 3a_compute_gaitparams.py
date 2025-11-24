#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# === Load metadata ===
meta_file_path = r'C:\LocoGaze\data\metadata.csv'
meta_df = pd.read_csv(meta_file_path, nrows=1)
reldirectory = meta_df.at[0, 'reldir']

# Leg length (cm -> m), fallback to 0.9 m
try:
    leg_length_cm = float(meta_df.at[0, 'leg_length'])
    leg_length = leg_length_cm / 100.0
    if not np.isfinite(leg_length) or leg_length <= 0:
        leg_length = 0.9
except Exception:
    leg_length = 0.9

# Thresholds
rms_thresh_g = meta_df.get('rms_thresh_g', pd.Series([0.5])).iat[0]
try:
    rms_thresh_g = float(rms_thresh_g)
except Exception:
    rms_thresh_g = 0.5

# Directories
data_directory = r'C:/LocoGaze/data/'
input_directory = os.path.join(data_directory, reldirectory)
output_folder = os.path.join(input_directory, 'output')
os.makedirs(output_folder, exist_ok=True)


# ---------- Helpers ----------

def instantaneous_cadence(time, acc_x, acc_y, acc_z,
                          min_peak_dist_s=0.35,
                          smooth_sigma=1):
    """Cadence (spm) from 3-axis accel magnitude using STEP peaks."""
    if len(time) < 2:
        return np.array([])
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    if acc_mag.size > 3 and smooth_sigma > 0:
        acc_mag = gaussian_filter1d(acc_mag, sigma=smooth_sigma)
    fs_est = 1.0 / np.median(np.diff(time))
    min_dist = max(1, int(min_peak_dist_s * fs_est))
    prom = np.nanstd(acc_mag) * 0.25
    if not np.isfinite(prom) or prom <= 0:
        prom = 0.05
    peaks, _ = find_peaks(acc_mag, distance=min_dist, prominence=prom)
    if peaks.size < 2:
        return np.array([])
    t_peaks = time[peaks]
    intervals = np.diff(t_peaks)
    valid = (intervals > 0.3) & (intervals < 2.0)
    if valid.sum() < 1:
        return np.array([])
    cadence = 60.0 / intervals[valid]
    if cadence.size > 1 and smooth_sigma > 0:
        cadence = gaussian_filter1d(cadence, sigma=smooth_sigma)
    return cadence


def build_gpx_speed_series(gpx_df):
    """Per-second GPS speed series (m/s) from cumulative distance (km)."""
    gpx_df = gpx_df.copy()
    if 'time_seconds' in gpx_df.columns and 'Timestamp' not in gpx_df.columns:
        gpx_df = gpx_df.rename(columns={'time_seconds': 'Timestamp'})
    gpx_df = gpx_df.sort_values('Timestamp').reset_index(drop=True)

    if 'cum_dist_km' in gpx_df.columns:
        t = gpx_df['Timestamp'].to_numpy(float)
        d_km = gpx_df['cum_dist_km'].to_numpy(float)
        dt = np.diff(t)
        dd_m = np.diff(d_km) * 1000.0
        spd = np.divide(dd_m, dt, out=np.zeros_like(dd_m), where=dt > 0)
        spd = np.r_[spd, spd[-1] if spd.size else 0.0]
        gpx_df['speed_ms'] = spd
    else:
        gpx_df['speed_ms'] = np.nan

    t0 = int(np.floor(gpx_df['Timestamp'].iloc[0]))
    t1 = int(np.ceil(gpx_df['Timestamp'].iloc[-1]))
    sec_index = pd.Index(np.arange(t0, t1 + 1), name='Timestamp')
    return (gpx_df[['Timestamp','speed_ms']]
            .set_index('Timestamp')
            .reindex(sec_index)
            .interpolate(limit_direction='both')['speed_ms'])


def merge_close_peaks(t, pks, min_interval=0.22):
    """
    Remove only unrealistically close duplicates (likely within-step ripples).
    Keeps the first peak, drops subsequent peaks occurring within min_interval seconds.
    """
    if pks.size == 0:
        return pks
    keep = [pks[0]]
    for idx in pks[1:]:
        if t[idx] - t[keep[-1]] >= min_interval:
            keep.append(idx)
    return np.asarray(keep, dtype=int)


def classify_peak_train(t, pks):
    """
    Classify peak train as 'step' or 'stride' based on median interval.
      - Step intervals typically ~0.35–1.2 s
      - Stride (same foot) intervals typically ~0.7–2.4 s
    If interval lies in both (overlap), choose based on threshold 1.0 s.
    """
    if pks.size < 2:
        return 'unknown', np.array([])
    intervals = np.diff(t[pks])
    med = np.median(intervals) if intervals.size else np.nan
    if not np.isfinite(med):
        return 'unknown', intervals
    # Ranges overlap; use a simple decision boundary at ~1.0 s
    if 0.35 <= med < 1.0:
        return 'step', intervals
    if med >= 1.0 and med <= 2.4:
        return 'stride', intervals
    # Fallback heuristics
    if med < 0.35:
        return 'step', intervals
    if med > 2.4:
        return 'stride', intervals
    return 'unknown', intervals


def compute_metrics_per_second(df, fs, centers, leg_length,
                               window_s=10.0,
                               gpx_speed_sec=None,
                               rms_thresh_g=0.5):
    """
    Per-second gait metrics with:
      - merged (not thinned-away) peaks
      - classification of peak train
      - cadence computed as steps/min (stride peaks doubled)
      - stride integration windows chosen per classification
    """
    g = 9.80665
    has_left_acc = all(c in df.columns for c in ['LEFT_ACC_X','LEFT_ACC_Y','LEFT_ACC_Z'])
    has_left_gyro = ('LEFT_GYRO_Y' in df.columns)

    if has_left_acc and 'LEFT_ACC_M' not in df.columns:
        df['LEFT_ACC_M'] = np.sqrt(df['LEFT_ACC_X']**2 + df['LEFT_ACC_Y']**2 + df['LEFT_ACC_Z']**2)

    half_w = window_s / 2.0
    half_walk = 0.5
    recs = []

    for tc in centers:
        win1  = df[(df['Timestamp'] >= tc - half_walk) & (df['Timestamp'] < tc + half_walk)]
        win10 = df[(df['Timestamp'] >= tc - half_w) & (df['Timestamp'] < tc + half_w)]

        rec = {
            'Timestamp': tc,
            'is_walking': False,
            'mean_rms_LEFT': np.nan,
            'stride_duration_LEFT': np.nan,
            'stride_length_LEFT': np.nan,
            'step_length_LEFT': np.nan,
            'cadence_LEFT': np.nan,
            'pace_LEFT': np.nan
        }
        if not has_left_acc or win10.empty:
            recs.append(rec); continue

        # Walk flag (1 s RMS in g)
        if not win1.empty:
            rms_l_win = float(np.sqrt(np.nanmean(win1['LEFT_ACC_M']**2)))
            rec['is_walking'] = bool(rms_l_win > rms_thresh_g)
            rec['mean_rms_LEFT'] = float(np.nanmean(win1['LEFT_ACC_M']) * g)

        # Signals
        t = win10['Timestamp'].to_numpy(float)
        acc_mag_ms2 = win10['LEFT_ACC_M'].to_numpy(float) * g
        acc_smooth = gaussian_filter1d(acc_mag_ms2, sigma=1) if acc_mag_ms2.size > 3 else acc_mag_ms2

        # Peaks
        if t.size > 1:
            fs_est = 1.0 / np.median(np.diff(t))
        else:
            fs_est = fs
        # Allow fairly close peaks; rely on prominence + merge for clean train
        min_dist = max(1, int(0.25 * fs_est))
        prom = np.nanstd(acc_smooth) * 0.35
        if not np.isfinite(prom) or prom <= 0:
            prom = 0.1
        pks, _ = find_peaks(acc_smooth, distance=min_dist, prominence=prom)
        # Merge near-duplicate peaks (<0.22 s apart)
        pks = merge_close_peaks(t, pks, min_interval=0.22)

        # Classify peak train and compute cadence as steps/min
        peak_kind, intervals = classify_peak_train(t, pks)
        c_mean = np.nan
        if intervals.size:
            mean_int = float(np.nanmean(intervals))
            if peak_kind == 'step':
                c_mean = 60.0 / mean_int
            elif peak_kind == 'stride':
                c_mean = 120.0 / mean_int  # stride → steps/min
            rec['cadence_LEFT'] = float(c_mean)

        # Stride durations (seconds)
        if peak_kind == 'step' and pks.size >= 3:
            stride_ints = t[pks[2:]] - t[pks[:-2]]            # i -> i+2
        elif peak_kind == 'stride' and pks.size >= 2:
            stride_ints = np.diff(t[pks])                     # i -> i+1
        else:
            stride_ints = np.array([])
        if stride_ints.size:
            rec['stride_duration_LEFT']     = float(np.nanmean(stride_ints))
        # Gyro integration for stride length
        stride_lens = []
        if has_left_gyro and pks.size >= (3 if peak_kind == 'step' else 2):
            omega = win10['LEFT_GYRO_Y'].to_numpy(float)
            if omega.size:
                w95 = np.nanpercentile(np.abs(omega), 95)
                if np.isfinite(w95) and w95 > 20:  # likely deg/s
                    omega = omega * (np.pi / 180.0)
            if omega.size == t.size:
                if peak_kind == 'step':
                    # integrate over (i -> i+2)
                    for i in range(len(pks) - 2):
                        i0, i2 = pks[i], pks[i+2]
                        seg_t = t[i0:i2+1]
                        seg_w = omega[i0:i2+1]
                        if seg_t.size < 2 or np.any(~np.isfinite(seg_w)):
                            continue
                        angle = abs(np.trapz(seg_w, seg_t))
                        angle = float(max(0.0, angle))
                        stride_len = 2.0 * leg_length * np.sin(angle / 2.0)
                        if stride_len > 0:
                            stride_lens.append(stride_len)
                elif peak_kind == 'stride':
                    # integrate over (i -> i+1)
                    for i in range(len(pks) - 1):
                        i0, i1 = pks[i], pks[i+1]
                        seg_t = t[i0:i1+1]
                        seg_w = omega[i0:i1+1]
                        if seg_t.size < 2 or np.any(~np.isfinite(seg_w)):
                            continue
                        angle = abs(np.trapz(seg_w, seg_t))
                        angle = float(max(0.0, angle))
                        stride_len = 2.0 * leg_length * np.sin(angle / 2.0)
                        if stride_len > 0:
                            stride_lens.append(stride_len)

        # Aggregate from gyro
        if stride_lens:
            stride_lens = np.array(stride_lens, dtype=float)
            rec['stride_length_LEFT']      = float(np.nanmean(stride_lens))
            rec['step_length_LEFT']        = float(np.nanmean(stride_lens) / 2.0)

        # GPS + cadence guardrails for plausibility
        if np.isfinite(c_mean) and c_mean > 0:
            v_ms = np.nan
            if gpx_speed_sec is not None:
                t0 = int(max(0, np.floor(tc - half_w)))
                t1 = int(np.floor(tc + half_w))
                sp = gpx_speed_sec.reindex(np.arange(t0, t1 + 1))
                if sp is not None:
                    v_ms = float(np.nanmean(sp.to_numpy()))
            if np.isfinite(v_ms) and v_ms > 0:
                expected_step   = v_ms / (c_mean / 60.0)
                expected_stride = 2.0 * expected_step
                # If computed stride exists but is <50% or >200% of expected, replace with GPS-implied
                if np.isfinite(rec['stride_length_LEFT']):
                    if (rec['stride_length_LEFT'] < 0.5 * expected_stride) or (rec['stride_length_LEFT'] > 2.0 * expected_stride):
                        rec['stride_length_LEFT'] = expected_stride
                        rec['step_length_LEFT']   = expected_step
                else:
                    rec['stride_length_LEFT'] = expected_stride
                    rec['step_length_LEFT']   = expected_step

        # Pace (km/h)
        if np.isfinite(rec['step_length_LEFT']) and np.isfinite(c_mean) and c_mean > 0:
            rec['pace_LEFT'] = float((rec['step_length_LEFT'] * c_mean / 60.0) * 3.6)

        recs.append(rec)

    return pd.DataFrame.from_records(recs)


# ---------- Main ----------

if __name__ == '__main__':
    # IMU
    imu_file = os.path.join(output_folder, 'realigned_feet.csv')
    if not os.path.exists(imu_file):
        raise FileNotFoundError(f"Cannot find '{imu_file}'")
    df = pd.read_csv(imu_file)
    if 'Timestamp' not in df.columns:
        raise ValueError("realigned_feet.csv must contain 'Timestamp'.")

    df['Timestamp'] = df['Timestamp'] - df['Timestamp'].iloc[0]
    df.sort_values('Timestamp', inplace=True, kind='mergesort')
    df.reset_index(drop=True, inplace=True)

    # GPX
    gpx_file = os.path.join(output_folder, 'realigned_gpx.csv')
    if not os.path.exists(gpx_file):
        raise FileNotFoundError(f"Cannot find '{gpx_file}'")
    gpx_df = pd.read_csv(gpx_file)
    gpx_speed_sec = build_gpx_speed_series(gpx_df)

    # Sampling
    fs = 100.0
    t_end = float(df['Timestamp'].iloc[-1])
    centers = np.arange(0.0, np.floor(t_end) + 1.0, 1.0)

    # Metrics
    gait_df = compute_metrics_per_second(
        df=df,
        fs=fs,
        centers=centers,
        leg_length=leg_length,
        window_s=10.0,
        gpx_speed_sec=gpx_speed_sec,
        rms_thresh_g=rms_thresh_g
    )

    # GPS average speed (km/h)
    gpx_tmp = gpx_df.copy()
    if 'time_seconds' in gpx_tmp.columns and 'Timestamp' not in gpx_tmp.columns:
        gpx_tmp = gpx_tmp.rename(columns={'time_seconds': 'Timestamp'})
    gpx_tmp = gpx_tmp.sort_values('Timestamp')
    if 'cum_dist_km' in gpx_tmp.columns and not gait_df.empty:
        max_dist_km = float(np.nanmax(gpx_tmp['cum_dist_km'].to_numpy()))
        last_time_s = float(np.nanmax(gait_df['Timestamp'].to_numpy()))
        gps_speed = max_dist_km / (last_time_s / 3600.0) if last_time_s > 0 else np.nan
    else:
        gps_speed = np.nan

    # IMU average pace (km/h)
    imu_speed = float(np.nanmean(gait_df[['pace_LEFT']].to_numpy()))

    # Scale pace to GPS
    scale = gps_speed / imu_speed if (np.isfinite(gps_speed) and np.isfinite(imu_speed) and imu_speed > 0) else 1.0
    if np.isfinite(scale) and scale > 0 and scale != 1.0:
        pace_cols = [c for c in gait_df.columns if c.startswith('pace_')]
        for col in pace_cols:
            gait_df[col] = gait_df[col] * scale

    # Merge GPX labels
    gpx_align = gpx_df.copy()
    if 'time_seconds' in gpx_align.columns and 'Timestamp' not in gpx_align.columns:
        gpx_align = gpx_align.rename(columns={'time_seconds': 'Timestamp'})
    gpx_align = gpx_align.sort_values('Timestamp')
    keep_cols = [c for c in ['Timestamp','area_label','latitude','longitude'] if c in gpx_align.columns]
    if 'Timestamp' not in keep_cols:
        raise ValueError("realigned_gpx.csv must contain 'Timestamp' or 'time_seconds'.")
    gpx_subset = gpx_align[keep_cols]
    gait_full = pd.merge(gait_df, gpx_subset, on='Timestamp', how='left')

    # Save
    out_file = os.path.join(output_folder, 'gait_metrics.csv')
    gait_full.to_csv(out_file, index=False)
    print(f"Saved gait_metrics.csv ({len(gait_full)} rows) with GPX labels & coords")
