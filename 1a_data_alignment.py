import os
import json
import pyxdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Config knobs (tweak as needed)
# ==============================
SENSOR_RATE_GAZE      = 100.0   # Hz
SENSOR_RATE_HEAD_IMU  = 100.0   # Hz
SENSOR_RATE_EMOTIBIT  = 25.0    # Hz

SHORT_GAP_S_GAZE      = 0.10    # max duration to interpolate gaze gaps
SHORT_GAP_S_HEAD_FFILL= 0.20    # cap forward-fill for head IMU
ASOF_TOL_HEAD         = 0.02    # s tolerance merging head IMU streams
ASOF_TOL_FEET         = 0.04    # s tolerance aligning feet/chest

# ==============
# Helper funcs
# ==============
def interpolate_short_gaps(df, value_cols, max_gap_s=0.1, rate_hint=None):
    """
    Interpolate only small runs of NaNs. Longer runs remain NaN to avoid slow drifts.
    """
    out = df.copy()
    # estimate sampling rate if not provided
    if rate_hint is None and len(out) > 1:
        dt = np.median(np.diff(out['Timestamp'].to_numpy()))
        if dt > 0:
            rate_hint = 1.0 / dt
    if not rate_hint:
        return out

    max_consecutive = int(round(max_gap_s * rate_hint))
    if max_consecutive <= 0:
        return out

    out[value_cols] = out[value_cols].interpolate(
        method='linear',
        limit=max_consecutive,
        limit_direction='both',
        limit_area='inside'
    )
    return out

def ffill_short_gaps(df, value_cols, max_gap_s, rate_hint):
    """
    Forward-fill but only for short gaps; longer gaps remain NaN.
    """
    out = df.copy()
    limit = int(round(max_gap_s * rate_hint))
    if limit <= 0:
        return out
    out[value_cols] = out[value_cols].fillna(method='ffill', limit=limit)
    return out

def asof_align_to(base_ts, src_ts, src_vals, max_gap_s):
    """
    Nearest-neighbour alignment with tolerance; leaves NaN when no close sample.
    """
    base = pd.DataFrame({'Timestamp': base_ts})
    src  = pd.DataFrame({'Timestamp': src_ts, 'val': src_vals})
    aligned = pd.merge_asof(
        base.sort_values('Timestamp'),
        src.sort_values('Timestamp'),
        on='Timestamp',
        tolerance=max_gap_s,
        direction='nearest'
    )
    return aligned['val'].to_numpy()  # may contain NaNs

def make_pad(ts_pad, data_cols):
    """Create a NaN pad DataFrame with a Timestamp column matching ts_pad length."""
    pad = pd.DataFrame({'Timestamp': ts_pad})
    for c in data_cols:
        pad[c] = np.nan
    return pad

def realign_and_pad_template(df, lag, ref_end_ts, rate_hz, zero_base=True, clip_negative=True):
    """
    Generic realign: (optional) zero-base, add lag, clip negatives, NaN-pad start and end, and trim by end timestamp.
    """
    out = df.copy()
    data_cols = [c for c in out.columns if c != 'Timestamp']

    # Zero-base if requested, then add lag
    if zero_base and len(out):
        out['Timestamp'] = out['Timestamp'] - out['Timestamp'].min()
    out['Timestamp'] = out['Timestamp'] + float(lag)

    # Clip negatives to start at 0 exactly
    if clip_negative:
        out = out[out['Timestamp'] >= 0].reset_index(drop=True)

    # If everything got clipped (e.g., huge negative lag), build a fully padded frame 0..ref_end_ts
    if len(out) == 0:
        if np.isfinite(ref_end_ts) and ref_end_ts > 0:
            ts_pad = np.arange(0, ref_end_ts + 1e-12, 1.0 / rate_hz)
        else:
            ts_pad = np.array([0.0])
        return make_pad(ts_pad, data_cols)

    # Prepend NaN rows if first timestamp is after 0
    first_ts = float(out['Timestamp'].iloc[0])
    if first_ts > 0:
        n_rows = int(round(first_ts * rate_hz))
        if n_rows > 0:
            ts_pad = (1.0 / rate_hz) * np.arange(0, n_rows)
            pad = make_pad(ts_pad, data_cols)
            out = pd.concat([pad, out], ignore_index=True)

    # Pad/truncate at the end to match reference end
    end_ts = float(out['Timestamp'].iloc[-1])
    if np.isfinite(ref_end_ts):
        diff = ref_end_ts - end_ts
        if diff > 0:
            n_rows = int(round(diff * rate_hz))
            if n_rows > 0:
                ts_pad = end_ts + (1.0 / rate_hz) * np.arange(1, n_rows + 1)
                pad = make_pad(ts_pad, data_cols)
                out = pd.concat([out, pad], ignore_index=True)
        else:
            # Trim by timestamp (more robust than row counts)
            out = out[out['Timestamp'] <= ref_end_ts].reset_index(drop=True)

    return out


# ======================
# === Load metadata ====
# ======================
meta_file_path = r'C:\LocoGaze\data\metadata.csv'
meta_df = pd.read_csv(meta_file_path, nrows=1)
reldirectory = meta_df.at[0, 'reldir']
lag_value     = meta_df.at[0, 'lag']

data_directory  = 'C:/LocoGaze/data/'
input_directory = os.path.join(data_directory, reldirectory)
output_folder   = os.path.join(input_directory, 'output')
os.makedirs(output_folder, exist_ok=True)

# ===========================
# === Read gaze JSON TXT ====
# ===========================
def read_gaze_json_txt(file_path):
    timestamps, gaze2d_data = [], []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                js = json.loads(line)
                if 'gaze2d' in js.get('data', {}):
                    timestamps.append(js['timestamp'])
                    gaze2d_data.append(js['data']['gaze2d'])
            except json.JSONDecodeError:
                continue
    if not gaze2d_data:
        return pd.DataFrame(columns=['Timestamp','X','Y'])
    df = pd.DataFrame(gaze2d_data, columns=['X','Y'])
    df['Timestamp'] = timestamps
    return df[['Timestamp','X','Y']]

# ==========================
# === Read XDF 2D gaze  ====
# ==========================
def read_xdf_gaze2d(xdf_path, max_gap_s=SHORT_GAP_S_GAZE):
    streams, _ = pyxdf.load_xdf(xdf_path)
    for s in streams:
        if s['info']['name'][0] == 'Gaze2d':
            data = np.asarray(s['time_series'])
            if data.ndim == 1:
                data = data.reshape(-1, 2)
            df = pd.DataFrame(data, columns=['X','Y'])
            ts = np.asarray(s['time_stamps'])
            df['Timestamp'] = ts - ts[0]
            # Only fill short gaps to avoid drifts
            df = interpolate_short_gaps(df, ['X','Y'], max_gap_s=max_gap_s)
            return df
    return None

# =========================================
# === Generic reader for N-D vector stream
# =========================================
def read_xdf_vector3d(xdf_path, stream_name, max_gap_s=SHORT_GAP_S_GAZE):
    streams, _ = pyxdf.load_xdf(xdf_path)
    for s in streams:
        if s['info']['name'][0] == stream_name:
            data = np.asarray(s['time_series'])
            ts   = np.asarray(s['time_stamps'])
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            dims = data.shape[1]
            if dims == 3:
                cols = ['X','Y','Z']
            elif dims == 6:
                cols = ['X_L','Y_L','Z_L','X_R','Y_R','Z_R']
            else:
                cols = [f'V{i}' for i in range(dims)]
            df = pd.DataFrame(data, columns=cols)
            df['Timestamp'] = ts - ts[0]
            df = interpolate_short_gaps(df, cols, max_gap_s=max_gap_s)
            return df
    return None

# ==================================
# === Plot comparison (2D only)  ====
# ==================================
def plot_full_gaze_comparison(gaze_txt, gaze_xdf, title, output_path):
    plt.figure(figsize=(12,5))
    plt.plot(gaze_txt['Timestamp'], gaze_txt['X'], label='TXT Gaze X', alpha=0.7)
    plt.plot(gaze_xdf['Timestamp'], gaze_xdf['X'], label='XDF Gaze X (aligned)', alpha=0.7)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Gaze X Position')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ====================================
# === Realign & pad 2D/3D gaze     ===
# ====================================
def realign_and_pad_gaze2d(df, lag, ref_df, sampling_rate=SENSOR_RATE_GAZE):
    ref_end = float(ref_df['Timestamp'].iloc[-1]) if len(ref_df) else 0.0
    return realign_and_pad_template(df, lag, ref_end, sampling_rate, zero_base=True, clip_negative=True)

def realign_and_pad_gaze3d(df, lag, ref_df, sampling_rate=SENSOR_RATE_GAZE):
    ref_end = float(ref_df['Timestamp'].iloc[-1]) if len(ref_df) else 0.0
    return realign_and_pad_template(df, lag, ref_end, sampling_rate, zero_base=True, clip_negative=True)

# =======================
# === Process head IMU ==
# =======================
def process_head_IMU(xdf_path):
    streams, _ = pyxdf.load_xdf(xdf_path)
    acc_df, gyro_df, magn_df = None, None, None

    for s in streams:
        name = s['info']['name'][0]
        data = np.array(s['time_series'])
        ts = np.array(s['time_stamps'])

        if data.ndim == 2 and data.shape[1] == 3:
            # Try to get channel labels
            if 'desc' in s['info'] and 'channels' in s['info']['desc'][0]:
                channel_info = s['info']['desc'][0]['channels'][0].get('channel', [])
                labels = [ch['label'][0] for ch in channel_info] if channel_info else ['X', 'Y', 'Z']
            else:
                labels = ['X', 'Y', 'Z']

            df = pd.DataFrame(data, columns=labels)
            df['Timestamp'] = ts

            if name == 'Accelerometer':
                acc_df = df.rename(columns={labels[0]: 'ACC_X', labels[1]: 'ACC_Y', labels[2]: 'ACC_Z'})
            elif name == 'Gyroscope':
                gyro_df = df.rename(columns={labels[0]: 'GYRO_X', labels[1]: 'GYRO_Y', labels[2]: 'GYRO_Z'})
            elif name == 'Magnetometer':
                magn_df = df.rename(columns={labels[0]: 'MAG_X', labels[1]: 'MAG_Y', labels[2]: 'MAG_Z'})

    # Merge the three dataframes on timestamp using asof (±ASOF_TOL_HEAD)
    if acc_df is not None and gyro_df is not None and magn_df is not None:
        merged = pd.merge_asof(
            acc_df.sort_values('Timestamp'),
            gyro_df.sort_values('Timestamp'),
            on='Timestamp', tolerance=ASOF_TOL_HEAD, direction='nearest'
        )
        merged = pd.merge_asof(
            merged.sort_values('Timestamp'),
            magn_df.sort_values('Timestamp'),
            on='Timestamp', tolerance=ASOF_TOL_HEAD, direction='nearest'
        )
        return merged[['Timestamp', 'ACC_X', 'ACC_Y', 'ACC_Z',
                       'GYRO_X', 'GYRO_Y', 'GYRO_Z',
                       'MAG_X', 'MAG_Y', 'MAG_Z']]
    return None

def realign_and_normalize_head_imu(df, lag, gaze_last_timestamp, sensor_rate=SENSOR_RATE_HEAD_IMU):
    # realign with NaN padding, then cap ffill small gaps
    aligned = realign_and_pad_template(
        df, lag, gaze_last_timestamp, sensor_rate,
        zero_base=True, clip_negative=True
    )
    value_cols = [c for c in aligned.columns if c != 'Timestamp']
    aligned = ffill_short_gaps(aligned, value_cols, max_gap_s=SHORT_GAP_S_HEAD_FFILL, rate_hint=sensor_rate)
    return aligned

# =======================
# === Process foot IMUs =
# =======================
def process_foot_IMUs(xdf_path, left_id, CHEST_ID, max_gap_s=ASOF_TOL_FEET):
    AXES = ['ACC_X','ACC_Y','ACC_Z','GYRO_X','GYRO_Y','GYRO_Z','MAG_X','MAG_Y','MAG_Z']
    streams, _ = pyxdf.load_xdf(xdf_path)

    def first(x): return x[0] if isinstance(x,(list,tuple)) else x
    data = {left_id:{}, CHEST_ID:{}}
    for s in streams:
        name = first(s['info'].get('name','')); sid = first(s['info'].get('source_id',''))
        if name in AXES and sid in data:
            ts = np.array(s['time_stamps']); vals = np.asarray(s['time_series']).ravel()
            data[sid][name] = (ts, vals)

    dfs = {}
    for sid,label in [(left_id,'LEFT'),(CHEST_ID,'CHEST')]:
        if 'ACC_X' not in data[sid]:
            dfs[label] = None
            continue
        ts0, v0 = data[sid]['ACC_X']
        df = pd.DataFrame({'Timestamp': ts0, f'{label}_ACC_X': v0})
        for axis in AXES:
            if axis == 'ACC_X':
                continue
            col = f'{label}_{axis}'
            if axis in data[sid]:
                t, v = data[sid][axis]
                df[col] = asof_align_to(ts0, t, v, max_gap_s=max_gap_s)
            else:
                df[col] = np.nan
        dfs[label] = df
    return dfs

# =======================
# === Synchronize feet ==
# =======================
def synchronize_feet(left_df, CHEST_df):
    # choose base as the longer df
    base_df  = left_df if len(left_df) >= len(CHEST_df) else CHEST_df
    merge_df = CHEST_df if base_df is left_df else left_df
    cols     = [c for c in merge_df.columns if c != 'Timestamp']

    aligned = pd.merge_asof(
        base_df[['Timestamp']].sort_values('Timestamp'),
        merge_df[['Timestamp'] + cols].sort_values('Timestamp'),
        on='Timestamp', tolerance=ASOF_TOL_FEET, direction='nearest'
    )

    # find region where any aligned column is non-null
    if not aligned[cols].notnull().any(axis=1).any():
        # no overlap found within tolerance
        return pd.DataFrame(columns=list(base_df.columns) + cols)

    valid = aligned[cols].notnull().any(axis=1)
    start_idx = valid.idxmax()
    end_idx   = valid[::-1].idxmax()
    aligned   = aligned.loc[start_idx:end_idx].reset_index(drop=True)
    bsub      = base_df.loc[start_idx:end_idx].reset_index(drop=True)

    # interpolate small gaps within aligned columns (row-wise linear is fine; long gaps remain NaN)
    aligned[cols] = aligned[cols].interpolate(limit=5, limit_area='inside')

    out = pd.concat([bsub, aligned[cols]], axis=1)
    out['Timestamp'] -= out['Timestamp'].min()
    return out

def realign_and_normalize_emotibit_imu(df, lag, gaze_last_timestamp, sensor_rate=SENSOR_RATE_EMOTIBIT):
    # df is already zero-based in our pipeline; do NOT zero-base again
    aligned = realign_and_pad_template(
        df, lag, gaze_last_timestamp, sensor_rate,
        zero_base=False, clip_negative=True
    )
    return aligned

# ===================
# === Main flow   ===
# ===================
gaze_txt_path = os.path.join(input_directory, 'gazedata.txt')
xdf_path       = os.path.join(input_directory, f'{reldirectory}.xdf')

# 1) 2D gaze
gaze_txt = read_gaze_json_txt(gaze_txt_path)
gaze2d   = read_xdf_gaze2d(xdf_path)

if gaze2d is not None and len(gaze_txt):
    g2 = realign_and_pad_gaze2d(gaze2d, lag_value, gaze_txt)
    g2.to_csv(os.path.join(output_folder,'realigned_gaze2d.csv'), index=False)
    plot_full_gaze_comparison(
        gaze_txt, g2,
        'Gaze2D After Lag Correction',
        os.path.join(output_folder,'gaze2d_comparison.png')
    )
else:
    print('No Gaze2d stream found or TXT gaze missing.')

# 2) All 3D streams: Gaze3d, Origin, Direction
streams_to_save = [
    ('Gaze3d',        'realigned_gaze3d.csv'),
    ('GazeOrigin',    'realigned_gaze_origin3d.csv'),
    ('GazeDirection', 'realigned_gaze_direction3d.csv'),
]
for name,fname in streams_to_save:
    vec = read_xdf_vector3d(xdf_path, name)
    if vec is not None and len(gaze_txt):
        aligned = realign_and_pad_gaze3d(vec, lag_value, gaze_txt)
        aligned.to_csv(os.path.join(output_folder, fname), index=False)
    else:
        print(f'No {name} stream found or TXT gaze missing.')

# 3) Foot IMUs
LEFT_ID, CHEST_ID = 'MD-V5-0001052', 'MD-V5-0000539'
feet = process_foot_IMUs(xdf_path, LEFT_ID, CHEST_ID)
lf, ch = feet.get('LEFT'), feet.get('CHEST')

if lf is not None:
    lf['Timestamp'] -= lf['Timestamp'].min()
if ch is not None:
    ch['Timestamp'] -= ch['Timestamp'].min()

glt = float(gaze_txt['Timestamp'].iloc[-1]) if len(gaze_txt) else 0.0

if lf is not None and ch is not None:
    merged_feet = synchronize_feet(lf, ch)
    emotibit_aligned = realign_and_normalize_emotibit_imu(merged_feet, lag_value, glt)
elif lf is not None:
    emotibit_aligned = realign_and_normalize_emotibit_imu(lf, lag_value, glt)
elif ch is not None:
    emotibit_aligned = realign_and_normalize_emotibit_imu(ch, lag_value, glt)
else:
    emotibit_aligned = None
    print('No foot IMU data')

if emotibit_aligned is not None and not emotibit_aligned.empty:
    emotibit_aligned.to_csv(os.path.join(output_folder, 'realigned_feet.csv'), index=False)

# 4) Head IMU
head_df = process_head_IMU(xdf_path)
if head_df is not None and len(head_df):
    head_aligned = realign_and_normalize_head_imu(head_df, lag_value, glt)
    head_aligned.to_csv(os.path.join(output_folder, 'realigned_head.csv'), index=False)

print('All data saved.')

# ===============================
# Plotting to assess alignment
# ===============================
# Re-read what we saved and plot TXT vs aligned XDF
try:
    csv_file = os.path.join(output_folder, "realigned_gaze2d.csv")
    df_csv = pd.read_csv(csv_file)
    # Re-parse TXT
    def load_json_data(txt_file):
        json_data = []
        with open(txt_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        json_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return json_data

    json_data = load_json_data(gaze_txt_path)
    timestamps_json, gaze2d_x_json = [], []
    for entry in json_data:
        try:
            gaze2d = entry['data']['gaze2d'][0]
            timestamps_json.append(entry['timestamp'])
            gaze2d_x_json.append(gaze2d)
        except KeyError:
            continue

    timestamps_json = np.array(timestamps_json)
    gaze2d_x_json = np.array(gaze2d_x_json)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_csv['Timestamp'], df_csv['X'], label='XDF Gaze2D X (aligned)')
    plt.plot(timestamps_json, gaze2d_x_json, label='TXT Gaze2D X', linestyle='--')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Gaze2D X Position')
    plt.title('Comparison of Gaze2D X: TXT vs Aligned XDF')
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Plotting skipped due to error: {e}")
