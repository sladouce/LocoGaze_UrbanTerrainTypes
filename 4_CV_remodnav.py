#!/usr/bin/env python3
"""
overlay_gaze_with_yolo_deepsort.py

2D gaze overlay + YOLOv8 detection + DeepSort tracking of people, bikes, cars, stop-signs.
Blurs persons, overlays gaze (2D & trails) and logs per-frame JSON with track IDs.
Integrates remodnav event classification (saccades/fixations).
"""
import os
import sys
import time
import json
import subprocess
import pandas as pd
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("Missing 'deep_sort_realtime' module. Install via: pip install deep_sort_realtime")
    sys.exit(1)

# ---- CONFIGURATION ----
start_time = time.time()

MODEL_NAME      = "yolov8x.pt"
CONF_THRESHOLD  = 0.20  # lowered to catch fainter detections
ALLOWED_CLASSES = [0,1,2,11]  # 0=person,1=bicycle,2=car,11=stop sign

# DeepSort params
MAX_AGE             = 10    # frames to keep lost tracks alive (occlusion tolerance)
N_INIT              = 12    # frames to confirm a new track (reduces ghost tracks)
MAX_COSINE_DISTANCE = 0.4   # appearance feature distance threshold for reID
MAX_IOU_DISTANCE    = 0.7   # reject detections with IoU < threshold

# Blur settings
PERSON_BLUR_KERNEL = (21,21)
PERSON_BLUR_SIGMA  = 7

# Gaze overlay settings
FPS_VIDEO      = 25.0
TRAIL_SEC      = 0.5
POINT_COLOR    = (255,0,0)
TRAIL_COLOR    = (255,255,255)
POINT_RADIUS   = 30
TRAIL_THICK    = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 1.10
FONT_THICK     = 2

# Paths & metadata
META_CSV       = r'C:/LocoGaze/data/metadata.csv'
meta           = pd.read_csv(META_CSV, nrows=1)
reldir         = meta.at[0,'reldir']
START_TIME     = meta.at[0,'START']
END_TIME       = meta.at[0,'END']
BASE_DIR       = r'C:/LocoGaze/data/'
input_dir      = os.path.join(BASE_DIR, reldir)
output_dir     = os.path.join(input_dir, 'output')

# Remodnav configuration
REMODNAV_INPUT_FILE  = os.path.join(output_dir, 'gaze2d_for_remodnav.txt')
REMODNAV_OUTPUT_FILE = os.path.join(output_dir, 'events_remodnav.tsv')
px2deg         = 0.0587
sampling_rate  = 100
savgol_length  = 0.05

# I/O files
GAZE2D_CSV     = os.path.join(output_dir, 'realigned_gaze2d.csv')
VIDEO_FILE     = os.path.join(input_dir, 'scenevideo.mp4')
OUT_VIDEO      = os.path.join(output_dir, 'scenevideo_deepsort.mp4')
OUT_JSON       = os.path.join(output_dir, 'tracking_gaze_deepsort.json')

SMOOTH_N = 3                    # how many frames to average over for smoothing
track_history = {}              # will map user_tid → deque of (cx,cy)

# Label mapping
COLOR_MAP = {0:(255,255,255),1:(0,0,255),2:(0,0,255),11:(255,0,255)}
CLASS_NAMES = {0:'person',1:'bicycle',2:'car',11:'stop sign'}

# ---- UTILS ----
def norm_to_px(xn, yn, W, H):
    """Normalized [0..1] → pixel; NaN-safe; returns None if invalid."""
    if not (np.isfinite(xn) and np.isfinite(yn)):
        return None
    # clamp to [0,1]
    xn = 0.0 if xn < 0.0 else (1.0 if xn > 1.0 else xn)
    yn = 0.0 if yn < 0.0 else (1.0 if yn > 1.0 else yn)
    return int(round(xn * (W - 1))), int(round(yn * (H - 1)))

def find_nearest(arr, t):
    if arr.size == 0:
        return None
    return int(np.abs(arr - t).argmin())

# ---- LOAD & PREPARE GAZE DATA ----
gaze2d = pd.read_csv(GAZE2D_CSV).rename(columns={'Timestamp':'time_s'})
gaze2d.sort_values('time_s', inplace=True)
gaze2d.reset_index(drop=True, inplace=True)

# Trim gaze data to [START_TIME, END_TIME]
mask = (
    (gaze2d['time_s'] >= START_TIME) &
    (gaze2d['time_s'] <= (END_TIME if END_TIME else gaze2d['time_s'].max()))
)
gaze_trimmed = gaze2d.loc[mask].reset_index(drop=True)

# Write remodnav input file (normalized X,Y per line)
with open(REMODNAV_INPUT_FILE, 'w') as fout:
    for _, row in gaze_trimmed.iterrows():
        fout.write(f"{row['X']}\t{row['Y']}\n")

# Run remodnav to classify saccades/fixations
cmd = [
    'remodnav',
    REMODNAV_INPUT_FILE,
    REMODNAV_OUTPUT_FILE,
    str(px2deg),
    str(sampling_rate),
    '--savgol-length', str(savgol_length)
]
subprocess.run(cmd, check=True)

# Parse remodnav output: onset (s), duration (s), event_type
events = []
with open(REMODNAV_OUTPUT_FILE, 'r') as fin:
    next(fin)  # skip header line
    for line in fin:
        onset, duration, ev_type, *rest = line.strip().split('\t')
        onset = float(onset) + START_TIME
        duration = float(duration)
        events.append((onset, duration, ev_type))

# Use trimmed gaze for overlay
gaze2d = gaze_trimmed
times2 = gaze2d['time_s'].to_numpy(dtype=float)

# Frame trimming indices
sf = int(START_TIME * FPS_VIDEO)
ef = int(END_TIME * FPS_VIDEO) if END_TIME else None

# ---- INITIALIZE MODELS ----
# YOLOv8 detector
try:
    model = YOLO(MODEL_NAME)
except Exception as e:
    print("Error loading YOLOv8:", e); sys.exit(1)

# DeepSort tracker
user_id_map = {}
next_user_id = 0

deepsort = DeepSort(
    max_age=MAX_AGE,
    n_init=N_INIT,
    max_cosine_distance=MAX_COSINE_DISTANCE,
    max_iou_distance=MAX_IOU_DISTANCE
)

# ---- VIDEO I/O SETUP ----
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened(): sys.exit(f"Cannot open video: {VIDEO_FILE}")
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS_VIDEO, (W,H))

# Seek to start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
frame_idx = sf
global_frame = 0
frame_logs = []

# Remodnav event tracking
current_event_idx = 0
last_event_type = 'unknown'

# ---- MAIN PROCESSING LOOP ----
while True:
    if ef and frame_idx > ef:
        break
    ret, frame = cap.read()
    if not ret:
        break

    t_frame = frame_idx / FPS_VIDEO

    # Update remodnav event
    if current_event_idx < len(events):
        onset, dur, etype = events[current_event_idx]
        if onset <= t_frame <= onset + dur:
            last_event_type = etype
        elif t_frame > onset + dur:
            current_event_idx += 1

    # 1) DETECTION
    results = model(frame, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    clss   = results.boxes.cls.cpu().numpy().astype(int)
    dets = []
    for (x1,y1,x2,y2), conf, cid in zip(boxes, scores, clss):
        if cid in ALLOWED_CLASSES and conf >= CONF_THRESHOLD:
            w, h = x2-x1, y2-y1
            dets.append(([x1,y1,w,h], conf, CLASS_NAMES.get(cid, str(cid))))

    # 2) UPDATE TRACKER
    tracks = deepsort.update_tracks(dets, frame=frame)
    tracked_objs = []
    for tr in tracks:
        if not tr.is_confirmed(): continue
        orig_tid = tr.track_id
        if orig_tid not in user_id_map:
            user_id_map[orig_tid] = next_user_id
            next_user_id += 1
        user_tid = user_id_map[orig_tid]

        l, t, r, b = tr.to_ltrb()
        cx, cy = int((l+r)/2), int((t+b)/2)

        hq = track_history.setdefault(user_tid, deque(maxlen=SMOOTH_N))
        hq.append((cx, cy))
        sx = sum(pt[0] for pt in hq)//len(hq)
        sy = sum(pt[1] for pt in hq)//len(hq)

        w_box, h_box = int(r-l), int(b-t)
        cx, cy = sx, sy
        cls_name = tr.get_det_class() or 'object'

        if cls_name == 'person':
            new_w = int(w_box * 1.5)
            new_h = int(h_box * 1.25)
            x1 = cx - new_w//2
            y1 = cy - new_h//2
            x2, y2 = x1 + new_w, y1 + new_h
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(W,x2), min(H,y2)
        else:
            x1 = cx - w_box//2
            y1 = cy - h_box//2
            x2, y2 = x1 + w_box, y1 + h_box

        x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
        w_box, h_box = x2-x1, y2-y1

        if cls_name=='person':
            roi = frame[y1:y2, x1:x2]
            if roi.size>0:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, PERSON_BLUR_KERNEL, PERSON_BLUR_SIGMA)

        cv2.putText(frame, cls_name, (x1, max(0,y1-8)), FONT, FONT_SCALE, (0,0,0), FONT_THICK+2, cv2.LINE_AA)
        cv2.putText(frame, cls_name, (x1, max(0,y1-8)), FONT, FONT_SCALE, COLOR_MAP.get(cid,(0,255,0)), FONT_THICK, cv2.LINE_AA)
        tracked_objs.append({'id':user_tid, 'class':cls_name, 'bbox':[x1,y1,w_box,h_box]})

    # 3) GAZE OVERLAY (NaN-safe)
    gaze_xy = [None, None]
    gi = find_nearest(times2, START_TIME + global_frame / FPS_VIDEO)
    if gi is not None and gi < len(times2):
        xn, yn = gaze2d.at[gi,'X'], gaze2d.at[gi,'Y']

        # Trailing path: use only valid points; draw if ≥2
        t_now = gaze2d.at[gi, 'time_s']
        mask_trail = (gaze2d['time_s'] >= t_now - TRAIL_SEC) & (gaze2d['time_s'] < t_now)
        if mask_trail.any():
            trail_xy = gaze2d.loc[mask_trail, ['X','Y']].dropna(how='any').values
            pts_list = []
            for x_t, y_t in trail_xy:
                p = norm_to_px(x_t, y_t, W, H)
                if p is not None:
                    pts_list.append(p)
            if len(pts_list) >= 2:
                cv2.polylines(frame, [np.array(pts_list, dtype=np.int32)], False, TRAIL_COLOR, TRAIL_THICK)

        # Current dot
        p = norm_to_px(xn, yn, W, H)
        if p is not None:
            cv2.circle(frame, p, POINT_RADIUS, POINT_COLOR, 2)
            gaze_xy = [int(p[0]), int(p[1])]

    # Overlay remodnav event label
    cv2.putText(frame,
                f"RV Event: {last_event_type}",
                (10, 60),
                FONT, FONT_SCALE, (0,255,0), FONT_THICK, cv2.LINE_AA)

    # 4) WRITE & LOG
    out.write(frame)
    frame_logs.append({
        'time_s': round(t_frame,3),
        'gaze': gaze_xy,              # [x,y] or [None,None]
        'objects': tracked_objs,
        'rv_event': last_event_type
    })
    frame_idx += 1
    global_frame += 1

# ---- CLEANUP ----
cap.release()
out.release()
with open(OUT_JSON, 'w') as f:
    json.dump(frame_logs, f, indent=2)

print(f"Saved video '{OUT_VIDEO}' and JSON '{OUT_JSON}'")
end_time = time.time()
elapsed = end_time - start_time
print(f"Elapsed time: {elapsed:.2f} seconds")
