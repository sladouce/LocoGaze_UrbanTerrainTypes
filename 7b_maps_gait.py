#!/usr/bin/env python3
"""
map_gait_metrics.py

Loads gait_metrics.csv, filters for walking bouts and a given time window (using START/END in seconds),
computes mean values for pace, cadence, stride length, and stride duration at each GPS coordinate,
spatially bins nearby points, and generates interactive Folium maps with circle markers colored by density
using the RdBu_r colormap. Saves CSVs and HTML maps.
"""
import os
import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import branca

# ---- Load metadata ----
META_CSV = r'C:/LocoGaze/data/metadata.csv'
meta_df = pd.read_csv(META_CSV, nrows=1)
reldir = meta_df.at[0, 'reldir']
# START and END are numeric seconds matching the Timestamp units
time_start = float(meta_df.at[0, 'START'])
time_end = float(meta_df.at[0, 'END'])

# ---- Construct file paths ----
BASE_DIR   = r'C:/LocoGaze/data/'
INPUT_DIR  = os.path.join(BASE_DIR, reldir, 'output')
GAIT_CSV   = os.path.join(INPUT_DIR, 'gait_metrics.csv')
OUTPUT_DIR = INPUT_DIR

# Metrics: column -> (vmin, vmax, legend caption)
METRICS = {
    'pace_LEFT':          (0, 6,    'Pace (m/s)'),
    'cadence_LEFT':       (80, 150, 'Cadence (steps/min)'),
    'stride_length_LEFT': (1, 3,    'Stride length (m)'),
    'stride_duration_LEFT': (1, 3,  'Stride duration (s)'),
}


def process_metric(df, metric, vmin, vmax, caption, out_csv, out_map):
    # Compute mean per GPS location
    mdf = (
        df.groupby(['latitude', 'longitude'], as_index=False)[metric]
          .mean()
          .rename(columns={metric: f'mean_{metric}'})
    )
    # Spatial binning by rounding coordinates
    eps = 3  # decimal places
    mdf['lat_bin'] = mdf['latitude'].round(eps)
    mdf['lon_bin'] = mdf['longitude'].round(eps)
    binned = (
        mdf.groupby(['lat_bin', 'lon_bin'], as_index=False)[f'mean_{metric}']
           .mean()
           .rename(columns={'lat_bin': 'latitude', 'lon_bin': 'longitude'})
    )
    # Save CSV
    binned.to_csv(out_csv, index=False)
    print(f"Saved {metric} data to {out_csv}")

    # Prepare colormap
    cmap = plt.get_cmap('RdBu_r')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    color_list = [mcolors.rgb2hex(cmap(x)) for x in np.linspace(0, 1, 256)]
    legend = branca.colormap.LinearColormap(
        colors=color_list,
        vmin=vmin,
        vmax=vmax,
        caption=caption
    )

    # Create map centered on data mean location
    center = [df['latitude'].mean(), df['longitude'].mean()]
    amap = folium.Map(location=center, zoom_start=14)

    # Add circle markers
    for r in binned.itertuples():
        val = max(vmin, min(getattr(r, f'mean_{metric}'), vmax))
        color = mcolors.rgb2hex(cmap(norm(val)))
        folium.CircleMarker(
            location=[r.latitude, r.longitude],
            radius=6,
            color=color,
            opacity=0.6,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"Mean {metric}: {getattr(r, f'mean_{metric}'): .2f}"
        ).add_to(amap)

    legend.add_to(amap)
    amap.save(out_map)
    print(f"Saved map for {metric} to {out_map}")


def main():
    # Load gait metrics
    gait_df = pd.read_csv(GAIT_CSV)
    # Filter for walking and within START/END seconds
    filt = (
        (gait_df['is_walking'] == True) &
        (gait_df['Timestamp'] >= time_start) &
        (gait_df['Timestamp'] <= time_end)
    )
    filt_df = gait_df.loc[filt].copy()
    print(f"Filtered to {len(filt_df)} walking records between {time_start} and {time_end} seconds")

    # Generate CSVs and maps for each metric
    for metric, (vmin, vmax, caption) in METRICS.items():
        fname = metric.replace('_LEFT', '')
        csv_out = os.path.join(OUTPUT_DIR, f"{fname}_binned.csv")
        map_out = os.path.join(OUTPUT_DIR, f"{fname}_map.html")
        process_metric(filt_df, metric, vmin, vmax, caption, csv_out, map_out)

if __name__ == '__main__':
    main()
