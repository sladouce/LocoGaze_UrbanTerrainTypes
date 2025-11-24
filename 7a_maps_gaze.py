#!/usr/bin/env python3
"""
map_people_and_depth.py

Loads visual_events.csv, computes mean number of people at each GPS coordinate
and mean depth (depth_d_s) under specific conditions, spatially bins nearby
points, and generates interactive Folium maps with heatmaps and circle markers
colored by density using the RdBu_r colormap (people: 0–5; depth: 0–3000).
Saves CSVs and HTML maps.
"""
import os
import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from folium.plugins import HeatMap
import branca

# ---- Load metadata ----
META_CSV = r'C:/LocoGaze/data/metadata.csv'
meta_df = pd.read_csv(META_CSV, nrows=1)
reldir = meta_df.at[0, 'reldir']

# ---- Construct file paths ----
BASE_DIR       = r'C:/LocoGaze/data/'
INPUT_DIR      = os.path.join(BASE_DIR, reldir, 'output')
VISUAL_CSV     = os.path.join(INPUT_DIR, 'visual_events.csv')
PEOPLE_CSV     = os.path.join(INPUT_DIR, 'people_presence_binned.csv')
PEOPLE_MAP     = os.path.join(INPUT_DIR, 'people_presence_map.html')
DEPTH_CSV      = os.path.join(INPUT_DIR, 'depth_binned.csv')
DEPTH_MAP      = os.path.join(INPUT_DIR, 'depth_map.html')

# ---- Load visual events ----
vis_df = pd.read_csv(VISUAL_CSV)

# === Section 1: People presence map ===
# Compute mean people per coordinate
people_df = (
    vis_df.groupby(['latitude', 'longitude'], as_index=False)
          ['number_people'].mean()
          .rename(columns={'number_people': 'mean_number_people'})
)
# Spatial binning
eps = 5  # decimal places for binning
people_df['lat_bin'] = people_df['latitude'].round(eps)
people_df['lon_bin'] = people_df['longitude'].round(eps)
people_binned = (
    people_df.groupby(['lat_bin', 'lon_bin'], as_index=False)
             ['mean_number_people'].mean()
)
people_binned.rename(columns={'lat_bin': 'latitude', 'lon_bin': 'longitude'}, inplace=True)
# Save CSV
people_binned.to_csv(PEOPLE_CSV, index=False)
print(f"Saved people presence to {PEOPLE_CSV}")
# Colormap for people (0–5)
people_cmap = plt.get_cmap('RdYlBu_r')
people_norm = mcolors.Normalize(vmin=0, vmax=5)
n_colors = 256
color_list = [
    mcolors.rgb2hex(people_cmap(x)) 
    for x in np.linspace(0, 1, n_colors)
]
people_legend = branca.colormap.LinearColormap(
    colors=color_list,
    vmin=0,                 # match your Normalize vmin
    vmax=5,                 # match your Normalize vmax
    caption='People encountered (0–5)'
)


# Create map
center = [vis_df['latitude'].mean(), vis_df['longitude'].mean()]
map_p = folium.Map(location=center, zoom_start=14)

# Circle markers
for r in people_binned.itertuples():
    val = max(0, min(r.mean_number_people, 5))
    color = mcolors.rgb2hex(people_cmap(people_norm(val)))
    folium.CircleMarker(
        location=[r.latitude, r.longitude],
        radius=6,
        color=color,
        opacity=0.5,
        fill=True,
        fill_color=color,
        fill_opacity=0.5,
        popup=f"Avg people: {r.mean_number_people:.2f}"
    ).add_to(map_p)
# Save map
people_legend.add_to(map_p)
map_p.save(PEOPLE_MAP)
print(f"Saved people map to {PEOPLE_MAP}")

# === Section 2: Depth map ===
# Filter for walking, looking_floor, depth_d_s > 0
depth_df = vis_df[
    (vis_df['is_walking'] == True) &
    (vis_df['depth_looking_floor'] == True) &
    (vis_df['depth_d_s'] > 0)
][['latitude', 'longitude', 'depth_d_s']]
# Compute mean depth per coordinate
depth_mean = (
    depth_df.groupby(['latitude', 'longitude'], as_index=False)
             ['depth_d_s'].mean()
             .rename(columns={'depth_d_s': 'mean_depth'})
)
# Spatial binning
depth_mean['lat_bin'] = depth_mean['latitude'].round(eps)
depth_mean['lon_bin'] = depth_mean['longitude'].round(eps)
depth_binned = (
    depth_mean.groupby(['lat_bin', 'lon_bin'], as_index=False)
              ['mean_depth'].mean()
)
depth_binned.rename(columns={'lat_bin': 'latitude', 'lon_bin': 'longitude'}, inplace=True)
# Save CSV
depth_binned.to_csv(DEPTH_CSV, index=False)
print(f"Saved depth data to {DEPTH_CSV}")
# Colormap for depth (0–3000)
depth_cmap = plt.get_cmap('bwr')
depth_norm = mcolors.Normalize(vmin=1000, vmax=3500)

n_colors = 256
color_list = [
    mcolors.rgb2hex(depth_cmap(x)) 
    for x in np.linspace(0, 1, n_colors)
]
depth_legend = branca.colormap.LinearColormap(
    colors=color_list,
    vmin=1000,                 # match your Normalize vmin
    vmax=3500,                 # match your Normalize vmax
    caption='Floor fixation distance (1-3.5m)'
)


# Create map
#map_d = folium.Map(location=center, zoom_start=14)
#map_d = folium.Map(location=center, tiles='CartoDB positron', zoom_start=14)
map_d = folium.Map(location=center, tiles='CartoDB dark_matter', zoom_start=14)

# Circle markers
for r in depth_binned.itertuples():
    val = max(1000, min(r.mean_depth, 3500))
    color = mcolors.rgb2hex(depth_cmap(depth_norm(val)))
    folium.CircleMarker(
        location=[r.latitude, r.longitude],
        radius=6,
        color=color,
        opacity=0.5,
        fill=True,
        fill_color=color,
        fill_opacity=0.5,
        popup=f"Avg depth: {r.mean_depth:.1f} m"
    ).add_to(map_d)
# Save map
depth_legend.add_to(map_d)
map_d.save(DEPTH_MAP)
print(f"Saved depth map to {DEPTH_MAP}")

# === Section 3: Floor‑looking map (continuous bwr) ===
# Copy only the columns you need
floor_df = vis_df[['latitude', 'longitude', 'depth_looking_floor']].copy()
# Convert True/False → int (1/0) and compute mean per bin
floor_df['floor_int'] = floor_df['depth_looking_floor'].astype(int)
floor_df['lat_bin'] = floor_df['latitude'].round(eps+1)
floor_df['lon_bin'] = floor_df['longitude'].round(eps+1)
floor_binned = (
    floor_df
    .groupby(['lat_bin', 'lon_bin'], as_index=False)['floor_int']
    .mean()
    .rename(columns={
        'lat_bin':'latitude',
        'lon_bin':'longitude',
        'floor_int':'mean_floor'
    })
)

# Save CSV
FLOOR_CSV = os.path.join(INPUT_DIR, 'floor_looking_binned.csv')
floor_binned.to_csv(FLOOR_CSV, index=False)
print(f"Saved floor‑looking data to {FLOOR_CSV}")

# Create map
FLOOR_MAP = os.path.join(INPUT_DIR, 'floor_looking_map.html')
map_f = folium.Map(location=center, tiles='CartoDB dark_matter', zoom_start=14)

# --- Define continuous bwr colormap ---
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

floor_cmap = plt.get_cmap('Reds')                 # blue-white-red
floor_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)  # your data range

# Draw a circle for each bin, colored by mean_floor
for r in floor_binned.itertuples():
    frac = r.mean_floor                         # between 0 and 1
    # clamp just in case, then map to hex color
    frac_clamped = max(0.0, min(frac, 1.0))
    color = mcolors.rgb2hex(floor_cmap(floor_norm(frac_clamped)))
    folium.CircleMarker(
        location=[r.latitude, r.longitude],
        radius=6,
        color=color,
        opacity=0.5,
        fill=True,
        fill_color=color,
        fill_opacity=0.5,
        popup=f"Floor‑look frac: {frac:.2f}"
    ).add_to(map_f)

# (Optional) add a legend using branca:
import branca
# sample the matplotlib cmap into a list of hexes
hexes = [mcolors.rgb2hex(floor_cmap(x)) for x in np.linspace(0,1,256)]
legend = branca.colormap.LinearColormap(
    colors=hexes,
    vmin=0,
    vmax=1,
    caption='Floor‑looking fraction (0 → 1)'
)
legend.add_to(map_f)

# Save
map_f.save(FLOOR_MAP)
print(f"Saved floor‑looking map to {FLOOR_MAP}")