import gpxpy
import pandas as pd
import numpy as np
from haversine import haversine
import geopandas as gpd
from shapely.geometry import Point, shape
import folium
import json
import os

# Step 0: Setting the directory
meta_file_path = r'C:\LocoGaze\data\metadata.csv'
meta_df = pd.read_csv(meta_file_path, nrows=1)
reldirectory = meta_df.at[0, 'reldir']
lagwatch_value = meta_df.at[0, 'watch_lag']

# Define the directory where all data files are located
data_directory = 'C:/LocoGaze/data/'  # Replace with your actual directory path
input_directory = data_directory + reldirectory

# === Step 1: Load & parse your custom GeoJSON format ===
with open(r'C:\LocoGaze\code\coordinates.json', 'r') as f:
    geojson_data = json.load(f)

features = []
for feature in geojson_data['features']:
    geom = shape(feature['geometry'])
    label = next(iter(feature['properties'].keys()))
    features.append({'geometry': geom, 'label': label})

areas = gpd.GeoDataFrame(features, crs='EPSG:4326')

# === Step 2: Load the GPX file & build raw DataFrame ===
gpx_file_path = input_directory + '/' + reldirectory + '.gpx'
with open(gpx_file_path, 'r') as gpx_file:
    gpx = gpxpy.parse(gpx_file)

rows = []
for trk in gpx.tracks:
    for seg in trk.segments:
        for pt in seg.points:
            rows.append({
                'time': pt.time,         # datetime
                'latitude': pt.latitude,
                'longitude': pt.longitude,
                'elevation': pt.elevation
            })

df = pd.DataFrame(rows)

# === Step 3: Convert to elapsed seconds & up-sample to every second ===
# ensure datetime
df['time'] = pd.to_datetime(df['time'])

# drop duplicate timestamps (down to the second)
df['second'] = df['time'].dt.floor('S')
df = df.drop_duplicates(subset='second', keep='first').drop(columns='second')

# set datetime index (now guaranteed unique)
df.set_index('time', inplace=True)

# resample to 1 Hz
df = df.resample('1S').asfreq()
df = df[~df.index.duplicated(keep='first')]
df[['latitude', 'longitude', 'elevation']] = df[['latitude', 'longitude', 'elevation']].interpolate()

# reset index and create elapsed time in seconds
df = df.reset_index()
df['elapsed_sec'] = np.arange(len(df))

dupes = df[df.duplicated(subset='elapsed_sec', keep=False)]
print(dupes)

# === Step 4: Compute cumulative distance & smoothed speed ===
distances = [0.0]
for i in range(1, len(df)):
    prev = (df.loc[i - 1, 'latitude'], df.loc[i - 1, 'longitude'])
    curr = (df.loc[i, 'latitude'], df.loc[i, 'longitude'])
    distances.append(distances[-1] + haversine(prev, curr))
df['cum_dist_km'] = distances

window = 10  # seconds
df['rolling_dist'] = df['cum_dist_km'].diff(periods=window)
df['rolling_time'] = df['time'].diff(periods=window).dt.total_seconds() / 3600
df['speed_kmh_smooth'] = (
    df['rolling_dist'] / df['rolling_time']
).clip(lower=0, upper=25).fillna(method='bfill')

# === Step 5: Spatial join to assign area labels ===
df['geometry'] = df.apply(lambda r: Point(r['longitude'], r['latitude']), axis=1)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

joined = gpd.sjoin(gdf, areas, how='left', predicate='within')
joined = joined.drop_duplicates(subset='elapsed_sec', keep='first')

# Raw terrain label from polygons; default to 'flat' when no polygon is hit
joined['area_label'] = joined['label'].fillna('flat')

# Apply watch lag
joined['elapsed_sec'] += lagwatch_value

# === Step 5b: Temporal majority-vote smoothing of terrain labels ===
# Sort by time to ensure correct temporal order
joined = joined.sort_values('elapsed_sec').reset_index(drop=True)

# Convert to categorical codes for efficient majority voting
labels_cat = joined['area_label'].astype('category')
codes = labels_cat.cat.codes  # integer codes, one per label

# Define majority-vote window (e.g., 5-second centered window)
mv_window = 5  # seconds; each sample considers ±2 s neighborhood

def majority_code(x):
    x = x.astype(int)
    # bincount over non-negative codes
    counts = np.bincount(x)
    return np.argmax(counts)

codes_smoothed = (
    codes.rolling(window=mv_window, center=True, min_periods=1)
         .apply(majority_code, raw=True)
         .astype(int)
)

# Map smoothed codes back to labels
joined['area_label'] = pd.Categorical.from_codes(
    codes_smoothed,
    categories=labels_cat.cat.categories
).astype(str)

# === Step 6: Tidy up & save as CSV ===
# replace the original datetime with elapsed seconds in your output
joined = joined.drop(columns=['time', 'label', 'index_right'])
joined = joined.rename(columns={'elapsed_sec': 'time_seconds'})
gpx_outfile_path = input_directory + '/' + '/output/'
csv_output = os.path.join(os.path.dirname(gpx_outfile_path),
                          'gpx_with_smooth_speed_labeled.csv')
joined.drop(columns='geometry').to_csv(csv_output, index=False)
print(f"Labeled CSV saved to: {csv_output}")

# === Step 7: Draw a folium map color-coded by area ===
color_map = {
    'green': '#367564',
    'cobblestone': '#E25012',
    'cobblestones': '#E25012',
    'flat': '#941C79'
}

m = folium.Map(
    location=[joined.iloc[0].latitude, joined.iloc[0].longitude],
    tiles="CartoDB.VoyagerLabelsUnder",
    zoom_start=14
)

for i in range(1, len(joined)):
    p1 = [joined.iloc[i - 1].latitude, joined.iloc[i - 1].longitude]
    p2 = [joined.iloc[i].latitude, joined.iloc[i].longitude]
    lbl = joined.iloc[i]['area_label']
    folium.PolyLine(
        [p1, p2],
        color=color_map.get(lbl, '#808080'),  # default gray
        weight=10
    ).add_to(m)

map_output = os.path.join(os.path.dirname(gpx_outfile_path),
                          'segmentedmap.html')

m.save(map_output)
print(f"Segmented map saved to: {map_output}")

# ----------------------------------------------------------------------
# Realignment of GPX to feet timestamps
# ----------------------------------------------------------------------
import os
import pandas as pd
import numpy as np

# === Load metadata ===
meta_file_path = r'C:\LocoGaze\data\metadata.csv'
meta_df = pd.read_csv(meta_file_path, nrows=1)
reldirectory = meta_df.at[0, 'reldir']

# === Directories & file paths ===
data_directory = 'C:/LocoGaze/data/'
input_dir = os.path.join(data_directory, reldirectory, 'output')
feet_file = os.path.join(input_dir, 'realigned_feet.csv')
gpx_file = os.path.join(input_dir, 'gpx_with_smooth_speed_labeled.csv')
output_file = os.path.join(input_dir, 'realigned_gpx.csv')

# === Load data ===
feet_df = pd.read_csv(feet_file)
gpx_df = pd.read_csv(gpx_file)

# === Prepare GPX DataFrame ===
# rename time_seconds to Timestamp for alignment
if 'time_seconds' in gpx_df.columns:
    gpx_df = gpx_df.rename(columns={'time_seconds': 'Timestamp'})
else:
    gpx_df = gpx_df.rename(columns={gpx_df.columns[0]: 'Timestamp'})

# === Determine full timestamp range from feet ===
feet_min = int(np.floor(feet_df['Timestamp'].min()))
feet_max = int(np.ceil(feet_df['Timestamp'].max()))
full_index = np.arange(feet_min, feet_max + 1)

# === Reindex GPX at 1 Hz using nearest existing data ===
gpx_indexed = gpx_df.set_index('Timestamp')
# ensure monotonic index
gpx_indexed = gpx_indexed.sort_index()
# reindex with nearest method
aligned = gpx_indexed.reindex(full_index, method='nearest')
aligned.index.name = 'Timestamp'
aligned_gpx = aligned.reset_index()

# === Save aligned GPX ===
aligned_gpx.to_csv(output_file, index=False)
print(f"Aligned GPX saved to {output_file}")
