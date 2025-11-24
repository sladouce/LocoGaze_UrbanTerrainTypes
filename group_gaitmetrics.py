#!/usr/bin/env python3
"""
collect_group_pace_social.py  (wide, all metrics, one row per participant)

Part A (existing):
- Converts 'social' to boolean
- Keeps only area_label == 'all'
- Averages duplicates within each social level
- Pivots ALL numeric measures into columns
- Renames prefixes:
      social_true_  -> present_
      social_false_ -> absent_
- Removes trailing '_LEFT' from metric names
- Output: C:\LocoGaze\data\group\group_pace_social.csv

Part B (UPDATED):
- Loads gait_stats_peoplebins.csv per participant
- Keeps only area_label == 'all'
- Averages duplicates within each people_bin (0,1,2plus)
- Pivots ALL numeric measures into columns with prefixes:
      bin0_, bin1_, bin2plus_
- Removes trailing '_LEFT' from metric names
- Output: C:\LocoGaze\data\group\group_gait_peoplebins.csv
"""

import os
import re
import warnings
import pandas as pd

# ---- Paths ----
BASE_DIR   = r"C:\LocoGaze\data"
META_CSV   = os.path.join(BASE_DIR, "metadata_all.csv")
GROUP_DIR  = os.path.join(BASE_DIR, "group")
OUTPUT_SOCIAL_CSV = os.path.join(GROUP_DIR, "group_pace_social.csv")
OUTPUT_BINS_CSV   = os.path.join(GROUP_DIR, "group_gait_peoplebins.csv")

def coerce_social_to_bool(s: pd.Series) -> pd.Series:
    """Coerce a 'social' series to boolean, handling strings like 'True'/'False'."""
    if s.dtype == bool:
        return s
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .map({
             "true": True, "false": False,
             "1": True, "0": False,
             "yes": True, "no": False
         })
         .astype("boolean")
         .fillna(False)
         .astype(bool)
    )

def rename_drop_left_suffix(col: str) -> str:
    """Helper used in both outputs: drop trailing _LEFT."""
    if col == "participant":
        return col
    return re.sub(r"_LEFT$", "", col)

# ----------------------------
# Part A: group_pace_social.csv
# ----------------------------
def collect_social_wide():
    os.makedirs(GROUP_DIR, exist_ok=True)

    # Load participants
    try:
        meta = pd.read_csv(META_CSV)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: Cannot find metadata file: {META_CSV}")
    if "participant" not in meta.columns:
        raise SystemExit("ERROR: 'participant' column not found in metadata_all.csv")

    participants = (
        meta["participant"].dropna().astype(str).str.strip().tolist()
    )
    if not participants:
        raise SystemExit("ERROR: No participants found in 'participant' column.")

    collected_rows = []
    missing_files = []
    all_metrics_list, seen_metrics = [], set()

    for pid in participants:
        csv_path = os.path.join(BASE_DIR, pid, "output", "stats", "gait_stats_social.csv")
        if not os.path.exists(csv_path):
            missing_files.append(pid)
            continue

        df = pd.read_csv(csv_path, sep=None, engine="python")

        # Handle occasional typo 'areal_label'
        if "area_label" not in df.columns and "areal_label" in df.columns:
            df = df.rename(columns={"areal_label": "area_label"})

        if "social" not in df.columns:
            warnings.warn(f"{csv_path} missing required column 'social'; skipping.")
            continue
        if "area_label" not in df.columns:
            warnings.warn(f"{csv_path} missing required column 'area_label'; skipping.")
            continue

        # Only area_label == 'all'
        sub = df[df["area_label"].astype(str).str.strip() == "all"].copy()
        if sub.empty:
            continue

        # Coerce and pick numeric metrics
        sub["social"] = coerce_social_to_bool(sub["social"])
        exclude = {"area_label", "areal_label", "social"}
        candidate_cols = [c for c in sub.columns if c not in exclude]
        tmp = sub[candidate_cols].apply(pd.to_numeric, errors="coerce")
        numeric_metrics = [c for c in tmp.columns if tmp[c].notna().any()]
        if not numeric_metrics:
            continue

        # Average duplicates within each social level
        tmp["social"] = sub["social"].values
        agg = tmp.groupby("social", as_index=False).mean(numeric_only=True)

        # Build wide row
        row = {"participant": pid}
        for _, r in agg.iterrows():
            lvl_prefix = "social_true_" if bool(r["social"]) else "social_false_"
            for m in numeric_metrics:
                row[f"{lvl_prefix}{m}"] = r[m]

        for m in numeric_metrics:
            if m not in seen_metrics:
                seen_metrics.add(m)
                all_metrics_list.append(m)

        collected_rows.append(row)

    if not collected_rows:
        raise SystemExit("ERROR: No data collected for Part A. Check 'all' rows and numeric measures in gait_stats_social.csv.")

    group_df = pd.DataFrame(collected_rows)

    # Ensure consistent columns across participants
    desired_cols = ["participant"] + \
        [f"social_false_{m}" for m in all_metrics_list] + \
        [f"social_true_{m}"  for m in all_metrics_list]

    for c in desired_cols:
        if c not in group_df.columns:
            group_df[c] = pd.NA
    group_df = group_df[desired_cols].sort_values("participant", ignore_index=True)

    # Rename columns before saving
    def rename_cols(c: str) -> str:
        if c == "participant":
            return c
        # social_* -> present_/absent_
        c = c.replace("social_true_", "present_").replace("social_false_", "absent_")
        # drop trailing _LEFT
        c = rename_drop_left_suffix(c)
        return c

    group_df = group_df.rename(columns=rename_cols)

    group_df.to_csv(OUTPUT_SOCIAL_CSV, index=False)
    print(f"Saved (wide, all metrics by social) to: {OUTPUT_SOCIAL_CSV}")

    if missing_files:
        print("Missing gait_stats_social.csv for participants:")
        for pid in missing_files:
            print(f"  - {pid}")

# ------------------------------------
# Part B: group_gait_peoplebins.csv
# ------------------------------------
def collect_peoplebins_wide():
    os.makedirs(GROUP_DIR, exist_ok=True)

    # Load participants
    try:
        meta = pd.read_csv(META_CSV)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: Cannot find metadata file: {META_CSV}")
    if "participant" not in meta.columns:
        raise SystemExit("ERROR: 'participant' column not found in metadata_all.csv")

    participants = (
        meta["participant"].dropna().astype(str).str.strip().tolist()
    )
    if not participants:
        raise SystemExit("ERROR: No participants found in 'participant' column.")

    collected_rows = []
    missing_files = []
    seen_metrics, all_metrics = set(), []

    # UPDATED ordered bins
    ORDERED_BINS = ["0", "1", "2plus"]

    for pid in participants:
        csv_path = os.path.join(BASE_DIR, pid, "output", "stats", "gait_stats_peoplebins.csv")
        if not os.path.exists(csv_path):
            missing_files.append(pid)
            continue

        df = pd.read_csv(csv_path, sep=None, engine="python")

        # Handle occasional typo 'areal_label'
        if "area_label" not in df.columns and "areal_label" in df.columns:
            df = df.rename(columns={"areal_label": "area_label"})

        # Required columns
        if "people_bin" not in df.columns:
            warnings.warn(f"{csv_path} missing required column 'people_bin'; skipping.")
            continue
        if "area_label" not in df.columns:
            warnings.warn(f"{csv_path} missing required column 'area_label'; skipping.")
            continue

        # Only area_label == 'all'
        sub = df[df["area_label"].astype(str).str.strip() == "all"].copy()
        if sub.empty:
            # If no 'all', fall back to averaging across areas (optional)
            fallback = df.copy()
            fallback["people_bin"] = fallback["people_bin"].astype(str).str.strip()
            sub = fallback.groupby("people_bin", as_index=False).mean(numeric_only=True)
            warnings.warn(f"{csv_path} has no area_label == 'all'; using across-area mean as fallback.")

        # Coerce bin labels and ensure order
        sub["people_bin"] = sub["people_bin"].astype(str).str.strip()
        sub = sub[sub["people_bin"].isin(ORDERED_BINS)].copy()

        # Pick numeric metrics (exclude non-numeric)
        exclude = {"area_label", "areal_label", "people_bin"}
        candidate_cols = [c for c in sub.columns if c not in exclude]
        tmp = sub[candidate_cols].apply(pd.to_numeric, errors="coerce")
        numeric_metrics = [c for c in tmp.columns if tmp[c].notna().any()]
        if not numeric_metrics:
            continue

        # Average duplicates within each bin (e.g., multiple rows per bin)
        tmp["people_bin"] = sub["people_bin"].values
        agg = tmp.groupby("people_bin", as_index=False).mean(numeric_only=True)

        # Build wide row
        row = {"participant": pid}
        for _, r in agg.iterrows():
            bin_tag = str(r["people_bin"])
            if bin_tag == "0":
                prefix = "bin0_"
            elif bin_tag == "1":
                prefix = "bin1_"
            else:  # "2plus"
                prefix = "bin2plus_"
            for m in numeric_metrics:
                row[f"{prefix}{m}"] = r[m]

        for m in numeric_metrics:
            if m not in seen_metrics:
                seen_metrics.add(m)
                all_metrics.append(m)

        collected_rows.append(row)

    if not collected_rows:
        raise SystemExit("ERROR: No data collected for Part B. Check gait_stats_peoplebins.csv per participant.")

    group_df = pd.DataFrame(collected_rows)

    # Desired column order: by bin, then metric
    desired_cols = ["participant"]
    for bin_tag in ["0", "1", "2plus"]:
        if bin_tag == "0":
            prefix = "bin0_"
        elif bin_tag == "1":
            prefix = "bin1_"
        else:
            prefix = "bin2plus_"
        for m in all_metrics:
            desired_cols.append(f"{prefix}{m}")

    # Ensure columns exist & reorder
    for c in desired_cols:
        if c not in group_df.columns:
            group_df[c] = pd.NA
    group_df = group_df[desired_cols].sort_values("participant", ignore_index=True)

    # Drop trailing _LEFT from metric names
    rename_map = {c: rename_drop_left_suffix(c) for c in group_df.columns}
    group_df = group_df.rename(columns=rename_map)

    group_df.to_csv(OUTPUT_BINS_CSV, index=False)
    print(f"Saved (wide, all metrics by people bins) to: {OUTPUT_BINS_CSV}")

    if missing_files:
        print("Missing gait_stats_peoplebins.csv for participants:")
        for pid in missing_files:
            print(f"  - {pid}")

if __name__ == "__main__":
    # Part A: present vs absent (non-cyclist people)
    collect_social_wide()
    # Part B: 0,1,2plus bins (non-cyclist people)
    collect_peoplebins_wide()
