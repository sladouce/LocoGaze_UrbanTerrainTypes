#!/usr/bin/env python3
"""
collect_group_env_objects.py

Reads participant IDs from:
  C:\LocoGaze\data\metadata_all.csv  (expects a 'participant' column)

For each participant PXX, loads:
  C:\LocoGaze\data\PXX\output\stats\environment_objects_proportion.csv

Creates two outputs in C:\LocoGaze\data\group:
  1) group_gaze_fix_objects.csv
     columns: participant, object, area_label, proportion_fixated_when_present
     (for object in {'floor','background'}, uses proportion_fixated instead)

  2) group_numberofpeoplepresent.csv
     columns: participant, area_label, avg_number_when_present, proportion_present
     (prefers object == 'people_nocyclist'; falls back to 'people')
"""

import os
import pandas as pd

# ---- Paths ----
BASE_DIR   = r"C:\LocoGaze\data"
META_CSV   = os.path.join(BASE_DIR, "metadata_all.csv")
GROUP_DIR  = os.path.join(BASE_DIR, "group")
OUTPUT_FIX = os.path.join(GROUP_DIR, "group_gaze_fix_objects.csv")
OUTPUT_NUM = os.path.join(GROUP_DIR, "group_numberofpeoplepresent.csv")

REQUIRED = {
    "object",
    "area_label",
    "proportion_present",
    "proportion_fixated",
    "proportion_fixated_when_present",
    "avg_number_when_present",
}

def load_participants(meta_csv: str) -> list[str]:
    meta = pd.read_csv(meta_csv)
    if "participant" not in meta.columns:
        raise SystemExit("ERROR: 'participant' column not found in metadata_all.csv")
    parts = meta["participant"].dropna().astype(str).str.strip().tolist()
    if not parts:
        raise SystemExit("ERROR: No participants found.")
    return parts

def main():
    os.makedirs(GROUP_DIR, exist_ok=True)
    participants = load_participants(META_CSV)

    fix_rows = []
    num_rows = []
    missing  = []

    for pid in participants:
        csv_path = os.path.join(BASE_DIR, pid, "output", "stats", "environment_objects_proportion.csv")
        if not os.path.exists(csv_path):
            missing.append(pid)
            continue

        # Auto-detect delimiter (comma/tab)
        df = pd.read_csv(csv_path, sep=None, engine="python")

        # Tolerate occasional typos in 'area_label'
        if "area_label" not in df.columns:
            if "areal_label" in df.columns:
                df = df.rename(columns={"areal_label": "area_label"})
            elif "area_labe" in df.columns:
                df = df.rename(columns={"area_labe": "area_label"})

        # Check columns
        missing_cols = REQUIRED - set(df.columns)
        if missing_cols:
            raise SystemExit(
                f"ERROR: {csv_path} missing columns: {sorted(missing_cols)}\n"
                f"Found: {list(df.columns)}"
            )

        # Normalize object/area strings
        df["object"] = df["object"].astype(str).str.strip().str.lower()
        df["area_label"] = df["area_label"].astype(str).str.strip()

        # ---- Output 1: fixation proportion (with floor/background override) ----
        sub = df[["object", "area_label",
                  "proportion_fixated_when_present", "proportion_fixated"]].copy()

        # Default to conditional; override for floor & background to use absolute fixation proportion
        sub["proportion_effective"] = sub["proportion_fixated_when_present"]
        override_mask = sub["object"].isin(["floor", "background"])
        sub.loc[override_mask, "proportion_effective"] = sub.loc[override_mask, "proportion_fixated"]

        out_fix = sub[["object", "area_label", "proportion_effective"]].rename(
            columns={"proportion_effective": "proportion_fixated_when_present"}
        )
        out_fix.insert(0, "participant", pid)
        fix_rows.append(out_fix)

        # ---- Output 2: avg number & presence proportion for (non-cyclist) people ----
        # Prefer 'people_nocyclist', fall back to 'people' if needed
        ppl_mask = df["object"].eq("people_nocyclist")
        if not ppl_mask.any():
            ppl_mask = df["object"].eq("people")

        people_cols = ["area_label", "avg_number_when_present", "proportion_present"]
        people = df.loc[ppl_mask, people_cols].copy()
        if not people.empty:
            people.insert(0, "participant", pid)
            num_rows.append(people)

    if not fix_rows:
        raise SystemExit("ERROR: No environment_objects_proportion.csv files were read.")

    group_fix = pd.concat(fix_rows, ignore_index=True)
    group_fix = group_fix.sort_values(["participant", "object", "area_label"], ignore_index=True)
    group_fix.to_csv(OUTPUT_FIX, index=False)

    if num_rows:
        group_num = pd.concat(num_rows, ignore_index=True)
        group_num = group_num.sort_values(["participant", "area_label"], ignore_index=True)
        group_num.to_csv(OUTPUT_NUM, index=False)
    else:
        # Still write an empty file with headers for consistency
        pd.DataFrame(columns=["participant", "area_label", "avg_number_when_present", "proportion_present"]).to_csv(OUTPUT_NUM, index=False)

    print(f"Saved:\n  {OUTPUT_FIX}\n  {OUTPUT_NUM}")
    if missing:
        print("Missing environment_objects_proportion.csv for participants:")
        for pid in missing:
            print(f"  - {pid}")

if __name__ == "__main__":
    main()
