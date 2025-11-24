#!/usr/bin/env python3
"""
compute_spatialbin_correlations_all.py

Computes:
- Per-subject binned features -> correlations (+ p-values) for ALL walking data
- Group Fisher z-mean matrix (+ permutation p-values + FDR q-values)
- Terrain-specific per-subject correlations and group matrices (+ permutation p-values + FDR q-values)
- Saves a manifest.json, participants lists, and vars_order(.json) for plotting reuse

Notes:
- Inverts gaze_y so TOP has higher values: gaze_y := 1080 - gaze_y
- Head pitch sign unchanged (negative = looking down)
- Includes FDR (Benjamini–Hochberg) correction on group permutation p-values:
    * ALL-data: FDR on that single matrix
    * Terrains: configurable to correct jointly across all terrains (recommended) or per terrain
- Quick preview figures mask by FDR q < FDR_Q_ALPHA by default
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches, colors

# ---- Optional stats ----
try:
    from scipy.stats import spearmanr, pearsonr
    _SCIPY_OK = True
except Exception:
    print("[WARN] scipy not found; per-subject p-value masking disabled. Install: pip install scipy")
    _SCIPY_OK = False

# ---------------- CONFIG ----------------
BASE_DIR   = r"C:/LocoGaze/data/"
META_ALL   = os.path.join(BASE_DIR, "metadata_all.csv")

GROUP_DIR  = os.path.join(BASE_DIR, "group")
OUT_DIR    = os.path.join(GROUP_DIR, "corr")
os.makedirs(OUT_DIR, exist_ok=True)

# Bin precision / inclusion
EPS = 3
MIN_SAMPLES_PER_BIN       = 0
MIN_FLOOR_SAMPLES_PER_BIN = 0  # require some floor-looking for stable depth means

# Correlations
CORR_METHOD   = 'spearman'  # or 'pearson'
ALPHA_SUBJECT = 0.01        # (used elsewhere for per-subject plots)

# Group permutations
N_PERM      = 5000
RNG_SEED    = 2025

# --- Multiple comparisons (FDR) ---
FDR_Q_ALPHA            = 0.05   # threshold used for masking in preview figures
FDR_JOINT_ACROSS_TERRAINS = True  # True: one FDR across all terrain matrices; False: per terrain
# If you prefer raw p for masking in quick figs, set USE_FDR_FOR_MASK=False
USE_FDR_FOR_MASK       = True

# Colormap bins
BIN_EDGES = np.arange(-1.0, 1.0 + 0.25, 0.25)  # 8 bins
MAX_CELL_FILL = 0.99

# Scene geometry
X_MAX = 1920.0
Y_MAX = 1080.0
X_CENTER = 960.0

# Feature mapping (from visual_events.csv)
FEAT_MEAN = {
    # Gait
    'pace_LEFT'           : 'pace',
    'cadence_LEFT'        : 'cadence',
    'stride_length_LEFT'  : 'stride_length',
    'stride_duration_LEFT': 'stride_duration',
    # Gaze
    'horizontal_ecc'      : 'horizontal_ecc',   # absolute |gaze_x - 960|
    'gaze_y'              : 'gaze_y',          # will be inverted before grouping
    'depth_head_pitch_deg': 'head_pitch_deg',
    # Floor-looking fraction
    'depth_looking_floor' : 'floor_looking_frac',
}
VAR_TARGETS = {
    'stride_length_LEFT'  : 'stride_length_var',
    'stride_duration_LEFT': 'stride_duration_var',
}
FEATURE_ORDER = [
    'pace','cadence','stride_length','stride_duration',
    'stride_length_var','stride_duration_var',
    'gaze_depth_m','horizontal_ecc','gaze_y','floor_looking_frac','head_pitch_deg',
]

# ---------------- HELPERS ----------------
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None

def fisher_mean(rhos):
    rhos = np.array(rhos, dtype=float)
    rhos = rhos[np.isfinite(rhos)]
    if rhos.size == 0:
        return np.nan
    rhos = np.clip(rhos, -0.999999, 0.999999)
    z = np.arctanh(rhos)
    return np.tanh(np.nanmean(z))

def _corr_pair(x, y, method='spearman'):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, 0
    if method == 'pearson':
        if _SCIPY_OK:
            r, _ = pearsonr(x[mask], y[mask])
        else:
            r = np.corrcoef(x[mask], y[mask])[0,1]
    else:
        if _SCIPY_OK:
            r, _ = spearmanr(x[mask], y[mask])
        else:
            xr = pd.Series(x[mask]).rank().values
            yr = pd.Series(y[mask]).rank().values
            r = np.corrcoef(xr, yr)[0,1]
    return r, mask.sum()

def _norm_label(v):
    if pd.isna(v):
        return None
    return str(v).strip().lower()

def _norm_series(s):
    return s.astype('string').str.strip().str.lower()

def compute_per_bin_features(vis, terrain=None):
    """
    Return per-bin features (optionally filtering to a terrain label).
    - Filters: walking-only done by caller
    - Invert gaze_y (top high): gaze_y := Y_MAX - gaze_y
    - Horizontal_ecc = |gaze_x - 960|
    - Floor-only depth mean (mm->m)
    """
    if terrain is not None:
        if 'area_label' not in vis.columns:
            return pd.DataFrame()
        norm_t = _norm_label(terrain)
        s = _norm_series(vis['area_label'])
        vis = vis[s == norm_t].copy()

    if not {'latitude','longitude'}.issubset(vis.columns):
        return pd.DataFrame()

    tmp = vis.dropna(subset=['latitude','longitude']).copy()
    if tmp.empty:
        return pd.DataFrame()

    # Gaze numeric + invert vertical
    tmp['gaze_x'] = pd.to_numeric(tmp.get('gaze_x', np.nan), errors='coerce')
    tmp['gaze_y'] = pd.to_numeric(tmp.get('gaze_y', np.nan), errors='coerce')
    tmp['gaze_y'] = Y_MAX - tmp['gaze_y']  # top = high

    # Horizontal eccentricity (absolute deviation from center)
    tmp['horizontal_ecc'] = np.abs(tmp['gaze_x'] - X_CENTER)

    # Floor-looking helpers
    tmp['depth_looking_floor'] = tmp.get('depth_looking_floor', False).astype(bool)
    tmp['floor_int'] = tmp['depth_looking_floor'].astype(int)

    # Floor-only depth (mm -> m) averaged later per bin
    if 'depth_d_s' in tmp.columns:
        tmp['depth_d_s'] = pd.to_numeric(tmp['depth_d_s'], errors='coerce')
        tmp['depth_m_flooronly'] = np.where(tmp['floor_int'] == 1, tmp['depth_d_s']/1000.0, np.nan)
    else:
        tmp['depth_m_flooronly'] = np.nan

    # Spatial bins
    tmp['lat_bin'] = tmp['latitude'].round(EPS)
    tmp['lon_bin'] = tmp['longitude'].round(EPS)
    gb = tmp.groupby(['lat_bin','lon_bin'], sort=False)

    out = gb['latitude'].size().to_frame(name='n_samples_bin').reset_index()
    out = out.rename(columns={'lat_bin':'latitude', 'lon_bin':'longitude'})

    # Means for basic features (floor fraction via floor_int)
    for src_col, out_name in FEAT_MEAN.items():
        if src_col == 'depth_looking_floor':
            out[out_name] = gb['floor_int'].mean().values
        elif src_col in tmp.columns:
            out[out_name] = gb[src_col].mean().values
        else:
            out[out_name] = np.nan

    # Variances for gait
    for src_col, out_name in VAR_TARGETS.items():
        out[out_name] = gb[src_col].var(ddof=1).values if src_col in tmp.columns else np.nan

    # Depth mean (floor-only)
    out['gaze_depth_m'] = gb['depth_m_flooronly'].mean().values
    out['n_floor_samples_bin'] = gb['floor_int'].sum().values.astype(int)

    # Keep populated bins
    out = out[out['n_samples_bin'] >= MIN_SAMPLES_PER_BIN].reset_index(drop=True)
    if MIN_FLOOR_SAMPLES_PER_BIN > 0:
        out = out[out['n_floor_samples_bin'] >= MIN_FLOOR_SAMPLES_PER_BIN].reset_index(drop=True)
    return out

def compute_corr_and_pvals(df_bins, method='spearman'):
    present = [c for c in FEATURE_ORDER if c in df_bins.columns]
    if len(present) < 2:
        return pd.DataFrame(), pd.DataFrame(), present
    X = df_bins[present].copy()
    nunique = X.nunique(dropna=True)
    present = [c for c in present if nunique.get(c, 0) > 1]
    X = X[present]
    if X.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame(), present

    n = len(present)
    R = np.full((n, n), np.nan); P = np.full((n, n), np.nan)
    if not _SCIPY_OK:
        for i in range(n):
            for j in range(n):
                r,_ = _corr_pair(X[present[i]].values, X[present[j]].values, method=method)
                R[i,j] = r
        return pd.DataFrame(R, index=present, columns=present), pd.DataFrame(P, index=present, columns=present), present

    for i in range(n):
        for j in range(n):
            xi, xj = X[present[i]].values, X[present[j]].values
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() >= 3:
                if method=='pearson':
                    r,p = pearsonr(xi[mask], xj[mask])
                else:
                    r,p = spearmanr(xi[mask], xj[mask])
                R[i,j], P[i,j] = r,p
    return pd.DataFrame(R, index=present, columns=present), pd.DataFrame(P, index=present, columns=present), present

def compute_group_perm_pvals(per_subject_bins, vars_order, per_subject_corrs, method='spearman',
                             n_perm=2000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n = len(vars_order)
    P = np.full((n,n), np.nan); Zobs = np.full((n,n), np.nan)

    subj_C = {s: C.reindex(index=vars_order, columns=vars_order) for s, C, _ in per_subject_corrs}
    subj_arrays = {}
    for s, bins_df in per_subject_bins.items():
        subj_arrays[s] = {v: (bins_df[v].values if v in bins_df.columns else None) for v in vars_order}

    for i in range(1,n):
        for j in range(0,i):
            r_list, valid = [], []
            for s, C in subj_C.items():
                rij = C.iat[i,j] if C is not None else np.nan
                if np.isfinite(rij):
                    r_list.append(rij); valid.append(s)
            if not r_list:
                continue

            z_obs = np.arctanh(np.clip(r_list, -0.999999, 0.999999))
            z_mean = np.nanmean(z_obs); Zobs[i,j] = z_mean

            z_null = []
            for _ in range(n_perm):
                r_perm = []
                for s in valid:
                    xi_full = subj_arrays[s][vars_order[i]]
                    xj_full = subj_arrays[s][vars_order[j]]
                    if xi_full is None or xj_full is None: continue
                    mask = np.isfinite(xi_full) & np.isfinite(xj_full)
                    if mask.sum() < 3: continue
                    xi = xi_full[mask]; xj = xj_full[mask].copy()
                    rng.shuffle(xj)
                    r,_ = _corr_pair(xi, xj, method=method)
                    if np.isfinite(r): r_perm.append(r)
                if r_perm:
                    z_null.append(np.nanmean(np.arctanh(np.clip(r_perm, -0.999999, 0.999999))))
            z_null = np.array(z_null, float)
            z_null = z_null[np.isfinite(z_null)]
            P[i,j] = np.mean(np.abs(z_null) >= np.abs(z_mean)) if z_null.size else np.nan
    return pd.DataFrame(P, index=vars_order, columns=vars_order), pd.DataFrame(Zobs, index=vars_order, columns=vars_order)

# -------- FDR (Benjamini–Hochberg) on lower triangles --------
def fdr_bh_on_lower_triangle(p_df_list):
    """
    p_df_list: list of DataFrames (permutation p-values) to correct together.
    Returns list of q-value DataFrames with same shapes.
    """
    if not p_df_list:
        return []
    flats, coords = [], []
    for li, P in enumerate(p_df_list):
        vals = P.values
        n = vals.shape[0]
        for i in range(1, n):
            for j in range(0, i):
                if np.isfinite(vals[i, j]):
                    flats.append(vals[i, j])
                    coords.append((li, i, j))
    if not flats:
        return [df.copy() for df in p_df_list]

    p = np.asarray(flats, float)
    m = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order); ranks[order] = np.arange(1, m+1)
    q = p * m / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_adj = np.empty_like(q); q_adj[order] = np.minimum(q_sorted, 1.0)

    out = [df.copy() for df in p_df_list]
    for (li, i, j), qv in zip(coords, q_adj):
        out[li].iat[i, j] = qv
        out[li].iat[j, i] = qv  # symmetric write (optional)
    return out

# ---------- plotting helper (quick previews) ----------
def plot_corr_square_triangle(C, vars_order, title, outfile, pvals=None, alpha=None,
                              cmap_name='coolwarm', bin_edges=BIN_EDGES,
                              max_cell_fill=MAX_CELL_FILL, annotate=False, annot_fmt=".2f", fs=0.7):
    if C.empty: return
    C = C.reindex(index=vars_order, columns=vars_order)
    P = pvals.reindex(index=vars_order, columns=vars_order) if (pvals is not None and not pvals.empty) else None
    vals = C.values; n = len(vars_order)

    fig, ax = plt.subplots(figsize=(max(6, fs*n), max(6, fs*n)))
    Nbins = len(bin_edges) - 1
    cmap_discrete = plt.get_cmap(cmap_name, Nbins)
    norm_discrete = colors.BoundaryNorm(bin_edges, Nbins, clip=True)

    # grid
    for i in range(n):
        ax.axhline(i-0.5, color='lightgray', lw=0.5, zorder=0)
        ax.axvline(i-0.5, color='lightgray', lw=0.5, zorder=0)

    # diagonal (grey)
    d = max_cell_fill
    for i in range(n):
        ax.add_patch(patches.Rectangle((i-d/2, i-d/2), d, d, facecolor='#d0d0d0', edgecolor='white', lw=0.8, zorder=2))

    # lower-triangle squares
    for i in range(1, n):
        for j in range(0, i):
            r = vals[i, j]
            if not np.isfinite(r): continue
            if (P is not None) and np.isfinite(P.values[i,j]) and (alpha is not None) and (P.values[i,j] >= alpha):
                continue
            side = (np.abs(r)) * max_cell_fill
            if side <= 0: continue
            ax.add_patch(patches.Rectangle((j-side/2, i-side/2), side, side,
                                           facecolor=cmap_discrete(norm_discrete(r)),
                                           edgecolor='white', lw=0.8, zorder=3))
            if annotate:
                ax.text(j, i, format(r, annot_fmt), ha='center', va='center', fontsize=8, color='black', zorder=4)

    ax.set_xticks(range(n)); ax.set_xticklabels(vars_order, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(vars_order)
    ax.set_xlim(-0.5, n-0.5); ax.set_ylim(n-0.5, -0.5); ax.set_aspect('equal')
    ax.set_title(title, pad=14)

    # mask upper triangle
    ax.add_patch(patches.Polygon([(-0.5,-0.5),(n-0.5,-0.5),(n-0.5,n-0.5)], closed=True,
                                 facecolor='white', edgecolor='none', zorder=1))
    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_discrete, norm=norm_discrete); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=bin_edges); cb.set_label('Correlation (r)')
    plt.tight_layout(); fig.savefig(outfile, dpi=300); plt.close(fig)

# ---------------- MAIN ----------------
def main():
    np.random.seed(RNG_SEED)
    meta = safe_read_csv(META_ALL)
    if meta is None or meta.empty or 'reldir' not in meta.columns:
        print(f"[ERROR] Could not read metadata_all.csv or missing 'reldir' at {META_ALL}")
        return

    # Collect outputs
    terrains_global = set()
    per_subject_corrs_all = []     # list of (subj, C_df, P_df)
    per_subject_bins_all  = {}     # subj -> bins df
    per_subject_corrs_byT = {}     # terrain -> list of (subj, C_df, P_df)
    per_subject_bins_byT  = {}     # terrain -> subj -> bins df
    any_vars_all          = set()
    any_vars_byT          = {}
    used_subjects_all     = []

    for _, row in meta.iterrows():
        subj = str(row['reldir'])
        vis_path = os.path.join(BASE_DIR, subj, 'output', 'visual_events.csv')
        vis = safe_read_csv(vis_path)
        if vis is None or vis.empty: continue

        # walking only
        if 'is_walking' in vis.columns:
            vis = vis[vis['is_walking'] == True].copy()
        if vis.empty: continue

        # terrains present in THIS subject
        terrains_this = []
        if 'area_label' in vis.columns:
            terrains_this = [_norm_label(t) for t in vis['area_label'].dropna().unique().tolist()]
            terrains_global.update(terrains_this)

        # ---- ALL data ----
        bins = compute_per_bin_features(vis, terrain=None)
        if not bins.empty:
            per_subject_bins_all[subj] = bins
            C,P,_ = compute_corr_and_pvals(bins, method=CORR_METHOD)
            if not C.empty:
                C.to_csv(os.path.join(OUT_DIR, f"{subj}_corr.csv"))
                P.to_csv(os.path.join(OUT_DIR, f"{subj}_corr_pvals.csv"))
                per_subject_corrs_all.append((subj, C, P))
                any_vars_all.update(list(C.columns))
                used_subjects_all.append(subj)

        # ---- per terrain (only those seen in this subject) ----
        for t in terrains_this:
            bt = compute_per_bin_features(vis, terrain=t)
            if bt.empty: continue
            per_subject_bins_byT.setdefault(t, {})[subj] = bt
            Ct, Pt, _ = compute_corr_and_pvals(bt, method=CORR_METHOD)
            if Ct.empty: continue
            osuffix = f"__terrain-{t}"
            Ct.to_csv(os.path.join(OUT_DIR, f"{subj}_corr{osuffix}.csv"))
            Pt.to_csv(os.path.join(OUT_DIR, f"{subj}_corr_pvals{osuffix}.csv"))
            per_subject_corrs_byT.setdefault(t, []).append((subj, Ct, Pt))
            any_vars_byT.setdefault(t, set()).update(list(Ct.columns))

    if not per_subject_corrs_all:
        print("[WARN] No subject correlation matrices were produced.")
        return

    # Vars order (global) and per-terrain
    vars_order_all = [v for v in FEATURE_ORDER if v in any_vars_all]
    with open(os.path.join(OUT_DIR, "vars_order.json"), "w") as f:
        json.dump(vars_order_all, f, indent=2)
    with open(os.path.join(OUT_DIR, "participants_used.txt"), "w") as f:
        f.write("\n".join(used_subjects_all))

    terrains = sorted([t for t in terrains_global if t is not None])
    for t in terrains:
        vset = any_vars_byT.get(t, set())
        vt = [v for v in FEATURE_ORDER if v in vset]
        with open(os.path.join(OUT_DIR, f"vars_order__terrain-{t}.json"), "w") as f:
            json.dump(vt, f, indent=2)
        plist = sorted(per_subject_bins_byT.get(t, {}).keys())
        with open(os.path.join(OUT_DIR, f"participants_used__terrain-{t}.txt"), "w") as f:
            f.write("\n".join(plist))

    # ---- GROUP (ALL) ----
    aligned = [C.reindex(index=vars_order_all, columns=vars_order_all).values for _, C, _ in per_subject_corrs_all]
    stack = np.stack(aligned, axis=0)
    n = stack.shape[1]
    group = np.full((n,n), np.nan)
    for i in range(n):
        for j in range(n):
            group[i,j] = fisher_mean(stack[:, i, j])
    group_df = pd.DataFrame(group, index=vars_order_all, columns=vars_order_all)
    group_df.to_csv(os.path.join(OUT_DIR, "group_corr_mean.csv"))

    print(f"[INFO] Running ALL-data group permutations with N_PERM={N_PERM} ...")
    gP, gZ = compute_group_perm_pvals(per_subject_bins=per_subject_bins_all,
                                      vars_order=vars_order_all,
                                      per_subject_corrs=per_subject_corrs_all,
                                      method=CORR_METHOD, n_perm=N_PERM, rng_seed=RNG_SEED)
    gP.to_csv(os.path.join(OUT_DIR, "group_corr_perm_pvals.csv"))
    gZ.to_csv(os.path.join(OUT_DIR, "group_corr_perm_zobs.csv"))

    # FDR (ALL-data)
    gQ, = fdr_bh_on_lower_triangle([gP])
    gQ.to_csv(os.path.join(OUT_DIR, "group_corr_perm_qvals.csv"))

    # Quick figs (mask by q if enabled; else by raw p with alpha=0.01 for legacy)
    if USE_FDR_FOR_MASK:
        mask_df = gQ; mask_alpha = FDR_Q_ALPHA
        title_suffix = f" (FDR q<{FDR_Q_ALPHA})"
    else:
        mask_df = gP; mask_alpha = 0.01
        title_suffix = f" (p<0.01)"
    plot_corr_square_triangle(group_df, vars_order_all,
        f"Group — {CORR_METHOD.capitalize()} r (Fisher z-mean){title_suffix}",
        os.path.join(OUT_DIR, "group_corr_mean_heatmap.png"),
        pvals=mask_df, alpha=mask_alpha, annotate=False)
    plot_corr_square_triangle(group_df, vars_order_all,
        f"Group — {CORR_METHOD.capitalize()} r (Fisher z-mean) — annotated{title_suffix}",
        os.path.join(OUT_DIR, "group_corr_mean_heatmap_annotated.png"),
        pvals=mask_df, alpha=mask_alpha, annotate=True)

    # ---- GROUP per TERRAIN ----
    terrain_p_list = []
    terrain_keys   = []
    terrain_vars_orders = {}
    for t in terrains:
        subj_list = per_subject_corrs_byT.get(t, [])
        if not subj_list:
            continue
        vt_path = os.path.join(OUT_DIR, f"vars_order__terrain-{t}.json")
        if os.path.exists(vt_path):
            with open(vt_path, "r") as f: vars_order_t = json.load(f)
        else:
            vseen = set().union(*[set(C.columns) for _, C, _ in subj_list])
            vars_order_t = [v for v in FEATURE_ORDER if v in vseen]
        terrain_vars_orders[t] = vars_order_t

        aligned_t = [C.reindex(index=vars_order_t, columns=vars_order_t).values for _, C, _ in subj_list]
        stack_t = np.stack(aligned_t, axis=0)
        n = stack_t.shape[1]
        group_t = np.full((n,n), np.nan)
        for i in range(n):
            for j in range(n):
                group_t[i,j] = fisher_mean(stack_t[:, i, j])
        group_t_df = pd.DataFrame(group_t, index=vars_order_t, columns=vars_order_t)
        group_t_df.to_csv(os.path.join(OUT_DIR, f"group_corr_mean__terrain-{t}.csv"))

        print(f"[INFO] Running group permutations for terrain='{t}' with N_PERM={N_PERM} ...")
        bins_t = per_subject_bins_byT.get(t, {})
        gP_t, gZ_t = compute_group_perm_pvals(per_subject_bins=bins_t,
                                              vars_order=vars_order_t,
                                              per_subject_corrs=subj_list,
                                              method=CORR_METHOD, n_perm=N_PERM, rng_seed=RNG_SEED)
        gP_t.to_csv(os.path.join(OUT_DIR, f"group_corr_perm_pvals__terrain-{t}.csv"))
        gZ_t.to_csv(os.path.join(OUT_DIR, f"group_corr_perm_zobs__terrain-{t}.csv"))

        # stash for joint FDR if requested
        terrain_p_list.append(gP_t)
        terrain_keys.append(t)

    # FDR for terrains
    if terrain_p_list:
        if FDR_JOINT_ACROSS_TERRAINS:
            # correct across all terrain matrices together
            q_list = fdr_bh_on_lower_triangle(terrain_p_list)
            for t, qdf in zip(terrain_keys, q_list):
                qdf.to_csv(os.path.join(OUT_DIR, f"group_corr_perm_qvals__terrain-{t}.csv"))
        else:
            # correct per terrain independently
            for t, Pdf in zip(terrain_keys, terrain_p_list):
                qdf, = fdr_bh_on_lower_triangle([Pdf])
                qdf.to_csv(os.path.join(OUT_DIR, f"group_corr_perm_qvals__terrain-{t}.csv"))

        # quick figs for terrains (mask by q if enabled)
        for t in terrain_keys:
            vars_order_t = terrain_vars_orders[t]
            C_path = os.path.join(OUT_DIR, f"group_corr_mean__terrain-{t}.csv")
            C_t = pd.read_csv(C_path, index_col=0)
            if USE_FDR_FOR_MASK:
                Pmask_t = pd.read_csv(os.path.join(OUT_DIR, f"group_corr_perm_qvals__terrain-{t}.csv"), index_col=0)
                alpha_t = FDR_Q_ALPHA
                cap = f"(FDR q<{FDR_Q_ALPHA})"
            else:
                Pmask_t = pd.read_csv(os.path.join(OUT_DIR, f"group_corr_perm_pvals__terrain-{t}.csv"), index_col=0)
                alpha_t = 0.01
                cap = "(p<0.01)"
            plot_corr_square_triangle(C_t, vars_order_t,
                f"Group — {t} {cap}",
                os.path.join(OUT_DIR, f"group_corr_mean_heatmap__terrain-{t}.png"),
                pvals=Pmask_t, alpha=alpha_t, annotate=False)

    # ---- MANIFEST for plotting script ----
    manifest = {
        "BASE_DIR": BASE_DIR,
        "OUT_DIR": OUT_DIR,
        "EPS": EPS,
        "MIN_SAMPLES_PER_BIN": MIN_SAMPLES_PER_BIN,
        "MIN_FLOOR_SAMPLES_PER_BIN": MIN_FLOOR_SAMPLES_PER_BIN,
        "CORR_METHOD": CORR_METHOD,
        "ALPHA_SUBJECT": ALPHA_SUBJECT,
        "N_PERM": N_PERM,
        "RNG_SEED": RNG_SEED,
        "BIN_EDGES": BIN_EDGES.tolist(),
        "terrains": terrains,
        "participants_used_all": used_subjects_all,
        "FDR_Q_ALPHA": FDR_Q_ALPHA,
        "FDR_JOINT_ACROSS_TERRAINS": FDR_JOINT_ACROSS_TERRAINS,
        "USE_FDR_FOR_MASK": USE_FDR_FOR_MASK
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Saved all outputs to {OUT_DIR}")

if __name__ == "__main__":
    main()
