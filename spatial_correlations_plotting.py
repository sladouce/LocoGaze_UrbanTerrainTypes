#!/usr/bin/env python3
"""
plot_spatialbin_correlations_panels.py

Adds an "All terrains" matrix in addition to per-terrain matrices.

Features:
- Mirrors FDR settings from compute:
  * USE_FDR_FOR_MASK (bool), FDR_Q_ALPHA (float) from manifest.json
  * If True, uses *_qvals*.csv (falls back to p if q missing)
  * If False, masks by ALPHA_GROUP (default 0.01)
- Subject grid (fixed size), labels hidden for tight packing
- Saves BOTH standard and ANNOTATED versions:
  * Combined figure with All + per-terrain (standard + annotated)
  * Individual PNGs for All and for each terrain (standard + annotated)

Inputs expected in OUT_DIR (from compute script):
- group_corr_mean.csv
- group_corr_perm_qvals.csv (or group_corr_perm_pvals.csv)
- group_corr_mean__terrain-*.csv
- group_corr_perm_qvals__terrain-*.csv (or ...pvals__terrain-*.csv)
- vars_order.json (+ per-terrain variants)
- participants_used.txt
- manifest.json
"""

import os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches, colors

# ----------------- CONFIG -----------------
OUT_DIR = r"C:/LocoGaze/data/group/corr"
SUBJECT_GRID_ROWS = 3
SUBJECT_GRID_COLS = 7

# ----------------- Helpers -----------------
def load_manifest(out_dir):
    with open(os.path.join(out_dir, "manifest.json"), "r") as f:
        return json.load(f)

def load_vars_order(out_dir, terrain=None):
    if terrain is None:
        path = os.path.join(out_dir, "vars_order.json")
    else:
        path = os.path.join(out_dir, f"vars_order__terrain-{terrain}.json")
        if not os.path.exists(path):
            path = os.path.join(out_dir, "vars_order.json")
    with open(path, "r") as f:
        return json.load(f)

def _norm_str(x: str) -> str:
    return str(x).strip().lower() if x is not None else ""

def _terrain_rank_and_label(t_raw: str):
    s = _norm_str(t_raw)
    if "flat" in s:
        return 0, "Flat", "flat"
    if "cobble" in s:
        return 1, "Cobblestones", "cobblestones"
    if "green" in s:
        return 2, "Green", "green"
    return 99, (t_raw.strip().title() if t_raw else "Other"), (s or "other")

def order_terrains(terrains_list):
    triples, labmap = [], {}
    for t in terrains_list:
        rank, disp, canon = _terrain_rank_and_label(t)
        triples.append((rank, t, disp, canon))
        labmap[t] = (disp, canon)
    triples.sort(key=lambda z: (z[0], z[1]))
    return [z[1] for z in triples], labmap

def plot_corr_square_triangle_ax(ax, C, vars_order, bin_edges, pvals=None, alpha=None,
                                 cmap_name='coolwarm', max_cell_fill=0.99,
                                 annotate=False, annot_fmt=".2f", show_labels=True,
                                 label_fontsize=7, grid_lw=0.4):
    if C.empty:
        ax.set_axis_off()
        return
    C = C.reindex(index=vars_order, columns=vars_order)
    P = pvals.reindex(index=vars_order, columns=vars_order) if (pvals is not None and not pvals.empty) else None
    vals = C.values; n = len(vars_order)

    Nbins = len(bin_edges) - 1
    cmap_discrete = plt.get_cmap(cmap_name, Nbins)
    norm_discrete = colors.BoundaryNorm(bin_edges, Nbins, clip=True)

    # grid
    for i in range(n):
        ax.axhline(i-0.5, color='lightgray', lw=grid_lw, zorder=0)
        ax.axvline(i-0.5, color='lightgray', lw=grid_lw, zorder=0)

    # diagonal (grey)
    d = max_cell_fill
    for i in range(n):
        ax.add_patch(patches.Rectangle((i-d/2, i-d/2), d, d, facecolor='#d0d0d0',
                                       edgecolor='white', lw=0.6, zorder=2))

    # lower triangles
    for i in range(1, n):
        for j in range(0, i):
            r = vals[i, j]
            if not np.isfinite(r):
                continue
            if (P is not None) and np.isfinite(P.values[i, j]) and (alpha is not None) and (P.values[i, j] >= alpha):
                continue
            side = (np.abs(r)) * max_cell_fill
            if side <= 0:
                continue
            ax.add_patch(patches.Rectangle((j-side/2, i-side/2), side, side,
                                           facecolor=cmap_discrete(norm_discrete(r)),
                                           edgecolor='white', lw=0.6, zorder=3))
            if annotate:
                ax.text(j, i, format(r, annot_fmt), ha='center', va='center',
                        fontsize=6, color='black', zorder=4)

    ax.set_xlim(-0.5, n-0.5); ax.set_ylim(n-0.5, -0.5); ax.set_aspect('equal')
    if show_labels:
        ax.set_xticks(range(n)); ax.set_xticklabels(vars_order, rotation=90, fontsize=label_fontsize)
        ax.set_yticks(range(n)); ax.set_yticklabels(vars_order, fontsize=label_fontsize)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")

    # mask upper triangle
    ax.add_patch(patches.Polygon([(-0.5,-0.5),(n-0.5,-0.5),(n-0.5,n-0.5)], closed=True,
                                 facecolor='white', edgecolor='none', zorder=1))

# ----------------- MAIN -----------------
def main():
    manifest = load_manifest(OUT_DIR)
    BIN_EDGES         = np.array(manifest["BIN_EDGES"])
    ALPHA_SUBJECT     = manifest["ALPHA_SUBJECT"]
    terrains_raw      = manifest["terrains"]

    # FDR optics
    FDR_Q_ALPHA       = manifest.get("FDR_Q_ALPHA", 0.05)
    USE_FDR_FOR_MASK  = manifest.get("USE_FDR_FOR_MASK", True)
    ALPHA_GROUP       = manifest.get("ALPHA_GROUP", 0.01)  # for raw-p fallback

    vars_order_all = load_vars_order(OUT_DIR, None)

    # ---------- SUBJECT GRID (labels removed) ----------
    part_list_path = os.path.join(OUT_DIR, "participants_used.txt")
    if os.path.exists(part_list_path):
        with open(part_list_path, "r") as f:
            subjects = [s.strip() for s in f.read().splitlines() if s.strip()]
    else:
        subjects = sorted([fn.replace("_corr.csv","")
                           for fn in os.listdir(OUT_DIR)
                           if fn.endswith("_corr.csv") and "__terrain-" not in fn])

    n_subj = len(subjects)
    rows, cols = SUBJECT_GRID_ROWS, SUBJECT_GRID_COLS
    fig_w = max(10, 2.4*cols)
    fig_h = max(8,  2.4*rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(rows, cols)

    for idx, subj in enumerate(subjects[:rows*cols]):
        r = idx // cols; c = idx % cols
        ax = axes[r, c]
        C_path = os.path.join(OUT_DIR, f"{subj}_corr.csv")
        P_path = os.path.join(OUT_DIR, f"{subj}_corr_pvals.csv")
        if not os.path.exists(C_path):
            ax.set_axis_off(); ax.set_title(subj, fontsize=9); continue
        C = pd.read_csv(C_path, index_col=0)
        P = pd.read_csv(P_path, index_col=0) if os.path.exists(P_path) else pd.DataFrame()
        plot_corr_square_triangle_ax(ax, C, vars_order_all, BIN_EDGES,
                                     pvals=P, alpha=ALPHA_SUBJECT,
                                     show_labels=False)
        ax.set_title(subj, fontsize=8, pad=2)

    # blank leftovers
    for k in range(min(n_subj, rows*cols), rows*cols):
        r = k // cols; c = k % cols
        axes[r, c].set_axis_off()

    plt.subplots_adjust(wspace=0.05, hspace=0.10)

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm', len(BIN_EDGES)-1),
                               norm=colors.BoundaryNorm(BIN_EDGES, len(BIN_EDGES)-1, clip=True))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01, ticks=BIN_EDGES)
    cbar.set_label("Correlation (r)")
    fig.savefig(os.path.join(OUT_DIR, f"subjects_corr_grid_{rows}x{cols}.png"), dpi=300)
    plt.close(fig)

    # ---------- GROUP: ALL + BY TERRAIN ----------
    terrains = terrains_raw or []
    terrains_ordered, labelmap = order_terrains(terrains)
    # Build panel list: first "ALL", then terrains in order
    panel_specs = [("ALL", "All terrains", "all")] + [(t, labelmap[t][0], labelmap[t][1]) for t in terrains_ordered]
    nP = len(panel_specs)

    # Combined (standard)
    fig_std, axs_std = plt.subplots(1, nP, figsize=(6*nP, 6), squeeze=False); axs_std = axs_std[0]
    # Combined (annotated)
    fig_ann, axs_ann = plt.subplots(1, nP, figsize=(6*nP, 6), squeeze=False); axs_ann = axs_ann[0]

    for i, (key, disp_label, canon) in enumerate(panel_specs):
        if key == "ALL":
            C_path = os.path.join(OUT_DIR, "group_corr_mean.csv")
            if USE_FDR_FOR_MASK:
                mask_path_primary = os.path.join(OUT_DIR, "group_corr_perm_qvals.csv")
                mask_alpha = FDR_Q_ALPHA
                title_cap = f"(FDR q<{FDR_Q_ALPHA})"
                if os.path.exists(mask_path_primary):
                    Pmask = pd.read_csv(mask_path_primary, index_col=0)
                else:
                    mask_path_fallback = os.path.join(OUT_DIR, "group_corr_perm_pvals.csv")
                    Pmask = pd.read_csv(mask_path_fallback, index_col=0) if os.path.exists(mask_path_fallback) else pd.DataFrame()
                    mask_alpha = ALPHA_GROUP
                    title_cap = f"(p<{ALPHA_GROUP})"
            else:
                mask_path_primary = os.path.join(OUT_DIR, "group_corr_perm_pvals.csv")
                mask_alpha = ALPHA_GROUP
                title_cap = f"(p<{ALPHA_GROUP})"
                Pmask = pd.read_csv(mask_path_primary, index_col=0) if os.path.exists(mask_path_primary) else pd.DataFrame()
        else:
            # terrain key
            t = key
            C_path = os.path.join(OUT_DIR, f"group_corr_mean__terrain-{t}.csv")
            if USE_FDR_FOR_MASK:
                mask_path_primary = os.path.join(OUT_DIR, f"group_corr_perm_qvals__terrain-{t}.csv")
                mask_alpha = FDR_Q_ALPHA
                title_cap = f"(FDR q<{FDR_Q_ALPHA})"
                if os.path.exists(mask_path_primary):
                    Pmask = pd.read_csv(mask_path_primary, index_col=0)
                else:
                    mask_path_fallback = os.path.join(OUT_DIR, f"group_corr_perm_pvals__terrain-{t}.csv")
                    Pmask = pd.read_csv(mask_path_fallback, index_col=0) if os.path.exists(mask_path_fallback) else pd.DataFrame()
                    mask_alpha = ALPHA_GROUP
                    title_cap = f"(p<{ALPHA_GROUP})"
            else:
                mask_path_primary = os.path.join(OUT_DIR, f"group_corr_perm_pvals__terrain-{t}.csv")
                mask_alpha = ALPHA_GROUP
                title_cap = f"(p<{ALPHA_GROUP})"
                Pmask = pd.read_csv(mask_path_primary, index_col=0) if os.path.exists(mask_path_primary) else pd.DataFrame()

        if not os.path.exists(C_path):
            for ax in (axs_std[i], axs_ann[i]):
                ax.set_axis_off(); ax.set_title(f"{disp_label} (no data)")
            continue

        C = pd.read_csv(C_path, index_col=0)

        # STANDARD
        plot_corr_square_triangle_ax(axs_std[i], C, vars_order_all, BIN_EDGES,
                                     pvals=Pmask, alpha=mask_alpha,
                                     show_labels=True, label_fontsize=8)
        axs_std[i].set_title(f"Group — {disp_label} {title_cap}", fontsize=12)

        # ANNOTATED
        plot_corr_square_triangle_ax(axs_ann[i], C, vars_order_all, BIN_EDGES,
                                     pvals=Pmask, alpha=mask_alpha,
                                     show_labels=True, label_fontsize=8,
                                     annotate=True, annot_fmt=".2f")
        axs_ann[i].set_title(f"Group — {disp_label} (annotated) {title_cap}", fontsize=12)

        # INDIVIDUAL SAVES
        fig_t, ax_t = plt.subplots(figsize=(6.5,6.5))
        plot_corr_square_triangle_ax(ax_t, C, vars_order_all, BIN_EDGES,
                                     pvals=Pmask, alpha=mask_alpha,
                                     show_labels=True, label_fontsize=9)
        sm_t = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm', len(BIN_EDGES)-1),
                                     norm=colors.BoundaryNorm(BIN_EDGES, len(BIN_EDGES)-1, clip=True))
        sm_t.set_array([])
        cb_t = fig_t.colorbar(sm_t, ax=ax_t, fraction=0.046, pad=0.04, ticks=BIN_EDGES)
        cb_t.set_label('Correlation (r)')
        ax_t.set_title(f"Group — {disp_label} {title_cap}", fontsize=12)
        plt.tight_layout()
        out_name = ("group_corr_mean_heatmap.png" if key == "ALL"
                    else f"group_corr_mean_heatmap__terrain-{canon}_PLOT.png")
        fig_t.savefig(os.path.join(OUT_DIR, out_name), dpi=300)
        plt.close(fig_t)

        # ANNOTATED INDIVIDUAL
        fig_ta, ax_ta = plt.subplots(figsize=(6.5,6.5))
        plot_corr_square_triangle_ax(ax_ta, C, vars_order_all, BIN_EDGES,
                                     pvals=Pmask, alpha=mask_alpha,
                                     show_labels=True, label_fontsize=9,
                                     annotate=True, annot_fmt=".2f")
        sm_ta = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm', len(BIN_EDGES)-1),
                                      norm=colors.BoundaryNorm(BIN_EDGES, len(BIN_EDGES)-1, clip=True))
        sm_ta.set_array([])
        cb_ta = fig_ta.colorbar(sm_ta, ax=ax_ta, fraction=0.046, pad=0.04, ticks=BIN_EDGES)
        cb_ta.set_label('Correlation (r)')
        ax_ta.set_title(f"Group — {disp_label} (annotated) {title_cap}", fontsize=12)
        plt.tight_layout()
        out_name_ann = ("group_corr_mean_heatmap_annotated.png" if key == "ALL"
                        else f"group_corr_mean_heatmap__terrain-{canon}_PLOT_annotated.png")
        fig_ta.savefig(os.path.join(OUT_DIR, out_name_ann), dpi=300)
        plt.close(fig_ta)

    # shared colorbars for combined figs
    sm_c = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm', len(BIN_EDGES)-1),
                                 norm=colors.BoundaryNorm(BIN_EDGES, len(BIN_EDGES)-1, clip=True))
    sm_c.set_array([])
    cb_std = fig_std.colorbar(sm_c, ax=axs_std.tolist(),
                              fraction=0.046/max(1, nP), pad=0.04, ticks=BIN_EDGES)
    cb_std.set_label('Correlation (r)')
    plt.tight_layout()
    fig_std.savefig(os.path.join(OUT_DIR, "group_corr_mean_heatmap_all_and_by_terrain.png"), dpi=300)
    plt.close(fig_std)

    sm_c2 = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm', len(BIN_EDGES)-1),
                                  norm=colors.BoundaryNorm(BIN_EDGES, len(BIN_EDGES)-1, clip=True))
    sm_c2.set_array([])
    cb_ann = fig_ann.colorbar(sm_c2, ax=axs_ann.tolist(),
                              fraction=0.046/max(1, nP), pad=0.04, ticks=BIN_EDGES)
    cb_ann.set_label('Correlation (r)')
    plt.tight_layout()
    fig_ann.savefig(os.path.join(OUT_DIR, "group_corr_mean_heatmap_all_and_by_terrain_annotated.png"), dpi=300)
    plt.close(fig_ann)

    print("[OK] Plots saved to", OUT_DIR)

if __name__ == "__main__":
    main()
