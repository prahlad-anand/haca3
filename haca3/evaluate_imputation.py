#!/usr/bin/env python3
"""
Imputation evaluation for HACA3 (non-contrastive / theta-encoder variant).

For each of the 4 target contrasts (T1, T2, PD, FLAIR) and every non-empty
subset of {T1, T2, PD, FLAIR} as input sources (15 combos per target = 60
total scenarios), compute SSIM and PSNR.  Only slices where all 4 contrasts
are present are used.

Model loading, dataset loading, and image preprocessing follow evaluate.py
exactly – no extra normalisation is applied before the model forward pass.
"""

import argparse
import sys
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import combinations
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from modules.model import HACA3
from modules.dataset import HACA3Dataset, contrast_names
from modules.utils import mkdir_p


# ── display helpers ──────────────────────────────────────────────────────────
CONTRAST_DISPLAY = {
    'T1PRE': 'T$_1$-w',
    'T2':    'T$_2$-w',
    'PD':    'PD-w',
    'FLAIR': 'FLAIR',
}
CONTRAST_SHORT = {
    'T1PRE': 'T1',
    'T2':    'T2',
    'PD':    'PD',
    'FLAIR': 'FLAIR',
}
# colours per source-set size (matching figure palette)
N_SOURCE_COLORS = {4: '#5B9BD5', 3: '#ED9234', 2: '#70B050', 1: '#EF8BB8'}
# colours for dot-matrix rows (T1, T2, PD, FLAIR)
DOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#e377c2']


# ── helpers ──────────────────────────────────────────────────────────────────
def get_all_subsets():
    """All 15 non-empty subsets of {0,1,2,3}, largest first then lexicographic."""
    subsets = []
    for r in range(4, 0, -1):
        for combo in combinations(range(4), r):
            subsets.append(list(combo))
    return subsets


def compute_ssim_psnr(img1, img2, mask=None):
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    if mask is not None:
        mask = mask.astype(bool)
        if mask.sum() == 0:
            return 0.0, 0.0
        img1, img2 = img1.copy(), img2.copy()
        img1[~mask] = 0
        img2[~mask] = 0
    try:
        sv = ssim(img1, img2, data_range=1.0)
    except Exception:
        sv = 0.0
    try:
        pv = psnr(img1, img2, data_range=1.0)
    except Exception:
        pv = 0.0
    return sv, pv


def combo_to_str(combo):
    return '-'.join(CONTRAST_SHORT[contrast_names[i]] for i in combo)


# ── figure ────────────────────────────────────────────────────────────────────
def make_boxplot_figure(all_metrics, all_subsets, metric_name, out_path):
    """
    2×2 panel figure.  Each panel = one target contrast.
    Each panel = boxplot strip + dot matrix, as in Fig. 11 of the paper.
    """
    n_combos = len(all_subsets)
    fig = plt.figure(figsize=(22, 16))
    outer = GridSpec(2, 2, figure=fig, hspace=0.60, wspace=0.28)

    for target_idx in range(4):
        target_name = contrast_names[target_idx]
        inner = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[target_idx // 2, target_idx % 2],
            height_ratios=[4, 1],
            hspace=0.04,
        )
        ax_box  = fig.add_subplot(inner[0])
        ax_dots = fig.add_subplot(inner[1])

        # ── collect per-combo data (pool all sites) ──────────────────────
        box_data   = []
        box_colors = []
        for combo in all_subsets:
            vals = []
            for site_vals in all_metrics[target_idx][tuple(combo)].values():
                if metric_name == 'ssim':
                    vals.extend(v[0] for v in site_vals)
                else:
                    vals.extend(v[1] for v in site_vals)
            box_data.append(vals if vals else [0.0])
            box_colors.append(N_SOURCE_COLORS[len(combo)])

        # ── boxplots ─────────────────────────────────────────────────────
        bp = ax_box.boxplot(
            box_data,
            positions=list(range(n_combos)),
            widths=0.55,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker='o', markersize=2.5, alpha=0.4, markeredgewidth=0.5),
            showfliers=True,
        )
        for patch, col in zip(bp['boxes'], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

        ax_box.set_title(
            f'Target contrast: {CONTRAST_DISPLAY[target_name]}', fontsize=11, pad=4
        )
        ax_box.set_ylabel(metric_name.upper(), fontsize=10)
        ax_box.grid(axis='y', alpha=0.3)
        ax_box.set_xlim(-0.5, n_combos - 0.5)

        if metric_name == 'ssim':
            ax_box.set_ylim(0.6, 1.0)
            ax_box.set_yticks(np.arange(0.6, 1.01, 0.05))
        elif metric_name == 'psnr':
            ax_box.set_ylim(10.0, 35.0)
            ax_box.set_yticks(np.arange(10.0, 35.1, 2.5))

        # ── dot matrix ───────────────────────────────────────────────────
        ax_dots.set_xlim(-0.5, n_combos - 0.5)
        ax_dots.set_ylim(-0.5, 3.5)
        for ci, combo in enumerate(all_subsets):
            for ri in range(4):          # ri 0=T1PRE … 3=FLAIR
                dot_y = 3 - ri            # T1PRE at top
                col = DOT_COLORS[ri]
                if ri in combo:
                    ax_dots.scatter(ci, dot_y, c=col, s=28, zorder=3)
                else:
                    ax_dots.scatter(ci, dot_y, c='none',
                                    edgecolors=col, s=28, linewidths=0.8, zorder=3)
        ax_dots.set_yticks([3, 2, 1, 0])
        ax_dots.set_yticklabels(
            [CONTRAST_DISPLAY[contrast_names[i]] for i in range(4)], fontsize=7
        )
        ax_dots.set_xticks([])
        ax_dots.tick_params(left=False, pad=2)
        for spine in ax_dots.spines.values():
            spine.set_visible(False)

    # shared legend
    legend_patches = [
        mpatches.Patch(color=N_SOURCE_COLORS[n], alpha=0.75,
                       label=f'{n} source{"s" if n > 1 else ""}')
        for n in [4, 3, 2, 1]
    ]
    fig.legend(handles=legend_patches, loc='upper right',
               bbox_to_anchor=(0.99, 0.99), fontsize=9, framealpha=0.9)

    plt.suptitle(f'HACA3 Imputation – {metric_name.upper()}', fontsize=14, y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────
def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(
        description='HACA3 imputation evaluation – all contrast source combinations.'
    )
    parser.add_argument('--dataset-dirs',  type=str, nargs='+', required=True)
    parser.add_argument('--contrasts',     type=str, nargs='+', required=True)
    parser.add_argument('--orientations',  type=str, nargs='+',
                        default=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--out-dir',       type=str, required=True)
    parser.add_argument('--pretrained-haca3',        type=str, required=True)
    parser.add_argument('--pretrained-eta-encoder',  type=str, default=None)
    parser.add_argument('--beta-dim',     type=int, default=5)
    parser.add_argument('--theta-dim',    type=int, default=2)
    parser.add_argument('--eta-dim',      type=int, default=2)
    parser.add_argument('--batch-size',   type=int, default=8)
    parser.add_argument('--gpu-id',       type=int, default=0)
    parser.add_argument('--max-batches',  type=int, default=None,
                        help='Limit to N batches (for debugging)')
    args = parser.parse_args(args)

    sep = '=' * 10
    print(f'{sep} BEGIN HACA3 IMPUTATION EVALUATION {sep}')

    # ── HACA3 model (identical to evaluate.py) ────────────────────────────
    print(f'{sep} Loading Model {sep}')
    haca3 = HACA3(
        beta_dim=args.beta_dim,
        theta_dim=args.theta_dim,
        eta_dim=args.eta_dim,
        pretrained_haca3=args.pretrained_haca3,
        pretrained_eta_encoder=args.pretrained_eta_encoder,
        gpu_id=args.gpu_id,
    )
    device = haca3.device

    haca3.beta_encoder.eval()
    haca3.eta_encoder.eval()
    haca3.theta_encoder.eval()
    haca3.decoder.eval()
    haca3.patchifier.eval()
    haca3.attention_module.eval()

    mkdir_p(args.out_dir)
    site_names = [os.path.basename(os.path.normpath(d)) for d in args.dataset_dirs]
    print(f'Sites: {site_names}')

    # ── dataset (identical to evaluate.py) ───────────────────────────────
    print(f'{sep} Loading Dataset {sep}')
    valid_dataset = HACA3Dataset(
        args.dataset_dirs, args.contrasts, args.orientations,
        'valid', normalization_method='01'
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    all_subsets = get_all_subsets()   # 15 non-empty subsets of {0,1,2,3}
    print(f'Source combinations per target: {len(all_subsets)}  '
          f'(1×4-src + 4×3-src + 6×2-src + 4×1-src)')

    # all_metrics[target_idx][combo_tuple][site_name] = [(ssim, psnr), ...]
    all_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    slice_rows  = []
    n_total, n_valid = 0, 0

    print(f'{sep} Running Evaluation {sep}')
    with torch.no_grad():
        for batch_id, image_dicts in enumerate(tqdm(valid_loader, desc='Batches')):
            if args.max_batches is not None and batch_id >= args.max_batches:
                break

            B = image_dicts[0]['image'].shape[0]
            n_total += B

            # ── filter: only slices where all 4 contrasts exist ──────────
            exists_stack = torch.stack(
                [d['exists'] for d in image_dicts], dim=-1
            )  # [B, 4]
            valid_mask = (exists_stack == 1).all(dim=-1)   # [B]
            if not valid_mask.any():
                continue
            valid_idxs = valid_mask.nonzero(as_tuple=True)[0].tolist()
            n_valid += len(valid_idxs)

            # ── move images to device (no extra normalisation) ────────────
            imgs = [d['image'].to(device) for d in image_dicts]   # 4 × [B,1,H,W]

            # ── encode all 4 contrasts once for this batch ────────────────
            logits_all, _ = haca3.calculate_beta(imgs)
            # list input → (list of [B,theta_dim,1,1], _, _)
            # theta is already [B, theta_dim, 1, 1] – no unsqueeze needed
            thetas_all, _, _ = haca3.calculate_theta(imgs)
            # list input → list of [B, eta_dim, 1, 1]
            etas_all = haca3.calculate_eta(imgs)

            # masks – exactly as in evaluate.py
            mask  = image_dicts[0]['mask'].to(device)                              # [B,1,H,W]
            masks = torch.stack([d['mask'] for d in image_dicts], dim=-1).to(device)  # [B,1,H,W,4]

            # keys for all 4 sources: list of [B, theta_dim+eta_dim, 1, 1]
            keys_all = [
                torch.cat([thetas_all[i], etas_all[i]], dim=1)
                for i in range(4)
            ]

            # ── iterate over targets × source combos ─────────────────────
            for target_idx in range(4):
                target_name = contrast_names[target_idx]

                # target theta embedding (for decode query)
                theta_target = thetas_all[target_idx]                          # [B,theta_dim,1,1]
                eta_target   = etas_all[target_idx]                            # [B,eta_dim,1,1]
                query        = torch.cat([theta_target, eta_target], dim=1)   # [B,theta_dim+eta_dim,1,1]

                for source_combo in all_subsets:
                    combo_key = tuple(source_combo)

                    # available_contrast_id [B, 4]: 1 if contrast is a source
                    available = torch.zeros(B, 4, device=device)
                    for src_idx in source_combo:
                        available[:, src_idx] = 1.0

                    rec_images, _, _, _, _ = haca3.decode(
                        logits_all, theta_target, query,
                        keys_all, available, masks,
                        contrast_dropout=False, contrast_id_to_drop=None,
                    )

                    # ── record metrics for all-4-available slices ─────────
                    for si in valid_idxs:
                        site_id   = image_dicts[0]['site_id'][si].item()
                        site_name = (site_names[site_id]
                                     if site_id < len(site_names)
                                     else f'site_{site_id}')

                        rec_np = rec_images[si, 0].cpu().numpy()
                        tgt_np = imgs[target_idx][si, 0].cpu().numpy()
                        msk_np = mask[si, 0].cpu().numpy()

                        sv, pv = compute_ssim_psnr(rec_np, tgt_np, msk_np)

                        all_metrics[target_idx][combo_key][site_name].append((sv, pv))
                        slice_rows.append({
                            'target_contrast': CONTRAST_SHORT[target_name],
                            'source_combo':    combo_to_str(source_combo),
                            'n_sources':       len(source_combo),
                            'site':            site_name,
                            'batch_id':        batch_id,
                            'sample_id':       si,
                            'ssim':            sv,
                            'psnr':            pv,
                        })

    print(f'\nTotal slices seen: {n_total} | All-4-available (evaluated): {n_valid}')

    # ── per-slice CSV ─────────────────────────────────────────────────────
    slice_csv = os.path.join(args.out_dir, 'imputation_per_slice.csv')
    slice_fields = ['target_contrast', 'source_combo', 'n_sources', 'site',
                    'batch_id', 'sample_id', 'ssim', 'psnr']
    with open(slice_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=slice_fields)
        w.writeheader()
        w.writerows(slice_rows)
    print(f'Saved per-slice CSV : {slice_csv}')

    # ── boxplot stats CSV ─────────────────────────────────────────────────
    stat_fields = [
        'target_contrast', 'source_combo', 'n_sources', 'site', 'metric',
        'n_samples', 'mean', 'std', 'median',
        'q1', 'q3', 'iqr', 'whisker_low', 'whisker_high',
        'min', 'max',
    ]
    stats_rows = []
    for target_idx in range(4):
        tgt_short = CONTRAST_SHORT[contrast_names[target_idx]]
        for source_combo in all_subsets:
            combo_key = tuple(source_combo)
            combo_str = combo_to_str(source_combo)
            n_src     = len(source_combo)
            site_dict = all_metrics[target_idx][combo_key]

            # per-site rows
            for site_name in sorted(site_dict.keys()):
                site_vals = site_dict[site_name]
                ssim_arr  = np.array([v[0] for v in site_vals])
                psnr_arr  = np.array([v[1] for v in site_vals])
                for metric, arr in [('ssim', ssim_arr), ('psnr', psnr_arr)]:
                    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
                    iqr    = q3 - q1
                    stats_rows.append({
                        'target_contrast': tgt_short,
                        'source_combo':    combo_str,
                        'n_sources':       n_src,
                        'site':            site_name,
                        'metric':          metric,
                        'n_samples':       len(arr),
                        'mean':            float(np.mean(arr)),
                        'std':             float(np.std(arr)),
                        'median':          float(np.median(arr)),
                        'q1':              float(q1),
                        'q3':              float(q3),
                        'iqr':             float(iqr),
                        'whisker_low':     float(q1 - 1.5 * iqr),
                        'whisker_high':    float(q3 + 1.5 * iqr),
                        'min':             float(np.min(arr)),
                        'max':             float(np.max(arr)),
                    })

            # overall (all sites pooled)
            all_vals = [v for sv in site_dict.values() for v in sv]
            if all_vals:
                ssim_arr = np.array([v[0] for v in all_vals])
                psnr_arr = np.array([v[1] for v in all_vals])
                for metric, arr in [('ssim', ssim_arr), ('psnr', psnr_arr)]:
                    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
                    iqr    = q3 - q1
                    stats_rows.append({
                        'target_contrast': tgt_short,
                        'source_combo':    combo_str,
                        'n_sources':       n_src,
                        'site':            'OVERALL',
                        'metric':          metric,
                        'n_samples':       len(arr),
                        'mean':            float(np.mean(arr)),
                        'std':             float(np.std(arr)),
                        'median':          float(np.median(arr)),
                        'q1':              float(q1),
                        'q3':              float(q3),
                        'iqr':             float(iqr),
                        'whisker_low':     float(q1 - 1.5 * iqr),
                        'whisker_high':    float(q3 + 1.5 * iqr),
                        'min':             float(np.min(arr)),
                        'max':             float(np.max(arr)),
                    })

    stats_csv = os.path.join(args.out_dir, 'imputation_boxplot_stats.csv')
    with open(stats_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=stat_fields)
        w.writeheader()
        w.writerows(stats_rows)
    print(f'Saved boxplot stats CSV: {stats_csv}')

    # ── figures ───────────────────────────────────────────────────────────
    for metric_name in ['ssim', 'psnr']:
        fig_path = os.path.join(args.out_dir, f'imputation_boxplot_{metric_name}.png')
        make_boxplot_figure(all_metrics, all_subsets, metric_name, fig_path)

    print(f'\n{sep} IMPUTATION EVALUATION COMPLETE {sep}')


if __name__ == '__main__':
    main()
