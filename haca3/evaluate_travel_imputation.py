#!/usr/bin/env python3
"""
Travel-subjects imputation evaluation for HACA3 (contrastive / ContrastiveImageEncoder variant).

Data layout
-----------
  travel_root/
    TRAVEL-HC01/
      HOME1-01/proc/*.nii.gz   (3-D NIfTI volumes; contrast encoded in filename)
      HOME1-02/proc/*.nii.gz
      ...
    TRAVEL-HC02/...

Runs the same within-site imputation evaluation as evaluate_imputation.py,
restricted to the target site supplied via --target-site, and additionally
saves example source/synthesised/target image panels.
"""

import argparse
import sys
import os
import re
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from itertools import combinations
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from collections import defaultdict
import nibabel as nib
from torchvision.transforms import Compose, Pad, CenterCrop

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from modules.model import HACA3
from modules.dataset import contrast_names
from modules.utils import mkdir_p
from modules.network import ContrastiveImageEncoder


# ── display helpers ───────────────────────────────────────────────────────────
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
N_SOURCE_COLORS = {4: '#5B9BD5', 3: '#ED9234', 2: '#70B050', 1: '#EF8BB8'}
DOT_COLORS      = ['#1f77b4', '#ff7f0e', '#2ca02c', '#e377c2']
N_EXAMPLES      = 3   # example images to collect per target contrast


# ── volume helpers ────────────────────────────────────────────────────────────
_default_transform = Compose([Pad(40), CenterCrop([224, 224])])
_vol_cache = {}


def extract_contrast_from_filename(fname):
    """Return HACA3 contrast name from a travel proc filename.

    Filename example:
      TRAVEL-HC01_HOME1-01_01-05_BRAIN-T1-IRFSPGR-3D-SAGITTAL-PRE_hdrfix_n4_reg.nii.gz
    Checks FLAIR first to avoid matching 'T1' inside other tokens.
    """
    fname_up = os.path.basename(fname).upper()
    if 'FLAIR' in fname_up:
        return 'FLAIR'
    m = re.search(r'BRAIN-(T1|T2|PD)', fname_up)
    if m:
        c = m.group(1)
        return 'T1PRE' if c == 'T1' else c
    return None


def load_volume_slice(fpath, axis, slice_idx, normalization_method='01'):
    """Return a [1, 224, 224] float32 tensor for one slice of a 3-D volume.

    Volumes are cached in memory after first load (per worker process).
    axis: 2=axial, 1=coronal, 0=sagittal
    """
    global _vol_cache
    if fpath not in _vol_cache:
        if not os.path.exists(fpath):
            _vol_cache[fpath] = None
        else:
            vol = np.squeeze(nib.load(fpath).get_fdata().astype(np.float32))
            if normalization_method == 'wm':
                vol = vol / 2.0
            else:
                p95 = np.percentile(vol, 95)
                vol = vol / (p95 + 1e-5)
                vol = np.clip(vol, 0.0, 5.0)
            _vol_cache[fpath] = vol
    vol = _vol_cache[fpath]
    if vol is None or vol.ndim < 3:
        return torch.ones(1, 224, 224)
    if axis == 2:
        s = vol[:, :, slice_idx]
    elif axis == 1:
        s = vol[:, slice_idx, :]
    else:
        s = vol[slice_idx, :, :]
    # transpose consistent with existing 2D dataset (transpose([1, 0]))
    t = torch.from_numpy(s.T.copy()).unsqueeze(0)   # [1, H, W]
    return _default_transform(t)


def _background_mask(image_dicts):
    """In-place: multiply all images by their joint non-zero mask."""
    mask = torch.ones(1, 224, 224)
    for d in image_dicts:
        mask = mask * d['image'].ge(1e-8)
    for d in image_dicts:
        d['image'] = d['image'] * mask
        d['mask']  = mask.bool()
    return image_dicts


# ── dataset ───────────────────────────────────────────────────────────────────
class TravelVolumeDataset(Dataset):
    """
    One item = one 2-D slice extracted from aligned 3-D volumes for a given
    (subject, target_site) pair.  Each subject maps to one site_id index so
    per-subject metrics are reported analogously to per-site metrics elsewhere.
    """
    _AXIS = {'axial': 2, 'coronal': 1, 'sagittal': 0}

    def __init__(self, travel_root, target_site, contrasts, orientations,
                 normalization_method='01'):
        self.normalization_method = normalization_method
        self.subjects = []
        self.items    = []   # (subj_idx, axis, slice_idx, contrast_files_dict)

        for subject in sorted(os.listdir(travel_root)):
            site_dir = os.path.join(travel_root, subject, target_site, 'proc')
            if not os.path.isdir(site_dir):
                continue

            contrast_files = {}
            for fname in sorted(os.listdir(site_dir)):
                if not fname.endswith('_reg.nii.gz'):
                    continue
                c = extract_contrast_from_filename(fname)
                if c and c in contrasts and c not in contrast_files:
                    contrast_files[c] = os.path.join(site_dir, fname)

            if not contrast_files:
                continue

            ref   = contrast_files.get('T1PRE') or next(iter(contrast_files.values()))
            shape = nib.load(ref).header.get_data_shape()
            if len(shape) < 3:
                continue

            subj_idx = len(self.subjects)
            self.subjects.append(subject)

            for ori in orientations:
                axis  = self._AXIS[ori]
                n_slc = int(shape[axis])
                for si in range(n_slc):
                    self.items.append((subj_idx, axis, si, contrast_files))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        subj_idx, axis, slice_idx, cfiles = self.items[idx]
        image_dicts = []
        for c_idx, cname in enumerate(contrast_names):
            fpath = cfiles.get(cname)
            image = (load_volume_slice(fpath, axis, slice_idx,
                                       self.normalization_method)
                     if fpath else torch.ones(1, 224, 224))
            image_dicts.append({
                'image':       image,
                'site_id':     subj_idx,
                'contrast_id': c_idx,
                'exists':      0 if image[0, 0, 0] > 0.9999 else 1,
                'mask':        torch.ones(1, 224, 224, dtype=torch.bool),
            })
        return _background_mask(image_dicts)


# ── helpers ───────────────────────────────────────────────────────────────────
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


# ── boxplot figure (consistent with evaluate_imputation.py) ───────────────────
def make_boxplot_figure(all_metrics, all_subsets, metric_name, out_path, target_site):
    """
    2×2 panel figure.  Each panel = one target contrast.
    Each panel = boxplot strip + dot matrix, as in evaluate_imputation.py.
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

        box_data, box_colors = [], []
        for combo in all_subsets:
            vals = []
            for site_vals in all_metrics[target_idx][tuple(combo)].values():
                if metric_name == 'ssim':
                    vals.extend(v[0] for v in site_vals)
                else:
                    vals.extend(v[1] for v in site_vals)
            box_data.append(vals if vals else [0.0])
            box_colors.append(N_SOURCE_COLORS[len(combo)])

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

        ax_box.set_title(f'Target contrast: {CONTRAST_DISPLAY[target_name]}',
                         fontsize=11, pad=4)
        ax_box.set_ylabel(metric_name.upper(), fontsize=10)
        ax_box.grid(axis='y', alpha=0.3)
        ax_box.set_xlim(-0.5, n_combos - 0.5)

        if metric_name == 'ssim':
            ax_box.set_ylim(0.6, 1.0)
            ax_box.set_yticks(np.arange(0.6, 1.01, 0.05))
        elif metric_name == 'psnr':
            ax_box.set_ylim(10.0, 35.0)
            ax_box.set_yticks(np.arange(10.0, 35.1, 2.5))

        ax_dots.set_xlim(-0.5, n_combos - 0.5)
        ax_dots.set_ylim(-0.5, 3.5)
        for ci, combo in enumerate(all_subsets):
            for ri in range(4):
                dot_y = 3 - ri
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

    legend_patches = [
        mpatches.Patch(color=N_SOURCE_COLORS[n], alpha=0.75,
                       label=f'{n} source{"s" if n > 1 else ""}')
        for n in [4, 3, 2, 1]
    ]
    fig.legend(handles=legend_patches, loc='upper right',
               bbox_to_anchor=(0.99, 0.99), fontsize=9, framealpha=0.9)

    plt.suptitle(
        f'HACA3 Travel Imputation – {metric_name.upper()}  [{target_site}]',
        fontsize=14, y=1.01,
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# ── example image figure ──────────────────────────────────────────────────────
def make_example_figure(examples, target_idx, out_path, target_site):
    """
    Plot up to N_EXAMPLES rows showing all 4 source contrasts, the synthesised
    image and the ground-truth target for the all-4-source combo.

    examples: list of (src_np[4], rec_np, tgt_np, ssim_val, psnr_val, subject)
    Columns: [T1-src | T2-src | PD-src | FLAIR-src | Synthesised | Ground Truth]
    """
    n = min(len(examples), N_EXAMPLES)
    if n == 0:
        return

    target_name = contrast_names[target_idx]
    n_cols = 6
    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 2.8, n * 2.8))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = (
        [f'{CONTRAST_DISPLAY[contrast_names[i]]}\n(source)' for i in range(4)]
        + ['Synthesised', 'Ground Truth']
    )
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=8)

    for row, (src_np, rec_np, tgt_np, sv, pv, subj) in enumerate(examples[:n]):
        for ci in range(4):
            axes[row, ci].imshow(np.clip(src_np[ci], 0, 1),
                                 cmap='gray', vmin=0, vmax=1)
            axes[row, ci].axis('off')

        ax = axes[row, 4]
        ax.imshow(np.clip(rec_np, 0, 1), cmap='gray', vmin=0, vmax=1)
        ax.set_xlabel(f'SSIM={sv:.3f}  PSNR={pv:.1f} dB', fontsize=7)
        ax.xaxis.set_label_coords(0.5, -0.04)
        ax.axis('off')

        axes[row, 5].imshow(np.clip(tgt_np, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[row, 5].axis('off')

        axes[row, 0].text(
            -0.12, 0.5, subj,
            transform=axes[row, 0].transAxes,
            fontsize=6, va='center', ha='right',
        )

    plt.suptitle(
        f'Target: {CONTRAST_DISPLAY[target_name]}  |  Site: {target_site}'
        f'  (all 4 sources)',
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────
def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(
        description='HACA3 travel-subjects imputation evaluation (contrastive).'
    )
    parser.add_argument('--travel-root',  type=str, required=True,
                        help='Root dir containing TRAVEL-HC* subject folders')
    parser.add_argument('--target-site',  type=str, required=True,
                        help='Site folder to evaluate on, e.g. HOME1-01')
    parser.add_argument('--contrasts',    type=str, nargs='+', required=True)
    parser.add_argument('--orientations', type=str, nargs='+',
                        default=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--out-dir',      type=str, required=True)
    parser.add_argument('--pretrained-haca3',            type=str, required=True)
    parser.add_argument('--pretrained-eta-encoder',      type=str, default=None)
    parser.add_argument('--pretrained-contrast-encoder', type=str, default=None)
    parser.add_argument('--beta-dim',     type=int, default=5)
    parser.add_argument('--contrast-dim', type=int, default=2)
    parser.add_argument('--eta-dim',      type=int, default=2)
    parser.add_argument('--batch-size',   type=int, default=8)
    parser.add_argument('--gpu-id',       type=int, default=0)
    parser.add_argument('--max-batches',  type=int, default=None,
                        help='Limit to N batches (for debugging)')
    args = parser.parse_args(args)

    sep = '=' * 10
    print(f'{sep} HACA3 TRAVEL IMPUTATION  [{args.target_site}] {sep}')

    # ── contrast encoder ──────────────────────────────────────────────────
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    contrast_encoder = ContrastiveImageEncoder(embedding_dim=args.contrast_dim)
    if args.pretrained_contrast_encoder is not None:
        ckpt = torch.load(args.pretrained_contrast_encoder, map_location=device)
        contrast_encoder.load_state_dict(ckpt['model_state_dict'])
    contrast_encoder.eval()
    contrast_encoder.to(device)

    # ── model ─────────────────────────────────────────────────────────────
    print(f'{sep} Loading Model {sep}')
    haca3 = HACA3(
        beta_dim=args.beta_dim,
        contrast_dim=args.contrast_dim,
        eta_dim=args.eta_dim,
        pretrained_haca3=args.pretrained_haca3,
        pretrained_eta_encoder=args.pretrained_eta_encoder,
        contrast_encoder=contrast_encoder,
        gpu_id=args.gpu_id,
    )
    device = haca3.device
    for m in [haca3.beta_encoder, haca3.eta_encoder, haca3.contrast_encoder,
              haca3.decoder, haca3.patchifier, haca3.attention_module]:
        m.eval()

    mkdir_p(args.out_dir)

    # ── dataset ───────────────────────────────────────────────────────────
    print(f'{sep} Loading Dataset {sep}')
    dataset = TravelVolumeDataset(
        args.travel_root, args.target_site,
        args.contrasts, args.orientations,
        normalization_method='01',
    )
    print(f'Subjects ({len(dataset.subjects)}): {dataset.subjects}')
    print(f'Total slices: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

    all_subsets = get_all_subsets()
    print(f'Source combinations per target: {len(all_subsets)}  '
          f'(1×4-src + 4×3-src + 6×2-src + 4×1-src)')

    # all_metrics[target_idx][combo_tuple][subject] = [(ssim, psnr), ...]
    all_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    slice_rows  = []
    n_total, n_valid = 0, 0
    FULL_COMBO  = tuple(range(4))
    examples    = defaultdict(list)  # target_idx → [(src_np[4], rec, tgt, sv, pv, subj)]

    print(f'{sep} Running Evaluation {sep}')
    with torch.no_grad():
        for batch_id, image_dicts in enumerate(tqdm(loader, desc='Batches')):
            if args.max_batches is not None and batch_id >= args.max_batches:
                break

            B = image_dicts[0]['image'].shape[0]
            n_total += B

            # filter: only slices where all 4 contrasts exist
            exists_stack = torch.stack(
                [d['exists'] for d in image_dicts], dim=-1)   # [B, 4]
            valid_mask = (exists_stack == 1).all(dim=-1)
            if not valid_mask.any():
                continue
            valid_idxs = valid_mask.nonzero(as_tuple=True)[0].tolist()
            n_valid += len(valid_idxs)

            imgs = [d['image'].to(device) for d in image_dicts]   # 4 × [B,1,H,W]

            logits_all, _  = haca3.calculate_beta(imgs)
            contrasts_all  = haca3.calculate_contrast(imgs)   # list of [B, contrast_dim]
            etas_all       = haca3.calculate_eta(imgs)         # list of [B, eta_dim, 1, 1]

            mask  = image_dicts[0]['mask'].to(device)                              # [B,1,H,W]
            masks = torch.stack([d['mask'] for d in image_dicts],
                                dim=-1).to(device)                                 # [B,1,H,W,4]

            # keys for all 4 sources: [B, contrast_dim+eta_dim, 1, 1]
            keys_all = [
                torch.cat([
                    contrasts_all[i].unsqueeze(2).unsqueeze(3),  # [B,contrast_dim,1,1]
                    etas_all[i],                                  # [B,eta_dim,1,1]
                ], dim=1)
                for i in range(4)
            ]

            for target_idx in range(4):
                target_name     = contrast_names[target_idx]
                contrast_target = contrasts_all[target_idx].unsqueeze(2).unsqueeze(3)  # [B,contrast_dim,1,1]
                eta_target      = etas_all[target_idx]                                 # [B,eta_dim,1,1]
                query           = torch.cat([contrast_target, eta_target], dim=1)     # [B,contrast_dim+eta_dim,1,1]

                for source_combo in all_subsets:
                    combo_key = tuple(source_combo)

                    available = torch.zeros(B, 4, device=device)
                    for src_idx in source_combo:
                        available[:, src_idx] = 1.0

                    rec_images, _, _, _, _ = haca3.decode(
                        logits_all, contrast_target, query,
                        keys_all, available, masks,
                        contrast_dropout=False, contrast_id_to_drop=None,
                    )

                    for si in valid_idxs:
                        site_id   = image_dicts[0]['site_id'][si].item()
                        subj_name = (dataset.subjects[site_id]
                                     if site_id < len(dataset.subjects)
                                     else f'subject_{site_id}')

                        rec_np = rec_images[si, 0].cpu().numpy()
                        tgt_np = imgs[target_idx][si, 0].cpu().numpy()
                        msk_np = mask[si, 0].cpu().numpy()

                        sv, pv = compute_ssim_psnr(rec_np, tgt_np, msk_np)

                        all_metrics[target_idx][combo_key][subj_name].append((sv, pv))
                        slice_rows.append({
                            'target_contrast': CONTRAST_SHORT[target_name],
                            'source_combo':    combo_to_str(source_combo),
                            'n_sources':       len(source_combo),
                            'subject':         subj_name,
                            'batch_id':        batch_id,
                            'sample_id':       si,
                            'ssim':            sv,
                            'psnr':            pv,
                        })

                        if (combo_key == FULL_COMBO
                                and len(examples[target_idx]) < N_EXAMPLES):
                            examples[target_idx].append((
                                [imgs[i][si, 0].cpu().numpy() for i in range(4)],
                                rec_np, tgt_np, sv, pv, subj_name,
                            ))

    print(f'\nTotal slices seen: {n_total} | All-4-available (evaluated): {n_valid}')

    # ── per-slice CSV ──────────────────────────────────────────────────────
    slice_csv    = os.path.join(args.out_dir, 'travel_imputation_per_slice.csv')
    slice_fields = ['target_contrast', 'source_combo', 'n_sources', 'subject',
                    'batch_id', 'sample_id', 'ssim', 'psnr']
    with open(slice_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=slice_fields)
        w.writeheader()
        w.writerows(slice_rows)
    print(f'Saved per-slice CSV: {slice_csv}')

    # ── boxplot stats CSV ──────────────────────────────────────────────────
    stat_fields = [
        'target_contrast', 'source_combo', 'n_sources', 'subject', 'metric',
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
            subj_dict = all_metrics[target_idx][combo_key]

            for subj_name in sorted(subj_dict.keys()):
                subj_vals = subj_dict[subj_name]
                ssim_arr  = np.array([v[0] for v in subj_vals])
                psnr_arr  = np.array([v[1] for v in subj_vals])
                for metric, arr in [('ssim', ssim_arr), ('psnr', psnr_arr)]:
                    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
                    iqr    = q3 - q1
                    stats_rows.append({
                        'target_contrast': tgt_short,
                        'source_combo':    combo_str,
                        'n_sources':       n_src,
                        'subject':         subj_name,
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

            # overall (all subjects pooled)
            all_vals = [v for sv in subj_dict.values() for v in sv]
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
                        'subject':         'OVERALL',
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

    stats_csv = os.path.join(args.out_dir, 'travel_imputation_boxplot_stats.csv')
    with open(stats_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=stat_fields)
        w.writeheader()
        w.writerows(stats_rows)
    print(f'Saved boxplot stats CSV: {stats_csv}')

    # ── boxplot figures ────────────────────────────────────────────────────
    for metric_name in ['ssim', 'psnr']:
        fig_path = os.path.join(
            args.out_dir,
            f'travel_imputation_boxplot_{metric_name}.png',
        )
        make_boxplot_figure(all_metrics, all_subsets, metric_name, fig_path,
                            args.target_site)

    # ── example image figures ──────────────────────────────────────────────
    ex_dir = os.path.join(args.out_dir, 'examples')
    mkdir_p(ex_dir)
    for target_idx in range(4):
        tgt_short = CONTRAST_SHORT[contrast_names[target_idx]]
        ex_path   = os.path.join(
            ex_dir,
            f'examples_{tgt_short}_{args.target_site.replace("/", "_")}.png',
        )
        make_example_figure(examples[target_idx], target_idx, ex_path,
                            args.target_site)

    print(f'\n{sep} TRAVEL IMPUTATION COMPLETE  [{args.target_site}] {sep}')


if __name__ == '__main__':
    main()
