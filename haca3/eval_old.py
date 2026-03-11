import argparse
import sys
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.utils import make_grid
import imageio
from collections import defaultdict

from modules.model import HACA3
from modules.dataset import HACA3Dataset, contrast_names
from modules.utils import mkdir_p

import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

def plot_theta_and_attention(
    contrasts_source,
    contrast_target,
    attention,
    batch_idx,
    out_dir,
    sample_idx=0
):
    """
    contrasts_source: list of tensors [B, 2, 1, 1]
    contrast_target: tensor [B, 2, 1, 1]
    attention: tensor [B, num_sources] or [B, num_sources, ...]
    """

    os.makedirs(out_dir, exist_ok=True)

    # --- extract theta (2D) ---
    src_thetas = [
        c[sample_idx, :, 0, 0].detach().cpu().numpy()
        for c in contrasts_source
    ]
    tgt_theta = contrast_target[sample_idx, :, 0, 0].detach().cpu().numpy()

    # --- theta scatter plot ---
    plt.figure(figsize=(6, 6))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, th in enumerate(src_thetas):
        plt.scatter(
            th[0], th[1],
            s=80,
            color=colors[i % len(colors)],
            label=f'Source {i}'
        )

    plt.scatter(
        tgt_theta[0], tgt_theta[1],
        s=140,
        marker='*',
        color='black',
        label='Target'
    )

    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(f'Theta space (batch {batch_idx}, sample {sample_idx})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f'theta_b{batch_idx:05d}_s{sample_idx}.png')
    )
    plt.close()

    # --- attention histogram ---
    att = attention[sample_idx]
    if att.dim() > 1:
        att = att.view(att.shape[0])

    att = att.detach().cpu().numpy()

    plt.figure(figsize=(5, 4))
    plt.bar(range(len(att)), att)
    plt.ylim(0, 1)
    plt.xlabel('Source index')
    plt.ylabel('Attention weight')
    plt.title(f'Attention (sum={att.sum():.3f})')
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f'attention_b{batch_idx:05d}_s{sample_idx}.png')
    )
    plt.close()


def colorize_betas(beta_tensor):
    """
    Convert beta maps [B, C, H, W] into RGB images [B, 3, H, W]
    using colorblind-friendly colors per beta channel.
    """
    # Okabe–Ito colorblind-safe palette (RGB in [0,1])
    COLORS = torch.tensor([
        [0.90, 0.62, 0.00],  # orange
        [0.00, 0.45, 0.70],  # blue
        [0.84, 0.37, 0.00],  # vermillion
        [0.00, 0.62, 0.45],  # bluish green
        [0.80, 0.47, 0.65],  # reddish purple
    ], device=beta_tensor.device)

    B, C, H, W = beta_tensor.shape
    assert C <= COLORS.shape[0], "Not enough colors for beta channels"

    beta_tensor = beta_tensor.clamp(0, 1)

    rgb = torch.zeros((B, 3, H, W), device=beta_tensor.device)

    for c in range(C):
        rgb += beta_tensor[:, c:c+1] * COLORS[c].view(1, 3, 1, 1)

    # Normalize to [0,1] for visualization
    rgb = rgb / (rgb.max(dim=1, keepdim=True)[0] + 1e-6)
    return rgb


def compute_ssim_psnr(img1, img2, mask=None):
    """
    Compute SSIM and PSNR between two images.

    Args:
        img1: numpy array (H, W)
        img2: numpy array (H, W)
        mask: optional mask to focus computation

    Returns:
        ssim_val, psnr_val
    """
    # Ensure images are in valid range
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    if mask is not None:
        mask = mask.astype(bool)
        # Only compute on masked region
        if mask.sum() > 0:
            img1_masked = img1.copy()
            img2_masked = img2.copy()
            img1_masked[~mask] = 0
            img2_masked[~mask] = 0
        else:
            return 0.0, 0.0
    else:
        img1_masked = img1
        img2_masked = img2

    # Compute SSIM
    try:
        ssim_val = ssim(img1_masked, img2_masked, data_range=1.0)
    except:
        ssim_val = 0.0

    # Compute PSNR
    try:
        psnr_val = psnr(img1_masked, img2_masked, data_range=1.0)
    except:
        psnr_val = 0.0

    return ssim_val, psnr_val


def save_grid_png(images, file_path, nrow=4):
    """
    Save a grid of images as PNG.

    Args:
        images: list of tensors [B, 1, H, W] or single tensor
        file_path: output path
        nrow: number of images per row
    """
    mkdir_p(os.path.dirname(file_path))

    if isinstance(images, list):
        # Concatenate all images
        image_tensors = []
        for img in images:
            if img.dim() == 4:
                image_tensors.append(img[:4, [0], ...].cpu())
            else:
                image_tensors.append(img.unsqueeze(0).unsqueeze(0).cpu())
        image_save = torch.cat(image_tensors, dim=0)
    else:
        image_save = images.cpu()

    # Create grid
    grid = make_grid(image_save, nrow=nrow, normalize=True, value_range=(0, 1))

    # Convert to numpy and save
    grid_np = grid.detach().cpu().numpy().transpose(1, 2, 0)
    grid_np = (grid_np * 255).astype(np.uint8)

    # Handle grayscale
    if grid_np.shape[-1] == 1:
        grid_np = grid_np.squeeze(-1)

    imageio.imwrite(file_path, grid_np)


class HACA3Evaluator:
    def __init__(self, haca3_model, device):
        self.haca3 = haca3_model
        self.device = device

    def evaluate_batch(self, image_dicts):
        """
        Perform image-to-image translation and compute metrics.

        Returns:
            dict with reconstructed images, targets, and metrics
        """
        source_images = []
        for i in range(len(image_dicts)):
            source_images.append(image_dicts[i]['image'].to(self.device))

        mask = image_dicts[0]['mask'].to(self.device)
        masks = torch.stack([d['mask'] for d in image_dicts], dim=-1).to(self.device)

        # Get target image
        target_image, contrast_id_for_decoding = self.haca3.select_available_contrasts(image_dicts)
        available_contrast_id = torch.stack([d['exists'] for d in image_dicts], dim=-1).to(self.device)

        # Encode
        logits, betas = self.haca3.calculate_beta(source_images)
        contrasts_source = [b.unsqueeze(2).unsqueeze(3) for b in self.haca3.calculate_contrast(source_images)]
        etas_source = self.haca3.calculate_eta(source_images)

        contrast_target = self.haca3.calculate_contrast(target_image).unsqueeze(2).unsqueeze(3)
        eta_target = self.haca3.calculate_eta(target_image)

        query = torch.cat([contrast_target, eta_target], dim=1)
        keys = [torch.cat([contrast, eta], dim=1) for (contrast, eta) in zip(contrasts_source, etas_source)]

        # Decode
        rec_image, attention, logit_fusion, beta_fusion = self.haca3.decode(
            logits, contrast_target, query, keys,
            available_contrast_id, masks,
            contrast_dropout=False, contrast_id_to_drop=None
        )


        return {
            'source_images': source_images,
            'rec_image': rec_image,
            'target_image': target_image,
            'betas': betas,
            'beta_fusion': beta_fusion,
            'mask': mask,
            'site_ids': [d['site_id'] for d in image_dicts]
        }
    
        # return {
        #     'source_images': source_images,
        #     'rec_image': rec_image,
        #     'target_image': target_image,
        #     'betas': betas,
        #     'beta_fusion': beta_fusion,
        #     'attention': attention,
        #     'contrast_target': contrast_target,
        #     'contrasts_source': contrasts_source,
        # }



def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='Evaluate HACA3 model with SSIM/PSNR metrics.')
    parser.add_argument('--dataset-dirs', type=str, nargs='+', required=True)
    parser.add_argument('--contrasts', type=str, nargs='+', required=True)
    parser.add_argument('--orientations', type=str, nargs='+', default=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--pretrained-haca3', type=str, required=True)
    parser.add_argument('--pretrained-eta-encoder', type=str, default=None)
    parser.add_argument('--pretrained-contrast-encoder', type=str, default=None)
    parser.add_argument('--beta-dim', type=int, default=5)
    parser.add_argument('--contrast-dim', type=int, default=2)
    parser.add_argument('--eta-dim', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=50, help='Save PNG grid every N batches')
    parser.add_argument('--max-batches', type=int, default=None, help='Max batches to evaluate (for testing)')

    args = parser.parse_args(args)

    text_div = '=' * 10
    print(f'{text_div} BEGIN HACA3 EVALUATION {text_div}')

    # Create output directories
    mkdir_p(args.out_dir)
    mkdir_p(os.path.join(args.out_dir, 'grids'))
    mkdir_p(os.path.join(args.out_dir, 'slices'))

    # Extract site names from dataset_dirs
    site_names = [os.path.basename(os.path.normpath(d)) for d in args.dataset_dirs]
    print(f'Sites: {site_names}')

    # Initialize model
    print(f'{text_div} Loading Model {text_div}')
    haca3 = HACA3(
        beta_dim=args.beta_dim,
        contrast_dim=args.contrast_dim,
        eta_dim=args.eta_dim,
        pretrained_haca3=args.pretrained_haca3,
        pretrained_eta_encoder=args.pretrained_eta_encoder,
        pretrained_contrast_encoder=args.pretrained_contrast_encoder,
        gpu_id=args.gpu_id
    )
    device = haca3.device

    # Set to eval mode
    haca3.beta_encoder.eval()
    haca3.eta_encoder.eval()
    haca3.contrast_encoder.eval()
    haca3.decoder.eval()
    haca3.patchifier.eval()
    haca3.attention_module.eval()

    # Load validation dataset
    print(f'{text_div} Loading Dataset {text_div}')
    valid_dataset = HACA3Dataset(
        args.dataset_dirs, args.contrasts, args.orientations,
        'valid', normalization_method='01'
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    evaluator = HACA3Evaluator(haca3, device)

    # Metrics storage
    # Per-site metrics
    site_ssim = defaultdict(list)
    site_psnr = defaultdict(list)
    # Per-slice metrics (for individual results)
    slice_metrics = []
    # Overall metrics
    all_ssim = []
    all_psnr = []

    print(f'{text_div} Running Evaluation {text_div}')

    with torch.set_grad_enabled(False):
        for batch_id, image_dicts in enumerate(tqdm(valid_loader, desc='Evaluating')):
            if args.max_batches is not None and batch_id >= args.max_batches:
                break

            # Run inference
            results = evaluator.evaluate_batch(image_dicts)

            rec_images = results['rec_image']
            target_images = results['target_image']
            masks = results['mask']

            batch_size = rec_images.shape[0]

            # Compute metrics for each sample in batch
            for i in range(batch_size):
                rec_np = rec_images[i, 0].cpu().numpy()
                target_np = target_images[i, 0].cpu().numpy()
                mask_np = masks[i, 0].cpu().numpy()

                # Get site_id (from first contrast, all should be same)
                site_id = image_dicts[0]['site_id'][i].item()
                site_name = site_names[site_id] if site_id < len(site_names) else f'site_{site_id}'

                # Compute metrics
                ssim_val, psnr_val = compute_ssim_psnr(rec_np, target_np, mask_np)

                # Store per-site
                site_ssim[site_name].append(ssim_val)
                site_psnr[site_name].append(psnr_val)

                # Store overall
                all_ssim.append(ssim_val)
                all_psnr.append(psnr_val)

                # Store slice-level
                slice_metrics.append({
                    'batch_id': batch_id,
                    'sample_id': i,
                    'site': site_name,
                    'ssim': ssim_val,
                    'psnr': psnr_val
                })
        # for batch_id, image_dicts in enumerate(tqdm(valid_loader, desc='Evaluating')):
        #     if args.max_batches is not None and batch_id >= args.max_batches:
        #         break

        #     # Run inference
        #     results = evaluator.evaluate_batch(image_dicts)

        #     batch_size = results['contrast_target'].shape[0]
        #     num_to_plot = min(4, batch_size)

        #     for i in range(num_to_plot):
        #         plot_theta_and_attention(
        #             contrasts_source=results['contrasts_source'],
        #             contrast_target=results['contrast_target'],
        #             attention=results['attention'],
        #             batch_idx=batch_id,
        #             out_dir=args.out_dir,
        #             sample_idx=i
        #         )

        #     rec_images = results['rec_image']
        #     target_images = results['target_image']
        #     masks = results['mask']

        #     batch_size = rec_images.shape[0]

        #     # Compute metrics for each sample in batch
        #     for i in range(batch_size):
        #         rec_np = rec_images[i, 0].cpu().numpy()
        #         target_np = target_images[i, 0].cpu().numpy()
        #         mask_np = masks[i, 0].cpu().numpy()

        #         # Get site_id (from first contrast, all should be same)
        #         site_id = image_dicts[0]['site_id'][i].item()
        #         site_name = site_names[site_id] if site_id < len(site_names) else f'site_{site_id}'

        #         # Compute metrics
        #         ssim_val, psnr_val = compute_ssim_psnr(rec_np, target_np, mask_np)

        #         # Store per-site
        #         site_ssim[site_name].append(ssim_val)
        #         site_psnr[site_name].append(psnr_val)

        #         # Store overall
        #         all_ssim.append(ssim_val)
        #         all_psnr.append(psnr_val)

        #         # Store slice-level
        #         slice_metrics.append({
        #             'batch_id': batch_id,
        #             'sample_id': i,
        #             'site': site_name,
        #             'ssim': ssim_val,
        #             'psnr': psnr_val
        #         })

            # Save PNG grids periodically
            if batch_id % args.save_every == 0:
                # Create grid: source images | reconstruction | target | betas | beta_fusion
                # Colorize betas
                colored_betas = [
                    colorize_betas(b)
                    for b in results['betas']
                ]

                colored_beta_fusion = colorize_betas(results['beta_fusion'])

                grid_images = (
                    results['source_images'] +
                    [results['rec_image']] +
                    [results['target_image']] +
                    colored_betas +
                    [colored_beta_fusion]
                )


                grid_path = os.path.join(
                    args.out_dir, 'grids',
                    f'batch_{str(batch_id).zfill(5)}.png'
                )
                save_grid_png(grid_images, grid_path, nrow=4)

                # Also save individual slice comparisons
                for i in range(min(4, batch_size)):
                    slice_grid = torch.stack([
                        results['source_images'][0][i:i+1, 0],  # T1
                        results['rec_image'][i:i+1, 0],         # Reconstruction
                        results['target_image'][i:i+1, 0],      # Target
                        (results['rec_image'][i:i+1, 0] - results['target_image'][i:i+1, 0]).abs()  # Difference
                    ], dim=0)

                    slice_path = os.path.join(
                        args.out_dir, 'slices',
                        f'batch_{str(batch_id).zfill(5)}_slice_{i}.png'
                    )
                    save_grid_png(slice_grid, slice_path, nrow=4)

    # Compute summary statistics
    print(f'\n{text_div} RESULTS {text_div}')

    # Per-site results
    print('\n=== Per-Site Metrics ===')
    site_summary = []
    for site_name in sorted(site_ssim.keys()):
        ssim_arr = np.array(site_ssim[site_name])
        psnr_arr = np.array(site_psnr[site_name])

        site_result = {
            'site': site_name,
            'n_samples': len(ssim_arr),
            'ssim_mean': np.mean(ssim_arr),
            'ssim_std': np.std(ssim_arr),
            'psnr_mean': np.mean(psnr_arr),
            'psnr_std': np.std(psnr_arr)
        }
        site_summary.append(site_result)

        print(f'{site_name}: n={site_result["n_samples"]}, '
              f'SSIM={site_result["ssim_mean"]:.4f}+/-{site_result["ssim_std"]:.4f}, '
              f'PSNR={site_result["psnr_mean"]:.2f}+/-{site_result["psnr_std"]:.2f}')

    # Overall results
    all_ssim_arr = np.array(all_ssim)
    all_psnr_arr = np.array(all_psnr)

    print('\n=== Overall Metrics ===')
    print(f'Total samples: {len(all_ssim_arr)}')
    print(f'SSIM: {np.mean(all_ssim_arr):.4f} +/- {np.std(all_ssim_arr):.4f}')
    print(f'PSNR: {np.mean(all_psnr_arr):.2f} +/- {np.std(all_psnr_arr):.2f}')

    # Save results to CSV
    # Per-site summary
    site_csv_path = os.path.join(args.out_dir, 'metrics_per_site.csv')
    with open(site_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['site', 'n_samples', 'ssim_mean', 'ssim_std', 'psnr_mean', 'psnr_std'])
        writer.writeheader()
        writer.writerows(site_summary)
        # Add overall row
        writer.writerow({
            'site': 'OVERALL',
            'n_samples': len(all_ssim_arr),
            'ssim_mean': np.mean(all_ssim_arr),
            'ssim_std': np.std(all_ssim_arr),
            'psnr_mean': np.mean(all_psnr_arr),
            'psnr_std': np.std(all_psnr_arr)
        })
    print(f'\nSaved per-site metrics to: {site_csv_path}')

    # Per-slice metrics
    slice_csv_path = os.path.join(args.out_dir, 'metrics_per_slice.csv')
    with open(slice_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['batch_id', 'sample_id', 'site', 'ssim', 'psnr'])
        writer.writeheader()
        writer.writerows(slice_metrics)
    print(f'Saved per-slice metrics to: {slice_csv_path}')

    # Overall summary
    summary_csv_path = os.path.join(args.out_dir, 'metrics_summary.csv')
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'std', 'min', 'max', 'n_samples'])
        writer.writerow(['ssim', np.mean(all_ssim_arr), np.std(all_ssim_arr),
                        np.min(all_ssim_arr), np.max(all_ssim_arr), len(all_ssim_arr)])
        writer.writerow(['psnr', np.mean(all_psnr_arr), np.std(all_psnr_arr),
                        np.min(all_psnr_arr), np.max(all_psnr_arr), len(all_psnr_arr)])
    print(f'Saved summary metrics to: {summary_csv_path}')

    print(f'\n{text_div} EVALUATION COMPLETE {text_div}')


if __name__ == '__main__':
    main()
