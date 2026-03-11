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
import matplotlib.pyplot as plt

from modules.model import HACA3
from modules.dataset import HACA3Dataset, contrast_names
from modules.utils import mkdir_p

def create_theta_attention_grid(
    source_images,
    rec_image,
    target_image,
    contrasts_source,
    contrast_target,
    contrast_rec,
    attention,
    betas,
    beta_fusion,
    batch_idx,
    out_dir,
    num_samples=4
):
    """
    Create a combined visualization grid with:
    - Row 1: Source images, reconstructed image, target image
    - Row 2: Colored betas and beta_fusion
    - Row 3: Theta space plot and attention histogram for each sample

    Uses colorblind-friendly (Okabe-Ito) colors:
    - Blue (#0077BB) for source thetas
    - Orange (#EE7733) for target theta
    - Teal (#009988) for reconstructed theta

    Args:
        source_images: list of tensors [B, 1, H, W]
        rec_image: tensor [B, 1, H, W]
        target_image: tensor [B, 1, H, W]
        contrasts_source: list of tensors [B, 2, 1, 1] - theta for each source
        contrast_target: tensor [B, 2, 1, 1] - theta for target
        contrast_rec: tensor [B, 2, 1, 1] - theta for reconstructed image
        attention: tensor [B, num_sources] - attention weights per modality
        betas: list of tensors [B, beta_dim, H, W]
        beta_fusion: tensor [B, beta_dim, H, W]
        batch_idx: batch index for filename
        out_dir: output directory
        num_samples: number of samples to visualize from batch
    """
    os.makedirs(out_dir, exist_ok=True)

    batch_size = rec_image.shape[0]
    num_samples = min(num_samples, batch_size)
    num_sources = len(source_images)

    # print("source_images[0]: ", source_images[0].min(), source_images[0].max())
    # print("source_images[1]: ", source_images[1].min(), source_images[1].max())
    # print("source_images[2]: ", source_images[2].min(), source_images[2].max())
    # print("source_images[3]: ", source_images[3].min(), source_images[3].max())
    # print("rec_image: ", rec_image.min(), rec_image.max())
    # print("target_image: ", target_image.min(), target_image.max())
    # print("betas[0]: ", betas[0].min(), betas[0].max(), betas[0].unique())
    # print("betas[1]: ", betas[1].min(), betas[1].max(), betas[1].unique())
    # print("betas[2]: ", betas[2].min(), betas[2].max(), betas[2].unique())
    # print("betas[3]: ", betas[3].min(), betas[3].max(), betas[3].unique())
    # print("betas[4]: ", betas[4].min(), betas[4].max(), betas[4].unique())

    # DEBUGGY print("num_sources", num_sources)

    # Colorblind-friendly palette (Okabe-Ito)
    SOURCE_COLORS = [
        '#0077BB',  # blue
        '#33BBEE',  # cyan
        '#EE3377',  # magenta
        '#CC3311',  # red
        '#009988',  # teal (backup)
    ]
    TARGET_COLOR = '#EE7733'   # orange
    REC_COLOR = '#009988'      # teal

    for sample_idx in range(num_samples):
        # Create figure with custom layout:
        # Top: images (sources + rec + target)
        # Middle: betas + beta_fusion
        # Bottom: theta space plot | attention histogram
        num_img_cols = num_sources + 2  # sources + rec + target
        num_beta_cols = num_sources + 1  # betas + beta_fusion

        fig = plt.figure(figsize=(3 * max(num_img_cols, 3), 10))

        # Use GridSpec for flexible layout
        gs = fig.add_gridspec(3, max(num_img_cols, num_beta_cols, 2),
                              height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.2)

        # --- Row 1: Source images, Reconstructed, Target ---
        for i, src_img in enumerate(source_images):
            ax = fig.add_subplot(gs[0, i])
            img = src_img[sample_idx, 0].cpu().numpy()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Source {i+1}', fontsize=10)
            ax.axis('off')

        ax = fig.add_subplot(gs[0, num_sources])
        img = rec_image[sample_idx, 0].cpu().numpy()
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title('Reconstructed', fontsize=10)
        ax.axis('off')

        ax = fig.add_subplot(gs[0, num_sources + 1])
        img = target_image[sample_idx, 0].cpu().numpy()
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title('Target', fontsize=10)
        ax.axis('off')

        # --- Row 2: Colored betas and beta_fusion ---
        # for i, beta in enumerate(betas):
        #     ax = fig.add_subplot(gs[1, i])
        #     colored = colorize_betas(beta[sample_idx:sample_idx+1])
        #     img = colored[0].permute(1, 2, 0).cpu().numpy()
        #     ax.imshow(img)
        #     ax.set_title(f'Beta {i+1}', fontsize=10)
        #     ax.axis('off')

        # --- Row 2: Binary visualization of betas[0] ---
        beta0 = betas[0][sample_idx, 0]  # [H, W]
        unique_vals = [0.0, 0.2, 0.4, 0.6, 0.8]

        for i, val in enumerate(unique_vals):
            ax = fig.add_subplot(gs[1, i])

            # Create binary mask
            mask = (torch.isclose(beta0, torch.tensor(val, device=beta0.device))).float()

            img = mask.cpu().numpy()

            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Beta {int(val*5)}', fontsize=10)
            ax.axis('off')


        # ax = fig.add_subplot(gs[1, num_sources])
        # colored = colorize_betas(beta_fusion[sample_idx:sample_idx+1])
        # img = colored[0].permute(1, 2, 0).cpu().numpy()
        # ax.imshow(img)
        # ax.set_title('Beta Fusion', fontsize=10)
        # ax.axis('off')

        # --- Row 3: Theta space plot (left) | Attention histogram (right) ---

        # Theta space plot
        ax_theta = fig.add_subplot(gs[2, :max(num_img_cols, num_beta_cols, 2)//2])

        # Extract theta values
        src_thetas = [
            c[sample_idx, :, 0, 0].detach().cpu().numpy()
            for c in contrasts_source
        ]
        # DEBUGGY print("src_thetas", src_thetas)
        tgt_theta = contrast_target[sample_idx, :, 0, 0].detach().cpu().numpy()
        # DEBUGGY print("tgt_theta", tgt_theta)
        rec_theta = contrast_rec[sample_idx, :, 0, 0].detach().cpu().numpy()
        # DEBUGGY print("rec_theta", rec_theta)

        # Plot source thetas
        for i, th in enumerate(src_thetas):
            color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
            ax_theta.scatter(th[0], th[1], s=100, color=color,
                           label=f'Source {i+1}', edgecolors='black', linewidths=0.5)

        # Plot target theta (star marker)
        ax_theta.scatter(tgt_theta[0], tgt_theta[1], s=200, marker='*',
                        color=TARGET_COLOR, label='Target',
                        edgecolors='black', linewidths=0.5)

        # Plot reconstructed theta (diamond marker)
        ax_theta.scatter(rec_theta[0], rec_theta[1], s=150, marker='D',
                        color=REC_COLOR, label='Reconstructed',
                        edgecolors='black', linewidths=0.5)

        ax_theta.set_xlabel(r'$\theta_1$ (Contrast dim 1)', fontsize=10)
        ax_theta.set_ylabel(r'$\theta_2$ (Contrast dim 2)', fontsize=10)
        ax_theta.set_title('Theta (Contrast) Space', fontsize=11)
        ax_theta.legend(loc='best', fontsize=8)
        ax_theta.grid(True, alpha=0.3)

        # Attention histogram (right side)
        ax_att = fig.add_subplot(gs[2, max(num_img_cols, num_beta_cols, 2)//2:])

        att = attention[sample_idx]

        # if att.dim() > 1:
        #     att = att.mean(dim=tuple(range(1, att.dim())))  # Average over spatial dims if present

        att = att.detach().cpu().numpy().squeeze().squeeze()

        bars = ax_att.bar(range(len(att)), att,
                         color=[SOURCE_COLORS[i % len(SOURCE_COLORS)] for i in range(len(att))],
                         edgecolor='black', linewidth=0.5)

        ax_att.set_ylim(0, 1)
        ax_att.set_xlabel('Source Modality', fontsize=10)
        ax_att.set_ylabel('Attention Weight (Normalized)', fontsize=10)
        ax_att.set_title(f'Attention Scores (sum={att.sum():.3f})', fontsize=11)
        ax_att.set_xticks(range(len(att)))
        ax_att.set_xticklabels([f'Src {i+1}' for i in range(len(att))], fontsize=9)
        ax_att.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, att):
            ax_att.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f'grid_b{batch_idx:05d}_s{sample_idx}.png'),
            dpi=150, bbox_inches='tight'
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
        # for i in range(len(image_dicts)):
        #     img = image_dicts[i]['image'].to(self.device)
        #     img = img.squeeze(0)  # remove batch dim
        #     source_images.append(img)

        for i in range(len(image_dicts)):
            # print("min-max at image_dicts: ", image_dicts[i]['image'].min(), image_dicts[i]['image'].max())
            source_images.append(image_dicts[i]['image'].to(self.device))

        # print("HOLUP: ", source_images.shape)

        # source_images = torch.stack(source_images)  # [4, 1, H, W]

        # DEBUGGY

        # for i, img in enumerate(source_images):

        #     # Remove batch dimension
        #     # img = img.squeeze(0)  # -> [C, H, W]

        #     # If single channel, keep it single channel
        #     if img.dim() == 2:
        #         img = img.unsqueeze(0)

        #     # Normalize to 0-1 for PNG safety
        #     img_min = img.min()
        #     img_max = img.max()
        #     if img_max > img_min:
        #         img = (img - img_min) / (img_max - img_min)
        #     else:
        #         img = torch.zeros_like(img)

        #     save_path = os.path.join("/iacl/pg23/prahlad/4imgs", f"source_images_contrast{i}.png")
        #     vutils.save_image(img, save_path)

        #     print(f"Saved {save_path}")

        # print(source_images[0].shape)


        mask = image_dicts[0]['mask'].to(self.device)
        masks = torch.stack([d['mask'] for d in image_dicts], dim=-1).to(self.device)

        # Get target image
        target_image, contrast_id_for_decoding = self.haca3.select_available_contrasts(image_dicts)
        available_contrast_id = torch.stack([d['exists'] for d in image_dicts], dim=-1).to(self.device)

        # Encode
        logits, betas = self.haca3.calculate_beta(source_images)
        # DEBUGGY print("source")
        thetas_source, _, _ = self.haca3.calculate_theta(source_images)
        # DEBUGGY print("contrasts_source", contrasts_source)
        etas_source = self.haca3.calculate_eta(source_images)

        # DEBUGGY print("target")
        theta_target, _, _ = self.haca3.calculate_theta(target_image)
        # print("contrast_target", contrast_target)
        # print(t)
        eta_target = self.haca3.calculate_eta(target_image)

        query = torch.cat([theta_target, eta_target], dim=1)
        # DEBUGGY print((contrasts_source[0].shape), (etas_source[0].shape))
        keys = [torch.cat([contrast, eta], dim=1) for (contrast, eta) in zip(thetas_source, etas_source)]

        # DEBUGGY print("rec_image")
        # Decode
        rec_image, attention, logit_fusion, beta_fusion, contrastwise_attention_scores = self.haca3.decode(
            logits, theta_target, query, keys,
            available_contrast_id, masks,
            contrast_dropout=False, contrast_id_to_drop=None
        )

        # print(attention[0].unique())
        # print(attention[0][0].unique())
        # print(attention[0][1].unique())
        # print(attention[0][2].unique())
        # print(attention[0][3].unique())
        # print(attention[0].shape)

        # from torchvision.utils import save_image

        # from torchvision.transforms.functional import to_pil_image

        # save_dir = "/iacl/pg23/prahlad"
        # os.makedirs(save_dir, exist_ok=True)

        # tensors = [attention[0][0], attention[0][1], attention[0][2], attention[0][3]]  # your 4 tensors

        # for i, t in enumerate(tensors):

        #     x = t.cpu().numpy()

        #     p1, p99 = np.percentile(x, [1, 99])
        #     x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0, 1)

        #     img = to_pil_image(torch.tensor(x))
        #     img.save(f"{save_dir}/attn_{i}.png")

        # txt_path = os.path.join(save_dir, "attention_stats.txt")

        # torch.set_printoptions(threshold=float('inf'))
        # with open(txt_path, "w") as f:
        #     f.write(f"Contrast 1: {attention[0][0]}\n")
        #     f.write(f"Contrast 2: {attention[0][1]}\n")
        #     f.write(f"Contrast 3: {attention[0][2]}\n")
        #     f.write(f"Contrast 4: {attention[0][3]}\n")
        

        # attention: tensor [4,224,224]
        # att = attention[0].detach().cpu()

        # which modality dominates each pixel
        # dominant = torch.argmax(att, dim=0).numpy()  # [224,224]

        # modalities = ["T1", "T2", "PD", "FLAIR"]

        # # nice distinct colors
        # colors = np.array([
        #     [0.8, 0.1, 0.1],   # red
        #     [0.1, 0.6, 0.9],   # blue
        #     [0.1, 0.8, 0.3],   # green
        #     [0.9, 0.6, 0.1]    # orange
        # ])

        # rgb = colors[dominant]

        # plt.figure(figsize=(6,6))
        # plt.imshow(rgb)
        # plt.axis("off")

        # # legend
        # handles = [
        #     plt.Line2D([0],[0], marker='s', color='w', markerfacecolor=colors[i], markersize=12)
        #     for i in range(len(modalities))
        # ]

        # plt.legend(handles, modalities, loc="lower right")

        # plt.tight_layout()
        # plt.savefig(f"{save_dir}/attention_dominance.png", dpi=200)
        # plt.close()

        # print(h)

        # Compute theta for reconstructed image
        theta_rec, _, _ = self.haca3.calculate_theta(rec_image) # .repeat(1, 3, 1, 1)

        return {
            'source_images': source_images,
            'rec_image': rec_image,
            'target_image': target_image,
            'betas': betas,
            'beta_fusion': beta_fusion,
            'mask': mask,
            'site_ids': [d['site_id'] for d in image_dicts],
            'attention': contrastwise_attention_scores,
            'contrasts_source': thetas_source,
            'contrast_target': theta_target,
            'contrast_rec': theta_rec,
        }
    
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
    parser.add_argument('--theta-dim', type=int, default=2)
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
        theta_dim=args.theta_dim,
        eta_dim=args.eta_dim,
        pretrained_haca3=args.pretrained_haca3,
        pretrained_eta_encoder=args.pretrained_eta_encoder,
        gpu_id=args.gpu_id
    )
    device = haca3.device

    # Set to eval mode
    haca3.beta_encoder.eval()
    haca3.eta_encoder.eval()
    haca3.theta_encoder.eval()
    haca3.decoder.eval()
    haca3.patchifier.eval()
    haca3.attention_module.eval()

    # Load validation dataset
    print(f'{text_div} Loading Dataset {text_div}')
    valid_dataset = HACA3Dataset(
        args.dataset_dirs, args.contrasts, args.orientations,
        'valid', normalization_method='01'
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

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

            # Save PNG grids periodically
            if batch_id % args.save_every == 0:
                # Create grid: source images | reconstruction | target | betas | beta_fusion
                # Colorize betas
                colored_betas = [
                    colorize_betas(b) if b.dim() == 4 else b
                    for b in results['betas']
                ]

                colored_beta_fusion = (
                    colorize_betas(results['beta_fusion'])
                    if results['beta_fusion'].dim() == 4
                    else results['beta_fusion']
                )

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

                theta_attention_dir = os.path.join(args.out_dir, 'theta_attention')
                create_theta_attention_grid(
                    source_images=results['source_images'],
                    rec_image=results['rec_image'],
                    target_image=results['target_image'],
                    contrasts_source=results['contrasts_source'],
                    contrast_target=results['contrast_target'],
                    contrast_rec=results['contrast_rec'],
                    attention=results['attention'],
                    betas=results['betas'],
                    beta_fusion=results['beta_fusion'],
                    batch_idx=batch_id,
                    out_dir=theta_attention_dir,
                    num_samples=min(4, batch_size)
                )

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
