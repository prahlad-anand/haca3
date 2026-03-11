from tqdm import tqdm
import numpy as np
import random
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.transforms import ToTensor
from datetime import datetime
import nibabel as nib
from torch.cuda.amp import autocast

from .utils import *
from .dataset import HACA3Dataset
from .network import UNet, ContrastiveImageEncoder, EtaEncoder, Patchifier, AttentionModule, FusionNet

import csv


class HACA3:
    def __init__(self, beta_dim, contrast_dim, eta_dim, pretrained_haca3=None, pretrained_eta_encoder=None, contrast_encoder=None, gpu_id=0):
        self.beta_dim = beta_dim
        self.contrast_dim = contrast_dim
        self.eta_dim = eta_dim
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.timestr = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.train_loader, self.valid_loader = None, None
        self.out_dir = None
        self.optimizer = None
        self.scheduler = None
        self.writer, self.writer_path = None, None
        self.checkpoint = None

        self.l1_loss, self.contrastive_loss, self.perceptual_loss = None, None, None

        # define networks
        self.beta_encoder = UNet(in_ch=1, out_ch=self.beta_dim, base_ch=8, final_act='none')
        self.contrast_encoder = contrast_encoder
        self.eta_encoder = EtaEncoder(in_ch=1, out_ch=self.eta_dim)
        self.attention_module = AttentionModule(self.contrast_dim + self.eta_dim, v_ch=self.beta_dim)
        self.decoder = UNet(in_ch=1 + self.contrast_dim, out_ch=1, base_ch=16, final_act='relu')
        self.patchifier = Patchifier(in_ch=1, out_ch=128)

        if pretrained_eta_encoder is not None:
            checkpoint_eta_encoder = torch.load(pretrained_eta_encoder, map_location=self.device)
            self.eta_encoder.load_state_dict(checkpoint_eta_encoder['eta_encoder'])
        if pretrained_haca3 is not None:
            self.checkpoint = torch.load(pretrained_haca3, map_location=self.device)
            self.beta_encoder.load_state_dict(self.checkpoint['beta_encoder'])
            self.eta_encoder.load_state_dict(self.checkpoint['eta_encoder'])
            self.decoder.load_state_dict(self.checkpoint['decoder'])
            self.attention_module.load_state_dict(self.checkpoint['attention_module'])
            self.patchifier.load_state_dict(self.checkpoint['patchifier'])
        self.beta_encoder.to(self.device)
        self.eta_encoder.to(self.device)
        self.decoder.to(self.device)
        self.attention_module.to(self.device)
        self.patchifier.to(self.device)
        self.start_epoch = 0

    def initialize_training(self, out_dir, lr):
        # define loss functions
        self.l1_loss = nn.L1Loss(reduction='none')
        # self.kld_loss = KLDivergenceLoss()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(self.device)
        self.perceptual_loss = PerceptualLoss(vgg)
        self.contrastive_loss = PatchNCELoss()

        # define optimizer and learning rate scheduler
        self.optimizer = Adam(list(self.beta_encoder.parameters()) +
                              list(self.decoder.parameters()) +
                              list(self.attention_module.parameters()) +
                              list(self.patchifier.parameters()), lr=lr)
        self.scheduler = CyclicLR(self.optimizer, base_lr=4e-4, max_lr=7e-4, cycle_momentum=False)
        if self.checkpoint is not None:
            self.start_epoch = self.checkpoint['epoch']
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.scheduler.load_state_dict(self.checkpoint['scheduler'])
            if 'timestr' in self.checkpoint:
                self.timestr = self.checkpoint['timestr']
        self.start_epoch = self.start_epoch + 1

        self.out_dir = out_dir
        mkdir_p(self.out_dir)
        mkdir_p(os.path.join(self.out_dir, f'training_results_{self.timestr}'))
        mkdir_p(os.path.join(self.out_dir, f'training_models_{self.timestr}'))

        self.writer_path = os.path.join(self.out_dir, self.timestr)
        self.writer = SummaryWriter(self.writer_path)

    def load_dataset(self, dataset_dirs, contrasts, orientations, batch_size, normalization_method='01'):
        train_dataset = HACA3Dataset(dataset_dirs, contrasts, orientations, 'train', normalization_method)
        valid_dataset = HACA3Dataset(dataset_dirs, contrasts, orientations, 'valid', normalization_method)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    def calculate_beta(self, images):
        logits, betas = [], []
        for image in images:
            logit = self.beta_encoder(image)
            beta = self.channel_aggregation(reparameterize_logit(logit))
            logits.append(logit)
            betas.append(beta)
        return logits, betas
    
    def calculate_contrast(self, images):
        if isinstance(images, list):
            contrasts = []
            # print("pass-through: ")
            # DEBUGGY print("\n IMAGES IN CALCULATE_CONTRAST: ")
            # DEBUGGY import torchvision.utils as vutils
            for i, img in enumerate(images):

                # DEBUGGY img_min = img.min()
                # DEBUGGY img_max = img.max()
                # DEBUGGY if img_max > img_min:
                    # DEBUGGY img = (img - img_min) / (img_max - img_min)
                # DEBUGGY else:
                    # DEBUGGY img = torch.zeros_like(img)

                # DEBUGGY save_path = os.path.join("/iacl/pg23/prahlad/4imgs", f"cc_{i}.png")
                # DEBUGGY vutils.save_image(img, save_path)

                # DEBUGGY print(f"Saved {save_path}")

                # DEBUGGY print("shape:", img.shape)
                # DEBUGGY print(img.min(), img.max(), "\n")


                # image_rgb = img.repeat(1, 3, 1, 1)
                # print("holup", img.min(), img.max())
                contrast = self.contrast_encoder(img)
                # print("this img min-max: ", image.min(), image.max())
                # print("CONTRAST", contrast)
                contrasts.append(contrast)
                # print(self.contrast_encoder.training)
                # print(raucushallelujahchorus)
            # DEBUGGY print("\nCONTRASTS IN CALCULATE_CONTRAST: ", contrasts, "\n")

        else:
            contrasts = self.contrast_encoder(images) # .repeat(1, 3, 1, 1)
            # DEBUGGY print(h)

        # DEBUGGY print("Done: ", contrasts)
        return contrasts

    def calculate_eta(self, images):
        if isinstance(images, list):
            etas = []
            for image in images:
                eta = self.eta_encoder(image)
                etas.append(eta)
        else:
            etas = self.eta_encoder(images)
        return etas

    def prepare_source_images(self, image_dicts):
        num_contrasts = len(image_dicts)
        num_contrasts_with_degradation = np.random.permutation(num_contrasts)[0]
        degradation_ids = sorted(np.random.choice(range(num_contrasts),
                                                  num_contrasts_with_degradation,
                                                  replace=False))
        source_images = []
        for i in range(num_contrasts):
            if i in degradation_ids:
                source_images.append(image_dicts[i]['image_degrade'].to(self.device))
            else:
                source_images.append(image_dicts[i]['image'].to(self.device))
        return source_images

    def channel_aggregation(self, beta_onehot_encode):
        """
        Combine multi-channel one-hot encoded beta into one channel (label-encoding).

        ===INPUTS===
        * beta_onehot_encode: torch.Tensor (batch_size, self.beta_dim, image_dim, image_dim)
            One-hot encoded beta variable. At each pixel location, only one channel will take value of 1,
            and other channels will be 0.
        ===OUTPUTS===
        * beta_label_encode: torch.Tensor (batch_size, 1, image_dim, image_dim)
            The intensity value of each pixel will be determined by the channel index with value of 1.
        """
        batch_size = beta_onehot_encode.shape[0]
        image_dim = beta_onehot_encode.shape[3]
        value_tensor = (torch.arange(0, self.beta_dim) * 1.0).to(self.device)
        value_tensor = value_tensor.view(1, self.beta_dim, 1, 1).repeat(batch_size, 1, image_dim, image_dim)
        beta_label_encode = beta_onehot_encode * value_tensor.detach()
        return beta_label_encode.sum(1, keepdim=True) / self.beta_dim

    def select_available_contrasts(self, image_dicts):
        """
        Select available contrasts as target.

        ===INPUTS===
        * image_dicts: list (num_contrasts, )
            List of dictionaries. Each element is a dictionary received from dataloader. See dataset.py for details.

        ===OUTPUTS===
        * target_image: torch.Tensor (batch_size, 1, image_dim=224, image_dim=224)
            Images as target for I2I.
        *  selected_contrast_id: torch.Tensor (batch_size, num_contrasts)
            Indicates which contrast has been selected as target image.
        """
        target_image_combined = torch.cat([d['image'] for d in image_dicts], dim=1)
        # (batch_size, num_contrasts)
        available_contrasts = torch.stack([d['exists'] for d in image_dicts], dim=-1)
        subject_ids = available_contrasts.nonzero(as_tuple=True)[0]
        contrast_ids = available_contrasts.nonzero(as_tuple=True)[1]
        unique_subject_ids = list(torch.unique(subject_ids))
        selected_contrast_ids = []
        for i in unique_subject_ids:
            selected_contrast_ids.append(random.choice(contrast_ids[subject_ids == i]))
        target_image = target_image_combined[unique_subject_ids, selected_contrast_ids, ...].unsqueeze(1).to(
            self.device)
        selected_contrast_id = torch.zeros_like(available_contrasts).to(self.device)
        selected_contrast_id[unique_subject_ids, selected_contrast_ids, ...] = 1.0
        return target_image, selected_contrast_id

    def decode(self, logits, target_contrast, query, keys, available_contrast_id, mask, contrast_dropout=False,
               contrast_id_to_drop=None):
        """
        HACA3 decoding.

        ===INPUTS===
        * logits: list (num_contrasts, )
            Encoded logit of each source image.
            Each element has shape (batch_size, self.beta_dim, image_dim, image_dim).
        * target_contrast: torch.Tensor (batch_size, self.contrast_dim, 1, 1)
            contrast values of target images used for decoding.
        * query: torch.Tensor (batch_size, self.contrast_dim+self.eta_dim, 1, 1)
            query variable. Concatenation of "target_contrast" and "target_eta".
        * keys: list (num_contrasts, )
            keys variable. Each element has shape (batch_size, self.contrast_dim+self.eta_dim)
        * available_contrast_id: torch.Tensor (batch_size, num_contrasts)
            Indicates which contrasts are available. 1: if available, 0: if unavailable.
        * contrast_dropout: bool
            Indicates if available contrasts will be randomly dropped out.

        ===OUTPUTS===
        * rec_image: torch.Tensor (batch_size, 1, image_dim, image_dim)
            Synthetic image after decoding.
        * attention: torch.Tensor (batch_size, num_contrasts)
            Learned attention of each source image contrast.
        * logit_fusion: torch.Tensor (batch_size, self.beta_dim, image_dim, image_dim)
            Optimal logit after fusion.
        * beta_fusion: torch.Tensor (batch_size, self.beta_dim, image_dim, image_dim)
            Optimal beta after fusion. beta_fusion = reparameterize_logit(logit_fusion).
        * attention_map: torch.Tensor (batch_size, num_contrasts)
            Learned attention map of each source image contrast.
        """
        num_contrasts = len(logits)
        batch_size = logits[0].shape[0]
        image_dim = logits[0].shape[-1]

        # logits_combined: (batch_size, self.beta_dim, num_contrasts, image_dim * image_dim)
        logits_combined = torch.stack(logits, dim=-1).permute(0, 1, 4, 2, 3)
        logits_combined = logits_combined.view(batch_size, self.beta_dim, num_contrasts, image_dim * image_dim)

        # value: (batch_size, self.beta_dim, image_dim*image_dim, num_contrasts)
        v = logits_combined.permute(0, 1, 3, 2)
        # key: (batch_size, self.contrast_dim+self.eta_dim, 1, num_contrasts)
        k = torch.cat(keys, dim=-1)
        # query: (batch_size, self.contrast_dim+self.eta_dim, 1)
        q = query.view(batch_size, self.contrast_dim + self.eta_dim, 1)

        if contrast_dropout:
            available_contrast_id = dropout_contrasts(available_contrast_id, contrast_id_to_drop)
        logit_fusion, attention, contrastwise_attention_scores = self.attention_module(q, k, v, mask, modality_dropout=1 - available_contrast_id,
                                                        temperature=10.0)
        beta_fusion = self.channel_aggregation(reparameterize_logit(logit_fusion))
        combined_map = torch.cat([beta_fusion, target_contrast.repeat(1, 1, image_dim, image_dim)], dim=1)
        rec_image = self.decoder(combined_map)# * mask
        return rec_image, attention, logit_fusion, beta_fusion, contrastwise_attention_scores

    def calculate_features_for_contrastive_loss(self, betas, source_images, available_contrast_id):
        """
        Prepare query, positive, and negative examples for calculating contrastive loss.

        ===INPUTS===
        * betas: list (num_contrasts, )
            Each element: torch.Tensor, (batch_size, self.beta_dim, 224, 224)
        * source_images: list(num_contrasts, )
            Each element: torch.Tensor, (batch_size, 1, 224, 224)
        * available_contrast_id: torch.Tensor (batch_size, num_contrasts)
            Indicates which contrasts are available. 1: if available, 0: if unavailable.

        ===OUTPUTS===
        * query_features: torch.Tensor (batch_size, 128, num_query_patches=49)
            Also called anchor features. Number of feature dimension (128) and
            number of patches (49) are determined by self.patchifier.
        * positive_features: torch.Tensor (batch_size, 128, num_positive_patches=49)
            Positive features are encouraged to be as close to query features as possible.
            Number of positive patches should be equal to the number of query patches.
        * negative_features: torch.Tensor (batch_size, 128, num_negative_patches)
            Negative features served as negative examples. They are pushed away from query features during training.
            Number of negative patches does not necessarily equal to "num_query_patches" or "num_positive_patches".
        """
        batch_size = betas[0].shape[0]
        betas_stack = torch.stack(betas, dim=-1)
        source_images_stack = torch.stack(source_images, dim=-1)
        query_contrast_ids, positive_contrast_ids = [], []
        for subject_id in range(batch_size):
            contrast_id_tmp = random.sample(list(set(available_contrast_id[subject_id].nonzero(as_tuple=True)[0])), 2)
            query_contrast_ids.append(contrast_id_tmp[0])
            positive_contrast_ids.append(contrast_id_tmp[1])
        query_example = torch.cat([betas_stack[[subject_id], :, :, :, query_contrast_ids[subject_id]]
                                   for subject_id in range(batch_size)], dim=0)
        query_feature = self.patchifier(query_example).view(batch_size, 128, -1)
        positive_example = torch.cat([betas_stack[[subject_id], :, :, :, positive_contrast_ids[subject_id]]
                                      for subject_id in range(batch_size)], dim=0)
        positive_feature = self.patchifier(positive_example).view(batch_size, 128, -1)
        num_positive_patches = positive_feature.shape[-1]
        negative_feature = torch.cat([
            self.patchifier(torch.cat([source_images_stack[[subject_id], :, :, :, query_contrast_ids[subject_id]]
                                       for subject_id in range(batch_size)], dim=0)).view(batch_size, 128, -1),
            self.patchifier(torch.cat([source_images_stack[[subject_id], :, :, :, positive_contrast_ids[subject_id]]
                                       for subject_id in range(batch_size)], dim=0)).view(batch_size, 128, -1),
            self.patchifier(torch.cat([betas_stack[[subject_id], :, :, :, query_contrast_ids[subject_id]]
                                       for subject_id in range(batch_size)], dim=0)).view(batch_size, 128, -1)[:, :,
            torch.randperm(num_positive_patches)],
            self.patchifier(torch.cat([betas_stack[[subject_id], :, :, :, query_contrast_ids[subject_id]]
                                       for subject_id in range(batch_size)], dim=0)).view(batch_size, 128, -1)[
            torch.randperm(batch_size), :, :],
            self.patchifier(torch.cat([betas_stack[[subject_id], :, :, :, positive_contrast_ids[subject_id]]
                                       for subject_id in range(batch_size)], dim=0)).view(batch_size, 128, -1)[:, :,
            torch.randperm(num_positive_patches)],
            self.patchifier(torch.cat([betas_stack[[subject_id], :, :, :, positive_contrast_ids[subject_id]]
                                       for subject_id in range(batch_size)], dim=0)).view(batch_size, 128, -1)[
            torch.randperm(batch_size), :, :]
        ], dim=-1)
        return query_feature, positive_feature, negative_feature

    def calculate_loss(self, rec_image, ref_image, mask, betas, source_images, available_contrast_id,
                       is_train=True):
        """
        Calculate losses for HACA3 training and validation.

        """
        # 1. reconstruction loss
        rec_loss = self.l1_loss(rec_image[mask], ref_image[mask]).mean()
        perceptual_loss = self.perceptual_loss(rec_image, ref_image).mean()

        # 2. KLD loss
        # kld_loss = self.kld_loss(mu, logvar).mean()

        # 3. beta contrastive loss
        query_feature, \
            positive_feature, \
            negative_feature = self.calculate_features_for_contrastive_loss(betas, source_images, available_contrast_id)
        beta_loss = self.contrastive_loss(query_feature, positive_feature.detach(), negative_feature.detach())

        # COMBINE LOSSES
        total_loss = 10 * rec_loss + 5e-1 * perceptual_loss + 5e-1 * beta_loss # + 1e-5 * kld_loss
        if is_train:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        loss = {'rec_loss': rec_loss.item(),
                'per_loss': perceptual_loss.item(),
                # 'kld_loss': kld_loss.item(),
                'beta_loss': beta_loss.item(),
                'total_loss': total_loss.item()}
        return loss

    def calculate_cycle_consistency_loss(self, eta_rec, eta_ref, beta_rec, beta_ref,
                                         is_train=True):
        # contrast_loss = self.l1_loss(contrast_rec, contrast_ref).mean()
        eta_loss = self.l1_loss(eta_rec, eta_ref).mean()
        beta_loss = self.l1_loss(beta_rec, beta_ref).mean()

        cycle_loss = eta_loss + 5e-2 * beta_loss
        if is_train:
            self.optimizer.zero_grad()
            (5e-2 * cycle_loss).backward()
            self.optimizer.step()
            self.scheduler.step()
        # 'contrast_cyc': contrast_loss.item(),
        loss = {'eta_cyc': eta_loss.item(),
                'beta_cyc': beta_loss.item()}
        return loss

    def write_tensorboard(self, loss, epoch, batch_id, train_or_valid='train', cycle_loss=None):
        if train_or_valid == 'train':
            curr_iteration = (epoch - 1) * len(self.train_loader) + batch_id
            self.writer.add_scalar(f'{train_or_valid}/learning rate', self.scheduler.get_last_lr()[0], curr_iteration)
        else:
            curr_iteration = (epoch - 1) * len(self.valid_loader) + batch_id
        self.writer.add_scalar(f'{train_or_valid}/reconstruction loss', loss['rec_loss'], curr_iteration)
        self.writer.add_scalar(f'{train_or_valid}/perceptual loss', loss['per_loss'], curr_iteration)
        # self.writer.add_scalar(f'{train_or_valid}/kld loss', loss['kld_loss'], curr_iteration)
        self.writer.add_scalar(f'{train_or_valid}/beta loss', loss['beta_loss'], curr_iteration)
        self.writer.add_scalar(f'{train_or_valid}/total loss', loss['total_loss'], curr_iteration)
        if cycle_loss is not None:
            # self.writer.add_scalar(f'{train_or_valid}/contrast cycle loss', cycle_loss['contrast_cyc'], curr_iteration)
            self.writer.add_scalar(f'{train_or_valid}/eta cycle loss', cycle_loss['eta_cyc'], curr_iteration)
            self.writer.add_scalar(f'{train_or_valid}/beta cycle loss', cycle_loss['beta_cyc'], curr_iteration)

    def save_model(self, epoch, file_name):
        state = {'epoch': epoch,
                 'timestr': self.timestr,
                 'beta_encoder': self.beta_encoder.state_dict(),
                 'eta_encoder': self.eta_encoder.state_dict(),
                 'decoder': self.decoder.state_dict(),
                 'attention_module': self.attention_module.state_dict(),
                 'patchifier': self.patchifier.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict()}
        torch.save(obj=state, f=file_name)

    def image_to_image_translation(self, batch_id, epoch, image_dicts, train_or_valid):
        if train_or_valid == 'train':
            contrast_dropout = True
            is_train = True
        else:
            contrast_dropout = False
            is_train = False

        source_images = self.prepare_source_images(image_dicts)
        mask = image_dicts[0]['mask'].to(self.device)
        masks = torch.stack([d['mask'] for d in image_dicts], dim=-1).to(self.device)
        # print(f'Mask in model is: {mask.shape}')
        # print(f'Length of image_dicts[0] is: {len(image_dicts[0])}')
        # print(f'Length of image_dicts is: {len(image_dicts)}')
        # print(f'Keys in image_dicts[0] are: {list(image_dicts[0].keys())}')


        target_image, contrast_id_for_decoding = self.select_available_contrasts(image_dicts)
        # available_contrast_id: (batch_size, num_contrasts). 1: if available, 0: otherwise.
        available_contrast_id = torch.stack([d['exists'] for d in image_dicts], dim=-1).to(self.device)
        # print(f'available_contrast_id in model is: {available_contrast_id.shape}')
        batch_size = source_images[0].shape[0]

        # ====== 1. INTRA-SITE IMAGE-TO-IMAGE TRANSLATION ======
        logits, betas = self.calculate_beta(source_images)
        # DEBUGGY print("sources")
        # print("target_image.shape", target_image.shape)
        contrasts_source = [b.unsqueeze(2).unsqueeze(3) for b in self.calculate_contrast(source_images)] # self.calculate_contrast(source_images)
        etas_source = self.calculate_eta(source_images)
        # print(len(contrasts_source), len(etas_source))
        # print(contrasts_source[0].shape, etas_source[0].shape)
        # DEBUGGY print("target")
        contrast_target = self.calculate_contrast(target_image).unsqueeze(2).unsqueeze(3)
        eta_target = self.calculate_eta(target_image)
        # print(contrast_target.shape, eta_target.shape)
        query = torch.cat([contrast_target, eta_target], dim=1)
        keys = [torch.cat([contrast, eta], dim=1) for (contrast, eta) in zip(contrasts_source, etas_source)]
        if torch.rand((1,)) > 0.2:
            contrast_id_to_drop = contrast_id_for_decoding
        else:
            contrast_id_to_drop = None
        rec_image, attention, logit_fusion, beta_fusion = self.decode(logits, contrast_target, query, keys,
                                                                      available_contrast_id,
                                                                      masks,
                                                                      contrast_dropout=contrast_dropout,
                                                                      contrast_id_to_drop=contrast_id_to_drop)
        loss = self.calculate_loss(rec_image, target_image, mask,
                                   betas, source_images, available_contrast_id, is_train=is_train)

        # ====== 2. SAVE IMAGES OF INTRA-SITE I2I ======
        if batch_id % 100 == 1:
            file_name = os.path.join(self.out_dir, f'training_results_{self.timestr}',
                                     f'{train_or_valid}_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(4)}'
                                     '_intra-site.nii.gz')
            save_image(source_images + [rec_image] + [target_image] + betas + [beta_fusion], file_name)
            # save_image(source_images, file_name+"source_images")
            # save_image([rec_image], file_name+"rec_image")
            # save_image([target_image], file_name+"target_image")
            # save_image(betas, file_name+"betas")
            # save_image([beta_fusion], file_name+"beta_fusion")

        # ====== 3. INTER-SITE IMAGE-TO-IMAGE TRANSLATION ======
        if epoch > 1:
            random_index = torch.randperm(batch_size)
            target_image_shuffled = target_image[random_index, ...]
            logits, betas = self.calculate_beta(source_images)
            contrasts_source = [b.unsqueeze(2).unsqueeze(3) for b in self.calculate_contrast(source_images)] # self.calculate_contrast(source_images)
            etas_source = self.calculate_eta(source_images)
            contrast_target = self.calculate_contrast(target_image_shuffled).unsqueeze(2).unsqueeze(3)
            eta_target = self.calculate_eta(target_image_shuffled)
            query = torch.cat([contrast_target, eta_target], dim=1)
            keys = [torch.cat([contrast, eta], dim=1) for (contrast, eta) in zip(contrasts_source, etas_source)]
            rec_image, attention, logit_fusion, beta_fusion = self.decode(logits, contrast_target, query, keys,
                                                                            available_contrast_id, masks,
                                                                            contrast_dropout=True)
            # contrast_recon = self.contrast_encoder(rec_image)
            eta_recon = self.eta_encoder(rec_image)
            beta_recon = self.channel_aggregation(reparameterize_logit(self.beta_encoder(rec_image)))
            cycle_loss = self.calculate_cycle_consistency_loss(eta_recon, eta_target.detach(),
                                                                beta_recon, beta_fusion.detach(),
                                                                is_train=is_train)

        # ====== 4. SAVE IMAGES FOR INTER-SITE I2I ======
        if epoch > 1 and batch_id % 100 == 1:
            file_name = os.path.join(self.out_dir, f'training_results_{self.timestr}',
                                     f'{train_or_valid}_epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(4)}'
                                     '_inter-site.nii.gz')
            save_image(source_images + [rec_image] + [target_image_shuffled] + betas + [beta_fusion], file_name)
            # save_image(source_images, file_name+"source_images")
            # save_image([rec_image], file_name+"rec_image")
            # save_image([target_image], file_name+"target_image")
            # save_image(betas, file_name+"betas")
            # save_image([beta_fusion], file_name+"beta_fusion")

        # ====== 5. VISUALIZE LOSSES FOR INTRA- AND INTER-SITE I2I ======
        if epoch > 1:
            if is_train:
                self.train_loader.set_description((f'epoch: {epoch}; '
                                                   f'rec: {loss["rec_loss"]:.3f}; '
                                                   f'per: {loss["per_loss"]:.3f}; '
                                                #    f'kld: {loss["kld_loss"]:.3f}; '
                                                   f'beta: {loss["beta_loss"]:.3f}; '
                                                #    f'contrast_c: {cycle_loss["contrast_cyc"]:.3f}; '
                                                   f'eta_c: {cycle_loss["eta_cyc"]:.3f}; '
                                                   f'beta_c: {cycle_loss["beta_cyc"]:.3f}; '))
                with open(self.out_dir+'/train.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, f'{loss["rec_loss"]:.3f}', f'{loss["per_loss"]:.3f}', f'{loss["beta_loss"]:.3f}', f'{cycle_loss["eta_cyc"]:.3f}', f'{cycle_loss["beta_cyc"]:.3f}'])
                
            else:
                self.valid_loader.set_description((f'epoch: {epoch}; '
                                                   f'rec: {loss["rec_loss"]:.3f}; '
                                                   f'per: {loss["per_loss"]:.3f}; '
                                                #    f'kld: {loss["kld_loss"]:.3f}; '
                                                   f'beta: {loss["beta_loss"]:.3f}; '
                                                #    f'contrast_c: {cycle_loss["contrast_cyc"]:.3f}; '
                                                   f'eta_c: {cycle_loss["eta_cyc"]:.3f}; '
                                                   f'beta_c: {cycle_loss["beta_cyc"]:.3f}; '))
                with open(self.out_dir+'/val.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, f'{loss["rec_loss"]:.3f}', f'{loss["per_loss"]:.3f}', f'{loss["beta_loss"]:.3f}', f'{cycle_loss["eta_cyc"]:.3f}', f'{cycle_loss["beta_cyc"]:.3f}'])
            self.write_tensorboard(loss, epoch, batch_id, train_or_valid, cycle_loss)
        else:
            if is_train:
                self.train_loader.set_description((f'epoch: {epoch}; '
                                                   f'rec: {loss["rec_loss"]:.3f}; '
                                                   f'per: {loss["per_loss"]:.3f}; '
                                                #    f'kld: {loss["kld_loss"]:.3f}; '
                                                   f'beta: {loss["beta_loss"]:.3f}; '))
                with open(self.out_dir+'/train.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'rec_loss', 'per_loss', 'beta_loss', 'eta_cyc', 'beta_cyc'])
            else:
                self.valid_loader.set_description((f'epoch: {epoch}; '
                                                   f'rec: {loss["rec_loss"]:.3f}; '
                                                   f'per: {loss["per_loss"]:.3f}; '
                                                #    f'kld: {loss["kld_loss"]:.3f}; '
                                                   f'beta: {loss["beta_loss"]:.3f}; '))
                with open(self.out_dir+'/val.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['epoch', 'rec_loss', 'per_loss', 'beta_loss', 'eta_cyc', 'beta_cyc'])                    
            self.write_tensorboard(loss, epoch, batch_id, train_or_valid)

        # ====== 6. SAVE TRAINED MODELS ======
        if batch_id % 2000 == 0 and is_train:
            file_name = os.path.join(self.out_dir, f'training_models_{self.timestr}',
                                     f'epoch{str(epoch).zfill(3)}_batch{str(batch_id).zfill(4)}_model.pt')
            self.save_model(epoch, file_name)

    def train(self, epochs):
        for epoch in range(self.start_epoch, epochs + 1):
            # ====== 1. TRAINING ======
            self.train_loader = tqdm(self.train_loader)
            self.eta_encoder.eval()
            self.contrast_encoder.eval()
            self.beta_encoder.train()
            self.decoder.train()
            self.attention_module.train()
            self.patchifier.train()
            for batch_id, image_dicts in enumerate(self.train_loader):
                self.image_to_image_translation(batch_id, epoch, image_dicts, train_or_valid='train')

            # ====== 2. VALIDATION ======
            self.valid_loader = tqdm(self.valid_loader)
            self.beta_encoder.eval()
            self.eta_encoder.eval()
            self.contrast_encoder.eval()
            self.decoder.eval()
            self.patchifier.eval()
            self.attention_module.eval()
            with torch.set_grad_enabled(False):
                for batch_id, image_dicts in enumerate(self.valid_loader):
                    self.image_to_image_translation(batch_id, epoch, image_dicts, train_or_valid='valid')

    def harmonize(self, source_images, target_images, target_contrast, target_eta, out_paths,
                  recon_orientation, norm_vals, header=None, num_batches=4, save_intermediate=True, intermediate_out_dir=None):
        if out_paths is not None:
            for out_path in out_paths:
                mkdir_p(out_path.parent)
        if save_intermediate:
            mkdir_p(intermediate_out_dir)
        if out_paths is not None:
            prefix = out_paths[0].name.replace('.nii.gz', '')
        with torch.set_grad_enabled(False):
            self.beta_encoder.eval()
            self.contrast_encoder.eval()
            self.eta_encoder.eval()
            self.decoder.eval()

            # === 1. CALCULATE BETA, contrast, ETA FROM SOURCE IMAGES ===
            logits, betas, keys, masks = [], [], [], []
            for source_image in source_images:
                source_image = source_image.unsqueeze(1)
                source_image_batches = divide_into_batches(source_image, num_batches)
                mask_tmp, logit_tmp, beta_tmp, key_tmp = [], [], [], []
                for source_image_batch in source_image_batches:
                    batch_size = source_image_batch.shape[0]
                    source_image_batch = source_image_batch.to(self.device)
                    #mask = (source_image_batch > 1e-6) * 1.0
                    mask = (source_image_batch > 1e-2) * 1.0
                    logit = self.beta_encoder(source_image_batch)
                    beta = self.channel_aggregation(reparameterize_logit(logit))
                    contrast_source = self.contrast_encoder(source_image_batch)
                    eta_source = self.eta_encoder(source_image_batch).view(batch_size, self.eta_dim, 1, 1)
                    mask_tmp.append(mask)
                    logit_tmp.append(logit)
                    beta_tmp.append(beta)
                    key_tmp.append(torch.cat([contrast_source, eta_source], dim=1))
                masks.append(torch.cat(mask_tmp, dim=0))
                logits.append(torch.cat(logit_tmp, dim=0))
                betas.append(torch.cat(beta_tmp, dim=0))
                keys.append(torch.cat(key_tmp, dim=0))

            # === 2. CALCULATE contrast, ETA FOR TARGET IMAGES (IF NEEDED) ===
            if target_contrast is None:
                queries, contrasts_target = [], []
                for target_image in target_images:
                    target_image = target_image.to(self.device).unsqueeze(1)
                    contrast_target, _ = self.contrast_encoder(target_image)
                    contrast_target = contrast_target.mean(dim=0, keepdim=True)
                    eta_target = self.eta_encoder(target_image).mean(dim=0, keepdim=True).view(1, self.eta_dim, 1, 1)
                    contrasts_target.append(contrast_target)
                    queries.append(
                        torch.cat([contrast_target, eta_target], dim=1).view(1, self.contrast_dim + self.eta_dim, 1))
                if save_intermediate:
                    # save contrast and eta of target images
                    with open(intermediate_out_dir / f'{prefix}_targets.txt', 'w') as fp:
                        fp.write(','.join(['img'] + [f'contrast{i}' for i in range(self.contrast_dim)] +
                                          [f'eta{i}' for i in range(self.eta_dim)]) + '\n')
                        for i, img_query in enumerate([query.squeeze().cpu().numpy().tolist() for query in queries]):
                            fp.write(','.join([f'target{i}'] + ['%.6f' % val for val in img_query]) + '\n')
            else:
                queries, contrasts_target = [], []
                for target_contrast_tmp, target_eta_tmp in zip(target_contrast, target_eta):
                    contrasts_target.append(target_contrast_tmp.view(1, self.contrast_dim, 1, 1).to(self.device))
                    queries.append(torch.cat([target_contrast_tmp.view(1, self.contrast_dim, 1).to(self.device),
                                              target_eta_tmp.view(1, self.eta_dim, 1).to(self.device)], dim=1))

            # === 3. SAVE ENCODED VARIABLES (IF REQUESTED) ===
            if save_intermediate and header is not None:
                if recon_orientation == 'axial':
                    # 3a. source images
                    for i, source_img in enumerate(source_images):
                        img_save = source_img.squeeze().permute(1, 2, 0).permute(1, 0, 2).cpu().numpy()
                        img_save = img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96]
                        nib.Nifti1Image(img_save, None, header).to_filename(
                            intermediate_out_dir / f'{prefix}_source{i}.nii.gz'
                        )
                    # 3b. beta images
                    beta = torch.stack(betas, dim=-1)
                    if len(beta.shape) > 4:
                        beta = beta.squeeze()
                    beta = beta.permute(1, 2, 0, 3).permute(1, 0, 2, 3).cpu().numpy()
                    img_save = nib.Nifti1Image(beta[112 - 96:112 + 96, :, 112 - 96:112 + 96, :], None, header)
                    file_name = intermediate_out_dir / f'{prefix}_source_betas.nii.gz'
                    nib.save(img_save, file_name)
                    # 3c. contrast/eta values
                    with open(intermediate_out_dir / f'{prefix}_sources.txt', 'w') as fp:
                        fp.write(','.join(['img', 'slice'] + [f'contrast{i}' for i in range(self.contrast_dim)] +
                                          [f'eta{i}' for i in range(self.eta_dim)]) + '\n')
                        for i, img_key in enumerate([key.squeeze().cpu().numpy().tolist() for key in keys]):
                            for j, slice_key in enumerate(img_key):
                                fp.write(','.join([f'source{i}', f'slice{j:03d}'] +
                                                  ['%.6f' % val for val in slice_key]) + '\n')

            # ===4. DECODING===
            for tid, (contrast_target, query, norm_val) in enumerate(zip(contrasts_target, queries, norm_vals)):
                if out_paths is not None:
                    out_prefix = out_paths[tid].name.replace('.nii.gz', '')
                rec_image, beta_fusion, logit_fusion, attention = [], [], [], []
                for batch_id in range(num_batches):
                    keys_tmp = [divide_into_batches(ks, num_batches)[batch_id] for ks in keys]
                    logits_tmp = [divide_into_batches(ls, num_batches)[batch_id] for ls in logits]
                    masks_tmp = [divide_into_batches(ms, num_batches)[batch_id] for ms in masks]
                    batch_size = keys_tmp[0].shape[0]
                    query_tmp = query.view(1, self.contrast_dim + self.eta_dim, 1).repeat(batch_size, 1, 1)
                    k = torch.cat(keys_tmp, dim=-1).view(batch_size, self.contrast_dim + self.eta_dim, 1, len(source_images))
                    v = torch.stack(logits_tmp, dim=-1).view(batch_size, self.beta_dim, 224 * 224, len(source_images))
                    
                    #expanded_mask = masks_tmp[0].unsqueeze(1)
                    #expanded_mask = masks_tmp.expand(-1, attention.size(1), -1, -1, -1).squeeze(2)

                    
                    logit_fusion_tmp, attention_tmp = self.attention_module(query_tmp, k, v, masks_tmp, None, 5.0)
                    beta_fusion_tmp = self.channel_aggregation(reparameterize_logit(logit_fusion_tmp))
                    combined_map = torch.cat([beta_fusion_tmp, contrast_target.repeat(batch_size, 1, 224, 224)], dim=1)
                    masks_cpu = [mask.cpu().numpy() for mask in masks_tmp]
                    union_mask = np.logical_or.reduce(masks_cpu)
                    union_mask = torch.from_numpy(union_mask).to(masks_tmp[0].device)
                    rec_image_tmp = self.decoder(combined_map) * union_mask
                    rec_image.append(rec_image_tmp)
                    beta_fusion.append(beta_fusion_tmp)
                    logit_fusion.append(logit_fusion_tmp)
                    attention.append(attention_tmp)

                rec_image = torch.cat(rec_image, dim=0)
                beta_fusion = torch.cat(beta_fusion, dim=0)
                logit_fusion = torch.cat(logit_fusion, dim=0)
                attention = torch.cat(attention, dim=0)

                # ===5. SAVE INTERMEDIATE RESULTS (IF REQUESTED)===
                # harmonized image
                if header is not None:
                    if recon_orientation == "axial":
                        img_save = np.array(rec_image.cpu().squeeze().permute(1, 2, 0).permute(1, 0, 2))
                    elif recon_orientation == "coronal":
                        img_save = np.array(rec_image.cpu().squeeze().permute(0, 2, 1).flip(2).permute(1, 0, 2))
                    else:
                        img_save = np.array(rec_image.cpu().squeeze().permute(2, 0, 1).flip(2).permute(1, 0, 2))
                    img_save = nib.Nifti1Image((img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96]) * norm_val, None,
                                               header)
                    file_name = out_path.parent / f'{out_prefix}_harmonized_{recon_orientation}.nii.gz'
                    nib.save(img_save, file_name)

                if save_intermediate and header is not None:
                    # 5a. beta fusion
                    if recon_orientation == 'axial':
                        img_save = beta_fusion.squeeze().permute(1, 2, 0).permute(1, 0, 2).cpu().numpy()
                        img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96], None, header)
                        file_name = intermediate_out_dir / f'{out_prefix}_beta_fusion.nii.gz'
                        nib.save(img_save, file_name)
                    # 5b. logit fusion
                    if recon_orientation == 'axial':
                        img_save = logit_fusion.permute(2, 3, 0, 1).permute(1, 0, 2, 3).cpu().numpy()
                        img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96, :], None, header)
                        file_name = intermediate_out_dir / f'{out_prefix}_logit_fusion.nii.gz'
                        nib.save(img_save, file_name)
                    # 5c. attention
                    if recon_orientation == 'axial':
                        img_save = attention.permute(2, 3, 0, 1).permute(1, 0, 2, 3).cpu().numpy()
                        img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96], None, header)
                        file_name = intermediate_out_dir / f'{out_prefix}_attention.nii.gz'
                        nib.save(img_save, file_name)
                    # # 5d. attention_map
                    # if recon_orientation == 'axial' and attention_map != []:
                    #     img_save = attention_map.permute(2, 3, 0, 1).permute(1, 0, 2, 3).cpu().numpy()
                    #     img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96], None, header)
                    #     file_name = intermediate_out_dir / f'{out_prefix}_attention_map.nii.gz'
                    #     nib.save(img_save, file_name)
        if header is None:
            return rec_image.cpu().squeeze()

    def combine_images(self, image_paths, out_path, norm_val, pretrained_fusion=None):
        # obtain images
        images = []
        for image_path in image_paths:
            image_pad = torch.zeros((224, 224, 224))
            image_obj = nib.load(image_path)
            image_vol, _ = normalize_intensity(torch.from_numpy(image_obj.get_fdata().astype(np.float32)))
            image_pad[112 - 96:112 + 96, :, 112 - 96:112 + 96] = image_vol
            image_header = image_obj.header
            images.append(image_pad.numpy())

        if pretrained_fusion is not None:
            checkpoint = torch.load(pretrained_fusion, map_location=self.device)
            fusion_net = FusionNet(in_ch=3, out_ch=1)
            fusion_net.load_state_dict(checkpoint['fusion_net'])
            fusion_net.to(self.device)
            fusion_net.eval()
            with autocast():
                image = torch.cat(
                    [ToTensor()(im).permute(2, 1, 0).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) for im in images],
                    dim=1).to(self.device)
                image_fusion = fusion_net(image).squeeze().detach().permute(1, 2, 0).permute(1, 0, 2).cpu().numpy()
        else:
            # calculate median
            image_cat = np.stack(images, axis=-1)
            image_fusion = np.median(image_cat, axis=-1)

        # save fusion_image
        img_save = image_fusion[112 - 96:112 + 96, :, 112 - 96:112 + 96] * norm_val
        img_save = nib.Nifti1Image(img_save, None, image_header)
        prefix = out_path.name.replace('.nii.gz', '')
        file_name = out_path.parent / f'{prefix}_harmonized_fusion.nii.gz'
        nib.save(img_save, file_name)
