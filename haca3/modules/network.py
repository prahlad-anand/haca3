import torch
from torch import nn
import torch.nn.functional as F
import math
from scipy.ndimage import label
import numpy as np
from .utils import normalize_attention, normalize_and_smooth_attention
from transformers import CLIPVisionModel


class FusionNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, 1, 1),
            nn.InstanceNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(8, 16, 3, 1, 1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch + 16, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(16, out_ch, 3, 1, 1),
            nn.ReLU())

    def forward(self, x):
        # return self.conv2(x + self.conv1(x))
        return self.conv2(torch.cat([x, self.conv1(x)], dim=1))

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act='noact'):
        super().__init__()
        self.final_act = final_act
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)

        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for lv in range(num_lvs):
            ch = base_ch * (2 ** lv)
            self.down_convs.append(ConvBlock2d(ch + conditional_ch, ch * 2, ch * 2))
            self.down_samples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.up_samples.append(Upsample(ch * 4))
            self.up_convs.append(ConvBlock2d(ch * 4, ch * 2, ch * 2))
        bottleneck_ch = base_ch * (2 ** num_lvs)
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch * 2, bottleneck_ch * 2)
        self.out_conv = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(base_ch, out_ch, 3, 1, 1))

    def forward(self, in_tensor, condition=None):
        encoded_features = []
        x = self.in_conv(in_tensor)
        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            if condition is not None:
                feature_dim = x.shape[-1]
                down_conv_out = down_conv(torch.cat([x, condition.repeat(1, 1, feature_dim, feature_dim)], dim=1))
            else:
                down_conv_out = down_conv(x)
            x = down_sample(down_conv_out)
            encoded_features.append(down_conv_out)
        x = self.bottleneck_conv(x)
        for encoded_feature, up_conv, up_sample in zip(reversed(encoded_features),
                                                       reversed(self.up_convs),
                                                       reversed(self.up_samples)):
            x = up_sample(x, encoded_feature)
            x = up_conv(x)
        x = self.out_conv(x)
        if self.final_act == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_act == "relu":
            x = torch.relu(x)
        elif self.final_act == 'tanh':
            x = torch.tanh(x)
        else:
            x = x
        return x


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.InstanceNorm2d(mid_ch),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor):
        return self.conv(in_tensor)


class Upsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = in_ch // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor, encoded_feature):
        up_sampled_tensor = F.interpolate(in_tensor, size=None, scale_factor=2, mode='bilinear', align_corners=False)
        up_sampled_tensor = self.conv(up_sampled_tensor)
        return torch.cat([encoded_feature, up_sampled_tensor], dim=1)


class EtaEncoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, 1, 2),  # (*, 16, 224, 224)
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 64, 3, 1, 1),  # (*, 64, 224, 224)
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.seq = nn.Sequential(
            nn.Conv2d(64 + in_ch, 32, 32, 32, 0),  # (*, 32, 7, 7)
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_ch, 7, 7, 0))

    def forward(self, x):
        return self.seq(torch.cat([self.in_conv(x), x], dim=1))


class Patchifier(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 32, 32, 0),  # (*, in_ch, 224, 224) --> (*, 64, 7, 7)
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, out_ch, 1, 1, 0))

    def forward(self, x):
        return self.conv(x)

# ============================================================================
# IMAGE ENCODER (SIMPLIFIED - REUSE FROM ORIGINAL)
# ============================================================================

    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet50."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512 * block.expansion

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("img_min_max", x.min(), x.max())
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    
def get_image_encoder(encoder_name, in_channels=3):
    """Factory function to create image encoders."""
    if encoder_name == 'resnet16':
        encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2], in_channels=in_channels)
        encoder.feature_dim = 512
    elif encoder_name == 'resnet18':
        encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2], in_channels=in_channels)
        encoder.feature_dim = 512
    elif encoder_name == 'resnet50':
        encoder = ResNetEncoder(Bottleneck, [3, 4, 6, 3], in_channels=in_channels)
        encoder.feature_dim = 2048
    elif encoder_name == 'vit-l':
        from torchvision.models.vision_transformer import VisionTransformer
        encoder = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096
        )
        encoder.feature_dim = 1024
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    return encoder

class MulticlassSVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.w = nn.Parameter(torch.randn(num_classes, input_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        return torch.matmul(x, self.w.T) + self.b


class MulticlassMLP(nn.Module):
    """MLP classifier without non-linearities."""

    def __init__(self, input_dim, num_classes, hidden_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        if hidden_dim is None:
            hidden_dim = max(input_dim * 2, num_classes * 4)

        self.hidden_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class ContrastiveImageEncoder(nn.Module):
    """Image encoder with tanh-bounded projection and classifier (SVM or MLP)."""

    def __init__(self, encoder_name='resnet16', embedding_dim=512, freeze_backbone=False,
                 num_contrasts=4, use_mlp=False, mlp_hidden_dim=None):
        super().__init__()

        self.backbone = get_image_encoder(encoder_name, in_channels=1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        backbone_dim = self.backbone.feature_dim
        self.embedding_dim = embedding_dim
        self.use_mlp = use_mlp

        # Tanh-bounded projection
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.Tanh()
        )

        # Classifier: SVM or MLP
        if use_mlp:
            self.contrast_classifier = MulticlassMLP(embedding_dim, num_contrasts, mlp_hidden_dim)
        else:
            self.contrast_classifier = MulticlassSVM(embedding_dim, num_contrasts)

        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x, return_contrast_logits=False, normalize_for_contrastive=False):
        # print("BATCH SHAPE: ", x.shape)
        # print("MIN-MAX OF IMG", x.min(), x.max())
        # DEBUGGY print("SHAPE-KUN", x.shape)
        # DEBUGGY print("before", x.min(), x.max())

        # import torchvision.utils as vutils
        # import os
        # for i, img in enumerate(x):

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

        #     save_path = os.path.join("/iacl/pg23/prahlad/4imgs", f"source_images_before{i}.png")
        #     vutils.save_image(img, save_path)

        #     print(f"Saved {save_path}")

        # print(x.shape)
        # print("\nbefore", x.min(), x.max())
        # # x = F.normalize(x, dim=1)
        # # x = x.repeat(1,3,1,1)
        # print("after", x.min(), x.max())

        # for i, img in enumerate(x):

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

        #     save_path = os.path.join("/iacl/pg23/prahlad/4imgs", f"source_images_after{i}.png")
        #     vutils.save_image(img, save_path)

        #     print(f"Saved {save_path}")

        # print(x.shape)

        features = self.backbone(x)
        embeddings = self.projector(features)
        # print("INITAL EMBEDDINGS", embeddings)

        

        if False: # normalize_for_contrastive
            embeddings_for_contrastive = F.normalize(embeddings, dim=1)
        else:
            embeddings_for_contrastive = embeddings

        if False: # return_contrast_logits
            contrast_logits = self.contrast_classifier(embeddings)
            return embeddings_for_contrastive, contrast_logits
        return embeddings_for_contrastive

class ThetaEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 17, 9, 4),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),  # (*, 32, 28, 28)
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1),  # (*, 64, 14, 14)
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1))  # (* 64, 7, 7)
        self.mean_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_ch, 6, 6, 0))
        self.logvar_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_ch, 6, 6, 0))

    def forward(self, x):
        M = self.conv(x)
        mu = self.mean_conv(M)
        logvar = self.logvar_conv(M)
        return mu, logvar

# class ThetaEncoder(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, 32, 32, 32, 0),  # (*, in_ch, 224, 244) --> (*, 32, 7, 7)
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(32, 64, 1, 1, 0),
#             # nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.1))
#         self.mu_conv = nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(64, out_ch, 7, 7, 0))
#         self.logvar_conv = nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(64, out_ch, 7, 7, 0))
#
#     def forward(self, x, patch_shuffle=False):
#         m = self.conv(x)
#         if patch_shuffle:
#             batch_size = m.shape[0]
#             num_features = m.shape[1]
#             num_patches_per_dim = m.shape[-1]
#             m = m.view(batch_size, num_features, -1)[:, :, torch.randperm(num_patches_per_dim ** 2)]
#             m = m.view(batch_size, num_features, num_patches_per_dim, num_patches_per_dim)
#         mu = self.mu_conv(m)
#         logvar = self.logvar_conv(m)
#         return mu, logvar


class AttentionModule(nn.Module):
    def __init__(self, dim, v_ch=5):
        super().__init__()
        self.dim = dim
        self.v_ch = v_ch
        self.q_fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 16),
            nn.LayerNorm(16))
        self.k_fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 16),
            nn.LayerNorm(16))

        self.scale = self.dim ** (-0.5)
        
    def forward(self, q, k, v, mask, modality_dropout=None, temperature=10.0):
        """
        Attention module for optimal anatomy fusion.

        ===INPUTS===
        * q: torch.Tensor (batch_size, feature_dim_q, num_q_patches=1)
            Query variable. In HACA3, query is the concatenation of target \theta and target \eta.
        * k: torch.Tensor (batch_size, feature_dim_k, num_k_patches=1, num_contrasts=4)
            Key variable. In HACA3, keys are \theta and \eta's of source images.
        * v: torch.Tensor (batch_size, self.v_ch=5, num_v_patches=224*224, num_contrasts=4)
            Value variable. In HACA3, values are multi-channel logits of source images.
            self.v_ch is the number of \beta channels.
        * modality_dropout: torch.Tensor (batch_size, num_contrasts=4)
            Indicates which contrast indexes have been dropped out. 1: if dropped out, 0: if exists.
        """
        batch_size, feature_dim_q, num_q_patches = q.shape
        _, feature_dim_k, _, num_contrasts = k.shape
        num_v_patches = v.shape[2]
        assert (
                feature_dim_k == feature_dim_q or feature_dim_q == self.feature_dim
        ), 'Feature dimensions do not match.'

        # q.shape: (batch_size, num_q_patches=1, 1, feature_dim_q)
        q = q.reshape(batch_size, feature_dim_q, num_q_patches, 1).permute(0, 2, 3, 1)
        # k.shape: (batch_size, num_k_patches=1, num_contrasts=4, feature_dim_k)
        k = k.permute(0, 2, 3, 1)
        # v.shape: (batch_size, num_v_patches=224*224, num_contrasts=4, v_ch=5)
        v = v.permute(0, 2, 3, 1)
        q = self.q_fc(q)
        # k.shape: (batch_size, num_k_patches=1, feature_dim_k, num_contrasts=4)
        k = self.k_fc(k).permute(0, 1, 3, 2)

        # dot_prod.shape: (batch_size, num_q_patches=1, 1, num_contrasts=4)
        dot_prod = (q @ k) * self.scale

        contrastwise_attention_scores = torch.softmax(dot_prod / temperature, dim=-1) # [32, 1, 1, 4]

        interpolation_factor = int(math.sqrt(num_v_patches // num_q_patches))

        q_spatial_dim = int(math.sqrt(num_q_patches))
        dot_prod = dot_prod.view(batch_size, q_spatial_dim, q_spatial_dim, num_contrasts)

        image_dim = int(math.sqrt(num_v_patches))
        # dot_prod_interp.shape: (batch_size, image_dim, image_dim, num_contrasts)
        dot_prod_interp = dot_prod.repeat(1, interpolation_factor, interpolation_factor, 1)
        if modality_dropout is not None:
            modality_dropout = modality_dropout.view(batch_size, num_contrasts, 1, 1).permute(0, 2, 3, 1)
            dot_prod_interp = dot_prod_interp - (modality_dropout.repeat(1, image_dim, image_dim, 1).detach() * 1e5)

        attention = (dot_prod_interp / temperature).softmax(dim=-1)
        # v = attention.view(batch_size, num_v_patches, 1, num_contrasts) @ v
        # v = v.view(batch_size, image_dim, image_dim, self.v_ch).permute(0, 3, 1, 2)
        # attention = attention.view(batch_size, image_dim, image_dim, num_contrasts).permute(0, 3, 1, 2)

        #print(f"Mask type: {mask.dtype}, shape: {mask.shape}")

        if isinstance(mask, list):
            mask = torch.stack(mask)

        #print(mask.shape)

        # Transpose the mask to match the order of dimensions in attention
        if len(mask.shape)==5 and mask.shape[0]!= batch_size:
            mask = mask.permute(1, 2, 3, 4, 0)  # This changes the order to [batch_size, num_contrasts, 1, 224, 224]
            #mask = mask.squeeze(1)  # Squeeze the -- dimension to reduce the shape to [56, 224, 224, 3]
        # print(mask.shape)
        mask = mask.squeeze(1)

        #print(f"Attention type: {attention.dtype}, shape: {attention.shape}")
        #print(f"Mask type: {mask.dtype}, shape: {mask.shape}")

        attention_map = attention * mask
        # print(f"Attention Map type: {attention_map.dtype}, shape: {attention_map.shape}")

        # Normalize the attention map
        # normalized_attention_map = normalize_attention(attention_map)
        normalized_attention_map = normalize_and_smooth_attention(attention_map,diff_threshold=0.2)
        # print(f"Normalization Attention Map type: {normalized_attention_map.dtype}, shape: {normalized_attention_map.shape}")

        # # Manual Attention Map (size:[56, 224, 224, 3])
        # normalized_attention_map[:, :112, :, 0] = 1  
        # normalized_attention_map[:, :112, :, 0] = 0 
        # normalized_attention_map[:, :112, :, 1] = 0 
        # normalized_attention_map[:, :112, :, 1] = 0.5
        # normalized_attention_map[:, :112, :, 2] = 0 
        # normalized_attention_map[:, :112, :, 2] = 0.5
        # # print(f"normalized_attention_map shape after manual modification: {normalized_attention_map.shape}")

        
        # Use the normalized attention map instead of the original attention for v calculation
        v = normalized_attention_map.view(batch_size, num_v_patches, 1, num_contrasts) @ v
        v = v.view(batch_size, image_dim, image_dim, self.v_ch).permute(0, 3, 1, 2)
        attention = normalized_attention_map.view(batch_size, image_dim, image_dim, num_contrasts).permute(0, 3, 1, 2)

        return v, attention, contrastwise_attention_scores
        
