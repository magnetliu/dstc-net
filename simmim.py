from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import cv2
from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer
from .dilated_swin_transformer import DilatedSwinTransformer
from .resnet import ResNet
from .config import get_config
import numpy as np
from SimMIM.utils.pixelshuffle3d import PixelShuffle3d

class ResNetForSimMIM(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #assert self.num_classes == 0

        #self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        #trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask, path, name):
        #x = x * (1. - mask)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask, path, name):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape

        H = W = D= 8
        x = x.reshape(B, C, H, W,D)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}
class DilatedSwinTransformerForSimMIM(DilatedSwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask, path, name):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        #i=0
        for layer in self.layers:
           # print('------%d------'%i)

            x = layer(x)
            #i+=1
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape

        H = W = D= 16
        x = x.reshape(B, C, H, W,D)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}

class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0
        self.ps = nn.PixelShuffle(16)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask, path, name):
        # print("x_shape:",x.shape)
        # print("mask_shape:",mask.shape)

        x = self.patch_embed(x)
        # print("x_shape:",x.shape)

        assert mask is not None
        B, L, _ = x.shape  # B,195,768

        mask_token = self.mask_token.expand(B, L, -1)  # B,196,
        # print("mask_token_shape:",mask_token.shape)
        # print("x_shape:",x.shape)

        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        #print("w_shape:",w.shape)

        x = x * (1 - w) + mask_token * w
        # print("x_shape:",x.shape)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)
        x = self.norm(x)
        print("x.shape",x.shape)
        x = x[:, 1:]
        B, L, C = x.shape  # B,196,768
        #print("x.shape",x.shape)

        H = W = D =8
        x = x.permute(0, 2, 1).reshape(B, C, H, W,D)
        return x

class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder1 = nn.Sequential(
            nn.Conv3d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 3, kernel_size=1),
            #nn.BatchNorm3d(32768),
            #nn.ReLU(True),

        )
        self.decoder2 = nn.Sequential(
            PixelShuffle3d(self.encoder_stride),
            #nn.Conv3d(1,1,1),
            #nn.LayerNorm([1,128,128,128]),
            nn.BatchNorm3d(1),
            #nn.LeakyReLU(),
            #nn.Tanh()

        )
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask, path, name):
        #print("x.shape",x.shape)
        z = self.encoder(x, mask, path, name)

        # x_rec = self.decoder2(z)

        # print("z_shape",z.shape)
        x_rec = self.decoder1(z)
        # print("x_rec1_shape:",x_rec1.shape)
        x_rec = self.decoder2(x_rec)
        x_rec_clone = x_rec.clone()
        """for i in range(x.shape[0]):
            x_max = torch.max(x_rec[i])
            x_min = torch.min(x_rec[i])
            x_rec_clone[i] = (x_rec[i]-x_min)/(x_max-x_min)
        x_rec_clone  = (x_rec_clone-0.5)/0.5"""


        # print(path)
        # print("x_rec_shape:",x_rec.shape)
        # print("x_shape:",x.shape)

        # print("mask_shape:",mask.shape)
        mask = mask.repeat_interleave(self.patch_size[0], 1).repeat_interleave(self.patch_size[1], 2).repeat_interleave(
            self.patch_size[2], 3).unsqueeze(1).contiguous()

        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss =  loss_recon.sum()/(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])#(loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans  #
        return loss,x_rec,x,mask

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

class MySimMIM(nn.Module):
    def __init__(self, encoder, encoder1,encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.encoder1 = encoder1
        self.decoder1_1 = nn.Sequential(
            nn.Conv3d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 3, kernel_size=1),
            PixelShuffle3d(self.encoder_stride),
            nn.BatchNorm3d(1),

            #nn.BatchNorm3d(32768),
            #nn.ReLU(True),

        )
        self.decoder1_2 = nn.Sequential(

            PixelShuffle3d(2),

            #nn.BatchNorm3d(32768),
            #nn.ReLU(True),

        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2, padding=0, output_padding=0)            #nn.Conv3d(1,1,1),
            #nn.LayerNorm([1,128,128,128]),
            #nn.LeakyReLU(),
            #nn.Tanh()

        )
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask, path, name):
        #print("x.shape",x.shape)
        z1 = self.encoder(x, mask, path, name)
        z2 = self.encoder1(x,mask, path, name)
        # x_rec = self.decoder2(z)

        # print("z_shape",z.shape)
        x_rec = self.decoder1_1(z1)
        #x_features_t = self.decoder1_2(z1)
        #x_features_r = self.decoder2(z2)
        # print("x_rec1_shape:",x_rec1.shape)
        x_rec_clone = x_rec.clone()
        """for i in range(x.shape[0]):
            x_max = torch.max(x_rec[i])
            x_min = torch.min(x_rec[i])
            x_rec_clone[i] = (x_rec[i]-x_min)/(x_max-x_min)
        x_rec_clone  = (x_rec_clone-0.5)/0.5"""


        # print(path)
        # print("x_rec_shape:",x_rec.shape)
        # print("x_shape:",x.shape)

        # print("mask_shape:",mask.shape)
        mask = mask.repeat_interleave(self.patch_size[0], 1).repeat_interleave(self.patch_size[1], 2).repeat_interleave(
            self.patch_size[2], 3).unsqueeze(1).contiguous()
        loss_features = F.l1_loss(z1,z2,reduction='none')
        loss_features = loss_features.sum()/(x.shape[0]*512*16*16*16)
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss_recon = loss_recon.sum()/(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])
        loss = loss_features+loss_recon#(loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans  #
        return loss,x_rec,x,mask

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim():
    model_type = 'mysimmim'
    config = get_config()
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(img_size=config.DATA.IMG_SIZE,
                                           patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                           in_chans=config.MODEL.SWIN.IN_CHANS,
                                           num_classes=0,
                                           embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                           depths=config.MODEL.SWIN.DEPTHS,
                                           num_heads=config.MODEL.SWIN.NUM_HEADS,
                                           window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                           mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                           qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                           qk_scale=config.MODEL.SWIN.QK_SCALE,
                                           drop_rate=config.MODEL.DROP_RATE,
                                           drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                           ape=config.MODEL.SWIN.APE,
                                           patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                           use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        encoder_stride = 16
        model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)
    if model_type == 'mysimmim':
        encoder = DilatedSwinTransformerForSimMIM(img_size=config.DATA.IMG_SIZE,
                                           patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                           in_chans=config.MODEL.SWIN.IN_CHANS,
                                           num_classes=0,
                                           embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                           depths=config.MODEL.SWIN.DEPTHS,
                                           num_heads=config.MODEL.SWIN.NUM_HEADS,
                                           window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                           mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                           qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                           qk_scale=config.MODEL.SWIN.QK_SCALE,
                                           drop_rate=config.MODEL.DROP_RATE,
                                           drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                           ape=config.MODEL.SWIN.APE,
                                           patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                           use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                           dilated = config.MODEL.SWIN.DILATED,
                                                  lambda1 = config.MODEL.SWIN.LAMBDA1)
        encoder1 = ResNetForSimMIM(layers = [2, 2, 2, 2])

        encoder_stride = 8
        model = MySimMIM(encoder=encoder,encoder1 = encoder1, encoder_stride=encoder_stride)
    elif model_type == 'vit':
        encoder = VisionTransformerForSimMIM(img_size=config.DATA.IMG_SIZE,
                                             patch_size=config.MODEL.VIT.PATCH_SIZE,
                                             in_chans=config.MODEL.VIT.IN_CHANS,
                                             num_classes=0,
                                             embed_dim=config.MODEL.VIT.EMBED_DIM,
                                             depth=config.MODEL.VIT.DEPTH,
                                             num_heads=config.MODEL.VIT.NUM_HEADS,
                                             mlp_ratio=config.MODEL.VIT.MLP_RATIO,
                                             qkv_bias=config.MODEL.VIT.QKV_BIAS,
                                             drop_rate=config.MODEL.DROP_RATE,
                                             drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             init_values=config.MODEL.VIT.INIT_VALUES,
                                             use_abs_pos_emb=config.MODEL.VIT.USE_APE,
                                             use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
                                             use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
                                             use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        encoder_stride = 16

        model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)

    elif model_type =='resnet':
        encoder = ResNetForSimMIM(layers = [2, 2, 2, 2])
        model = SimMIM_1(encoder=encoder, encoder_stride=32)
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")


    return model