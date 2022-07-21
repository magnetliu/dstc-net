import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
import SimpleITK as sitk
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from .new_dilated_swin_transformer import DilatedSwinTransformer
from .config import get_config
from .new_resnet import ResNet
from .pixelshuffle3d import PixelShuffle3d
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        if in_planes>16:
            self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_planes // 16, in_planes, 1, bias=False))
        else:
            self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // 1, 1, bias=False),
                                    nn.ReLU(),
                                    nn.Conv3d(in_planes // 1, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()

        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()

        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class ResNetForFinetune(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        extract_x = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        extract_x.append(x)
        x = self.layer1(x)
        extract_x.append(x)

        x = self.layer2(x)
        extract_x.append(x)

        x = self.layer3(x)
        extract_x.append(x)

        x = self.layer4(x)
        extract_x.append(x)

        return extract_x

class DilatedSwinTransformerForFinetune(DilatedSwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #assert self.num_classes == 0
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x):
        extract_x = []

        x = self.patch_embed(x)
        B, L, C = x.shape
        #print("B,L,C",B,L,C)

        extract_x.append(x.transpose(1,2).reshape(B,C,32,32,32))
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        i=0
        for layer in self.layers:
            # #print(('------%d------'%i)
            x = layer(x)
            B, L, C = x.shape
            #print("B,L,C", B, L, C)

            #i+=1
            if i<1:
                extract_x.append(x.transpose(1, 2).reshape(B, C, 32, 32, 32))
            elif i<2:
                extract_x.append(x.transpose(1, 2).reshape(B, C, 32, 32, 32))
            elif i<3:
                extract_x.append(x.transpose(1, 2).reshape(B, C, 16, 16, 16))
            i+=1
        x = self.norm(x)
        B, L, C = x.shape
        #print("B,L,C",B,L,C)
        extract_x.append(x.transpose(1, 2).reshape(B, C, 16, 16, 16))
        return extract_x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}
class DST_FPN(SegmentationNetwork):
    def __init__(self,encoder,encoder1,num_classes=2,lambda1=0.5,conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=False, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2)):
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}  #
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin  # nn.LeakyReLU
        self.nonlin_kwargs = nonlin_kwargs  # {'negative_slope': 1e-2, 'inplace': True}
        self.dropout_op_kwargs = dropout_op_kwargs  # {'p': 0, 'inplace': True}
        self.norm_op_kwargs = norm_op_kwargs  # {'eps': 1e-5, 'affine': True}
        self.weightInitializer = weightInitializer  # InitWeights_He(1e-2)
        self.conv_op = nn.Conv3d

        self.norm_op = norm_op  # nn.InstanceNorm3d
        self.dropout_op = dropout_op  # nn.Dropout3d

        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.encoder = encoder
        self.encoder1 = encoder1
        self.lambda1 = lambda1
        encoder_plane = 64
        encoder1_plane = 64
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(encoder_plane, encoder_plane),
                Conv3DBlock(encoder_plane, encoder_plane),
                nn.Upsample(scale_factor=2, mode='trilinear')

            )

        self.decoder1 = \
            nn.Sequential(
                Conv3DBlock(encoder_plane, encoder_plane ),
                Conv3DBlock(encoder_plane, encoder_plane),
                nn.Upsample(scale_factor=2, mode='trilinear')

                # Conv3DBlock(encoder_plane*2, encoder_plane*2),
                #PixelShuffle3d(2)

            )

        self.decoder2 = \
            nn.Sequential(
                Conv3DBlock(encoder_plane, encoder_plane),
                Conv3DBlock(encoder_plane, encoder_plane),
                nn.Upsample(scale_factor=2, mode='trilinear')

                # Conv3DBlock(16, 16),
                #PixelShuffle3d(2)

            )

        self.decoder3 = \
            nn.Sequential(
                Conv3DBlock(encoder_plane * 8, encoder_plane* 8),
                # Conv3DBlock(128, 128),
                Conv3DBlock(encoder_plane* 8, encoder_plane),
                nn.Upsample(scale_factor=2, mode='trilinear')

                #PixelShuffle3d(2)

                # PixelShuffle3d(2)    #(64,32,32,32)
            )

        self.decoder4 = \
            nn.Sequential(
                Conv3DBlock(encoder_plane * 8, encoder_plane * 8),
                Conv3DBlock(encoder_plane * 8, encoder_plane * 8),

                # PixelShuffle3d(2)       #(512,16,16,16)
            )

        self.decoder4_upsampler = \
            nn.Sequential(
                Conv3DBlock(512, 256),
                # Conv3DBlock(256, 128),
                #PixelShuffle3d(2)  # 64,32,32,32
                Conv3DBlock(256, 64),
                nn.Upsample(scale_factor=2, mode='trilinear')

            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                # Conv3DBlock(128, 128),
                Conv3DBlock(64,64),
                nn.Upsample(scale_factor=2,mode='trilinear')
                #PixelShuffle3d(2)
            )

        self.decoder2_upsampler = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64)
            )

        self.decoder1_upsampler = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64)

                #(2),
                #nn.BatchNorm3d(3),
                #nn.Softmax(1)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 32),
                Conv3DBlock(32,num_classes),
                nn.Upsample(scale_factor=2, mode='trilinear'),
                #SingleDeconv3DBlock(32, num_classes),
                nn.BatchNorm3d(num_classes)
                #nn.Softmax(1)

            )
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()

        self.ca5 = ChannelAttention(512)
        self.ca4 = ChannelAttention(64)
        self.ca3 = ChannelAttention(64)
        self.ca2 = ChannelAttention(64)
        self.ca1 = ChannelAttention(64)

        self.fpn_decoder0 = \
            nn.Sequential(
                #Deconv3DBlock(64, 64),

                Conv3DBlock(64, 64),
                nn.Upsample(scale_factor=2, mode='trilinear')

                #Deconv3DBlock(64, 64),

            )

        self.fpn_decoder1 = \
            nn.Sequential(
                #Deconv3DBlock(64, 64),

                Conv3DBlock(64, 64),
                nn.Upsample(scale_factor=2, mode='trilinear')

                #Deconv3DBlock(64, 64),

            )

        self.fpn_decoder2 = \
            nn.Sequential(
                #Deconv3DBlock(128, 64),

                Conv3DBlock(128, 128),
                #Conv3DBlock(64, 32),
                Conv3DBlock(128, 64),

                nn.Upsample(scale_factor=2, mode='trilinear')

                #Deconv3DBlock(128, 64),

            )

        self.fpn_decoder3 = \
            nn.Sequential(
                #Deconv3DBlock(256, 64),

                Conv3DBlock(256, 256),
                Conv3DBlock(256, 64),

                nn.Upsample(scale_factor=2, mode='trilinear')

                #Deconv3DBlock(256, 64),
            )

        self.fpn_decoder4 = \
            nn.Sequential(
                Conv3DBlock(512, 512),
                Conv3DBlock(512, 512),
                # Deconv3DBlock(256, 128),
            )

    def forward(self, x,sn):
        z = self.encoder(x)
        y = self.encoder1(x)
        y0, y1, y2, y3, y4 = y
        z0, z1, z2, z3, z4 = z
        path = '/home/ubuntu/liuyiyao/my_nnUNet_data_raw_base/hot_map'


        
        print('y0.shape', y0.shape)
        print('y1.shape', y1.shape)
        print('y2.shape', y2.shape)
        print('y3.shape', y3.shape)
        print('y4.shape', y4.shape)
        print('z0.shape', z0.shape)
        print('z1.shape', z1.shape)
        print('z2.shape', z2.shape)
        print('z3.shape', z3.shape)
        print('z4.shape', z4.shape)
        lambda1 = self.lambda1
        lambda2 = 1 - self.lambda1

        z4 = self.decoder4(z4)
        # print('z4.shape', z4.shape)

        y4 = self.fpn_decoder4(y4)
        # print('y4.shape', y4.shape)
        if sn != 0:
            z4_np = z4.detach().cpu().numpy()
            np.save(path+'/%d_z4.npy'%sn,z4_np[0,:,:,:,:])
            y4_np = y4.detach().cpu().numpy()
            np.save(path + '/%d_y4.npy' % sn, y4_np[0, :, :, :, :])
        y4_ca = self.ca5(y4)
        # print('y4_1.shape', y4_1.shape)
        z4_sa = self.sa5(z4)

        # print('z4_1.shape', z4_1.shape)

        z4 = lambda1 * (y4*z4_sa) + lambda2 * (z4*y4_ca)

        u4 = self.decoder4_upsampler(z4)

        z3 = self.decoder3(z3)
        ##print('z3.shape', z3.shape)
        y3 = self.fpn_decoder3(y3)
        if sn != 0:
            z3_np = z3.detach().cpu().numpy()
            np.save(path+'/%d_z3.npy'%sn,z3_np[0,:,:,:,:])
            y3_np = y3.detach().cpu().numpy()
            np.save(path + '/%d_y3.npy' % sn, y3_np[0, :, :, :, :])
        y3_ca = self.ca4(y3)
        # print('y4_1.shape', y4_1.shape)
        z3_sa = self.sa4(z3)

        # print('z4_1.shape', z4_1.shape)

        z3  = lambda1 * (y3*z3_sa) + lambda2 * (z3*y3_ca)

        ##print('z3.shape', z3.shape)

        u3 = self.decoder3_upsampler(torch.cat([z3, u4], dim=1))

        z2 = self.decoder2(z2)
        y2 = self.fpn_decoder2(y2)
        if sn != 0:
            z2_np = z2.detach().cpu().numpy()
            np.save(path+'/%d_z2.npy'%sn,z2_np[0,:,:,:,:])
            y2_np = y2.detach().cpu().numpy()
            np.save(path + '/%d_y2.npy' % sn, y2_np[0, :, :, :, :])
        y2_ca = self.ca3(y2)
        # print('y4_1.shape', y4_1.shape)
        z2_sa = self.sa3(z2)

        # print('z4_1.shape', z4_1.shape)

        z2 = lambda1 * (y2 * z2_sa) + lambda2 * (z2 * y2_ca)
        ##print('z2.shape', z2.shape)
        u2 = self.decoder2_upsampler(torch.cat([z2, u3], dim=1))
        ##print('z2.shape', z2.shape)

        z1 = self.decoder1(z1)
        y1 = self.fpn_decoder1(y1)
        if sn != 0:
            z1_np = z1.detach().cpu().numpy()
            np.save(path+'/%d_z1.npy'%sn,z1_np[0,:,:,:,:])
            y1_np = y1.detach().cpu().numpy()
            np.save(path + '/%d_y1.npy' % sn, y1_np[0, :, :, :, :])
        y1_ca = self.ca2(y1)
        # print('y4_1.shape', y4_1.shape)
        z1_sa = self.sa2(z1)

        # print('z4_1.shape', z4_1.shape)

        z1 = lambda1 * (y1 * z1_sa) + lambda2 * (z1 * y1_ca)
        ##print('z1.shape', z1.shape)




        u1 = self.decoder1_upsampler(torch.cat([z1, u2], dim=1))
        ##print('z2.shape', z2.shape)

        z0 = self.decoder0(z0)
        y0 = self.fpn_decoder0(y0)
        if sn != 0:
            z0_np = z0.detach().cpu().numpy()
            np.save(path+'/%d_z0.npy'%sn,z0_np[0,:,:,:,:])
            y0_np = y0.detach().cpu().numpy()
            np.save(path + '/%d_y0.npy' % sn, y0_np[0, :, :, :, :])
        y0_ca = self.ca1(y0)
        # print('y4_1.shape', y4_1.shape)
        z0_sa = self.sa1(z0)
        # print('z4_1.shape', z4_1.shape)
        z0 = lambda1 * (y0 * z0_sa) + lambda2 * (z0 * y0_ca)
        u0 = self.decoder0_header(torch.cat([z0, u1], dim=1))
        path = '/home/ubuntu/liuyiyao/my_nnUNet_data_raw_base/hot_map'
        if sn != 0:
            z0_np = u0.detach().cpu().numpy()
            np.save(path+'/%d_u0.npy'%sn,z0_np[0,:,:,:,:])
            z1_np = u1.detach().cpu().numpy()
            np.save(path + '/%d_u1.npy' % sn, z1_np[0, :, :, :, :])
            z2_np = u2.detach().cpu().numpy()
            np.save(path + '/%d_u2.npy' % sn, z2_np[0, :, :, :, :])
            z3_np = u3.detach().cpu().numpy()
            np.save(path + '/%d_u3.npy' % sn, z3_np[0, :, :, :, :])
            z4_np = u4.detach().cpu().numpy()
            np.save(path + '/%d_u4.npy' % sn, z4_np[0, :, :, :, :])
        x_np = x.detach().cpu().numpy()
        np.save(path + '/%d_ori.npy' % sn,x_np[0])
        x_img = sitk.GetImageFromArray(x_np[0,0])
        sitk.WriteImage(x_img,path + '/%d_ori.nii.gz' % sn)
        output = []
        output.append(u0)

        if self._deep_supervision and self.do_ds:
            return output
        else:
            return output[-1]
        ##print('output.shape', output.shape)


def build_newsc_dst_fpn(num_classes=3,lambda1 = 0.5):
    config = get_config()
    encoder = DilatedSwinTransformerForFinetune(img_size=config.DATA.IMG_SIZE,
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
    resnet = ResNetForFinetune(layers = [2, 2, 2, 2])
    model = DST_FPN(encoder=encoder,encoder1=resnet,num_classes=num_classes,lambda1=lambda1,conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2))
    return model