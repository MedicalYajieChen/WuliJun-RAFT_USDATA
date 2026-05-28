import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update import BasicUpdateBlock
from core.extractor_sam import BasicEncoder, twins_svt_large
from core.corr import CorrBlock, AlternateCorrBlock
from core.utils.utils import bilinear_sampler, coords_grid, upflow8
from core.image_encoder import ImageEncoderViT
# from update import BasicUpdateBlock
# from extractor_sam import BasicEncoder, sam_basic_context
# from corr import CorrBlock, AlternateCorrBlock
# from utils.utils import bilinear_sampler, coords_grid, upflow8
import argparse

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class ResidualBlock_context(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock_context, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        # self.fnet = twins_svt_large(pretrained=False)        
        # self.cnet = sam_basic_context(args.image_size)
        self.image_encoder = ImageEncoderViT(args.image_size)
        self.cnet = BasicEncoder(output_dim=256, norm_fn='batch', dropout=0.0)
        # self.channel_convertor = nn.Conv2d(256, 256, 1, padding=0, bias=False)

        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        in_channals = 512
        self.depthwise = nn.Conv2d(in_channals, in_channals, 3, padding=1, groups=512)
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_channals, in_channals, 1)

        self.residual_block1 = ResidualBlock_context(in_channals, in_channals)
        self.residual_block2 = ResidualBlock_context(in_channals, in_channals)
        self.residual_block3 = ResidualBlock_context(in_channals, in_channals)
        self.last_layer = nn.Conv2d(in_channals, in_channals//2, 1)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
            # imgs = torch.cat([image1, image2], dim=0)
            # feats = self.fnet(imgs)
            # feats = self.channel_convertor(feats)
            # B = feats.shape[0] // 2

            # fmap1 = feats[:B]
            # fmap2 = feats[B:]  
            # print(fmap1.shape)
            # print(image1.shape)      
        # print(fmap1.shape)
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        # if self.args.alternate_corr:
        corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        # else:
        # corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            # cnet = self.image_encoder(image1)

            ##sam
            fea_b = self.cnet(image1)
            fea_s = self.image_encoder(image1)

            assert fea_b.shape == fea_s.shape

            fea_cat = torch.cat([fea_b, fea_s], dim=1)
            fea_conv = self.residual_block1(fea_cat)
            fea_conv = self.residual_block2(fea_conv)
            fea_dep = self.depthwise(fea_cat)
            fea_dep = self.pointwise(fea_dep)
            cnet = fea_dep+ fea_conv
            cnet = self.residual_block3(cnet)
            cnet = self.last_layer(cnet)

            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-usdata', help="name your experiment")
    parser.add_argument('--stage', default='usdata', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', default='usdata', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    # parser.add_argument('--image_size', type=int, nargs='+', default=[376, 464])
    parser.add_argument('--image_size', type=int, nargs='+', default=[352, 376])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()
    img1 = torch.randn([2,3,512,512])
    img2 = torch.randn([2,3,512,512])
    model = RAFT(args)
    output = model(img1, img2)
    print('a')