import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 64*9, 1, padding=0))
            ###改  quarter
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        # print('8' + str(motion_features.shape))
        # print('9' + str(inp.shape))
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

class OMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(OMAUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.oma = OMA(r=1)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, itr, upsample=True):
        motion_features = self.encoder(flow, corr)
        # print('8' + str(motion_features.shape))
        # print('9' + str(inp.shape))
        motion_oma = self.oma(motion_features, itr)
        inp = torch.cat([inp, motion_oma], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

class OMA(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = r
    def cosine_similarity_update(self, x):
        """Calculate the cosine similarity between the center pixel and its 8 neighbors,
           and update the center pixel with a weighted average of the neighbor pixels based on
           the similarity scores.

        Args:
            x (torch.Tensor): A 3D tensor of shape (C, H, W), representing the input image.

        Returns:
            torch.Tensor: A 3D tensor of shape (C, H, W), representing the output image with
            updated center pixels.
        """

        Batch, C, H, W = x.shape
        update_m = []
        for i in range(Batch):
            padded_x = torch.nn.functional.pad(x[i], (1, 1, 1, 1), mode='constant', value=0)  # pad with zeros

            # Compute cosine similarity between center pixel and its 8 neighbors
            center = padded_x[:, 1:H + 1, 1:W + 1]
            neighbors = [padded_x[:, i:i + H, j:j + W] for i, j in
                         [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]]
            sims = [(torch.nn.functional.cosine_similarity(center.reshape(C, -1), neighbor.reshape(C, -1))) for neighbor
                    in neighbors]

            # for k in range(len(sims)):
            #     sims[k][sims[k]<0]=0
            # Compute weighted average of neighbor pixels based on similarity scores
            update = torch.stack([sim.view(C, 1, 1) * neighbor for sim, neighbor in zip(sims, neighbors)])
            # update = 0.5*x[i] + 0.5*(torch.sum(update, dim=0))
            update_m.append(torch.sum(update, dim=0))

        outputs = torch.stack(update_m, dim=0)

        return outputs

    def forward(self, *inputs):
        motion_features, itr = inputs
        if itr > 5:
            feat_o = self.cosine_similarity_update(motion_features)
        else:
            feat_o = motion_features
        return feat_o

class OMAUpdateBlock_pre(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(OMAUpdateBlock_pre, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.oma = OMA_pre(r=1)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, flow_pre=None, upsample=True):
        motion_features = self.encoder(flow, corr)
        # print('8' + str(motion_features.shape))
        # print('9' + str(inp.shape))
        if flow_pre is not None:
            motion_oma = self.oma(motion_features, flow_pre)
            inp = torch.cat([inp, motion_oma], dim=1)
        else:
            inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

class OMA_pre(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = r
    def cosine_similarity_update(self, x, flow):
        """Calculate the cosine similarity between the center pixel and its 8 neighbors,
           and update the center pixel with a weighted average of the neighbor pixels based on
           the similarity scores.

        Args:
            x (torch.Tensor): A 3D tensor of shape (C, H, W), representing the input image.

        Returns:
            torch.Tensor: A 3D tensor of shape (C, H, W), representing the output image with
            updated center pixels.
        """

        Batch, C, H, W = x.shape
        update_m = []
        for i in range(Batch):
            padded_x = torch.nn.functional.pad(x[i], (1, 1, 1, 1), mode='constant', value=0)  # pad with zeros

            # Compute cosine similarity between center pixel and its 8 neighbors
            center = padded_x[:, 1:H + 1, 1:W + 1]
            neighbors = [padded_x[:, i:i + H, j:j + W] for i, j in
                         [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]]

            padded_flow = torch.nn.functional.pad(flow[i], (1, 1, 1, 1), mode='constant', value=0)  # pad with zeros

            # Compute cosine similarity between center pixel and its 8 neighbors
            center_flow = padded_flow[:, 1:H + 1, 1:W + 1]
            neighbors_flow = [padded_flow[:, i:i + H, j:j + W] for i, j in
                         [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]]
            sims = [torch.cosine_similarity(center_flow, neighbor, dim=0) for neighbor in neighbors_flow]
            
            update = torch.stack([sim * neighbor for sim, neighbor in zip(sims, neighbors)])
            update = 0.89*center + 0.11*(torch.sum(update, dim=0))
            # update = 0.5*x[i] + 0.5*(torch.sum(update, dim=0))
            update_m.append(update)

        outputs = torch.stack(update_m, dim=0)

        return outputs

    def forward(self, *inputs):
        motion_features, flow_pre = inputs

        feat_o = self.cosine_similarity_update(motion_features, flow_pre)
        return feat_o

