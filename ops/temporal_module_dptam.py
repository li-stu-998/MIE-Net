
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



class DPTAM(nn.Module):
    def __init__(self,
                 in_channels,
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(DPTAM, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        print('DPTAM with kernel_size {}.'.format(kernel_size))

        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)#context Modeling
        self.softmax = nn.Softmax(dim=2)
        self.p1_conv1= nn.Conv1d(in_channels , in_channels, 1, bias=False)


        self.dptam = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        nt, c, h, w = x.size()

        t = self.n_segment
        n_batch = nt // t
        new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3,4).contiguous()
        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))

        x_22=out.view(-1,c,t)
        x22_c_t = self.p1_conv1(x_22)
        x22 =x_22.mean(2,keepdim=True)
        x22 = self.p1_conv1(x22)
        x22 = x22_c_t * x22
        x22= x_22+x22

        local_activation = self.dptam(x22).view(n_batch, c, t, 1, 1)
        new_x = new_x * local_activation

        out = new_x.view(n_batch, c, t, h, w) #光local
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)

        return out


class Temporal2Bottleneck(nn.Module):
    def __init__(self,
                 net,
                 n_segment=8,
                 t_kernel_size=3,
                 t_stride=1,
                 t_padding=1):
        super(Temporal2Bottleneck, self).__init__()
        self.net = net
        assert isinstance(net, torchvision.models.resnet.Bottleneck)
        self.n_segment = n_segment
        self.tlam = DPTAM(in_channels=net.conv1.out_channels,
                       n_segment=n_segment,
                       kernel_size=t_kernel_size,
                       stride=t_stride,
                       padding=t_padding)

    def forward(self, x):
        identity = x

        out = self.net.conv1(x)
        out = self.net.bn1(out)
        out = self.net.relu(out)
        out = self.tlam(out)

        out = self.net.conv2(out)
        out = self.net.bn2(out)
        out = self.net.relu(out)


        out = self.net.conv3(out)
        out = self.net.bn3(out)

        if self.net.downsample is not None:
            identity = self.net.downsample(x)

        out += identity
        out = self.net.relu(out)

        return out

def make_temporal_c_modeling(net,
                           n_segment=8,
                           t_kernel_size=3,
                           t_stride=1,
                           t_padding=1):
    if isinstance(net, torchvision.models.ResNet):
        n_round = 1

        def make_block_c_temporal(stage,
                                this_segment,
                                t_kernel_size=3,
                                t_stride=1,
                                t_padding=1):
            blocks = list(stage.children())
            print('=> Processing this stage with {} blocks residual'.format(
                len(blocks)))
            for i, b in enumerate(blocks):
                # if i >= len(blocks)//2:
                if i % n_round == 0:
                    blocks[i] = Temporal2Bottleneck(b, this_segment,
                                                   t_kernel_size, t_stride,
                                                   t_padding)
            return nn.Sequential(*blocks)

        net.layer1 = make_block_c_temporal(net.layer1, n_segment, t_kernel_size,
                                         t_stride, t_padding)
        net.layer2 = make_block_c_temporal(net.layer2, n_segment, t_kernel_size,
                                         t_stride, t_padding)
        net.layer3 = make_block_c_temporal(net.layer3, n_segment, t_kernel_size,
                                         t_stride, t_padding)
        net.layer4 = make_block_c_temporal(net.layer4, n_segment, t_kernel_size,
                                         t_stride, t_padding)


if __name__ == '__main__':
    # test
    pass
