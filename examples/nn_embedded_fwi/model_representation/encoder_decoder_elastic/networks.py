import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, negative_slope=0.1):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope)
        self.init_weights()

    def init_weights(self):
        # normal initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class UpsampleingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, negative_slope=0.1):
        super(UpsampleingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.init_weights()

    def init_weights(self):
        # normal initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                m.bias.detach().zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.up(x)
        return x

class FWINET(nn.Module):
    def __init__(self, shot_counts, nsamples, nrecs, min_filters=8):
        super(FWINET, self).__init__()
        
        self.vx_compress = nn.Conv2d(shot_counts, 1, 3, 1, 1)
        self.vz_compress = nn.Conv2d(shot_counts, 1, 3, 1, 1)

        ksize = 5
        pad = (ksize-1)//2

        self.encoder1 = EncoderBlock(2, min_filters, ksize, 1, pad)
        self.downsample1 = nn.MaxPool2d(2, 2)
        self.encoder2 = EncoderBlock(min_filters, min_filters*2, ksize, 1, pad)
        self.downsample2 = nn.MaxPool2d(2, 2)
        self.encoder3 = EncoderBlock(min_filters*2, min_filters*4, ksize, 1, pad)
        self.downsample3 = nn.MaxPool2d(2, 2)
        self.encoder4 = EncoderBlock(min_filters*4, min_filters*8, ksize, 1, pad)
        self.downsample4 = nn.MaxPool2d(2, 2)

        dim1 = nsamples//16
        dim2 = nrecs//16

        self.linear1 = nn.Linear(min_filters*8*dim1*dim2, 16)
        self.linear2 = nn.Linear(16, 16*32*4) # depends on the velocity size

        self.up_vp_1 = UpsampleingBlock(min_filters*8, min_filters*4, ksize, 1, pad)
        self.up_vp_2 = UpsampleingBlock(min_filters*4, min_filters*2, ksize, 1, pad)
        self.up_vp_3 = UpsampleingBlock(min_filters*2, min_filters, ksize, 1, pad)
        self.up_vp_final = UpsampleingBlock(min_filters, min_filters, 1, 1, 0)
        self.up_vp_out = nn.Conv2d(min_filters, 1, 1, 1, 0)

        self.up_vs_1 = UpsampleingBlock(min_filters*8, min_filters*4, ksize, 1, pad)
        self.up_vs_2 = UpsampleingBlock(min_filters*4, min_filters*2, ksize, 1, pad)
        self.up_vs_3 = UpsampleingBlock(min_filters*2, min_filters, ksize, 1, pad)
        self.up_vs_final = UpsampleingBlock(min_filters, min_filters, 1, 1, 0)
        self.up_vs_out = nn.Conv2d(min_filters, 1, 1, 1, 0)

        self.up_rho_1 = UpsampleingBlock(min_filters*8, min_filters*4, 3, 1, 1)
        self.up_rho_2 = UpsampleingBlock(min_filters*4, min_filters*2, 3, 1, 1)
        self.up_rho_3 = UpsampleingBlock(min_filters*2, min_filters, 3, 1, 1)
        self.up_rho_final = UpsampleingBlock(min_filters, min_filters, 1, 1, 0)
        self.up_rho_out = nn.Conv2d(min_filters, 1, 1, 1, 0)

        self.init_linear_weights()

    def init_linear_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                m.bias.detach().zero_()

    def forward(self, vx, vz):

        vx = self.vx_compress(vx)
        vz = self.vz_compress(vz)

        x = torch.cat((vx, vz), dim=1)

        x = self.encoder1(x)
        x = self.downsample1(x)
        x = self.encoder2(x)
        x = self.downsample2(x)
        x = self.encoder3(x)
        x = self.downsample3(x)
        x = self.encoder4(x)
        x = self.downsample4(x)

        x = x.view(1, x.size(1)*x.size(2)*x.size(3))
        x = self.linear1(x)
        x = self.linear2(x)

        x = x.view(1, 64, 4, 8)

        vp = self.up_vp_1(x)
        vp = self.up_vp_2(vp)
        vp = self.up_vp_3(vp)
        vp = self.up_vp_final(vp)
        vp = self.up_vp_out(vp)

        vs = self.up_vs_1(x)
        vs = self.up_vs_2(vs)
        vs = self.up_vs_3(vs)
        vs = self.up_vs_final(vs)
        vs = self.up_vs_out(vs)

        rho = self.up_rho_1(x)
        rho = self.up_rho_2(rho)
        rho = self.up_rho_3(rho)
        rho = self.up_rho_final(rho)
        rho = self.up_rho_out(rho)

        return vp.squeeze(), vs.squeeze(), rho.squeeze()