from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class NoiseTranpose2d(nn.Module):
    def __init__(self, in_planes, out_planes, level):
        super(NoiseTranpose2d, self).__init__()

        self.noise = torch.randn(1,in_planes,1,1)
        self.level = level
        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x):
        tmp1 = x.data.shape
        tmp2 = self.noise.shape
        if tmp1[0] != tmp2[0]:
            if tmp1[0]<tmp2[0]:
                self.noise = self.noise[0:tmp1[0], :, :, :]
            else:
                expand_noise = torch.Tensor(tmp1[0], tmp2[1], tmp2[2], tmp2[3])
                expand_noise[:, :, :, :] = self.noise[0, 0, 0, 0]
                self.noise = expand_noise

        if (tmp1[1] != tmp2[1]) or (tmp1[2] != tmp2[2]) or (tmp1[3] != tmp2[3]):
            self.noise = (2*torch.rand(x.data.shape)-1)*self.level
            self.noise = self.noise.cuda()

        print (x.data.shape)
        print (self.noise.shape)
        x.data = x.data + self.noise
        x = self.layers(x)
        return x

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class P_Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(P_Generator, self).__init__()
        self.ngpu = ngpu
        self.noise_level = 0.2
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Upsample(scale_factor=2, mode='nearest'),
            NoiseTranpose2d(     nz, ngf * 8, 0.8),
            NoiseTranpose2d(     ngf * 8, ngf * 8, 0.8),
            # state size. (ngf*8) x 2 x 2
            nn.Upsample(scale_factor=2, mode='nearest'),
            NoiseTranpose2d(ngf * 8, ngf * 8, 0.6),
            NoiseTranpose2d(ngf * 8, ngf * 4, 0.6),
            # state size. (ngf*4) x 4 x 4
            nn.Upsample(scale_factor=2, mode='nearest'),
            NoiseTranpose2d(ngf * 4, ngf * 4, 0.3),
            NoiseTranpose2d(ngf * 4, ngf * 4, 0.3),
            # state size. (ngf*4) x 8 x 8
            nn.Upsample(scale_factor=2, mode='nearest'),
            NoiseTranpose2d(ngf * 4, ngf * 4, 0.2),
            NoiseTranpose2d(ngf * 4, ngf * 2, 0.2),
            # state size. (ngf*2) x 16 x 16
            nn.Upsample(scale_factor=2, mode='nearest'),
            NoiseTranpose2d(ngf * 2, ngf * 2, 0.2),
            NoiseTranpose2d(ngf * 2, ngf * 2, 0.2),
            NoiseTranpose2d(ngf * 2,     ngf, 0.2),
            # state size. (ngf) x 32 x 32
            nn.Upsample(scale_factor=2, mode='nearest'),
            NoiseTranpose2d(    ngf,      ngf, 0.2),
            NoiseTranpose2d(    ngf,      ngf, 0.2),
            NoiseTranpose2d(    ngf,      ngf, 0.2),
            NoiseTranpose2d(    ngf,      nc, 0.2),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu, nz, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class NoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, level):
        super(NoiseLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)
        #self.noise = torch.randn(1,in_planes,1,1)
        self.level = level
        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm2d(in_planes),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        tmp1 = x.data.shape
        tmp2 = self.noise.shape
        # if tmp1[0] != tmp2[0]:
        #     if tmp1[0]<tmp2[0]:
        #         self.noise = self.noise[0:tmp1[0], :, :, :]
        #     else:
        #         expand_noise = torch.Tensor(tmp1[0], tmp2[1], tmp2[2], tmp2[3])
        #         expand_noise[:, :, :, :] = self.noise[0, 0, 0, 0]
        #         self.noise = expand_noise

        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()
            self.noise = (2 * self.noise - 1) * self.level

        print ('input:',x.data.shape)
        print ('noise:',self.noise.shape)
        y = torch.add(x, self.noise)
        print ('afadd:',y.data.shape)
        z = self.layers(y)
        print ('outpt:',z.data.shape,'\n')
        return z

class NoiseBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2):
        super(NoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer(in_planes, planes, level),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            NoiseLayer(planes, planes, level),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class NoiseNoPoolBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2, isLastBN=True):
        super(NoiseNoPoolBlock, self).__init__()
        if isLastBN:
            self.layers = nn.Sequential(
                NoiseLayer(in_planes, planes, level),
                #nn.MaxPool2d(stride, stride),
                nn.BatchNorm2d(planes),
                nn.ReLU(True),
                NoiseLayer(planes, planes, level),
                nn.BatchNorm2d(planes),
            )
        else:
            self.layers = nn.Sequential(
                NoiseLayer(in_planes, planes, level),
                #nn.MaxPool2d(stride, stride),
                nn.BatchNorm2d(planes),
                nn.ReLU(True),
                NoiseLayer(planes, planes, level),
                #nn.BatchNorm2d(planes),
            )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class NoiseBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2):
        super(NoiseBottleneck, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            NoiseLayer(planes, planes, level),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class NoiseResNet(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level):
        super(NoiseResNet, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level) #(block type, output size, block num, level)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level)
        self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        x9 = self.sigmoid(x8.view(x8.size(0)))
        return x9

class NoiseResGenetator(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, level):
        super(NoiseResGenetator, self).__init__()
        self.in_planes = nfilters
        self.layer1 = self._make_layer(block, 8*nfilters, nblocks[0], level=level)
        self.layer2 = self._make_layer(block, 8*nfilters, nblocks[1], level=level)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], level=level)
        self.layer4 = self._make_layer(block, 4*nfilters, nblocks[3], level=level)
        self.layer5 = self._make_layer(block, 2*nfilters, nblocks[4], level=level)
        self.layer6 = self._make_layer(block, nfilters, nblocks[5], level=level)
        self.layer7 = self._make_layer(block, nchannels, nblocks[6], level=level, isLastBN=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.tanh = nn.Tanh()

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2, isLastBN=True):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            if i == nblocks and isLastBN == False:
                layers.append(block(self.in_planes, planes, level=level, isLastBN=False))
            else:
                layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = x # (64, nz,1,1)
        print ('x2')
        x2 = self.layer1(x1) # (64, 8*ngf, 1, 1)
        x3 = self.upsample(x2) # (64, 8*ngf, 2, 2)
        print ('x4')
        x4 = self.layer2(x3) # (64, 8*ngf, 2, 2)
        x5 = self.upsample(x4) # (64, 8*ngf, 4, 4)
        print ('x6')
        x6 = self.layer3(x5) # (64, 4*ngf, 4, 4)
        x7 = self.upsample(x6) # (64, 4*ngf, 8, 8)
        print ('x8')
        x8 = self.layer4(x7) # (64, 4*ngf, 8, 8)
        x9 = self.upsample(x8) # (64, 4*ngf, 16, 16)
        print ('x10')
        x10 = self.layer5(x9) # (64, 2*ngf, 16, 16)
        x11 = self.upsample(x10)# (64, 2*ngf, 32, 32)
        print ('x12')
        x12 = self.layer6(x11)# (64, ngf, 32, 32)
        x13 = self.upsample(x12)# (64, ngf, 64, 64)
        print ('x14')
        x14 = self.layer7(x13)# (64, nc, 64, 64)
        x15 = self.tanh(x14) # (64, nc, 64, 64)
        return x15

def noiseresnet18(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBasicBlock, [2,2,2,2], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresnet34(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBasicBlock, [3,4,6,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresnet101(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBottleneck, [3,4,23,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresnet152(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBottleneck, [3,8,36,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresgenerator26(nchannels, nfilters, nclasses, level=0.1):
    return NoiseResGenetator(NoiseNoPoolBlock, [2,2,2,2,2,2,2], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, level=level)

def noiseresgenerator76(nchannels, nfilters, nclasses, level=0.1):
    return NoiseResGenetator(NoiseNoPoolBlock, [3,4,6,8,10,12,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, level=level)

def noiseresgenerator101(nchannels, nfilters, nclasses, level=0.1):
    return NoiseResGenetator(NoiseNoPoolBlock, [3,4,23,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, level=level)

def noiseresgenerator152(nchannels, nfilters, nclasses, level=0.1):
    return NoiseResGenetator(NoiseNoPoolBlock, [3,8,36,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, level=level)
