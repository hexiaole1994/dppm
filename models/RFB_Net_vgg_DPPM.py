import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4


        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class RFBNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.base = nn.ModuleList(base)
        # conv_4
        self.Norm = BasicRFB_a(512,512,stride = 1,scale=1.0)
        self.extras = nn.ModuleList(extras)
        self.conv00 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv10 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv20 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv01 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv11 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv21 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv02 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv12 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv22 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv03 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv13 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv23 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv04 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv14 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv24 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv05 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv15 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(256, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv25 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(128, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv30 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv31 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv32 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv33 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv34 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.conv35 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3,
                                              padding=1, groups=1, bias=False),
                                    nn.BatchNorm2d(64, eps=1e-5, momentum=0.01, affine=True)
                                    )
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        lo = list()
        co = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = self.Norm(x)
        s0 = self.conv00(s)
        s1 = self.conv01(s)
        s2 = self.conv02(s)
        s3 = self.conv03(s)
        s4 = self.conv04(s)
        s5 = self.conv05(s)
        sources.append(s0)
        sources.append(s1)
        sources.append(s2)
        sources.append(s3)
        sources.append(s4)
        sources.append(s5)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k%2 ==0:
                if k==0:
                    s6 = self.conv10(x)
                    s7 = self.conv11(x)
                    s8 = self.conv12(x)
                    s9 = self.conv13(x)
                    s10 = self.conv14(x)
                    s11 = self.conv15(x)
                    sources.append(s6)
                    sources.append(s7)
                    sources.append(s8)
                    sources.append(s9)
                    sources.append(s10)
                    sources.append(s11)
                elif k==1:
                        s12 = self.conv20(x)
                        s13 = self.conv21(x)
                        s14 = self.conv22(x)
                        s15 = self.conv23(x)
                        s16 = self.conv24(x)
                        s17 = self.conv25(x)
                        sources.append(s12)
                        sources.append(s13)
                        sources.append(s14)
                        sources.append(s15)
                        sources.append(s16)
                        sources.append(s17)
                elif k==2:
                    s18 = self.conv30(x)
                    s19 = self.conv31(x)
                    s20 = self.conv32(x)
                    s21 = self.conv33(x)
                    s22 = self.conv34(x)
                    s23 = self.conv35(x)
                    sources.append(s18)
                    sources.append(s19)
                    sources.append(s20)
                    sources.append(s21)
                    sources.append(s22)
                    sources.append(s23)
                elif k==4 or k==6:
                    sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        lo.append(torch.cat([l for l in loc[:6]], 3))
        lo.append(torch.cat([l for l in loc[6:12]], 3))
        lo.append(torch.cat([l for l in loc[12:18]], 3))
        lo.append(torch.cat([l for l in loc[18:24]], 3))
        lo.append(loc[24])
        lo.append(loc[25])
        co.append(torch.cat([c for c in conf[:6]], 3))
        co.append(torch.cat([c for c in conf[6:12]], 3))
        co.append(torch.cat([c for c in conf[12:18]], 3))
        co.append(torch.cat([c for c in conf[18:24]], 3))
        co.append(conf[24])
        co.append(conf[25])

        #print([o.size() for o in loc])


        loc = torch.cat([o.view(o.size(0), -1) for o in lo], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in co], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def add_extras(size, cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=1)]
                else:
                    layers += [BasicRFB(in_channels, cfg[k+1], stride=2, scale = 1.0, visual=2)]
            else:
                layers += [BasicRFB(in_channels, v, scale = 1.0, visual=2)]
        in_channels = v
    if size == 512:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=4,stride=1,padding=1)]
    elif size ==300:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers

extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
}


def multibox(vgg, extra_layers, num_classes):
    loc_layers = []
    conf_layers = []
    for _ in range(6):
        loc_layers += [nn.Conv2d(128,
                                4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(128,
                                num_classes, kernel_size=3, padding=1)]
    for _ in range(6):
        loc_layers += [nn.Conv2d(256,
                                4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256,
                                num_classes, kernel_size=3, padding=1)]

    for _ in range(6):
        loc_layers += [nn.Conv2d(128,
                                4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(128,
                                num_classes, kernel_size=3, padding=1)]

    for _ in range(6):
        loc_layers += [nn.Conv2d(64,
                                4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(64,
                                num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(256,
                                4*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256,
                                4*num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(256,
                                4*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256,
                                4*num_classes, kernel_size=3, padding=1)]


    return vgg, extra_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return RFBNet(phase, size, *multibox( vgg(base[str(size)], 3),
                                add_extras(size, extras[str(size)], 1024),
                                 num_classes), num_classes)
