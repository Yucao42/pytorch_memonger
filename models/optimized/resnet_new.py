import torch.nn as nn
import math
import pdb, time, sys
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnet200', 'resnet1001']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(pretrained=False, **kwargs):
    model = PreActResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

# looks like the ResNet1001 was only defined for the cifar dataset
def resnet1001(pretrained=False, **kwargs):
    # the input data should be like 32 x 3 x 32 x 32
    model = PreActResNet(PreActBottleneck, [111, 111, 111], **kwargs)
    return model

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        #print(inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn2(out)
        out = self.relu3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class PreActResNet(nn.Module):

    def __init__(self, block, layers, dropout=0.5, num_classes=4):
        self.inplanes = 64
        super(PreActResNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpooling1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self._make_layer(block, 64, layers[0], stage=1)
        self._make_layer(block, 128, layers[1], stride=2, stage=2)
        self._make_layer(block, 256, layers[2], stride=2, stage=3)
        self._make_layer(block, 512, layers[3], stride=2, stage=4)
        #self.features.add_module('bn2', nn.BatchNorm2d(256 * block.expansion))
        #self.features.add_module('relu2', nn.ReLU(inplace=True))
        self.features.add_module('avgpool', nn.AvgPool2d(7, stride=1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, stage=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        #ayers = []
        #ayers.append(
        #   block(self.inplanes, planes, stride, downsample))
        #elf.inplanes = planes * block.expansion
        #or i in range(1, blocks):
        #   layers.append(
        #       block(self.inplanes, planes))
        #elf.features.add_module('layer%d' % stage, nn.Sequential(*layers))

        self.features.add_module(
            'layer%d__%d' % (stage, 0),
            block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.features.add_module(
                'layer%d__%d' % (stage, i),
                block(self.inplanes, planes))

    def forward(self, input_var, chunks=3):
        modules = [module for k, module in self._modules.items()][0]
        #print(modules)
        input_var = checkpoint_sequential(modules, chunks, input_var)
        #print(input_var.shape)
        input_var = input_var.view(input_var.size(0)//5, 5, -1).mean(1)
        input_var = self.fc(input_var)
        return input_var

def load_resnet(resnet_type='50', pretrained=False, dropout=0.0, **kwargs):
    if resnet_type == '18':
        block = BasicBlock
        layers = [2, 2, 2, 2]
    elif resnet_type == '34':
        block = BasicBlock
        layers = [3, 4, 6, 3]
    elif resnet_type == '50':
        block = Bottleneck
        layers = [3, 4, 6, 3]
    elif resnet_type == '101':
        block = Bottleneck
        layers = [3, 4, 23, 3]
    elif resnet_type == '152':
        block = Bottleneck
        layers = [3, 8, 36, 3]

    model = PreActResNet(block, layers, dropout=dropout, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet%s' % resnet_type])
        model_dict = list(model.modules())[1].state_dict()
        for k, v in model_dict.items():
            k_ = k.replace('__', '.')
            if k_ in state_dict.keys():
                v = state_dict[k_].data
                try:
                    model_dict[k].copy_(v)
                except:
                    print(k, k_, model_dict[k].shape, v.shape)
        #state_dict = {k.replace('__', '.'): v for k, v in state_dict.items() if k in model_dict}
        #model_dict.update(state_dict)
        #model.load_state_dict(model_dict)
    return model
