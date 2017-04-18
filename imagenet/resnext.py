import torch
import torch.nn as nn
import math
import copy
#import torch.utils.model_zoo as model_zoo


__all__ = ['ResNeXt', 'resnext50', 'resnext101',
           'resnext152']

"""
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
"""


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3Group(in_planes, out_planes, groups=32, stride=1):
    "3x3 group convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
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

class NeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, finer = 1):
        super(NeXtBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, groups=32 * finer, stride=stride, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
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
    
    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.logsoftmax = nn.LogSoftMax()
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.logsoftmax(x)

        return x


class ResNeXt(nn.Module):

    def __init__(self, block, layers, verticalfrac=False, fracparam=2, wider = 1, finer = 1,  num_classes=1000, \
                multiway = 0, L1mode = False):
        self.inplanes = 64
        self.verticalfrac = verticalfrac
        self.fracparam = fracparam
        self.multiway = multiway
        self.L1mode = L1mode
        
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #if self.verticalfrac==False:
        self.layer1 = self._make_layer(block, wider * 128, layers[0], finer = finer)
        self.layer2 = self._make_layer(block, wider * 256, layers[1], stride=2, finer = finer)
        self.layer3 = self._make_layer(block, wider * 512, layers[2], stride=2, finer = finer)
        self.layer4 = self._make_layer(block, wider * 1024, layers[3], stride=2, finer = finer)
        
        if self.verticalfrac == True:
            for bigidx, bigblock in enumerate([self.layer1,self.layer2,self.layer3,self.layer4]):
                for idx, layer in enumerate(bigblock):
                    exec('self.layer_{bigidx}_{idx} = layer'.format(bigidx=bigidx, idx=idx))
                
        self.avgpool = nn.AvgPool2d(7)
        if multiway <= 0:
            self.fc = nn.Linear(wider * 1024 * block.expansion, num_classes)
        else:
            for i in range(multiway):
                exec('self.fc_{0} = nn.Linear(wider*1024*block.expansion, num_classes)'.format(i))

        if L1mode == False:
            self.sm = nn.LogSoftmax()
        else:
            self.sm = nn.Softmax()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, finer=1):
        '''
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        '''
        
        downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, finer))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, finer=finer))

        if self.verticalfrac == False:
            return nn.Sequential(*layers)
        else:
            return copy.copy(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.verticalfrac == False:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        else:
            for bigidx, bigblock in enumerate([self.layer1,self.layer2,self.layer3,self.layer4]):
                #blockinit = x
                for idx, layer in enumerate(bigblock):
                    tmpjump = 1
                    if idx == 0:
                        #print 'out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(x)'\
                        #     .format(bigidx=bigidx,idx=idx)
                        exec('out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(x)'\
                             .format(bigidx=bigidx,idx=idx))
                    else:
                        #print 'out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(out_{bigidx}_{idxm})'\
                        #     .format(bigidx=bigidx,idx=idx,idxm=idx-tmpjump)
                        exec('out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(out_{bigidx}_{idxm})'\
                             .format(bigidx=bigidx,idx=idx,idxm=idx-tmpjump))
                        tmpjump = tmpjump * self.fracparam
                        while idx - tmpjump > 0:
                        #    print 'out_{bigidx}_{idx} = out_{bigidx}_{idx} + out_{bigidx}_{idxm}'\
                        #         .format(bigidx=bigidx,idx=idx,idxm=idx-tmpjump)
                            exec('out_{bigidx}_{idx} = out_{bigidx}_{idx} + out_{bigidx}_{idxm}'\
                                 .format(bigidx=bigidx,idx=idx,idxm=idx-tmpjump))
                            tmpjump = tmpjump * self.fracparam
                #print 'x = out_{bigidx}_{idx}'.format(bigidx=bigidx,idx=len(bigblock)-1)
                exec('x = out_{bigidx}_{idx}'.format(bigidx=bigidx,idx=len(bigblock)-1))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.multiway <=0:
            x = self.fc(x)
            x = self.sm(x)
        else:
            xtmp = self.fc_0(x) * ( 1.0 / self.multiway)
            for i in range(1, self.multiway):
                exec('xtmp = xtmp + self.sm(self.fc_{0}(x) * ( 1.0 / self.multiway))')
            x = xtmp
            
        return x
    
class ResNeXt_HGS(nn.Module):

    def __init__(self, block, layers, verticalfrac=False, verticalHGS = False, fracparam = 2, wider = 1, finer = 1,  num_classes=1000):
        self.inplanes = 64
        self.verticalfrac = verticalfrac
        self.verticalHGS = verticalHGS
        self.fracparam = fracparam
        
        super(ResNeXt_HGS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #if self.verticalfrac==False:
        self.layer1 = self._make_layer(block, wider * 128, layers[0], finer = finer)
        self.layer2 = self._make_layer(block, wider * 256, layers[1], stride=2, finer = finer)
        self.layer3 = self._make_layer(block, wider * 512, layers[2], stride=2, finer = finer)
        self.layer4 = self._make_layer(block, wider * 1024, layers[3], stride=2, finer = finer)
        
        self.layeridx = {}
        self.invlayeridx = {}
        self.invlayergroupind = {}
        layeridx = 0
        if self.verticalfrac == True:
            for bigidx, bigblock in enumerate([self.layer1,self.layer2,self.layer3,self.layer4]):
                for idx, layer in enumerate(bigblock):
                    exec('self.layer_{bigidx}_{idx} = layer'.format(bigidx=bigidx, idx=idx))
                    self.layeridx[layeridx] = (bigidx,idx)
                    self.invlayeridx[(bigidx,idx)] = layeridx
                    layeridx += 1
                
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(wider * 1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, finer=1):
        '''
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        '''
        
        downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, finer))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, finer=finer))

        if self.verticalfrac == False:
            return nn.Sequential(*layers)
        else:
            return copy.copy(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        #print 'forwarding'
        
        if self.verticalfrac == False:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        elif self.verticalHGS == False:
            for bigidx, bigblock in enumerate([self.layer1,self.layer2,self.layer3,self.layer4]):
                #blockinit = x
                for idx, layer in enumerate(bigblock):
                    tmpjump = 1
                    if idx == 0:
                        #print 'out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(x)'\
                        #     .format(bigidx=bigidx,idx=idx)
                        exec('out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(x)'\
                             .format(bigidx=bigidx,idx=idx))
                    else:
                        #print 'out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(out_{bigidx}_{idxm})'\
                        #     .format(bigidx=bigidx,idx=idx,idxm=idx-tmpjump)
                        exec('out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(out_{bigidx}_{idxm})'\
                             .format(bigidx=bigidx,idx=idx,idxm=idx-tmpjump))
                        tmpjump = tmpjump * self.fracparam
                        while idx - tmpjump > 0:
                        #    print 'out_{bigidx}_{idx} = out_{bigidx}_{idx} + out_{bigidx}_{idxm}'\
                        #         .format(bigidx=bigidx,idx=idx,idxm=idx-tmpjump)
                            exec('out_{bigidx}_{idx} = out_{bigidx}_{idx} + out_{bigidx}_{idxm}'\
                                 .format(bigidx=bigidx,idx=idx,idxm=idx-tmpjump))
                            tmpjump = tmpjump * self.fracparam
                #print 'x = out_{bigidx}_{idx}'.format(bigidx=bigidx,idx=len(bigblock)-1)
                exec('x = out_{bigidx}_{idx}'.format(bigidx=bigidx,idx=len(bigblock)-1))
        else:
            for bigidx, bigblock in enumerate([self.layer1,self.layer2,self.layer3,self.layer4]):
                for idx, layer in enumerate(bigblock):
                    tmpjump = 1
                    if bigidx == 0 and idx == 0:
                        #print 'out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(x)'\
                        #     .format(bigidx=bigidx,idx=idx)
                        exec('out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(x)'\
                             .format(bigidx=bigidx,idx=idx))
                    else:
                        totallayeridx = self.invlayeridx[(bigidx,idx)]
                        lastbigidx,lastidx = self.layeridx[totallayeridx-1]
                        #print 'out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(out_{lastbigidx}_{lastidx})'\
                        #     .format(bigidx=bigidx,idx=idx,lastbigidx=lastbigidx,lastidx=lastidx)
                        exec('out_{bigidx}_{idx} = self.layer_{bigidx}_{idx}(out_{lastbigidx}_{lastidx})'\
                             .format(bigidx=bigidx,idx=idx,lastbigidx=lastbigidx,lastidx=lastidx))
                        tmpjump = tmpjump * self.fracparam
                        while totallayeridx - tmpjump >= 0:
                            #print "totallayeridx=",totallayeridx,"tmpjump=", tmpjump
                            lastbigidx , lastidx = self.layeridx[totallayeridx - tmpjump]
                            npool = bigidx - lastbigidx
                            #print "npool=", npool
                            s = 'out_{lastbigidx}_{lastidx}'.format(lastbigidx=lastbigidx,lastidx=lastidx)
                            for _ in range(npool):
                                s = 'self.avgpool2({0})'.format(s)
                                s = 'torch.cat([{s},{s}],1)'.format(s=s)
                                #print s
                            #print 'out_{bigidx}_{idx} = out_{bigidx}_{idx} + {s}'.format(bigidx=bigidx,idx=idx,s=s)
                            exec('out_{bigidx}_{idx} = out_{bigidx}_{idx} + {s}'.format(bigidx=bigidx,idx=idx,s=s))
                            tmpjump = tmpjump * self.fracparam
                            #print "self.fracparam=",self.fracparam, "updatedtmpjump=", tmpjump
            #print 'x = out_{bigidx}_{idx}'.format(bigidx=3,idx=len(self.layer4)-1)
            exec('x = out_{bigidx}_{idx}'.format(bigidx=3,idx=len(self.layer4)-1))
     
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model



def resnext26(pretrained=False, **kwargs):
    """Constructs a ResNeXt-26 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [2, 2, 2, 2], **kwargs)
    
    return model



def resnext38(pretrained=False, **kwargs):
    """Constructs a ResNeXt-38 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [3, 3, 3, 3], **kwargs)
    
    return model



def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    
    return model

def resnext50(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [3, 4, 6, 3], **kwargs)

    return model

def resnext50my(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [3, 4, 6, 3], multiway = 10, **kwargs)
    
    return model

def resnext50myL1(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [3, 4, 6, 3], multiway = 10, L1mode=True, **kwargs)
    
    return model

def resnext50L1(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [3, 4, 6, 3], L1mode = True, **kwargs)
    
    return model



def resnext50v(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [3, 4, 6, 3], verticalfrac=True, **kwargs)

    return model


def resnext50hgs(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt_HGS(NeXtBottleneck, [3, 4, 6, 3], verticalfrac=True, verticalHGS=True,  **kwargs)
    
    return model


def resnext50x2(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [3, 4, 6, 3], wider=2, **kwargs)

    return model

def resnext50x2f(pretrained=False, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [3, 4, 6, 3], wider=2, finer=2,  **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model




def resnext101(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnext101v(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [3, 4, 23, 3], verticalfrac=True,  **kwargs)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



def resnext152(pretrained=False, **kwargs):
    """Constructs a ResNeXt-151 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt(NeXtBottleneck, [3, 8, 36, 3], **kwargs)

    return model

