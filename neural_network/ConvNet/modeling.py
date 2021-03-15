from .utils import IntermediateLayerGetter
from .backbone import resnet, resnetRGBD
import torch.nn as nn
import torch
import torch.nn.functional as F

def _convnet_resnet(backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        # aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        # aspp_dilate = [6, 12, 18]

    backbone1 = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    backbone2 = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    # low_level_planes = 256

    fuselayers = nn.Sequential(
            # nn.Conv2d(inplanes*2, inplanes*2, 3, padding=1, bias=False),
            # nn.BatchNorm2d(inplanes*2),
            # nn.ReLU(inplace=True),
            nn.Conv2d(inplanes*2, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
        )
    
    return_layers = {'layer4': 'out'}
    classifier = ConvNetHead(inplanes, num_classes)
    
    backbone1 = IntermediateLayerGetter(backbone1, return_layers=return_layers)
    backbone2 = IntermediateLayerGetter(backbone2, return_layers=return_layers)

    model = _MySegmentationModel(backbone1, backbone2, fuselayers, classifier)
    return model

class ConvNetHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ConvNetHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 128, 1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class _MySegmentationModel(nn.Module):
    def __init__(self, backbone1, backbone2, fuselayers, classifier):
        super(_MySegmentationModel, self).__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.fuselayers = fuselayers
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        x1 = x[:, :3, ...]
        x2 = x[:, 3:, ...]
        features1 = self.backbone1(x1)
        features2 = self.backbone2(x2)
        # print('feature1:', features1.keys())
        features = {}

        features['out'] = torch.cat([features1['out'], features2['out']], dim=1)
        features['out'] = self.fuselayers(features['out'])
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

def _load_model(backbone, num_classes, output_stride, pretrained_backbone):

    # if backbone=='mobilenetv2':
    #     model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    # elif backbone.startswith('resnet'):
    #     model = _segm_resnetRGBD(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    # else:
    #     raise NotImplementedError
    if backbone.startswith('resnet'):
        model = _convnet_resnet(backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    
    return model

def convnet_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
