import torch
import torch.nn as nn

from torchvision.models import resnet18

import timm
from timm.models import efficientnet_b0

def get_efficientnetb0(num_classes, num_channels=3):

    model = efficientnet_b0(pretrained=True, in_chans=num_channels)
    return model

def get_resnet18(num_classes, num_channels=3):
    model = resnet18(pretrained=True, in_chans=num_channels)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def get_ghostnet(num_classes):
    model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
    model.classifier = nn.Linear(
        in_features=model.classifier.in_features, 
        out_features=num_classes, 
        bias=True
        )
    return model


class CSPResNet50(nn.Module):

    def __init__(self, num_classes, num_channels=3):
        super(CSPResNet50, self).__init__()

        self.model = timm.create_model("cspresnet50", in_chans=num_channels)
        self.model.head.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    
    def forward(self, x):
        return self.model(x)

    def unfreeze_new_layers(self): 
        for layer in self.model.head.fc.parameters():
            layer.requires_grad = True

    def freeze_all_layers(self): 
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all_layers(self): 
        for param in self.model.parameters():
            param.requires_grad = True

class EfficientNetB8(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super(EfficientNetB8, self).__init__()

        self.model = timm.create_model("tf_efficientnet_b8", pretrained=True, in_chans=num_channels)
        self.model.classifier = nn.Linear(in_features=2816, out_features=num_classes, bias=True)
    
    def forward(self, x):
        return self.model(x)

def get_model(model_name, num_classes, input_channels, pretrained_path=None):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=input_channels)
    
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)
        
    return model


