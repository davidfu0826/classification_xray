import torch
import torch.nn as nn

from torchvision.models import resnet18

from timm.models import efficientnet_b0

def get_efficientnetb0(num_classes, num_channels=3):

    model = efficientnet_b0(pretrained=True)
    if num_channels==1:
       model.conv_stem = nn.Conv2d(
        1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier = nn.Linear(
        in_features=model.classifier.in_features, 
        out_features=num_classes, 
        bias=True
        )
    #model = models.resnet18(pretrained=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def get_resnet18(num_classes):
    model = resnet18(pretrained=True)
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
