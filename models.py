import os
import torch
from torchvision import models
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
      super(Identity,self).__init__()
    def forward(self,x):
      return x
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
# ***************** ResNet18 **************************
   
resnet = models.resnet18(pretrained=True)
# resnet_model.avgpool = Identity()
resnet.fc = nn.Sequential(
nn.Linear(512, 128),
nn.ReLU(),
nn.Dropout(p=0.5),
nn.Linear(128,12))

resnet = resnet.to(device)



# ***************** GoogLeNet **************************

googlenet = models.googlenet(pretrained=True)
# googlenet_model.avgpool = Identity()
googlenet.fc = nn.Sequential(
nn.Linear(1024, 512),
nn.ReLU(),
nn.Dropout(p=0.5),
nn.Linear(512,12))

googlenet = googlenet.to(device)
