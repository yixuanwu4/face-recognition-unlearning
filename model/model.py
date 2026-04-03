import torch.nn as nn
from torchvision import models

class FaceModel(nn.Module):
  def __init__(self, num_classes = 100):
    super().__init__()
    self.backbone = models.mobilenet_v2(pretrained = True)

    for p in self.backbone.parameters():
      p.requires_grad = False

      for p in self.backbone.features[-3:].parameters():
        p.requires_grad = True

      in_features = self.backbone.classifier[1].in_features
      self.backbone.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
      )
  
  def forward(self, x):
    return self.backbone(x)