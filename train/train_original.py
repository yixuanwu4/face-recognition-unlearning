import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from data.dataset import FaceDataset
from model.model import FaceModel

def train():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # we want the model to contain the "to-be-forgotten" user first
  train_dataset = FaceDataset("./data/lfw_processed/train")
  forget_dataset = FaceDataset("./data/lfw_processed/forget") 
  dataset = ConcatDataset([train_dataset, forget_dataset])
  loader = DataLoader(dataset, batch_size=8, shuffle=True)

  model = FaceModel().to(device)

  optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=1e-3)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(10):
    correct = 0
    total = 0

    for x,y in loader:
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      
      out = model(x)
      loss=criterion(out, y)
      
      loss.backward()
      optimizer.step()

      pred = out.argmax(1)
      correct += (pred == y).sum().item()
      total += y.size(0)

    print(f"Epoch {epoch} Acc: {correct/total:.3f}")

  os.makedirs("./checkpoints", exist_ok = True)
  torch.save(model.state_dict(), "./checkpoints/original.pth")