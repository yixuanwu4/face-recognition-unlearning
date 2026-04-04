import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from data.dataset import FaceDataset
from model.model import FaceModel

def unlearn(noise_std=0.05):
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

  model = FaceModel()
  model.load_state_dict(torch.load("./checkpoints/original.pth"))
  model.to(device)

  train_dataset = FaceDataset("./data/lfw_processed/train")
  forget_dataset = FaceDataset("./data/lfw_processed/forget") 
  dataset = ConcatDataset([train_dataset, forget_dataset])
  loader = DataLoader(dataset, batch_size=8, shuffle=True)

  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(5):
    for x, y in loader:
      x, y = x.to(device), y.to(device)

      optimizer.zero_grad()
      out = model(x)
      loss = criterion(out, y)
      loss.backward()

      # core step: gradient clipping
      nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      
      # add noise
      for p in model.parameters():
        if p.grad is not None:
          noise = torch.randn_like(p.grad) * noise_std
          p.grad += noise

      optimizer.step()

  os.makedirs("./checkpoints", exist_ok=True)
  torch.save(model.state_dict(), "./checkpoints/unlearned.pth")


