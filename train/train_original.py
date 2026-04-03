import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import FaceDataset
from model.model import FaceModel

def train():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  dataset = FaceDataset("./data/lfw_processed/train")
  loader = DataLoader(dataset, batch_size=8, shuffle=True)

  model = FaceModel().to(device)

  optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=1e-3)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(10):
    correct = 0
    total = 0

    for x,y in loader:
      x, y = x.to(device), y.to(device)

      out = model(x)
      loss=criterion(out, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      pred = out.argmax(1)
      correct += (pred == y).sum().item()
      total += y.size(0)

    print(f"Epoch {epoch} Acc: {correct/total:.3f}")