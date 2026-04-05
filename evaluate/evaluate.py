import torch
from data.dataset import FaceDataset
from torch.utils.data import DataLoader
from model.model import FaceModel

def evaluate(model_path, data_dir):
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

  model = FaceModel()
  model.load_state_dict(torch.load(model_path))
  model.to(device)
  model.eval()

  dataset = FaceDataset(data_dir)
  loader = DataLoader(dataset, batch_size=8)

  correct = 0
  total = 0

  with torch.no_grad():
    for x, y in loader:
      x, y = x.to(device), y.to(device)
      pred = model(x).argmax(1)
      correct += (pred == y).sum().item()
      total += y.size(0)
  
  print(f"{model_path} Accuracy: {correct/total:.3f}")