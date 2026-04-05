from data.preprocess import prepare_data
from train.train_original import train
from unlearning.unlearn import unlearn
from evaluate.evaluate import evaluate

if __name__ == "__main__":
  prepare_data()
  train()
  unlearn()

  print("\n=== Evaluation ===")
  evaluate("./checkpoints/original.pth", "./data/lfw_processed/forget")
  evaluate("./checkpoints/original.pth", "./data/lfw_processed/test")

  evaluate("./checkpoints/unlearned.pth", "./data/lfw_processed/forget")
  evaluate("./checkpoints/unlearned.pth", "./data/lfw_processed/test")