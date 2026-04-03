from data.preprocess import prepare_data
from train.train_original import train

if __name__ == "__main__":
  prepare_data()
  train()