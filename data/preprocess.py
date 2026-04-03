"""
Goal:
Convert the original LFW dataset into the structure required for the project:
- train (training set)
- test (test set)
- forget (users to be forgotten)

The core premise of 'forgetting learning' is that:
the model must 'explicitly know who is to be removed'
"""
import os
import numpy as np
import cv2
import json
from sklearn.datasets import fetch_lfw_people

def prepare_data(n_users=100, save_dir="./data/lfw_processed", random_seed=11):
  np.random.seed(random_seed)
  
  lfw = fetch_lfw_people(min_faces_per_person=10, resize=1.0)

  all_user_ids = np.unique(lfw.target)
  np.random.shuffle(all_user_ids)
  # pick the front n_users
  user_ids = all_user_ids[:n_users]

  # label mapping
  label_map = {int(old): int(new) for new, old in enumerate(user_ids)}
  print("Label mapping (original ID -> new ID):")
  print(label_map)

  os.makedirs(save_dir, exist_ok=True)

  for split in ["train", "test", "forget"]:
    target_dir= save_dir + "/" + str(split)
    os.makedirs(target_dir, exist_ok=True)

  forget_user = user_ids[5] # can be altered

  for i, (img, target) in enumerate(zip(lfw.images, lfw.target)):
    if target not in user_ids:
      continue
    
    new_label = label_map[target]

    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (160, 160))

    if target == forget_user:
      path = f"{save_dir}/forget/{new_label}_{i}.jpg"
    elif i % 5 == 0:
      path = f"{save_dir}/test/{new_label}_{i}.jpg"
    else:
      path = f"{save_dir}/train/{new_label}_{i}.jpg"

    cv2.imwrite(path, img)

  # save label map
  with open(f"{save_dir}/label_map.json", "w") as f:
    # print(int(label_map))
    json.dump(label_map, f)

  print("Data prepared!")

# prepare_data()
