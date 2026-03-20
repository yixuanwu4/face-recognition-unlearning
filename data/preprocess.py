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

def prepare_data(n_users=100, save_dir="./lfw_processed"):
  lfw = fetch_lfw_people(min_faces_per_person=10, resize=1.0)

  # pick the front n_users
  user_ids = np.unique(lfw.target)[:n_users]
  print(user_ids[:5])

  # label mapping
  label_map = {old: new for old, new in enumerate(user_ids)}

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
    json.dumps(label_map, f)

  print("Data prepared!")

prepare_data()
