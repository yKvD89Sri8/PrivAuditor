import os
import json
from datasets import load_dataset
import random

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return

def writer(data, save_dir, file_type):
    
    save_path = "/".join([save_dir, file_type + ".json"])
    
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


yelp_ds = load_dataset("yelp_review_full")
save_dir = "dataset/yelp_review_full"

yelp_train = yelp_ds["train"]
yelp_test = yelp_ds["test"]

create_dir(save_dir)

data_list = []
for sample in yelp_train:
    data_list.append({
        "instruction": "Please rate the review: ",
        "input": sample["text"],
        "output": sample["label"],
        "answer": sample["label"],
    })
random.shuffle(data_list)
writer(data_list[:-5000], save_dir, "train")    
writer(data_list[-5000:], save_dir, "val")

del data_list

data_list = []
for sample in yelp_test:
    data_list.append({
        "instruction": "Please rate the review: ",
        "input": sample["text"],
        "output": sample["label"],
        "answer": sample["label"],
    })
writer(data_list, save_dir, "test")

print("==============finish======================")