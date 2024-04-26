import numpy as np
import pickle
import os 
import cv2

from tqdm import tqdm
import sys

root_dir = "/esat/garnet/thess/data/miniimgnet"
final_name = "miniimgnet.pkl"

data_dirs = []

for dir in os.listdir(root_dir):
    #print(root_dir+"/"+dir)
    dir = root_dir+"/"+dir
    if os.path.isdir(dir):
        #print(dir)
        for sub_dir in os.listdir(dir):
            sub_dir = dir+"/"+sub_dir
            if os.path.isdir(sub_dir):
                data_dirs.append(sub_dir+"/")

print("data_dirs")
print(len(data_dirs))

dataset = {"data":[], "labels":[]}
for i, dir in tqdm(enumerate(data_dirs)):
    sample_paths = os.listdir(dir)
    for sample in sample_paths:
        sample = dir+sample
        if os.path.isfile(sample):
            img = cv2.imread(sample)
            dataset["data"].append(img)
            dataset["labels"].append(i)

print("Storing data")
with open(root_dir+"/"+final_name, 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Storing done...", root_dir+"/"+final_name)