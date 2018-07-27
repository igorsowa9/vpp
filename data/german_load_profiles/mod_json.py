"""
Modifies the data file to align time, originaly it aligned the time from 1,2,3 (as integers) i.e. every 15 minutes
into 0,5,10 i.e. every 5 minutes i.e. already in the minutes values

check input: names, raw_path_save, raw_path_load, extend_from (originally 15), extend_to (originally 5), regardless of
the input in the form 1,2,3,4,5...
"""


import sys
import json
from pprint import pprint as pp
import copy
import numpy as np
import os


def load_jsonfile(path):
    with open(path, 'r') as f:
        arr = json.load(f)
    return arr


names = [
    "l1_sept17",
    "g1_sept17",
    "g4_sept17",
    "h0_sept17"
]
raw_path_save = "/home/iso/PycharmProjects/vpp/data/german_load_profiles/min5/"
raw_path_load = "/home/iso/PycharmProjects/vpp/data/german_load_profiles/min15/"

for name in names:

    path = raw_path_load + name + ".json"
    org_list = load_jsonfile(path)

    extend_from = 15  # minutes
    extend_to = 5  # minutes

    ratio = int(extend_from/extend_to)
    new_list_len = len(org_list)*ratio
    new_list = np.zeros([new_list_len, 3])

    for i in range(len(org_list)):
        org = org_list[i]
        new = [0, org, org]
        new_time0 = extend_to*i*ratio
        for k in range(ratio):
            new[0] = new_time0 + k*extend_to
            new_list[ratio*i+k] = new

    new_list = new_list.tolist()

    new_path = raw_path_save + name + ".json"

    if not os.path.exists(raw_path_save):
        os.makedirs(raw_path_save)

    with open(new_path, 'w') as outfile:
        json.dump(new_list, outfile)

    print("Saving modified list to json (length: " + str(len(new_list)) + ", period: "+str(extend_to) +
          " min) to the file: \n" + new_path)
