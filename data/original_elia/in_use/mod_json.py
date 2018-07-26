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
    "PV_AIEG_9_97",
    "PV_AIESH_9_13",
    "PV_IVEG_26_27",
    "PV_Ores_15_97",
    "PV_Rew_7_25",
    "PV_Sibelgas_21_34",
    "PV_Tacteo_13_80",
    "wind_onshore_elia_158",
    "win_offshore_ellia_877"
]
raw_path_save = "/home/iso/PycharmProjects/vpp/data/original_elia/in_use/"
raw_path_load = "/home/iso/PycharmProjects/vpp/data/original_elia/"

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
        new = copy.deepcopy(org)
        new_time0 = extend_to*i*ratio
        for k in range(ratio):
            new[0] = new_time0 + k*extend_to
            new_list[ratio*i+k] = new

    new_list = new_list.tolist()

    path_save = raw_path_save + "min" + str(extend_to) + "/"
    new_path = path_save + name + ".json"

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    with open(new_path, 'w') as outfile:
        json.dump(new_list, outfile)

    print("Saving modified list to json (length: " + str(len(new_list)) + ", period: "+str(extend_to) +
          " min) to the file: \n" + new_path)
