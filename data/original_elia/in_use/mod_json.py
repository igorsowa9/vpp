import sys
import json
from pprint import pprint as pp
import copy
import numpy as np


def load_jsonfile(path):
    with open(path, 'r') as f:
        arr = json.load(f)
    return arr


names = ["PV_AIESH_9_13", "PV_ReW_7_25", "wind_offshore_Elia_877", "wind_onshore_elia_158"]
raw_path = "/home/iso/PycharmProjects/vpp/data/original_elia/in_use/"

for name in names:

    path = raw_path + name + ".json"
    org_list = load_jsonfile(path)

    extend = 15

    # new_list = np.empty([extend * len(org_list),3])
    # for i in range(len(org_list)):
    #     org = org_list[i]
    #     new_time0 = extend*i
    #     for x in range(extend):
    #         new_time = new_time0 + x
    #         new = copy.deepcopy(org)
    #         new[0] = new_time
    #         idx = extend*i+x
    #         new_list[idx] = new

    new_list = np.empty([len(org_list), 3])
    for i in range(len(org_list)):
        org = org_list[i]
        new_time0 = extend*i
        new = copy.deepcopy(org)
        new[0] = new_time0
        new_list[i] = new

    new_list = new_list.tolist()

    new_path = raw_path + name + "_ver_"+str(extend)+"min" + ".json"
    with open(new_path, 'w') as outfile:
        json.dump(new_list, outfile)

    print("Saving modified list to json (length: " + str(len(new_list)) + ", period: "+str(extend) +
          " min) to the file: \n" + new_path)
