from settings_4bus import *
import matplotlib.pyplot as plt
import json
import copy
import sys
from pypower.api import *
from pypower_mod.rundcopf_noprint import rundcopf
from pprint import pprint as pp
from time import gmtime, strftime
import os
import matplotlib.backends.backend_pdf
import pandas as pd
import time
import pickle
import csv
import numpy as np
import scipy.io as sio

path_save = '/home/iso/Desktop/vpp_some_results/2019_0216_0957_history_week1_oneshot_pri3_1_20/'



file_path = [
    '/home/iso/PycharmProjects/vpp/data/german_load_profiles/min5/g1_sept17.json',
    '/home/iso/PycharmProjects/vpp/data/german_load_profiles/min5/h0_sept17.json',
    '/home/iso/PycharmProjects/vpp/data/german_load_profiles/min5/g4_sept17.json',
    '/home/iso/PycharmProjects/vpp/data/german_load_profiles/min5/l1_sept17.json',
    '/home/iso/PycharmProjects/vpp/data/original_elia/in_use/min5/PV_AIEG_9_97.json',
    '/home/iso/PycharmProjects/vpp/data/original_elia/in_use/min5/PV_AIESH_9_13.json',
    '/home/iso/PycharmProjects/vpp/data/original_elia/in_use/min5/PV_IVEG_26_27.json',
    '/home/iso/PycharmProjects/vpp/data/original_elia/in_use/min5/PV_Ores_15_97.json',
    '/home/iso/PycharmProjects/vpp/data/original_elia/in_use/min5/PV_Rew_7_25.json',
    '/home/iso/PycharmProjects/vpp/data/original_elia/in_use/min5/PV_Sibelgas_21_34.json',
    '/home/iso/PycharmProjects/vpp/data/original_elia/in_use/min5/PV_Tacteo_13_80.json',
    '/home/iso/PycharmProjects/vpp/data/original_elia/in_use/min5/wind_offshore_elia_877.json',
    '/home/iso/PycharmProjects/vpp/data/original_elia/in_use/min5/wind_onshore_elia_158.json',

    '/home/iso/PycharmProjects/vpp/data/pvoutput_org/min5/Freakycat_20.100kW/Freakycat_20.100kW_20180618_20180701.json',
    '/home/iso/PycharmProjects/vpp/data/pvoutput_org/min5/GfB_mbH_-_Westnetz_GmbH_29.610kW/GfB_mbH_-_Westnetz_GmbH_29.610kW_20180618_20180701.json',
    '/home/iso/PycharmProjects/vpp/data/pvoutput_org/min5/HLB_Sunnyfarm_19.500kW/HLB_Sunnyfarm_19.500kW_20180618_20180701.json',
    '/home/iso/PycharmProjects/vpp/data/pvoutput_org/min5/Michiels_Wegberg_27.900kW/Michiels_Wegberg_27.900kW_20180618_20180701.json',
    '/home/iso/PycharmProjects/vpp/data/pvoutput_org/min5/PV-Anlage_dahoam_23.000kW/PV-Anlage_dahoam_23.000kW_20180618_20180701.json',
    '/home/iso/PycharmProjects/vpp/data/pvoutput_org/min5/race|result_92.750kW/race|result_92.750kW_20180618_20180701.json',
    '/home/iso/PycharmProjects/vpp/data/pvoutput_org/min5/SGjuk_12KW_29.000kW/SGjuk_12KW_29.000kW_20180618_20180701.json',
    '/home/iso/PycharmProjects/vpp/data/pvoutput_org/min5/WohnhausA1_20.240kW/WohnhausA1_20.240kW_20180618_20180701.json']

mat_names = ["g1", "h0", "g4", "l1", "aieg", "aiesh", "iveg", "ores", "rew", "sibelgas", "tecteo", "offshore", "onshore",
             "freak", "wetnetz", "sunnyfarm", "wegberg", "dahoam", "race", "sgjuk", "wohnhaus"]

adict = {}
for idx in range(0, len(file_path)):
    fp = file_path[idx]
    name = mat_names[idx]

    with open(fp, 'r') as f:
        arr = json.load(f)

    adict[name] = np.array(arr)

sio.savemat(path_save + 'all_profiles.mat', adict)

# mat_contents = sio.loadmat(path_save + 'g1mat.mat')
# print(mat_contents)

# sys.exit()

opf1_save_balcost_all = np.load(path_save + "opf1_save_balcost_all.npy")
opf1_save_genload_all = np.load(path_save + "opf1_save_genload_all.npy")
opf1_save_prices_all = np.load(path_save + "opf1_save_prices_all.npy")

vpp1exc = np.round(opf1_save_balcost_all[:, 0, 0], 2)
vpp1def = np.round(opf1_save_balcost_all[:, 0, 1], 2)

print(vpp1exc)
print(vpp1def)

vpp2exc = np.round(opf1_save_balcost_all[:, 1, 0], 2)
vpp2def = np.round(opf1_save_balcost_all[:, 1, 1], 2)

vpp3exc = np.round(opf1_save_balcost_all[:, 2, 0], 2)
vpp3def = np.round(opf1_save_balcost_all[:, 2, 1], 2)

vpp4exc = np.round(opf1_save_balcost_all[:, 3, 0], 2)
vpp4def = np.round(opf1_save_balcost_all[:, 3, 1], 2)

adict = {}
adict['vpp1exc'] = vpp1exc
adict['vpp1def'] = vpp1def
adict['vpp2exc'] = vpp2exc
adict['vpp2def'] = vpp2def
adict['vpp3exc'] = vpp3exc
adict['vpp3def'] = vpp3def
adict['vpp4exc'] = vpp4exc
adict['vpp4def'] = vpp4def

print(adict)
sys.exit()

sio.savemat('/home/iso/Desktop/vpp_some_results/2019_0216_0957_history_week1_oneshot_pri3_1_20/testmat.mat', adict)
