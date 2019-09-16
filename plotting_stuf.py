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
import scipy.io as sio

# path_save = '/home/iso/Desktop/vpp_some_results/2019_0710_0820_price_decrease_at_2114/'
path_save = '/home/iso/Desktop/vpp_some_results/2019_0710_1539_test/'

opf1_save_balcost_all = np.load(path_save + "opf1_save_balcost_all.npy")
opf1_save_genload_all = np.load(path_save + "opf1_save_genload_all.npy")
opf1_save_prices_all = np.load(path_save + "opf1_save_prices_all.npy")
opfe3_save_costs_all = np.load(path_save + "opfe3_save_costs_all.npy")

VPP_MAXEXC = 0   # max excess value
VPP_PBAL = 1   # value to balance - deficit
OBJF = 2   # objective function value from opf1
OBJF_NODSO = 3  # objective function if no dso buying costs

LOAD_FIX = 0  # max excess value
GEN_RES = 1  # value to balance - deficit
GEN_UP = 2  # objective function value from opf1
GEN_LOW = 3  # objective function if no dso buying costs

print("###################")
print("### PRINTING ######")
print("###################")

figure_counter = 0

# figures for different VPPs
for alias in data_names:
    vpp_idx = data_names_dict[alias]
    figure_counter += 1

    plt.figure(figure_counter, figsize=(figsizeH, figsizeL))
    plt.suptitle(str(alias) + ': balance and costs')

    plt.subplot(411)
    plt.title('total generation (green) and total load (red) in ' + str(alias))
    pb = plt.plot(np.sum(opf1_save_genload_all[:, vpp_idx, 1:, GEN_RES], axis=1))
    plt.setp(pb, 'color', 'g', 'linewidth', 2.0)
    pb = plt.plot(np.sum(opf1_save_genload_all[:, vpp_idx, 1:, LOAD_FIX], axis=1))
    plt.setp(pb, 'color', 'r', 'linewidth', 2.0)

    plt.subplot(412)
    plt.title('to DSO export (green) and from DSO import (red), at PCC.')
    pb = plt.plot(opf1_save_genload_all[:, vpp_idx, 0, LOAD_FIX])
    plt.setp(pb, 'color', 'g', 'linewidth', 2.0)
    pb = plt.plot(opf1_save_genload_all[:, vpp_idx, 0, GEN_RES])
    plt.setp(pb, 'color', 'r', 'linewidth', 2.0)

    plt.ylabel('power value')
    plt.axhline(0, color='black')

    plt.subplot(413)
    plt.title('excess (green) and power to balance (price=Y) (blue).')
    pb = plt.plot(opf1_save_balcost_all[:, vpp_idx, VPP_MAXEXC])
    plt.setp(pb, 'color', 'g', 'linewidth', 2.0)
    pb = plt.plot(opf1_save_balcost_all[:, vpp_idx, VPP_PBAL])
    plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
    plt.ylabel('power value')
    plt.axhline(0, color='black')

    plt.subplot(414)
    plt.title('cost in vpp0 (def from DSO - red), cost in vpp0 (def. costs excl. - blue), incl. negotiation: yellow')
    pb = plt.plot(opf1_save_balcost_all[:, vpp_idx, OBJF])
    plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
    pb = plt.plot(opf1_save_balcost_all[:, vpp_idx, OBJF_NODSO])
    plt.setp(pb, 'color', 'b', 'linewidth', 2.0)

    ####### AFTER NEGOTIATION #############
    pb = plt.plot(opfe3_save_costs_all[:, vpp_idx])
    plt.setp(pb, 'color', 'gold', 'linewidth', 2.0)
    ####################

    plt.ylabel('cost value')
    plt.axhline(0, color='black')
    plt.xlabel('time in minutes')

    figure_counter += 1
    plt.figure(figure_counter, figsize=(figsizeH, figsizeL))
    plt.suptitle(str(alias) + ': generators/loads with constraints. \n'
                              'Bus 1 is slack with no internal gens and loads.')

    n_bus = len(opf1_save_genload_all[0, vpp_idx, :, GEN_RES])
    with open(data_paths[data_names_dict[alias]], 'r') as f:
        arr = json.load(f)
    n_bus_real = arr['bus_n']


    for g in range(0, n_bus_real):
        if g == 0:
            plt.subplot(int(n_bus_real * 100 + 11))
            plt.title('fixed loads at all buses excl. 0 ')
            pb = plt.plot(opf1_save_genload_all[:, vpp_idx, 1:, LOAD_FIX])
            plt.setp(pb, 'color', 'g', 'linewidth', 2.0)
            continue
        plt.subplot(int(n_bus_real * 100 + 10 + g + 1))
        fixed_price = opf1_save_prices_all[0, vpp_idx, g]
        plt.title('generation at bus: ' + str(g+1) + ' price: ' + str(fixed_price))
        pb = plt.plot(opf1_save_genload_all[:, vpp_idx, g, GEN_RES])
        plt.setp(pb, 'color', 'g', 'linewidth', 2.0)
        pb = plt.plot(opf1_save_genload_all[:, vpp_idx, g, GEN_UP])
        plt.setp(pb, 'color', 'r', 'linewidth', 1.0, dashes=[6, 2])
        pb = plt.plot(opf1_save_genload_all[:, vpp_idx, g, GEN_LOW])
        plt.setp(pb, 'color', 'b', 'linewidth', 1.0, dashes=[6, 2])

############################
# Other figures for analysis
############################
figure_counter += 1
plt.figure(figure_counter, figsize=(figsizeH, figsizeL))

m_vpp3 = pd.read_pickle(path_save + "temp_ln_" + str(data_names_dict['vpp3']) + ".pkl")
m_vpp2 = pd.read_pickle(path_save + "temp_ln_" + str(data_names_dict['vpp2']) + ".pkl")

# m_vpp2.to_csv(path_save + "_" + "view_vpp2" + ".csv")
# m_vpp3.to_csv(path_save + "_" + "view_vpp3" + ".csv")

p1 = np.array(m_vpp3['exc_cost_range'].tolist())
p2 = m_vpp2['marginal_price_novpp3'].tolist()
# p3 = np.array(m_vpp3['price'].tolist())
# p3 = np.array(m_vpp3['price'])
p3 = m_vpp3['pcf'].tolist()
# p3 = [np.array(p3[0]) for element in p3]
# p3 = np.array(p3).flatten()

x = np.array(m_vpp3['t'])

plt.title('vpp3 (excess - learning) vs vpp2 (deficit - dumb)')

plt.subplot(211)
plt.title('vpp3 costs generation excess (light and dark green) vs vpp2 marginal price in deals (no vpp3) (red) and proposed price (purple)')
plt.setp(plt.plot(x, p1[:, 0]), color='lime', linewidth=1.0)
plt.setp(plt.plot(x, p1[:, 1]), color='darkgreen', linewidth=1.0)
plt.setp(plt.plot(x, p2), color='red', linewidth=1.0)
plt.setp(plt.plot(x, p3), color='purple', linewidth=2.0)

plt.fill_between(x, p2, p1[:, 0], facecolor='lightgreen', interpolate=True)
plt.ylim(-1, 18)

plt.ylabel('prices')
plt.axhline(0, color='black')
plt.xlabel('timestamp but only if a request exists')

plt.subplot(212)
plt.title('bids saldo of vpp3 (excess - learning)')
p4 = plt.plot(m_vpp3['bids_saldo'].tolist())
plt.setp(p4, 'color', 'orange', 'linewidth', 2.0)

plt.ylabel('revenue')
plt.axhline(0, color='black')
plt.xlabel('timestamp but only if a request exists')

######################3 learning - not universal!!!!

figure_counter += 1
plt.figure(figure_counter, figsize=(figsizeH, figsizeL))

um_vpp3 = pd.read_pickle(path_save + "updated_memory_ln_" + str(data_names_dict['vpp3']) + ".pkl")

p1u = np.array(um_vpp3[['t', 'mp_belief']])
boundry_t = max(p1u[p1u[:, 0] < 2109, 0])

initial_beliefs_arr = p1u[np.where(p1u[:, 0] == boundry_t), 1]

initial_beliefs = initial_beliefs_arr[0][0]['mp_factor_avg']
pcfs_upper_h = initial_beliefs_arr[0][0]['pcfs_upper_h']

updated_part = p1u[np.where(p1u[:, 0] >= min(x)-1), :]
updated_part = updated_part[0]
# print(updated_part.shape)
# sys.exit()

# print(initial_beliefs)
# print(pcfs_upper_h)
# sys.exit()

n_h = len(pcfs_upper_h)
n_upd = len(updated_part[:, 0])  # 66
mp_factor_m = np.zeros((n_h, n_upd+1))
mp_factor_m[:, 0] = initial_beliefs
# print(mp_factor_m.shape)
# print(mp_factor_m)
#
# print(np.arange(0, n_upd))

for k in np.arange(0, n_upd):
    sett = updated_part[k]
    # print(mp_factor_m[:, k])
    mp_factor_m[:, k+1] = sett[1]['mp_factor_avg']

x_with_init = np.append([min(x)-1], x)

# rgb = [0, 0.2, 0.4, 0.6, 1.0]
print(mp_factor_m)
print(x_with_init)
adict = {}
adict['mp_factor_m'] = mp_factor_m
adict['x_with_init'] = x_with_init
sio.savemat(path_save + 'mp_factor.mat', adict)
sys.exit()

rgb = 0
for h in np.arange(len(pcfs_upper_h)):
    upper_h = pcfs_upper_h[h]
    plt.setp(plt.plot(x_with_init, mp_factor_m[h, :]), color=(rgb**2, 1-rgb, 1-rgb, rgb), linewidth=2.5)
    rgb = rgb + 1/len(pcfs_upper_h)

# plt.ylim(10, 18)
plt.ylabel('belief')
plt.axhline(0, color='black')
plt.xlabel('timestamp but only if a request exists')


############################

if pdf:
    pdf = matplotlib.backends.backend_pdf.PdfPages(path_save + 'all_figs_plotting_stuff.pdf')
    for fig in range(1, figure_counter + 1):
        pdf.savefig(fig)
    pdf.close()
else:
    plt.show()