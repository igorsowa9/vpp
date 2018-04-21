from pypower.api import *
from case5_vpp import case5_vpp
from rundcopf_noprint import rundcopf
import copy, sys, json
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    with open(path, 'r') as f:
        arr = json.load(f)
    return arr

timesteps = 24*4
fload_mod = 0.25

gen2_pvReW = load_data('/home/iso/PycharmProjects/vpp/data/original_elia/in_use/PV_ReW_7_25.json')
gen3_wShore = load_data('/home/iso/PycharmProjects/vpp/data/original_elia/in_use/wind_onshore_elia_158.json')
gen4_pvAIESH = load_data('/home/iso/PycharmProjects/vpp/data/original_elia/in_use/PV_AIESH_9_13.json')

result_opf = np.zeros((timesteps, 3))  # pb, exc, obj
result_gen2 = np.zeros((timesteps, 3))  # P, Pmax
result_gen3 = np.zeros((timesteps, 3))  # P, Pmax
result_gen4 = np.zeros((timesteps, 3))  # P, Pmax
result_gen5 = np.zeros((timesteps, 3))  # P, Pmax

slack_idx = 0
for t in range(0, timesteps):

    ppc0 = case5_vpp()
    ppc_t = copy.deepcopy(ppc0)

    ppc_t['bus'][:, 2] = ppc0['bus'][:, 2] * fload_mod
    gens0 = ppc_t['gen'][:, 8]

    # gen1 is a virtual one
    gen2 = round(gen2_pvReW[t][2] * ppc0['gen'][1, 8], 4)  # pv
    gen3 = round(gen3_wShore[t][2] * ppc0['gen'][2, 8], 4) * 5  # wind
    gen4 = round(gen4_pvAIESH[t][2] * ppc0['gen'][3, 8], 4)  # pv
    # gen5 is always max

    max_generation = [1e4, gen2, gen3, gen4, ppc_t['gen'][4, 8]]
    print(max_generation)
    ppc_t['gen'][:, 8] = max_generation
    # also update load and price!

    res = rundcopf(ppc_t, ppoption(VERBOSE=0))
    if round(res['gen'][slack_idx, 1], 4) > 0:  # there's a need for external resources (generation at slack >0) i.e. DEFICIT
        power_balance = round(-1 * res['gen'][slack_idx, 1], 1)  # from vpp perspective i.e. negative if deficit
        objf_noslackcost = round(res['f'] - res['gen'][slack_idx, 1] * ppc_t['gencost'][slack_idx][4], 1)
        max_excess = 0

    else:  # no need for external power - BALANCE or EXCESS
        power_balance = round(-1 * res['gen'][slack_idx, 1], 1)
        max_excess = round(sum(ppc_t['gen'][:, 8]) - ppc_t['gen'][slack_idx, 8] - (sum(res['gen'][:, 1])
                                                                                   - res['gen'][slack_idx, 1]), 1)
        objf_noslackcost = round(res['f'] - res['gen'][slack_idx, 1] * ppc_t['gencost'][slack_idx][4], 1)

    result_opf[t][0] = power_balance
    result_opf[t][1] = max_excess
    result_opf[t][2] = objf_noslackcost

    result_gen2[t][0] = res['gen'][1, 1]
    result_gen2[t][1] = gen2

    result_gen3[t][0] = res['gen'][2, 1]
    result_gen3[t][1] = gen3

    result_gen4[t][0] = res['gen'][3, 1]
    result_gen4[t][1] = gen4

    result_gen5[t][0] = res['gen'][4, 1]
    result_gen5[t][1] = 30

plt.figure(1)
pb = plt.plot(result_opf[:, 0])
plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
plt.ylabel('power balance in vpp0')
plt.axhline(0, color='black')

plt.figure(2)
pb = plt.plot(result_opf[:, 1])
plt.setp(pb, 'color', 'g', 'linewidth', 2.0)
plt.ylabel('excess in vpp0')
plt.axhline(0, color='black')

plt.figure(3)
pb = plt.plot(result_opf[:, 2])
plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
plt.ylabel('cost with no slack in vpp0')
plt.axhline(0, color='black')

plt.figure(4)
plt.subplot(221)
pb = plt.plot(result_gen2[:, 0])
plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
pb = plt.plot(result_gen2[:, 0])
plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
plt.ylabel('gen2')
plt.axhline(0, color='black')

plt.subplot(222)
pb = plt.plot(result_gen3[:, 0])
plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
pb = plt.plot(result_gen3[:, 0])
plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
plt.ylabel('gen3')
plt.axhline(0, color='black')

plt.subplot(223)
pb = plt.plot(result_gen4[:, 0])
plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
# pb = plt.plot(result_gen4[:, 0])
# plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
plt.ylabel('gen4')
plt.axhline(0, color='black')

plt.subplot(224)
pb = plt.plot(result_gen5[:, 0])
plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
pb = plt.plot(result_gen5[:, 0])
plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
plt.ylabel('gen5')
plt.axhline(0, color='black')

plt.show()

