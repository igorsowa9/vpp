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

day = 3 # 1-30 of Sept 2017
timesteps = 24*4
load_mod = 1
wind_mod = 1

gen2_pvReW = load_data('/home/iso/PycharmProjects/vpp/data/original_elia/in_use/PV_ReW_7_25.json')
gen3_wShore = load_data('/home/iso/PycharmProjects/vpp/data/original_elia/in_use/wind_onshore_elia_158.json')
gen4_pvAIESH = load_data('/home/iso/PycharmProjects/vpp/data/original_elia/in_use/PV_AIESH_9_13.json')

load2_h0 = load_data('/home/iso/PycharmProjects/vpp/data/german_load_profiles/h0_sept17.json')
load3_g4 = load_data('/home/iso/PycharmProjects/vpp/data/german_load_profiles/g4_sept17.json')
load4_l1 = load_data('/home/iso/PycharmProjects/vpp/data/german_load_profiles/l1_sept17.json')
load5_g1 = load_data('/home/iso/PycharmProjects/vpp/data/german_load_profiles/g1_sept17.json')

result_opf = np.zeros((timesteps, 3))  # pb, exc, obj
result_gen2 = np.zeros((timesteps, 2))  # P, Pmax
result_gen3 = np.zeros((timesteps, 2))  # P, Pmax
result_gen4 = np.zeros((timesteps, 2))  # P, Pmax
result_gen5 = np.zeros((timesteps, 2))  # P, Pmax
result_total_gen = np.zeros(timesteps)

result_loads = np.zeros((timesteps, 4))  # load number,[]
result_total_load = np.zeros(timesteps)

slack_idx = 0
start_ts = (day-1)*4*24
for t_data in range(start_ts, start_ts + timesteps):
    t_save = t_data - start_ts
    ppc0 = case5_vpp()
    ppc_t = copy.deepcopy(ppc0)

    ppc_t['bus'][:, 2] = ppc0['bus'][:, 2] * load_mod
    gens0 = ppc_t['gen'][:, 8]

    # gen1 is a virtual one
    gen2 = round(gen2_pvReW[t_data][2] * ppc0['gen'][1, 8], 4)  # pv
    gen3 = round(gen3_wShore[t_data][2] * ppc0['gen'][2, 8], 4) * wind_mod  # wind
    gen4 = round(gen4_pvAIESH[t_data][2] * ppc0['gen'][3, 8], 4)  # pv
    gen5 = round(ppc0['gen'][4, 8], 4)  # gen5 is always at max

    max_generation_vector = [1e4, gen2, gen3, gen4, gen5]
    ppc_t['gen'][:, 8] = max_generation_vector

    # also update load and price!
    load2 = round(load2_h0[t_data] * ppc0['bus'][1, 2], 4)
    load3 = round(load3_g4[t_data] * ppc0['bus'][2, 2], 4)
    load4 = round(load4_l1[t_data] * ppc0['bus'][3, 2], 4)
    load5 = round(load5_g1[t_data] * ppc0['bus'][4, 2], 4)
    load_vector = [0, load2, load3, load4, load5]
    ppc_t['bus'][:, 2] = load_vector

    res = rundcopf(ppc_t, ppoption(VERBOSE=0))
    if round(res['gen'][slack_idx, 1], 4) > 0:  # there's a need for external resources (generation at slack >0) i.e. DEFICIT
        power_balance = round(-1 * res['gen'][slack_idx, 1], 4)  # from vpp perspective i.e. negative if deficit
        objf_noslackcost = round(res['f'] - res['gen'][slack_idx, 1] * ppc_t['gencost'][slack_idx][4], 4)
        max_excess = 0

    else:  # no need for external power - BALANCE or EXCESS
        power_balance = round(-1 * res['gen'][slack_idx, 1], 4)
        max_excess = round(sum(ppc_t['gen'][:, 8]) - ppc_t['gen'][slack_idx, 8] - (sum(res['gen'][:, 1])
                                                                                   - res['gen'][slack_idx, 1]), 4)
        objf_noslackcost = round(res['f'] - res['gen'][slack_idx, 1] * ppc_t['gencost'][slack_idx][4], 4)

    result_opf[t_save][0] = power_balance
    result_opf[t_save][1] = max_excess
    result_opf[t_save][2] = objf_noslackcost

    result_gen2[t_save][0] = res['gen'][1, 1]
    result_gen2[t_save][1] = gen2

    result_gen3[t_save][0] = res['gen'][2, 1]
    result_gen3[t_save][1] = gen3

    result_gen4[t_save][0] = res['gen'][3, 1]
    result_gen4[t_save][1] = gen4

    result_gen5[t_save][0] = res['gen'][4, 1]
    result_gen5[t_save][1] = res['gen'][4, 8]

    result_total_gen[t_save] = gen2 + gen3 + gen4 + gen5

    result_loads[t_save][0] = ppc_t['bus'][1, 2]
    result_loads[t_save][1] = ppc_t['bus'][2, 2]
    result_loads[t_save][2] = ppc_t['bus'][3, 2]
    result_loads[t_save][3] = ppc_t['bus'][4, 2]
    result_total_load[t_save] = ppc_t['bus'][1, 2] + ppc_t['bus'][2, 2] + ppc_t['bus'][3, 2] + ppc_t['bus'][4, 2]

# balance, excess, cost\slack
plt.figure(1)
plt.subplot(311)
pb = plt.plot(result_total_gen)
plt.setp(pb, 'color', 'g', 'linewidth', 2.0)
pb = plt.plot(result_total_load)
plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
plt.ylabel('total generation (green) and \ntotal load (red) in vpp0')
plt.axhline(0, color='black')

plt.subplot(312)
pb = plt.plot(result_opf[:, 1])
plt.setp(pb, 'color', 'g', 'linewidth', 2.0)
pb = plt.plot(result_opf[:, 0])
plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
plt.ylabel('excess (green) and power to balance \n(price=200) (blue) in vpp0')
plt.axhline(0, color='black')

plt.subplot(313)
pb = plt.plot(result_opf[:, 2])
plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
plt.ylabel('cost with no slack in vpp0')
plt.axhline(0, color='black')

# generators with max contraints
plt.figure(2)
plt.subplot(221)
pb = plt.plot(result_gen2[:, 0])
plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
pb = plt.plot(result_gen2[:, 1])
plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
plt.ylabel('gen2: price=4')
plt.axhline(0, color='black')

plt.subplot(222)
pb = plt.plot(result_gen3[:, 0])
plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
pb = plt.plot(result_gen3[:, 1])
plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
plt.ylabel('gen3: price=21')
plt.axhline(0, color='black')

plt.subplot(223)
pb = plt.plot(result_gen4[:, 0])
plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
pb = plt.plot(result_gen4[:, 1])
plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
plt.ylabel('gen4: price=32')
plt.axhline(0, color='black')

plt.subplot(224)
pb = plt.plot(result_gen5[:, 0])
plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
pb = plt.plot(result_gen5[:, 1])
plt.setp(pb, 'color', 'r', 'linewidth', 2.0)
plt.ylabel('gen5: price=103')
plt.axhline(0, color='black')

# loads
plt.figure(3)
pb = plt.plot(result_loads[:, 0])
plt.setp(pb, color='#c7d59f', linewidth=2.0)
pb = plt.plot(result_loads[:, 1])
plt.setp(pb, color='#b7ce63', linewidth=2.0)
pb = plt.plot(result_loads[:, 2])
plt.setp(pb, color='#8fb339', linewidth=2.0)
pb = plt.plot(result_loads[:, 3])
plt.setp(pb, color='#4b5842', linewidth=2.0)
plt.ylabel('loads at buses 2,3,4,5')

plt.show()

