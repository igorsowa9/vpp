from settings import *
import matplotlib.pyplot as plt
import json
from oct2py import octave
import copy, sys
from pprint import pprint as pp

from pypower.api import *
from case5_vpp import case5_vpp
from rundcopf_noprint import rundcopf

octave.addpath('/home/iso/PycharmProjects/vpp/matpow_cases')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0/t')


def system_printing_opf(mpc_t, t, data):
    """
    Analogous to system_update_state_and_balance in the agents' methods
    """
    generation = data['max_generation']
    load = data['fixed_load']
    price = data['price']
    slack_idx = data['slack_idx']

    mpc_t['bus'][:, 2] = load[t]
    mpc_t['gen'][:, 8] = generation[t]
    mpc_t['gencost'][:, 4] = price[t]

    res = rundcopf(mpc_t, ppoption(VERBOSE=0))
    if res['gen'][slack_idx, 1] > 0:  # there's a need for external resources (generation at slack >0) i.e. DEFICIT
        power_balance = round(-1 * res['gen'][slack_idx, 1], 1)  # from vpp perspective i.e. negative if deficit
        objf_noslackcost = round(res['f'] - res['gen'][slack_idx, 1] * mpc_t['gencost'][slack_idx][4], 1)
        max_excess = 0

    else:  # no need for external power - BALANCE or EXCESS
        power_balance = round(-1 * res['gen'][slack_idx, 1])
        max_excess = round(sum(mpc_t['gen'][:, 8]) - mpc_t['gen'][slack_idx, 8] - (sum(res['gen'][:, 1])
                                                                                   - res['gen'][slack_idx, 1]), 1)
        objf_noslackcost = round(res['f'] - res['gen'][slack_idx, 1] * mpc_t['gencost'][slack_idx][4], 1)
    return power_balance, objf_noslackcost, max_excess


def print_data():

    balance_sum = np.zeros(ts_n)
    for vi in range(vpp_n):
        with open(data_paths[vi], 'r') as f:
            ar = json.load(f)

        plt.figure(1)
        plt.subplot(vi + 221)
        price = plt.plot(list(zip(*ar['price']))[4])
        plt.setp(price, 'color', 'y', 'linewidth', 2.0)
        plt.ylabel('exemplary price (2.) in: ' + data_names[vi])

        plt.figure(2)
        ppc = cases[ar['case']]()
        power_balance_m = np.zeros(ts_n)
        for t in range(ts_n):
            power_balance_m[t], objf_noslackcost, max_reserve = system_printing_opf(copy.deepcopy(ppc), t, ar)
            balance_sum[t] = balance_sum[t] + power_balance_m[t]
        plt.subplot(vi + 221)
        pb = plt.plot(power_balance_m)
        plt.setp(pb, 'color', 'b', 'linewidth', 2.0)
        plt.ylabel('power balance in: ' + data_names[vi])
        plt.axhline(0, color='black')

    plt.figure(3)
    ps = plt.plot(balance_sum)
    plt.setp(ps, 'color', 'r', 'linewidth', 2.0)
    plt.ylabel('total balance power of the system')
    plt.axhline(0, color='black')

    plt.show()


def system_consensus_check(ns, global_time):
    n_consensus = 0
    for alias in ns.agents():
        a = ns.proxy(alias)
        if a.get_attr('consensus') == True:
            n_consensus = n_consensus + 1

    if n_consensus == vpp_n:
        print("- Multi-consensus reached (" + str(n_consensus) + "/" + str(vpp_n) + ") for time: ", global_time)
        print("----- Deals: -----")
        for alias in ns.agents():
            a = ns.proxy(alias)
            print(alias + " deals (with?, value buy+/-sell, price): ", a.get_attr('timestep_memory_mydeals'))
        return True
    else:
        print("- Multi-consensus NOT reached (" + str(n_consensus) + "/" + str(vpp_n) + ") for time: ", global_time)
        return False


def erase_iteration_memory(ns):
    print('--- iteration M erase ---')
    for vpp_idx in range(vpp_n):
        a = ns.proxy(data_names[vpp_idx])
        a.set_attr(iteration_memory_pc=[])
        a.set_attr(iteration_memory_bid=[])
        a.set_attr(iteration_memory_bid_accept=[])
        a.set_attr(n_bids=0)


def erase_timestep_memory(ns):
    print('--- timestamp M erase ---')
    for vpp_idx in range(vpp_n):
        a = ns.proxy(data_names[vpp_idx])
        a.set_attr(timestep_memory_mydeals=[])
        a.set_attr(n_iteration=0)
        a.set_attr(n_requests=0)
        a.set_attr(consensus=False)
        a.set_attr(requests=[])
        a.set_attr(opf1=[])
