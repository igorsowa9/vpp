from settings_3busML import *
import matplotlib.pyplot as plt
import json
import copy
from pypower.api import *
from pypower_mod.rundcopf_noprint import rundcopf


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
        power_balance = round(-1 * res['gen'][slack_idx, 1], 1)
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
        print("##################")
        print("##################")
        print("----- Deals: -----")
        print("##################")
        print("##################")
        for alias in ns.agents():
            a = ns.proxy(alias)
            print("\n\n\n>>>>>>>>" + alias + " deals (tobalance, excess): (" + str(a.get_attr('opf1')['power_balance']) + " "
                  + str(a.get_attr('opf1')['max_excess']) + ")")
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            print('Before negotiation (opf1, opfe2-for Exc):'
                  '\n\tobjf: ' + str(a.get_attr('opf1')['objf']) +
                  '\n\tobjf_noslackcost (opf1): ' + str(a.get_attr('opf1')['objf_noslackcost']))
            if a.get_attr('opfe2'):  # for excess agents
                incr_factor = a.load_data(data_paths[data_names_dict[alias]])['pc_matrix_price_increase_factor']
                print('\tMy price curve for other VPPs (gen_id/max.exc./price): (prices incr. factor: '+str(incr_factor)+')')
                print('\t' + str(a.get_attr('opfe2')['pc_matrix']))
                print('\t'+'objf_greentodso (opfe2): ' + str(a.get_attr('opfe2')['objf_greentodso']) +
                      '\n\tobjf_exportall (opfe2): ' + str(a.get_attr('opfe2')['objf_exportall']))

            if a.get_attr('opfe3'):  # for excess agents
                print('After: ' + str(a.get_attr('opfe3')['objf_bidsrevenue']))
            elif a.get_attr('opfd3'):  # for def agents
                print('After. Cost of buying bids: ' + str(a.get_attr('opfd3')['buybids_cost']))

            for deal_vpp in a.get_attr('timestep_memory_mydeals'):
                print("\tWith: " + str(deal_vpp[0]))
                print("\t\tBid values [vpp_idx (where selling generator is), gen_idx, value, price]: ")
                for bid in deal_vpp[1]:
                    print("\t\t" + str(bid))
        return True
    else:
        print("- Multi-consensus NOT reached (" + str(n_consensus) + "/" + str(vpp_n) + ") for time: ", global_time)
        return False


def erase_iteration_memory(ns):
    print('--- iteration M erase ---')
    for vpp_idx in range(vpp_n):
        a = ns.proxy(data_names[vpp_idx])
        a.set_attr(iteration_memory_received_pc=[])
        a.set_attr(iteration_memory_my_pc=[])
        a.set_attr(iteration_memory_bid=[])
        a.set_attr(iteration_memory_bid_accept=[])
        a.set_attr(iteration_memory_bid_finalanswer=[])
        a.set_attr(n_bids=0)
        a.set_attr(opfe3=0)
        a.set_attr(opfe2=0)


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
        a.set_attr(opf1_resgen=[])


def load_jsonfile(path):
    with open(path, 'r') as f:
        arr = json.load(f)
    return arr