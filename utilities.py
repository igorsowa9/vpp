from settings_4bus import *
import matplotlib.pyplot as plt
import json
import copy
import sys
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
        print("- n_iteration values: ")
        for alias in ns.agents():
            print(str(alias) + ": " + str(ns.proxy(alias).get_attr('n_iteration')))
        print("\n\n##################")
        print("##################")
        print("----- Deals: -----")
        print("##################")
        print("##################")
        for alias in ns.agents():
            a = ns.proxy(alias)
            print("\n\n\n>>>>>>>>" + alias + " deals (tobalance, excess): (" + str(a.get_attr('opf1')['power_balance']) + " "
                  + str(a.get_attr('opf1')['max_excess']) + ")")
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            print('Before negotiation (opf1, opfe2, no opfd2 so far):'
                  '\n\tobjf (opf1): ' + str(a.get_attr('opf1')['objf']) +
                  '\n\tobjf_noslackcost (i.e. dso) (opf1): ' + str(a.get_attr('opf1')['objf_noslackcost']))
            if a.get_attr('opfe2'):  # for excess agents
                incr_factor = a.load_data(data_paths[data_names_dict[alias]])['pc_matrix_price_increase_factor']
                print('\tMy initial price curve for other VPPs (opfe2): (prices incr. factor: '+str(incr_factor)+')')
                print('\t| gen_id\t| max.exc.\t | price\t |')
                for pp in np.array(a.get_attr('opfe2')['pc_matrix']):
                    strin = "\t| " + str(pp[0]) + "\t\t| " + str(pp[1]) + "\t\t| " + str(pp[2]) + "\t\t|"
                    print(strin)
                # print('\t' + str(np.array(a.get_attr('opfe2')['pc_matrix']).T))
                print('\t'+'objf_greentodso (opfe2): ' + str(a.get_attr('opfe2')['objf_greentodso']) +
                      '\n\tobjf_exportall (opfe2): ' + str(a.get_attr('opfe2')['objf_exportall']))
            else:
                print(
                    '\tI did not run opfe2 (i did not have requests), therefore no price curve available)')

            print('After:')
            if a.get_attr('opfe3'):  # for excess agents
                bid_rev = a.get_attr('opfe3')['objf_inclbidsrevenue']
                print('\tExcess VPP. Final cost after subtracting bids revenue objf_inclbidsrevenue (opfe3): ' + str(bid_rev))
            elif a.get_attr('opfd3'):  # for def agents
                print('\tDeficit VPP. Cost of buying from VPPs: ' + str(a.get_attr('opfd3')['buybids_cost']))
                total_withdso = a.get_attr('opf1')['objf']
                total_withbids = round(a.get_attr('opfd3')['buybids_cost'] + a.get_attr('opf1')['objf_noslackcost'], 4)
                print('\tTotal cost buying DSO (opf1) vs total cost with bids (opf1 and opf3): ' +
                      str(total_withdso) + " vs. " + str(total_withbids))
            else:
                print('\tI did not run opfe3 / opfd3.')

            print("Contracts/deals:")
            for deal_vpp in a.get_attr('timestep_memory_mydeals'):
                print("\tWith: " + str(deal_vpp[0]))
                print("\t\tBid values:")
                print("\t\t| vpp_idx (where selling generator is) "
                      "\n\t\t\t\t| gen_idx | value | price | ")
                for bid in deal_vpp[1]:
                    print("\t\t| " + str(bid[0]) + "\t| " +  str(bid[1]) + "\t| " + str(bid[2]) + "\t| " + str(bid[3]) + "\t| ")
                print("\n")

        # saving into results_history:
        # before negotiation: excess/deficit value, objf, objf_noslackcost,
        # after: objf_inclbidsrevenue (excess), cost of buying bids + final cost (deficit)
        vpp_result = np.zeros((vpp_n, 4))
        VPP_VALUE = 0
        OBJF1 = 1
        OBJF1_NODSO = 2
        OBJF_AFTER = 3

        # global results for example number of iterations
        global_result = np.zeros(1)

        for alias in ns.agents():
            a = ns.proxy(alias)
            if not a.get_attr('opf1')['max_excess']:
                save = a.get_attr('opf1')['power_balance']
            else:
                save = a.get_attr('opf1')['max_excess']
            vpp_result[data_names_dict[alias], VPP_VALUE] = save

            vpp_result[data_names_dict[alias], OBJF1] = a.get_attr('opf1')['objf']
            vpp_result[data_names_dict[alias], OBJF1_NODSO] = a.get_attr('opf1')['objf_noslackcost']

            if a.get_attr('opfe3'):
                after = a.get_attr('opfe3')['objf_inclbidsrevenue']
            elif a.get_attr('opfd3'):
                after = round(a.get_attr('opfd3')['buybids_cost'] + a.get_attr('opf1')['objf_noslackcost'], 4)
            else:
                after = False
            vpp_result[data_names_dict[alias], OBJF_AFTER] = after

            global_result[0] = a.get_attr('n_iteration')  # should be out of the loop

        save_results_history(global_time, global_result, vpp_result)

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
        a.set_attr(opfd3=0)
        a.set_attr(opfd2=0)
        a.set_attr(consensus=False)


def erase_timestep_memory(ns):
    print('--- timestamp M erase ---')
    for vpp_idx in range(vpp_n):
        a = ns.proxy(data_names[vpp_idx])
        a.set_attr(timestep_memory_mydeals=[])
        a.set_attr(n_iteration=0)
        a.set_attr(n_requests=0)
        a.set_attr(n_bidoffers=0)
        a.set_attr(requests=[])
        a.set_attr(opf1=[])
        a.set_attr(opf1_resgen=[])
        a.set_attr(pc_memory_exc=np.array([{} for _ in range(max_iteration)]))
        a.set_attr(opfe2=0)  # this is set only once, refers to pc_memory_exc, that's why it's here for now
        a.set_attr(pc_memory_def=np.array([{} for _ in range(max_iteration)]))

#
# def erase_persistent_memory(ns):
#     print('--- persistent M erase ---')
#     for vpp_idx in range(vpp_n):
#         a = ns.proxy(data_names[vpp_idx])
#         a.set_attr(results_history=[])


def load_jsonfile(path):
    with open(path, 'r') as f:
        arr = json.load(f)
    return arr


vpp_results = np.zeros((ts_n, vpp_n, 4))
global_results = np.zeros((ts_n, 1))


def save_results_history(global_time, global_result, vpp_result):
    vpp_results[global_time-ts_0, :, :] = vpp_result
    global_results[global_time-ts_0, :] = global_result
    return 1


def show_results_history():

    VPP_VALUE = 0
    OBJF1 = 1
    OBJF1_NODSO = 2
    OBJF_AFTER = 3

    print("###################")
    print("### PRINTING ######")
    print("###################")
    vpp_idx = 0

    plt.figure(1)
    # plt.subplot(311)
    pb = plt.plot(vpp_results[:, :, VPP_VALUE])
    plt.setp(pb, 'color', 'g', 'linewidth', 2.0)
    plt.ylabel('value for vpp1')
    plt.axhline(0, color='black')

    plt.figure(2)
    for ch in range(vpp_n):
        plt.subplot(411+ch)
        pb = plt.plot(vpp_results[:, ch, OBJF1])
        plt.setp(pb, color='r', linewidth=2.0)
        pb = plt.plot(vpp_results[:, ch, OBJF1_NODSO])
        plt.setp(pb, color='b', linewidth=2.0)
        pb = plt.plot(vpp_results[:, ch, OBJF_AFTER])
        plt.setp(pb, color='g', linewidth=2.0)
        plt.ylabel('objf-red, nodso-blue, after-green')
        plt.axhline(0, color='black')

    plt.show()
