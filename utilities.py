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
    time.sleep(small_wait)
    n_consensus = 0
    for alias in ns.agents():
        a = ns.proxy(alias)
        print(str(alias) + " consensus: " + str(a.get_attr('consensus')))
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
            if data_names_dict[alias] in vpp_learn:
                print("+++ Memorising ON +++: ")

            for deal_vpp in a.get_attr('timestep_memory_mydeals'):
                print("\tWith: " + str(deal_vpp[0]))
                print("\t\tBid values:")
                print("\t\t| vpp_idx (where selling generator is) "
                      "\n\t\t\t\t| gen_idx | value | price | ")
                for bid in deal_vpp[1]:
                    print("\t\t| " + str(bid[0]) + "\t| " + str(bid[1]) + "\t| " + str(bid[2]) + "\t| " + str(bid[3]) + "\t| ")
                print("\n")

        print("##################")
        print("##################\n\n")

        #############################
        #### saving for learning ####
        #############################

        # Memorize for all excess agents - ONLY SINGLE REQUESTS ASSUMED
        for alias in ns.agents():
            a = ns.proxy(alias)

            if a.get_attr('opf1')['max_excess'] > 0:  # if I am excess agent in this round

                if a.get_attr('requests'):  # if I am excess and I received requests
                    requests = a.get_attr('requests')
                    deals_memory = a.get_attr('timestep_memory_mydeals')
                    for req in requests:
                        req_alias = req['vpp_name']
                        deal_exist = False
                        for d in deals_memory:
                            deal_alias = d[0]
                            if deal_alias == req_alias:
                                a.save_deal_to_memory(d, global_time, req_alias, update_during_exploit)
                                deal_exist = True
                        if not deal_exist:
                            a.save_deal_to_memory(False, global_time, req_alias, update_during_exploit)

            else:# a.get_attr('opf1')['max_excess'] == 0 and a.get_attr('opf1')['power_balance'] > 0:  # if I am deficit agent

                a.save_if_deficit(global_time)

        print("\n\n##################")
        print("##################\n\n")

        #############################
        #############################
        #############################

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
        a.set_attr(price_increase_factor=[])
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
        # a.set_attr(agent_time=-1)
        a.set_attr(n_iteration=0)
        a.set_attr(n_requests=0)
        a.set_attr(n_bidoffers=0)
        a.set_attr(requests=[])
        a.set_attr(opf1=[])
        a.set_attr(opf1_resgen=[])
        a.set_attr(opf1_ppct=[])
        a.set_attr(opf1_res=[])
        a.set_attr(pc_memory_exc=np.array([{} for _ in range(max_iteration)]))
        a.set_attr(opfe2=0)  # this is set only once, refers to pc_memory_exc, that's why it's here for now
        a.set_attr(pc_memory_def=np.array([{} for _ in range(max_iteration)]))
        a.set_attr(similar_cases_chosen={})


def erase_learning_memory(vpp_learn):  # <---------------- now to files
    print('--- learning M erase (creating files) ---')
    for vpp_idx in vpp_learn:
        mem = pd.DataFrame({})
        mem.to_pickle(path_save + "temp_ln_" + str(vpp_idx) + ".pkl")


def load_jsonfile(path):
    with open(path, 'r') as f:
        arr = json.load(f)
    return arr


################ saving values to files #################

if not os.path.exists(path_save):
    os.makedirs(path_save)

vpp_results = np.zeros((ts_n, vpp_n, 4))
global_results = np.zeros((ts_n, 1))
def save_results_history(global_time, global_result, vpp_result):
    vpp_results[global_time-ts_0, :, :] = vpp_result
    global_results[global_time-ts_0, :] = global_result
    return 1


opf1_save_balcost_all = np.zeros((ts_n, vpp_n, 4))
opf1_save_genload_all = np.zeros((ts_n, vpp_n, 5, 4))  # as deep list due to different number of buses
opf1_save_prices_all = np.zeros((ts_n, vpp_n, 5))

np.save(path_save + "opf1_save_balcost_all", opf1_save_balcost_all)
np.save(path_save + "opf1_save_genload_all", opf1_save_genload_all)
np.save(path_save + "opf1_save_prices_all", opf1_save_prices_all)
def save_opf1_history(global_time, opf1_save_balcost, opf1_save_genload, opf1_save_prices):

    opf1_save_balcost_all = np.load(path_save + "opf1_save_balcost_all.npy")
    opf1_save_genload_all = np.load(path_save + "opf1_save_genload_all.npy")
    opf1_save_prices_all = np.load(path_save + "opf1_save_prices_all.npy")

    opf1_save_balcost_all[global_time - ts_0, :, :] = opf1_save_balcost
    opf1_save_genload_all[global_time - ts_0, :, :, :] = opf1_save_genload
    opf1_save_prices_all[global_time - ts_0, :, :] = opf1_save_prices

    np.save(path_save + "opf1_save_balcost_all", opf1_save_balcost_all)
    np.save(path_save + "opf1_save_genload_all", opf1_save_genload_all)
    np.save(path_save + "opf1_save_prices_all", opf1_save_prices_all)
    return


opfe3_save_costs_all = np.zeros((ts_n, vpp_n, 1))
np.save(path_save + "opfe3_save_costs_all", opfe3_save_costs_all)
def save_opfe3_history(global_time, opf3_history):
    opfe3_save_costs_all = np.load(path_save + "opfe3_save_costs_all.npy")
    opfe3_save_costs_all[global_time - ts_0, :, :] = opf3_history
    np.save(path_save + "opfe3_save_costs_all", opfe3_save_costs_all)
    return


def show_results_history(pdf):

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

    p1 = np.array(m_vpp3['exc_cost_range'].tolist())
    p2 = m_vpp2['marginal_price'].tolist()
    p3 = np.array(m_vpp3['price'].tolist())

    plt.title('vpp3 (excess - learning) vs vpp2 (deficit - dumb)')

    plt.subplot(211)
    plt.title('vpp3 costs generation excess (light and dark green) vs vpp2 marginal price in deals (red) and proposed price (purple)')
    plt.setp(plt.plot(p1[:, 0]), 'color', 'lime', 'linewidth', 1.0)
    plt.setp(plt.plot(p1[:, 1]), 'color', 'darkgreen', 'linewidth', 1.0)
    plt.setp(plt.plot(p2), 'color', 'red', 'linewidth', 1.0)
    plt.setp(plt.plot(p3[:, 0]), 'color', 'purple', 'linewidth', 1.0)
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

    if pdf:
        pdf = matplotlib.backends.backend_pdf.PdfPages(path_save + 'all_figs.pdf')
        for fig in range(1, figure_counter + 1):
            pdf.savefig(fig)
        pdf.close()
    else:
        plt.show()


def save_learning_memory(tocsv):

    for vpp_idx in vpp_learn:
        memory = pd.read_pickle(path_save + "temp_ln_" + str(vpp_idx) + ".pkl")
        print("\n\nAgent (id: " + str(vpp_idx) + ") memory: ")
        print(memory)
        print("\n\n\n")

        if tocsv:
            memory.to_csv(path_save + str(vpp_idx) + "_" + directory_tail + ".csv")

