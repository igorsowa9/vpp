from osbrain import Agent
import json
from settings_4bus import *
#from utilities import system_state_update_and_balance
import time
from pprint import pprint as pp
import copy
import sys
from create_ppc_file import create_ppc_file
import pandas as pd
import os

from pypower.api import *
from pypower_mod.rundcopf_noprint import rundcopf
from pypower_mod.rundcpf_noprint import rundcpf
from datetime import datetime, timedelta

from utilities import save_opfe3_history


class VPP_ext_agent(Agent):

    def load_data(self, path):
        """
        Loads data for VPP from file, from web, whatever necessary.
        :param path:
        :return:
        """
        with open(path, 'r') as f:
            arr = json.load(f)
        return arr

    def current_price(self, time):
        """
        Loads price of own resources / create price curve, at a time.
        :param time:
        :return: price value / price curve
        """
        json_data = self.load_data(data_paths[data_names_dict[self.name]])
        return json_data["price"][time]

    def modify_fromfile(self, t, if_fload=True, if_gen=True, if_price=True):

        vpp_file = self.load_data(data_paths[data_names_dict[self.name]])
        ppc0 = cases[vpp_file['case']]()
        ppc_t = copy.deepcopy(ppc0)

        fixed_load0 = copy.deepcopy(ppc0['bus'][:, 2])
        max_generation0 = copy.deepcopy(ppc0['gen'][:, 8])
        price0 = copy.deepcopy(ppc0['gencost'][:, 4])

        slack_idx = vpp_file['slack_idx']

        if if_fload:
            for idx in range(vpp_file['bus_n']):
                if idx == slack_idx:
                    continue
                # fixed loads values modification
                fixed_load_t = fixed_load0
                fload_path = vpp_file['fixed_load_profiles_paths'][idx]
                if not fload_path == "":
                    d = self.load_data(fload_path)
                    mod = d[t][2]
                    fixed_load_t[idx] = (mod + load_mod_offset) * fixed_load0[idx]
            ppc_t['bus'][:, 2] = fixed_load_t

        if if_gen:
            for idx in range(vpp_file['bus_n']):
                if idx == slack_idx:
                    continue
                # max generation constraints modification
                max_generation_t = max_generation0
                gen_path = vpp_file['generation_profiles_paths'][idx]
                if not gen_path == "":  # no modification of the original value
                    d = self.load_data(gen_path)
                    mod = d[t][MEASURED]
                    max_generation_t[idx] = (mod + gen_mod_offset) * max_generation0[idx]
            ppc_t['gen'][:, 8] = max_generation_t

        if if_price:
            for idx in range(vpp_file['bus_n']):
                if idx == slack_idx:
                    continue
                # prices modification (values directly from the file)
                price_t = copy.deepcopy(price0)
                price_path = vpp_file['price_profiles_paths'][idx]
                if not price_path == "":
                    d = self.load_data(price_path)
                    price_t[idx] = d[t]
            ppc_t['gencost'][:, 4] = price_t

        return ppc_t

    def runopf1(self, t):
        """
        This should be internal OPF in order to define excess/deficit.
        Updates the system ppc at time t according to data and runs opf.
        If excess, derive exces curve matrix (with generation prices).
        :param mpc_t:
        :param t:
        :param data:
        :return: balance needed at PCC, max possible excess, objf with subtracted virtual slack cost
        """

        ppc_t = self.modify_fromfile(t)
        create_ppc_file(str(self.name) + "_fromOPF1", ppc_t)
        slack_idx = self.load_data(data_paths[data_names_dict[self.name]])['slack_idx']
        res = rundcopf(ppc_t, ppoption(VERBOSE=opf1_verbose))
        if opf1_prinpf:
            #printpf(res)
            print("generators constraints from ppc_t:\n bus, actual, max, min, prices ")
            pp(np.round(res['gen'][:, [0, 1, 8, 9]], 4))
            pp(ppc_t['gencost'][:, 4])

        if res['success'] == 1:
            self.log_info("I have successfully run the OPF1.")
            self.set_attr(opf1_resgen=res['gen'])
            self.set_attr(opf1_ppct=ppc_t)
            self.set_attr(opf1_res=res)

            if round(res['gen'][slack_idx, 1], 4) > 0:  # there's a need for external resources (generation at slack >0) i.e. DEFICIT
                self.set_attr(current_status='D')
                power_balance = round(-1 * res['gen'][slack_idx, 1], 4)  # from vpp perspective i.e. negative if deficit

                # for deficit vpps objf includes costs of buying from DSO (at slack bus) for fixed higher price
                objf = round(res['f'], 4)
                objf_noslackcost = round(objf - res['gen'][slack_idx, 1] * ppc_t['gencost'][slack_idx][4], 4)

                self.set_attr(opf1={'power_balance': power_balance,
                                    'max_excess': False,
                                    'objf': objf,
                                    'objf_noslackcost': objf_noslackcost,
                                    'exc_matrix': False})

            else:  # no need for external power - BALANCE or EXCESS
                power_balance = -1 * round(res['gen'][slack_idx, 4])
                max_excess = round(sum(ppc_t['gen'][:, 8]) - ppc_t['gen'][slack_idx, 8] - (sum(res['gen'][:, 1])
                                                                                           - res['gen'][slack_idx, 1]), 4)

                if power_balance == 0 and max_excess == 0:
                    self.set_attr(current_status='B')
                else:
                    self.set_attr(current_status='D')

                # derive the price curve: (matrix having indeces, exces values per gen, their LINEAR prices
                idx = ppc_t['gen'][1:, 0] if slack_idx == 0 else print("Is slack at 0 bus?")
                pexc = ppc_t['gen'][1:, 8] - res['gen'][1:, 1]
                gcost = ppc_t['gencost'][1:, 4]
                gens_exc_price = np.round(np.matrix([idx, pexc, gcost]), 4)
                sorted_m = gens_exc_price[:, gens_exc_price[2, :].argsort()]
                sorted_nonzero = sorted_m[:, np.nonzero(sorted_m[1, :])]
                # sorted_nonzero_squeezed = np.reshape(sorted_nonzero, np.squeeze(sorted_nonzero).shape)

                # increase price by a factor
                exc_matrix = np.matrix(sorted_nonzero)
                if exc_matrix.shape[0] == 1:
                    exc_matrix = exc_matrix.T

                objf = round(res['f'], 4)
                objf_noslackcost = round(objf - res['gen'][slack_idx, 1] * ppc_t['gencost'][slack_idx][4], 4)

                self.set_attr(opf1={'power_balance': power_balance,
                                    'max_excess': max_excess,
                                    'objf': objf,
                                    'objf_noslackcost': objf_noslackcost,
                                    'exc_matrix': np.array(exc_matrix)})
            return True
        else:
            self.log_info("OPF1 does not converge. STOP.")
            sys.exit()

    def runopf_e2(self, t):
        """
        Excess agents' calculations, before sending the price curves.
        Derivation of the PC for this iteration step. From excess directly, or from memory.
        This include verification of transmitting the excess to the other vpps (only to the ones that send requests),
        but also to DSO.
        """
        exc_matrix = self.get_attr('opf1')['exc_matrix']
        generation_type = np.array(self.load_data(data_paths[data_names_dict[self.name]])['generation_type'])

        n_iteration = self.get_attr("n_iteration")

        ######## build price curve according to pc_matrix_price_increase_factor (gen prices * increase factor)
        # price increase factor modification should be calculated according to request number not to the timestamp
        # i.e. if the last request was at previous X timestamps then modify the factor
        price_increase_factor = self.load_data(data_paths[data_names_dict[self.name]])[
            'pc_matrix_price_increase_factor']

        if type(price_increase_factor) == list:

            currmem = pd.read_pickle(path_save + "temp_ln_" + str(data_names_dict[self.name]) + ".pkl")
            t_range = np.arange(t - max_ts_range_for_price_modification, t)

            previous_fulfilled = 0  # check if the time exist and if some conditions are fulfilled: success
            for test_t in t_range:
                if 't' in currmem.columns:
                    test_row = currmem.loc[currmem['t'] == test_t]
                    if not test_row.empty:
                        if (currmem['t'] == test_t).any() and test_row.iloc[0]['success'] == 1:
                            previous_fulfilled += 1

            if previous_fulfilled == max_ts_range_for_price_modification:

                previous_row = currmem.loc[currmem['t'] == t-1]  # take previous row
                previous_mod = previous_row.iloc[0]['pcf']

                if previous_mod < price_increase_factor[1]:
                    price_increase_factor = previous_mod + price_increase_factor[2]
                else:
                    price_increase_factor = price_increase_factor[0]

            else:  # if conditions not fulilled come back to the first value
                price_increase_factor = price_increase_factor[0]

        a = 'OFF'
        if exploit and self.name in vpp_exploit:
            self.log_info("I will apply memory through similarity.")
            price_increase_factor = self.similarity(t)
            a = 'ON'

        self.log_info("My runopf_e2 (exploit " + a + ") price_increase_factor: " + str(price_increase_factor))
        self.set_attr(price_increase_factor=price_increase_factor)

        if n_iteration == 0:  # if its 0 iteration, make according to pr_matrix_price_increase_factor (each vpp has own)
            pc_matrix_incr = copy.deepcopy(exc_matrix.T)

            pc_matrix_incr[:, 2] = pc_matrix_incr[:, 2] * price_increase_factor
            pc_matrix_incr = np.matrix(np.round(pc_matrix_incr, 4))

            self.log_info("OPFe2: PC matrix for requesters (i.e. exc_matrix increased): " + str(pc_matrix_incr))
            self.set_attr(opfe2={'pc_matrix': np.array(pc_matrix_incr)})
            self.get_attr('pc_memory_exc')[n_iteration].update({'all': np.array(pc_matrix_incr)})
        elif n_iteration > 0:
            if self.get_attr('pc_memory_exc')[n_iteration] == {}: # this should make PC for ALL if there is no particular PCs
                price_curve = copy.deepcopy(self.get_attr('pc_memory_exc')[0]['all'])
                # make a new pc according to "price increase policy" for now just linear increase for each vpp:
                prices = price_curve[:, 2]
                if type(prices) == np.float64:  # i.e. if there is only one excess generator, there is no list but float
                    new_prices = prices + price_increase_factor*self.get_attr("n_iteration")
                else:
                    new_prices = [x + price_increase_factor*self.get_attr("n_iteration") for x in prices]
                price_curve[:, 2] = new_prices

                pc_matrix_incr = copy.deepcopy(price_curve)
                self.get_attr('pc_memory_exc')[n_iteration].update({'all': np.array(pc_matrix_incr)})

            else:  # if it exists already (i.e. the particular PCs exist)
                pass


        ######### load, update, modify data according to files and opf1
        origin_opf1_resgen = self.get_attr('opf1_resgen')
        if opf1_prinpf:
            print('bus \ value \ pmax \ pmin (original opf1)')
            print(np.round(origin_opf1_resgen[:, [0, 1, 8, 9]], 4))
            print('Objf_noslack: ' + str(self.get_attr('opf1')['objf_noslackcost']))

        # max_generation from results of opf1, not from files
        ppc_t = self.modify_fromfile(t, True, False, True)
        slack_idx = self.load_data(data_paths[data_names_dict[self.name]])['slack_idx']
        ppc_t['gen'][0:, 8] = 0  # virtual slack generator removed
        ppc_t['gen'][0:, 9] = 0
        ppc_t['gen'][1:, 8] = np.round(origin_opf1_resgen[1:, 1] * (1 + relax_e2), 4)  # from OPF1, without slack
        ppc_t['gen'][1:, 9] = np.round(origin_opf1_resgen[1:, 1] * (1 - relax_e2), 4)  # both bounds

        ###### calculate prospective revenue if green energy sold to DSO
        self.log_info("I calculate prospective revenue if green energy sold to DSO or all energy to DSO (in OPFe2)")
        c1 = generation_type[exc_matrix[0, :].astype(int)]  # check gen types of the ones in pc_matrix
        c2 = np.isin(c1, green_sources)  # choose only green ones
        c3 = np.reshape(c2, np.squeeze(c2).shape)

        exc_matrix_green = exc_matrix[:, c3]

        if exc_matrix_green.shape[1] == 1:
            exc_matrix_green = np.matrix(np.squeeze(exc_matrix[:, c3])).T  # the most stupid line ever

        if exc_matrix_green.size > 0:
            self.get_attr('opfe2').update({'exc_matrix_green': exc_matrix_green})

            ####### exporting green (feasible) energy (e.g. option of selling to DSO)
            ppc_tg = copy.deepcopy(ppc_t)
            exc_matrix_greenT = np.array(exc_matrix_green.T)

            for gen_idx in np.unique(exc_matrix_greenT[:, 0]):  # loop through the generators from bids (1,2,3) there should be no slack - as constraints for OPF of same Pmin and Pmax

                c1 = np.where(exc_matrix_greenT[:, 0] == gen_idx)[0]  # rows where gen
                p_bid = np.round(np.sum(exc_matrix_greenT[c1, 1]), 4)  # total value of bidded power for generator

                exc_gen = np.array([int(gen_idx), 0, 0, 0, 0, 1, 100, 1, p_bid, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                ppc_tg['gen'] = np.concatenate((ppc_tg['gen'], np.matrix(exc_gen)), axis=0)
                gc = np.array([2, 0, 0, 2, low_price, 0])
                ppc_tg['gencost'] = np.concatenate((ppc_tg['gencost'], np.matrix(gc)), axis=0)

                neg_gen = [slack_idx, 0, 0, 0, 0, 1, 100, 1, 0, -p_bid, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ppc_tg['gen'] = np.concatenate((ppc_tg['gen'], np.matrix(neg_gen)), axis=0)
                ngc = np.array([2, 0, 0, 2, high_price, 0])
                ppc_tg['gencost'] = np.concatenate((ppc_tg['gencost'], np.matrix(ngc)), axis=0)

                ppc_tg['gen'] = np.array(ppc_tg['gen'])
                ppc_tg['gencost'] = np.array(ppc_tg['gencost'])

            # relax thermal constr.
            ppc_tg['branch'][:, [5, 6, 7]] = ppc_tg['branch'][:, [5, 6, 7]] * (1 + relax_e2)

            ppc_tg['gen'][:, 8][ppc_tg['gen'][:, 8] == 0] = 1e-6  # substitute generators upper bounds 0 with 1e-6
            create_ppc_file(str(self.name) + "_fromOPF_e2_green", ppc_tg)
            res_g = rundcopf(ppc_tg, ppoption(VERBOSE=opf1_verbose, PDIPM_GRADTOL=PDIPM_GRADTOL_mod))

            show = np.array(
                np.concatenate((np.matrix(res_g['gen'][:, [0, 1, 8, 9]]), np.matrix(ppc_tg['gencost'][:, 4]).T),
                               axis=1))
            show_realcost = copy.deepcopy(show)
            for g in range(len(show_realcost)):
                garr = np.array(show_realcost[g, :])
                if garr[4] == low_price:
                    g_idx = garr[0]
                    p = exc_matrix[2, exc_matrix[0, :] == g_idx]  # increased price of the resource
                    garr[4] = p
                    show_realcost[g, :] = garr
                if garr[4] == high_price:
                    g_idx = np.array(show_realcost[g - 1, 0])  # bus number from the raw above
                    garr[0] = g_idx
                    temp = np.array(pc_matrix_incr.T)
                    p = temp[2, temp[0, :] == g_idx]  # increased price of the resource
                    garr[4] = p
                    show_realcost[g, :] = garr

            real_cost = np.round(sum(show_realcost[:, 1] * show_realcost[:, 4]), 4)
            self.get_attr('opfe2').update({'objf_greentodso': real_cost})

            if opfe2_prinpf:
                print('bus \ value \ pmax \ pmin \ price (if green excess available)')
                print(np.round(show_realcost, 4))
                print("Real cost (objf by hand): " + str(real_cost))

            if res_g['success']:
                self.log_info('Feasible OPFe2 with green resources available to DSO.')
                f1 = True
            else:
                self.log_info('NOT feasible OPFe2 with green resources available to DSO. NOT GOOD - STOP.')
                f1 = False
                sys.exit()

        else:
            self.log_info("No green excess that could be sold to DSO.")
            self.get_attr('opfe2').update({'exc_matrix_green': False})
            self.get_attr('opfe2').update({'objf_greentodso': False})
            f1 = False

        ######## exporting whole (feasible) excess through PCC
        exc_matrixT = np.array(exc_matrix.T)

        for gen_idx in np.unique(exc_matrixT[:, 0]):  # loop through the generators from bids (1,2,3) there should be no slack - as constraints for OPF of same Pmin and Pmax

            c1 = np.where(exc_matrixT[:, 0] == gen_idx)[0]  # rows where gen
            p_bid = np.round(np.sum(exc_matrixT[c1, 1]), 4)  # total value of bidded power for generator

            exc_gen = np.array([int(gen_idx), 0, 0, 0, 0, 1, 100, 1, p_bid, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            ppc_t['gen'] = np.concatenate((ppc_t['gen'], np.matrix(exc_gen)), axis=0)
            gc = np.array([2, 0, 0, 2, low_price, 0])
            ppc_t['gencost'] = np.concatenate((ppc_t['gencost'], np.matrix(gc)), axis=0)

            neg_gen = [slack_idx, 0, 0, 0, 0, 1, 100, 1, 0, -p_bid, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ppc_t['gen'] = np.concatenate((ppc_t['gen'], np.matrix(neg_gen)), axis=0)
            ngc = np.array([2, 0, 0, 2, high_price, 0])
            ppc_t['gencost'] = np.concatenate((ppc_t['gencost'], np.matrix(ngc)), axis=0)

            ppc_t['gen'] = np.array(ppc_t['gen'])
            ppc_t['gencost'] = np.array(ppc_t['gencost'])

        # relax thermal constr.
        ppc_t['branch'][:, [5, 6, 7]] = ppc_t['branch'][:, [5, 6, 7]] * (1+relax_e2)

        ppc_t['gen'][:, 8][ppc_t['gen'][:, 8] == 0] = 1e-6  # substitute generators upper bounds 0 with 1e-6
        create_ppc_file(str(self.name) + "_fromOPF_e2_all", ppc_t)
        res = rundcopf(ppc_t, ppoption(VERBOSE=opf1_verbose, PDIPM_GRADTOL=PDIPM_GRADTOL_mod))

        show = np.array(np.concatenate((np.matrix(res['gen'][:, [0, 1, 8, 9]]), np.matrix(ppc_t['gencost'][:, 4]).T), axis=1))
        show_realcost = copy.deepcopy(show)
        for g in range(len(show_realcost)):
            garr = np.array(show_realcost[g, :])
            if garr[4] == low_price:
                g_idx = garr[0]
                p = exc_matrix[2, exc_matrix[0, :] == g_idx]  # increased price of the resource
                garr[4] = p
                show_realcost[g, :] = garr
            if garr[4] == high_price:
                g_idx = np.array(show_realcost[g - 1, 0])  # bus number from the raw above
                garr[0] = g_idx
                temp = np.array(pc_matrix_incr.T)
                p = temp[2, temp[0, :] == g_idx]  # increased price of the resource
                garr[4] = p
                show_realcost[g, :] = garr

        real_cost = np.round(sum(show_realcost[:, 1] * show_realcost[:, 4]), 4)
        self.get_attr('opfe2').update({'objf_exportall': real_cost})

        if opfe2_prinpf:
            # printpf(res)
            print('bus \ value \ pmax \ pmin \ price (if whole available excess)')
            print(np.round(show_realcost, 4))
            print("Real cost (objf by hand): " + str(real_cost))

        if res['success']:
            self.log_info('Feasible OPFe2 with overall excess available.')
            f2 = True
        else:
            self.log_info('NOT feasible OPFe2 with overall excess available. NOT GOOD - STOP')
            f2 = False
            sys.exit()

        return f1, f2

    def runopf_e3(self, all_bids_mod, t):
        """
        After an excess agent decides about the answers for the bids (original or modified),
        it should check feasibility of such answer through pf:
            - added bids as more generation,
            - added load at the pcc.
        In case of unfeasible solution, further modification of bids has to be done.
        Furthermore, the objective function should be compared if lower cost is reached.
        """
        all_bids_mod = np.array(all_bids_mod)

        # modify according to prospective accepted bids: loop through my generators (excluding slack)
        origin_opf1_resgen = self.get_attr('opf1_resgen')

        # max_generation from results of opf1, not from files
        ppc_t = self.modify_fromfile(t)

        # slack_idx = self.load_data(data_paths[data_names_dict[self.name]])['slack_idx']
        # ppc_t['gen'][0:, 8] = 0  # virtual slack generator removed
        # ppc_t['gen'][0:, 9] = 0
        # ppc_t['gen'][1:, 8] = np.round(origin_opf1_resgen[1:, 1] * (1 + relax_e3), 4)  # from OPF1, without slack
        # ppc_t['gen'][1:, 9] = np.round(origin_opf1_resgen[1:, 1] * (1 - relax_e3), 4)  # both bounds
        #
        # for gen_idx in np.unique(all_bids_mod[:, 2]):  # loop through the generators from bids (1,2,3) there should be no slack - as constraints for OPF of same Pmin and Pmax
        #
        #     c1 = np.where(all_bids_mod[:, 2] == gen_idx)[0]  # rows where gen
        #     p_bid = np.round(np.sum(all_bids_mod[c1, 3]), 4)  # total value of bidded power for generator
        #
        #     c2 = np.where(origin_opf1_resgen[:, 0] == gen_idx)[0]
        #     ppc_t['gen'][c2, [8, 9]] = np.array([0, 0])  # make the value 0 before modification
        #     ppc_t['gen'][c2, [8, 9]] += np.round(origin_opf1_resgen[c2, 1], 4)  # add value from opf1, to Pmax,Pmin
        #
        #     c3 = np.where(ppc_t['gen'][:, 0] == gen_idx)[0]
        #     ppc_t['gen'][c3, [8, 9]] += np.round(p_bid, 4)  # add value from bidding, to Pmax,Pmin

        # modify the load at the pcc i.e. slack bus id:0 - same formulation for PF and OPF
        bids_sum = np.round(np.sum(all_bids_mod[:, 3]), 4)
        ppc_t['bus'][0, 2] += bids_sum

        create_ppc_file(str(self.name) + "_fromOPF_e3", ppc_t)
        # with bids updated, verify the power flow if feasible - must be opf in order to include e.g. thermal limits
        res = rundcopf(ppc_t, ppoption(VERBOSE=opf1_verbose))

        if opfe3_prinpf:
            printpf(res)
            pp(ppc_t)

        # calculation of the costs: objective function minus revenue from selling
        bids_revenue = 0
        for bid in all_bids_mod:
            bids_revenue += bid[3] * bid[4]

        costs = np.round(res['f'] - bids_revenue, 4)  # costs of vpp as costs of operation (slackcost for E = 0) - revenue from bids

        self.set_attr(opfe3={'objf_inclbidsrevenue': costs, 'bids_revenue': bids_revenue})

        return res['success']

    def runopf_d2(self):
        """
        After a deficit agent receives all price curves, it should define bids through running internal opf
        as defined in this function.
        :return: bids that are sent back to the excess agents of interest
        """
        # opf asa system is integrated
        memory = self.get_attr('iteration_memory_received_pc')
        need = abs(self.get_attr('opf1')['power_balance'])

        self.log_info("I run runopf_d2 for iteration number: " + str(self.get_attr("n_iteration")))

        all_pc = []
        for mem in memory:
            if mem['value'] == False:
                continue
            else:
                for gn in range(0, mem['price_curve'].shape[0]):
                    all_pc.append([data_names_dict[mem["vpp_name"]],
                                   mem["price_curve"][gn, 0],
                                   mem["price_curve"][gn, 1],
                                   mem["price_curve"][gn, 2]])
        sorted_pc = sorted(all_pc, key=lambda price: price[3])
        self.set_attr(opfd2={"received_pc": sorted_pc})
        self.log_info("All current price curves together: " + "\n" + str(np.array(sorted_pc)) + " I need: " + str(need))
        bids = []
        for pc in sorted_pc:
            pc_vpp_idx = pc[0]
            pc_gen_idx = pc[1]
            pc_maxval = float(pc[2])
            pc_price = float(pc[3])
            if need != 0 and need >= pc_maxval:
                bids.append([pc_vpp_idx, pc_gen_idx, pc_maxval, pc_price])
                need = need - pc_maxval
            elif need != 0 and need < pc_maxval:
                bids.append([pc_vpp_idx, pc_gen_idx, need, pc_price])
                need = 0
            elif need == 0:
                break
        # fill the bids with the unbidded vpps for refuse messages
        bids = np.array(bids)
        b = np.matrix(all_pc)
        allb = np.unique(np.array(b[:, 0]))
        are = np.array(np.matrix(bids)[:, 0])
        missing = np.setdiff1d(allb, are)
        for m in missing:
            empty_bid = np.array([[m, 0, 0, 0]])
            bids = np.concatenate((bids, empty_bid), axis=0)

        self.log_info("All chosen price curves to be sent as bids (during n_iteration = " +
                      str(self.get_attr("n_iteration")) +
                      "): " + "\n" + str(np.array(bids)))
        # self.set_attr(opfd2={"bids": bids})  # set attribute instead of return
        self.get_attr('opfd2').update({'bids': bids})
        return

    def runopf_d3(self):
        """
        After a deficit agent receives the bid accepts, bid modifies etc. it should verify it finally or at least
        calculate the related costs, for now/
        :return: //, sets opfd3 with some cost calculations
        """

        mydeals = []
        for accepted_bid in self.get_attr('iteration_memory_bid_accept'):
            mydeals.append([accepted_bid['vpp_name'], accepted_bid['bid']])
        self.set_attr(timestep_memory_mydeals=mydeals)
        self.set_attr(consensus=True)
        self.log_info("I set consensus in runopf_d3.")

        cost = 0
        for deal_vpp in self.get_attr('timestep_memory_mydeals'):
            cost += np.sum(deal_vpp[1][:, 2] * deal_vpp[1][:, 3])
        cost = np.round(cost, 4)
        self.set_attr(opfd3={'buybids_cost': cost})

        # self.log_info("opfd3print: " + str(self.get_attr("opfd3")))

        # save_opf3_history(global_time, opf1_save_balcost, opf1_save_genload, opf1_save_prices)

        return True

    def set_consensus_if_norequest(self):
        """
        This is called in case an excess agent does not have any requests from deficit agents in the deficit loop.
        :return:
        """
        time.sleep(small_wait)
        if self.get_attr('n_requests') == 0:
            self.log_info('I am E and I nobody wants to buy from me (n_requests=' + str(self.get_attr('n_requests')) +
                          '). I set consensus.')
            self.set_attr(consensus=True)

    # def sys_octave_test(self, mpc):
    #     res = octave.rundcopf(mpc)
    #     return res['success']

    def sys_pypower_test(self, ppc):
        r = rundcopf(ppc)
        return r['success']

    def bids_alignment1(self, mypc0, all_bids0):

        mypc0 = mypc0[mypc0[:, 2].argsort()]  # make the pc in merit order in case it is not
        all_bids = copy.deepcopy(all_bids0)

        # fill the missing generators (0-bids)
        for vpp in np.unique(all_bids0[:, 0]):
            tobid_gens_n = len(mypc0)
            if sum(all_bids0[:, 0] == vpp) != tobid_gens_n:
                existing = set(all_bids0[all_bids0[:, 0] == vpp, 2])
                allgen = set(mypc0[:, 0])
                missing = np.array(list(allgen - existing))

                for mis in missing:
                    to_append = np.array([vpp,
                                          all_bids0[0, 1],
                                          mis,
                                          0,
                                          mypc0[mypc0[:, 0] == mis, 2]])

                    all_bids = np.append(all_bids, [to_append], axis=0)
        all_bids_sum = sum(all_bids0[:, 3])
        all_bids_mod = copy.deepcopy(all_bids)

        for pc_i in range(0, len(mypc0)):
            pc = mypc0[pc_i, :]
            gen_id = pc[0]
            gen_pmax = pc[1]
            bid_1gen = all_bids_mod[all_bids_mod[:, 2] == gen_id]
            sum_bids_1gen = sum(bid_1gen[:, 3])

            if gen_pmax < sum_bids_1gen:  # if resource exceeded for 1 generator, then share the excess
                self.log_info("Modification of bids of over-bidded gens... ")

                for vpp_idx in np.unique(all_bids_mod[:, 0]):

                    c1 = np.where(all_bids_mod[:, 0] == vpp_idx)[0]  # raws where vpp
                    c2 = np.where(all_bids_mod[:, 2] == gen_id)[0]  # raws where gens
                    c3 = np.intersect1d(c1, c2)

                    last_bid = all_bids_mod[c3, 3]

                    vpp_bids_sum = np.sum(all_bids_mod[c1, 3])
                    vpp_waga = vpp_bids_sum / all_bids_sum  # 55/65
                    vpp_assigned = vpp_waga * gen_pmax

                    all_bids_mod[c3, 3] = np.round(vpp_assigned, 4)  # 55/65 * 20

                    rest = last_bid - vpp_assigned

                    k1 = mypc0[pc_i+1, 0]  # id of next gen in merit order
                    k2 = np.where(all_bids_mod[:, 2] == k1)[0]
                    k3 = np.intersect1d(c1, k2)
                    all_bids_mod[k3, 3] += np.round(rest, 4)

        # new pc curves for particular vpps:
        pc_msg = np.array([])

        for v in np.unique(all_bids_mod[:, 0]):
            pc = np.array(all_bids_mod[all_bids_mod[:, 0] == v, :][:, 2:]).T
            pc_msg = np.append(pc_msg, [{"vpp_idx": int(v), "pc_curve": pc}])

        return all_bids_mod, pc_msg

    def save_if_deficit(self, global_time):

        memory = pd.read_pickle(path_save + "temp_ln_" + str(data_names_dict[self.name]) + ".pkl")

        my_deficit = self.get_attr('opf1')['power_balance']
        received_pc = self.get_attr('opfd2')['received_pc']

        memory = memory.append({'my_deficit': my_deficit,
                                'received_pc': np.array(received_pc),
                                't': global_time,
                                'final_deals': self.get_attr('timestep_memory_mydeals')
                                }, ignore_index=True)

        memory = memory[['my_deficit',
                         'received_pc',  # all received original price curves by the excess agents as response to request
                         't',
                         'final_deals']]

        # self.set_attr(learning_memory=memory)
        memory.to_pickle(path_save + "temp_ln_" + str(data_names_dict[self.name]) + ".pkl")
        self.log_info("My learning memory updated (saved to file)")

        return



    def save_deal_to_memory(self, deal, global_time, req_alias):
        """
        "with_idx": memory based on the deal with this VPP
        "success": successful negotiation of failure, for deal is ofc successful
        "n_it": number of iteration to successful deal
        "n_ref": number of refuse offers in this process of negotiation
        "price": final price
        "quantity": final quantity
        "dt": day time (00:00 - 23:59) in minutes 0 - 288
        "wt": week time 1-7
        "mt": month time 1-12
        "po_idx": list of prospective opponents (all)
        "po_wf": forecast numbers of the prospective opponents (all)
        "ton": estimated "topology of negotiation" i.e. opponents in this negotiation

        :return:
        """
        myself = self.name

        # self.log_info("save_deal_to_memory, d: " + str(deal))
        # self.log_info("save_deal_to_memory, req_alias: " + str(req_alias))

        # memory = self.get_attr("learning_memory") <------------ now from file
        memory = pd.read_pickle(path_save + "temp_ln_" + str(data_names_dict[self.name]) + ".pkl")

        my_req = self.get_attr('requests')
        for req in my_req:
            if req['vpp_name'] == req_alias:
                req_value = req['value']

        # define prospective opponents in case of such a deal
        prospective_opponents_idx = np.where(np.array(adj_matrix[data_names_dict[req_alias]]) == True)
        po_idx = []
        for i in prospective_opponents_idx[0]:
            if i == data_names_dict[myself] or i == data_names_dict[req_alias]:
                continue
            po_idx.append(int(i))

        # convert timestamp to other times
        start_date = datetime.strptime(start_datetime, "%d/%m/%Y %H:%M")
        global_delta = timedelta(minutes=global_time * 5)
        current_time = start_date + global_delta

        # download the weather forecast of the opponents
        # known:        installed generators' power and forecasted max power factors
        # not known:    loads, prices, topology

        po_wf = {}
        res_installed_power_factor = {}
        av_weather_factor = {}
        for vpp_idx in po_idx:
            vpp_file = self.load_data(data_paths[vpp_idx])

            ppc0 = cases[vpp_file['case']]()
            max_generation0 = copy.deepcopy(ppc0['gen'][:, 8])

            forecast_max_generation_factor = np.zeros(vpp_file['bus_n'])
            forecast_max_generation = np.zeros(vpp_file['bus_n'])
            for idx in range(vpp_file['bus_n']):
                gen_path = vpp_file['generation_profiles_paths'][idx]
                if not gen_path == "":
                    d = self.load_data(gen_path)
                    forecast_max_generation_factor[idx] = d[global_time][MEASURED] + gen_mod_offset  # should be FORECASTED but for now 100% accuracy
                    forecast_max_generation[idx] = np.round(forecast_max_generation_factor[idx] * max_generation0[idx], 4)
                po_wf.update({vpp_idx: forecast_max_generation})

            generation_type = np.array(self.load_data(data_paths[vpp_idx])['generation_type'])

            stack = np.vstack((max_generation0, forecast_max_generation, generation_type))
            stack = stack[:, 1:].T

            # print("stack: " + str(stack))

            res_weather = 0
            max_res_weather = 0
            max_other = 0
            max = np.sum(stack[:, 0])
            for s in stack:
                if s[2] in weather_dependent_sources:
                    res_weather += s[1]
                    max_res_weather += s[0]
                else:
                    max_other += s[0]
            res_installed_power_factor.update({vpp_idx: round(max_res_weather / max, 4)})
            av_weather_factor.update({vpp_idx: round(res_weather / max_res_weather, 4)})

        ton = "unknown"

        bids_saldo = 0
        if self.get_attr('opfe3'):
            bids_saldo = np.round(self.get_attr('opfe3')['bids_revenue'], 4)
        elif self.get_attr('opfd3'):
            bids_saldo = -1 * np.round(self.get_attr('opfd3')['buybids_cost'], 4)

        if deal:  # if deal has happened
            deal_with = deal[0]
            gen_deals = deal[1]

            if req_value == np.round(np.sum(np.abs(gen_deals[:, 2])), 4):
                ton = []

            # create the record
            memory = memory.append({'with_idx_req': np.array([int(data_names_dict[deal_with]), req_value]),
                                    'success': True,   # its based on successful deals so far only
                                    'n_it': self.get_attr('n_iteration') + 1,
                                    'n_ref': "tbd",
                                    'price': gen_deals[:, 3],
                                    'quantity': np.round(np.abs(gen_deals[:, 2]), 3),
                                    'percent_req': np.round(np.sum(np.abs(gen_deals[:, 2])) / req_value, 4),
                                    'bids_saldo': bids_saldo,
                                    'pcf': self.get_attr('price_increase_factor'),
                                    'my_excess': self.get_attr('opf1')['exc_matrix'].T,
                                    't': global_time,
                                    'minute_t': int(current_time.hour * 60 + current_time.minute),
                                    'week_t': int(current_time.weekday()),
                                    'month_t': int(current_time.month),
                                    # 'po_idx': po_idx, # redundant, included in po_wf actually
                                    'po_wf': po_wf,
                                    'res_inst': res_installed_power_factor,
                                    'av_weather': av_weather_factor,
                                    'ton': ton
                                    }, ignore_index=True)

        else:  # there were no deal on that request:

            memory = memory.append({'with_idx_req': np.array([int(data_names_dict[req_alias]), req_value]),
                                    'success': False,  # its based on successful deals so far only
                                    'n_it': self.get_attr('n_iteration') + 1,
                                    'n_ref': "tbd",
                                    'price': self.get_attr('pc_memory_exc')[self.get_attr('n_iteration')]['all'][:, 2],
                                    'quantity': np.array(self.get_attr('pc_memory_exc')[self.get_attr('n_iteration')]['all'])[:, 1],
                                    'percent_req': 0,
                                    'bids_saldo': bids_saldo,
                                    'pcf': self.get_attr('price_increase_factor'),
                                    'my_excess': self.get_attr('opf1')['exc_matrix'].T,
                                    't': global_time,
                                    'minute_t': int(current_time.hour * 60 + current_time.minute),
                                    'week_t': int(current_time.weekday()),
                                    'month_t': int(current_time.month),
                                    # 'po_idx':
                                    'po_wf': po_wf,
                                    'res_inst': res_installed_power_factor,
                                    'av_weather': av_weather_factor,
                                    'ton': ton
                                    }, ignore_index=True)

        memory = memory[['with_idx_req',  # id of the vpp that we negotiate with and its request value
                         'success',  # success 1 or 0 failure
                         'n_it',  # number of iteration in the negotiation
                         'n_ref',  # number of refuses during the negotiation
                         'price',  # sell price(s) of particular generators
                         'quantity',  # amount(s) sold from particular generators
                         'percent_req',  # how much of the particular request was filled by my resources
                         'bids_saldo',  # positive if revenue negative if have to be bought
                         'pcf',  # price increase factor, that could be modified according to the vpp settings in json
                         'my_excess',
                         #  np.round(np.sum(np.array([gen_deals[:, 3] * np.round(np.abs(gen_deals[:, 2]), 3)])), 4),
                         't',  # simple global time
                         'minute_t',  # time in minutes of the day, weekday and month
                         'week_t',
                         'month_t',
                         # 'po_idx',  # redundant, included in po_wf actually
                         'po_wf',  # forecast numbers of the prospective opponents (all)
                         'res_inst',    # it is constant for particular vpp, but can be used to derive "weight"
                                        # of vpp in negotiation due to more or less RES
                         'av_weather',  # factors based on installed power and forecasts
                         'ton']]  # topology of negotiation

        # self.set_attr(learning_memory=memory)
        memory.to_pickle(path_save + "temp_ln_" + str(data_names_dict[self.name]) + ".pkl")
        self.log_info("My learning memory updated (saved to file)")

        return

    def similarity(self, t):
        """
        Calculates the best price increase factor according to the existing memory.
        :return:
        """

        features_weights = {"mem_requests": 0.15,
                            "minute_t": 0.3,
                            "mem_week_t": 0.15,
                            "month_t": 0.1,
                            "mem_av_weather": 0.3}

        my_idx = data_names_dict[self.name]

        # Load history
        path_dir = '/home/iso/Desktop/vpp_some_results/2018_0822_1555_week1-2/'
        path_pickle = path_dir + "temp_ln_" + str(my_idx) + ".pkl"
        f = pd.read_pickle(path_pickle)
        learn_memory = f.iloc[0:333, :]  # only the first week as learning
        learn_memory_mod = copy.deepcopy(learn_memory)

        ### Prepare memory - change values in cells for the needs (only once)
        path_memory = path_dir + "memory_ln_" + str(my_idx) + ".pkl"

        if not os.path.isfile(path_memory):  # if memory file has not been prepared so far
            # new columns:
            learn_memory_mod['mem_requests'] = learn_memory_mod.with_idx_req.apply(lambda x: x[1])
            learn_memory_mod['mem_requesters'] = learn_memory_mod.with_idx_req.apply(lambda x: x[0])
            learn_memory_mod['mem_week_t'] = learn_memory_mod.week_t.apply(lambda x: x + 1)
            # week and month are ready in original version
            learn_memory_mod['mem_av_weather'] = ""

            for index, row in learn_memory_mod.iterrows():

                res_now_power_all = 0  # potencial renewable power of all opponents together in Watts
                av_weather = row.loc['av_weather']
                for vpp_idx in av_weather.keys():
                    vpp_file = self.load_data(data_paths[vpp_idx])
                    ppc0 = cases[vpp_file['case']]()
                    installed_power = np.round(np.sum(ppc0['gen'][1:, 8]), 4)

                    res_inst = row['res_inst'][vpp_idx]
                    av_weather_onevpp = row['av_weather'][vpp_idx]
                    res_now_power = installed_power * res_inst * av_weather_onevpp
                    res_now_power_all += res_now_power

                row['mem_av_weather'] = res_now_power_all
                learn_memory_mod.iloc[index] = row

            learn_memory_mod.to_pickle(path_memory)
            self.log_info("Memory prepared and saved.")
        else:
            self.log_info("Memory has been already prepared before.")

        ### Calculate range of values values necessary for similarity
        fmem = pd.read_pickle(path_memory)

        # weather data (necessary for the similarity calculation) should be downloaded from the public data:
        start_date = datetime.strptime(start_datetime, "%d/%m/%Y %H:%M")
        global_delta = timedelta(minutes=t*5)
        current_time = start_date + global_delta

        minute_t_now = int(current_time.hour * 60 + current_time.minute)
        week_t_now = int(current_time.weekday()) + 1
        month_t_now = int(current_time.month)

        # loop through the requests to calculate similarity and the best price for each
        for r in range(int(self.get_attr('n_requests'))):
            request_value = self.get_attr("requests")[r]['value']
            deficit_agent = self.get_attr("requests")[r]['vpp_name']

            # define prospective opponents in case of such request
            prospective_opponents_idx = np.where(np.array(adj_matrix[data_names_dict[deficit_agent]]) == True)
            po_idx = []
            for i in prospective_opponents_idx[0]:
                if i == my_idx or i == data_names_dict[deficit_agent]:
                    continue
                po_idx.append(int(i))
            # download the weather forecasts of the prospective opponents
            po_wf = {}
            res_now_power_all = 0
            max_res_weather_all = 0
            for vpp_idx in po_idx:
                vpp_file = self.load_data(data_paths[vpp_idx])

                ppc0 = cases[vpp_file['case']]()
                max_generation0 = copy.deepcopy(ppc0['gen'][:, 8])

                forecast_max_generation_factor = np.zeros(vpp_file['bus_n'])
                forecast_max_generation = np.zeros(vpp_file['bus_n'])
                for idx in range(vpp_file['bus_n']):
                    gen_path = vpp_file['generation_profiles_paths'][idx]
                    if not gen_path == "":
                        d = self.load_data(gen_path)
                        forecast_max_generation_factor[idx] = d[t][MEASURED] + gen_mod_offset  # should be FORECASTED but for now 100% accuracy
                        forecast_max_generation[idx] = np.round(forecast_max_generation_factor[idx] * max_generation0[idx], 4)
                    po_wf.update({vpp_idx: forecast_max_generation})

                generation_type = np.array(self.load_data(data_paths[vpp_idx])['generation_type'])

                stack = np.vstack((max_generation0, forecast_max_generation, generation_type))
                stack = stack[:, 1:].T
                res_weather = 0
                max_res_weather = 0
                max_other = 0
                maxp = np.sum(stack[:, 0])
                for s in stack:
                    if s[2] in weather_dependent_sources:
                        res_weather += s[1]
                        max_res_weather += s[0]
                    else:
                        max_other += s[0]

                installed_power = np.round(np.sum(ppc0['gen'][1:, 8]), 4)
                res_inst = max_res_weather / maxp
                av_weather_onevpp = res_weather / max_res_weather
                res_now_power = np.round(installed_power * res_inst * av_weather_onevpp, 4)
                res_now_power_all += res_now_power
                max_res_weather_all += max_res_weather

            features_now = {
                "mem_requests": request_value,
                "minute_t": minute_t_now,
                "mem_week_t": week_t_now,
                "month_t": month_t_now,
                "mem_av_weather": res_now_power_all}

            features_ranges = {
                "mem_requests": np.round(np.abs(fmem['mem_requests'].max() - fmem['mem_requests'].min()), 4),
                "minute_t": 24 * 60 / 2,
                "mem_week_t": 1,
                "month_t": 12 / 2,
                "mem_av_weather": max_res_weather_all}

            ### calculate similarity for each tuple in the memory:
            # make new column for similarity with zeros:
            fmem['sim'] = 0.0

            # loop for each tuple in memory
            fmem = fmem.drop(fmem.index[[range(2, 333)]])

            for mem_row in fmem.itertuples():
                index = int(mem_row.Index)

                sim_sum = 0
                for label in features_weights.keys():
                    now = features_now[label]
                    mem = mem_row[fmem.columns.get_loc(label)+1]
                    weight = features_weights[label]
                    ran = features_ranges[label]

                    # print("label: " + str(label))
                    # print("now: " + str(now))
                    # print("mem: " + str(mem))
                    # print("weight: " + str(weight))
                    # print("range: " + str(ran))

                    if label == "minute_t":
                        diff = min(min(now, mem) + 60*24 - max(now, mem), abs(now - mem))
                    elif label == "mem_week_t":
                        if (now in [1,2,3,4,5] and mem in [6,7]) or (now in [6,7] and mem in [1,2,3,4,5]):
                            diff = 1
                        elif (now == 6 and mem == 7) or (now == 6 and mem == 7):
                            diff = 0.2
                        elif now == mem:
                            diff = 0
                        else:
                            diff = 0.1
                    elif label == "month_t":
                        diff = min(min(now, mem) + 12 - max(now, mem), abs(now - mem))
                    else:
                        diff = abs(now - mem)

                    ratio = diff / ran
                    sim1 = weight * (1 - ratio)
                    sim_sum += sim1

                #     print("diff: " + str(diff))
                #     print("ratio: " + str(ratio))
                #     print("sim1: " + str(sim1))
                #     print("\n")
                #
                # print(sim_sum)
                # print("\n\n")

                fmem.at[index, 'sim'] = np.round(sim_sum, 4)

            # print("all sim assigned to tuples for r=" + str(r))

            ### Choosing the best option and price for the proposal

            # 0) save original memory to file
            fmem.to_csv(path_save + "_temp_step0.csv")

            # 1) exclude unsuccessful ones
            select = fmem.index[fmem['bids_saldo'] == 0].tolist()
            fmem_mod = fmem.drop(fmem.index[select])

            # 2) select the ones with similarity more than treshold
            fmem_mod = fmem_mod.loc[fmem_mod['sim'] > similarity_treshold]
            fmem_mod = fmem_mod.sort_values(by=['sim'], ascending=False)

            # 3) select top X in in bids_saldo
            fmem_mod = fmem_mod.head(top_selection_quantity)

            # 4) calculate average of pcfs of all selected cases with that similarity
            pcfs = fmem_mod['pcf'].tolist()
            pcf_avg = np.round(np.sum(pcfs)/len(pcfs), 4)

            # ### ALTERNATIVE:
            # # select top 30 revenue and choose the most similar one
            # fmem_mod = fmem_mod.sort_values(by=['bids_saldo'], ascending=False)
            # fmem_mod = fmem_mod.head(30)
            # fmem_mod = fmem_mod.sort_values(by=['sim'], ascending=False)
            # pcf_avg = np.round(fmem_mod['pcf'].tolist()[0], 4)

            # print(fmem_mod[['bids_saldo', 'sim', 'pcf']])
            # print(pcf_avg)

            self.get_attr('requests')[r].update({'pcf_learning': pcf_avg})

        return pcf_avg