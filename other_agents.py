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
        ppc0 = self.time_modifications(ppc0, data_names_dict[self.name], t)
        ppc_t = copy.deepcopy(ppc0)
        print("ppc_t")
        print(ppc_t)

        fixed_load0 = copy.deepcopy(ppc0['bus'][:, 2])
        max_generation0 = copy.deepcopy(ppc0['gen'][:, 8])
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

        price_increase_factor = "not active"
        price_absolute_increase = "not active"

        if self.name not in vpp_exploit or (self.name in vpp_exploit and price_increase_policy == 1):
            ######## build price curve according to pc_matrix_price_increase_factor (gen prices * increase factor)
            # price increase factor modification should be calculated according to request number not to the timestamp
            # i.e. if the last request was at previous X timestamps then modify the factor
            price_increase_factor = self.load_data(data_paths[data_names_dict[self.name]])[
                'pc_matrix_price_increase_factor']

            if type(price_increase_factor) == list:
                currmem = pd.read_pickle(path_save + "temp_ln_" + str(data_names_dict[self.name]) + ".pkl")
                rows_range = currmem.tail(max_ts_range_for_price_modification)

                previous_fulfilled = 0  # check if the time exist and if some conditions are fulfilled: success
                if not rows_range.empty:
                    previous_fulfilled = np.sum(rows_range['success'].tolist())

                if previous_fulfilled == max_ts_range_for_price_modification:

                    previous_row = rows_range.tail(1)
                    previous_mod = previous_row.iloc[0]['pcf']

                    if previous_mod < price_increase_factor[1]:
                        price_increase_factor = previous_mod + price_increase_factor[2]
                    else:
                        price_increase_factor = price_increase_factor[0]

                else:  # if conditions not fulilled come back to the first value
                    if not constant_environment:
                        price_increase_factor = price_increase_factor[0]

                    else:  # if we assume exploration in a constant environment...
                        test_row = currmem.tail(1)
                        if not test_row.empty:
                            previous_mod = test_row.iloc[0]['pcf']
                            if test_row.iloc[0]['success'] == 1:
                                if previous_mod < price_increase_factor[1]:
                                    price_increase_factor = previous_mod + price_increase_factor[2]
                                else:
                                    price_increase_factor = price_increase_factor[0]
                            else:
                                price_increase_factor = price_increase_factor[0]
                        else:
                            price_increase_factor = price_increase_factor[0]

        if self.name in vpp_exploit and price_increase_policy == 2:
            ######## build price curve according to pc_matrix_price_absolute_increase (from, to, step)
            # it is not driven by the current price of excess generator (it does is by the excess amount though i.e. occurs only when excess exists).
            # also increases in case of X successes, it resets on failure

            price_increase_settings = self.load_data(data_paths[data_names_dict[self.name]])[
                'pc_matrix_price_absolute_increase']
            middle_offset = self.load_data(data_paths[data_names_dict[self.name]])[
                'pc_matrix_price_absolute_increase'][2] / 2.0
            # price increase vector need to be the lower boundries of the hypotheses not middle!
            price_increase_vector = np.arange(price_increase_settings[0],
                                              price_increase_settings[1],
                                              price_increase_settings[2])

            currmem = pd.read_pickle(path_save + "temp_ln_" + str(data_names_dict[self.name]) + ".pkl")

            if not currmem.empty:
                self.log_warning("currmem_pcf: " + str(currmem[['pcf']]))

            rows_range = currmem.tail(max_ts_range_for_price_modification)
            previous_fulfilled = 0  # check if the time exist and if some conditions are fulfilled: success
            if not rows_range.empty:
                previous_fulfilled = np.sum(rows_range['success'].tolist())

            if previous_fulfilled == max_ts_range_for_price_modification:
                previous_row = rows_range.tail(1)
                self.log_warning("previous_row: " + str(previous_row))

                previous_mod = previous_row.iloc[0]['pcf']
                print("previous_mod: " + str(previous_mod))

                if previous_mod < price_increase_vector[-1]:  # price_increase_settings[1]-price_increase_settings[2]:
                    price_absolute_increase = previous_mod + price_increase_settings[2]
                else:
                    price_absolute_increase = price_increase_vector[0]

            else:  # if conditions not fulilled come back to the first value
                if not constant_environment:
                    price_absolute_increase = price_increase_vector[0]
                else:  # if we assume exploration in a constant environment... CONSTANT ENVIRONMENT DEPRECIATED - NOT UPDATED TO PRICE INCREASE VECTOR !!!!!
                    test_row = currmem.tail(1)
                    if not test_row.empty:
                        previous_mod = test_row.iloc[0]['pcf']
                        if test_row.iloc[0]['success'] == 1:
                            if previous_mod < price_increase_settings[1]:
                                price_absolute_increase = previous_mod + price_increase_settings[2]
                            else:
                                price_absolute_increase = price_increase_settings[0]
                        else:
                            price_absolute_increase = price_increase_settings[0]
                    else:
                        price_absolute_increase = price_increase_settings[0]

        a = 'OFF'
        if exploit_mode and self.name in vpp_exploit:  # not prepared yet for price_absolute_increase
            self.log_info("I will apply memory through similarity.")
            if update_during_exploit:
                self.log_info("Live update_during_exploit is on!")
            if price_increase_policy == 1:
                price_increase_factor = self.similarity(t, 1)
            if price_increase_policy == 2:
                price_absolute_increase = self.similarity(t, 2) # this needs to be done! otherwise the part below will work only for the exploration
                if price_absolute_increase == 0:
                    price_increase_factor = self.load_data(data_paths[data_names_dict[self.name]])[
                'pc_matrix_price_increase_factor'][0]  # as emergency if no absolute increase
            a = 'ON'

        self.log_info("My runopf_e2 (I exploit: " + a + ") price_increase_factor: " + str(price_increase_factor))
        self.log_info("My runopf_e2 (I exploit: " + a + ") price_absolute_increase: " + str(price_absolute_increase))
        # self.set_attr(price_increase_factor=price_increase_factor)
        # self.set_attr(price_absolute_increase=price_absolute_increase)

        if self.name in vpp_exploit and price_increase_policy == 2:  # only for ML agents with policy==2
            self.set_attr(price_increased=price_absolute_increase)  # common parameter for both policies
        else:
            self.set_attr(price_increased=price_increase_factor)

        if n_iteration == 0:  # if its 0 iteration, make according to pr_matrix_price_increase_factor (each vpp has own)
            pc_matrix_incr = copy.deepcopy(exc_matrix.T)

            if self.name in vpp_exploit and price_increase_policy == 2 and price_absolute_increase != 0:
                # substitute only prices of the generators that have cheaper production costs then the derived price

                if exploit_mode == False:
                    pc_matrix_incr[:, 2] = price_absolute_increase # + self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase'][2] / 2
                else:  # exploit_mode == True
                    for g in range(len(pc_matrix_incr[:, 1])):
                        gen_row = pc_matrix_incr[g, :]
                        if gen_row[2] < price_absolute_increase:
                            gen_row[2] = price_absolute_increase
                            pc_matrix_incr[g, :] = gen_row

            else:
                pc_matrix_incr[:, 2] = pc_matrix_incr[:, 2] * price_increase_factor

            pc_matrix_incr = np.matrix(np.round(pc_matrix_incr, 4))

            # self.log_info("OPFe2: original PC matrix: " + str(exc_matrix))
            self.log_info("OPFe2: PC matrix for requesters (i.e. exc_matrix increased): " + str(pc_matrix_incr))
            self.set_attr(opfe2={'pc_matrix': np.array(pc_matrix_incr)})
            self.get_attr('pc_memory_exc')[n_iteration].update({'all': np.array(pc_matrix_incr)})
        ################################
        elif n_iteration > 0:  # not aligned to price absolute increase
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
        print(b)
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

        marginal_price = 0
        for deal in self.get_attr('timestep_memory_mydeals'):
            if np.max(deal[1][:, 3]) > marginal_price:
                marginal_price = np.max(deal[1][:, 3])

        #### VERY NON-UNIVERSAL
        # calculate the maximum possible revenue (after excluding one vpp):
        vppidx_to_delete = 2
        received_pc = np.array(received_pc)

        # delete vpp3 bids
        received_pc_novpp3 = np.delete(received_pc, (np.where(received_pc[:, 0] == vppidx_to_delete)), axis=0)
        # delete bids cheaper than vpp3's generation cost
        vpp3_mingencost = 9
        received_pc_novpp3 = np.delete(received_pc_novpp3, (np.where(received_pc[:, 3] < vpp3_mingencost)), axis=0)
        # subtract the cheaper deals... self.get_attr('timestep_memory_mydeals')
        final_deals = self.get_attr('timestep_memory_mydeals')
        sum = 0
        for fd in final_deals:
            v = fd[0]
            if v == "vpp3":
                continue
            d = fd[1]
            for dg in d:
                if dg[3] < vpp3_mingencost:
                    sum = sum + dg[2]
        #########################################

        need0 = abs(self.get_attr('opf1')['power_balance'])  # need should be the max(value of vpp3 excess!, need) - sum of cheaper
        excess = np.sum(received_pc[received_pc[:, 0] == vppidx_to_delete, 2])
        need = min(need0, excess) - sum

        ##########################################

        bids = []
        for pc in received_pc_novpp3:
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
        if bids.size == 0:
            th_max_rev_forvpp3 = 0
            th_max_rev_forvpp3_res1 = 0
        else:
            th_max_rev_forvpp3 = np.round(np.sum(bids[:, 2]*bids[:, 3]), 4)
            th_max_rev_forvpp3_res1 = np.round(np.sum(bids[:, 2]*np.floor(bids[:, 3])), 4)

        ### marginal_price_novpp3
        print("debuggg:")
        print(bids)
        if not bids.size ==0:
            marginal_price_novpp3 = np.max(bids[:, 3])
        else:
            marginal_price_novpp3 = 0

        memory = memory.append({'t': global_time,
                                'my_deficit': my_deficit,
                                'received_pc': np.array(received_pc),
                                'final_deals': final_deals,
                                'marginal_price': marginal_price,
                                'received_pc_novpp3': np.array(received_pc_novpp3),
                                'bids_novpp3': bids,
                                'marginal_price_novpp3': marginal_price_novpp3,
                                'th_max_rev_forvpp3': th_max_rev_forvpp3,
                                'th_max_rev_forvpp3_res1': th_max_rev_forvpp3_res1
                                }, ignore_index=True)
        memory = memory[['t',
                         'my_deficit',
                         'received_pc',  # all received original price curves by the excess agents as response to request
                         'final_deals',
                         'marginal_price',
                         'received_pc_novpp3',
                         'bids_novpp3',
                         'marginal_price_novpp3',
                         'th_max_rev_forvpp3',
                         'th_max_rev_forvpp3_res1']]

        memory.to_pickle(path_save + "temp_ln_" + str(data_names_dict[self.name]) + ".pkl")
        self.log_info("My learning memory updated (saved to file)")

        return


    def save_deal_to_memory(self, deal, global_time, req_alias, memory_update=False):
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
        my_idx = data_names_dict[self.name]
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

        po_wf = {}
        res_installed_power_factor = {}
        av_weather_factor = {}
        for vpp_idx in po_idx:
            vpp_file = self.load_data(data_paths[vpp_idx])

            ppc0 = cases[vpp_file['case']]()
            ppc0 = self.time_modifications(ppc0, vpp_idx, global_time)
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

            # total bids generation costs (if successful), i.e. only to produce the amount dealt
            print(deal)
            print(self.get_attr('opf1')['exc_matrix'].T)

            print(np.sum(-1*deal[1][:, 2]))
            print(np.array(self.get_attr('opf1')['exc_matrix'].T[:, 2]))

            excm = self.get_attr('opf1')['exc_matrix'].T
            a1 = deal[1][:, 1]
            b1 = np.array(self.get_attr('opf1')['exc_matrix'].T[:, 0])
            todel = np.setdiff1d(b1, a1)
            excm = np.delete(excm, (np.where(excm[:, 0] == todel)), axis=0)
            bids_gen_cost = np.round(np.sum(-1*deal[1][:, 2] * excm[:, 2]), 4)

            # create the record
            to_append = {'with_idx_req': np.array([int(data_names_dict[deal_with]), req_value]),
                                    'success': True,   # its based on successful deals so far only
                                    'n_it': self.get_attr('n_iteration') + 1,
                                    'n_ref': "tbd",
                                    'price': gen_deals[:, 3],
                                    'quantity': np.round(np.abs(gen_deals[:, 2]), 3),
                                    'pc': self.get_attr('pc_memory_exc'),
                                    'deal': deal[1],
                                    'percent_req': np.round(np.sum(np.abs(gen_deals[:, 2])) / req_value, 4),
                                    'bids_saldo': bids_saldo,
                                    'bids_gen_cost': bids_gen_cost,
                                    'pcf': self.get_attr('price_increased'),
                                    'h_mid': "",
                                    'sim_cases': self.get_attr('selection'),
                                    'my_excess': self.get_attr('opf1')['exc_matrix'].T,
                                    'exc_cost_range': [np.min(self.get_attr('opf1')['exc_matrix'][2, :]),
                                                       np.max(self.get_attr('opf1')['exc_matrix'][2, :])],
                                    't': global_time,
                                    'minute_t': int(current_time.hour * 60 + current_time.minute),
                                    'week_t': int(current_time.weekday()),
                                    'month_t': int(current_time.month),
                                    'po_wf': po_wf,
                                    'res_inst': res_installed_power_factor,
                                    'av_weather': av_weather_factor,
                                    'ton': ton
                                    }

            if memory_update and self.name in vpp_exploit:  # if it is update of the memory, during exploitation, after initial preparation

                res_now_power_all = 0
                for vpp_idx in av_weather_factor.keys():
                    vpp_file = self.load_data(data_paths[vpp_idx])
                    ppc0 = cases[vpp_file['case']]()
                    ppc0 = self.time_modifications(ppc0, vpp_idx, global_time)
                    installed_power = np.round(np.sum(ppc0['gen'][1:, 8]), 4)

                    res_inst = res_installed_power_factor[vpp_idx]
                    av_weather_onevpp = av_weather_factor[vpp_idx]
                    res_now_power = installed_power * res_inst * av_weather_onevpp
                    res_now_power_all += res_now_power

                to_append.update({'mem_requests': req_value,
                                    'mem_requesters': int(data_names_dict[deal_with]),
                                    'mem_week_t': int(current_time.weekday()) + 1,
                                    'mem_av_weather': res_now_power_all,
                                    'codf': 'tbd',  # change of deal factor -> there is no deal now to compare with
                                    'esf': 'tbd',  # environment similarity factor (discount)
                                    'mp_factor': 'tbd'
                                  })

        else:  # there were no deal on that request:

            price_minimum = min(self.get_attr('pc_memory_exc')[self.get_attr('n_iteration')]['all'][:, 2])

            to_append = {'with_idx_req': np.array([int(data_names_dict[req_alias]), req_value]),
                                    'success': False,  # its based on successful deals so far only
                                    'n_it': self.get_attr('n_iteration') + 1,
                                    'n_ref': "tbd",
                                    'price': np.array([price_minimum]),
                                    'quantity': np.array([self.get_attr('pc_memory_exc')[self.get_attr('n_iteration')]['all'][np.argmin(price_minimum), 1]]),
                                    'pc': self.get_attr('pc_memory_exc'),
                                    'deal': 0,
                                    'percent_req': 0,
                                    'bids_saldo': bids_saldo,
                                    'bids_gen_cost': 0,
                                    'pcf': self.get_attr('price_increased'),
                                    'h_mid': "",
                                    'sim_cases': self.get_attr('selection'),
                                    'my_excess': self.get_attr('opf1')['exc_matrix'].T,
                                    'exc_cost_range': [np.min(self.get_attr('opf1')['exc_matrix'][2, :]),
                                                       np.max(self.get_attr('opf1')['exc_matrix'][2, :])],
                                    't': global_time,
                                    'minute_t': int(current_time.hour * 60 + current_time.minute),
                                    'week_t': int(current_time.weekday()),
                                    'month_t': int(current_time.month),
                                    'po_wf': po_wf,
                                    'res_inst': res_installed_power_factor,
                                    'av_weather': av_weather_factor,
                                    'ton': ton
                                    }

            if memory_update and self.name in vpp_exploit:  # if it is update of the memory, during exploitation, after initial preparation

                res_now_power_all = 0
                for vpp_idx in av_weather_factor.keys():
                    vpp_file = self.load_data(data_paths[vpp_idx])
                    ppc0 = cases[vpp_file['case']]()
                    ppc0 = self.time_modifications(ppc0, vpp_idx, global_time)
                    installed_power = np.round(np.sum(ppc0['gen'][1:, 8]), 4)

                    res_inst = res_installed_power_factor[vpp_idx]
                    av_weather_onevpp = av_weather_factor[vpp_idx]
                    res_now_power = installed_power * res_inst * av_weather_onevpp
                    res_now_power_all += res_now_power

                to_append.update({'mem_requests': req_value,
                                    'mem_requesters': int(data_names_dict[req_alias]),
                                    'mem_week_t': int(current_time.weekday()) + 1,
                                    'mem_av_weather': res_now_power_all,
                                    'codf': 'tbd',
                                    'esf': 'tbd',
                                    'mp_factor': 'tbd'
                                  })

        memory = memory.append(to_append, ignore_index=True)
        memory = memory[['with_idx_req',  # id of the vpp that we negotiate with and its request value
                         'success',  # success 1 or 0 failure
                         'n_it',  # number of iteration in the negotiation
                         'n_ref',  # number of refuses during the negotiation
                         'price',  # sell price(s) of particular generators
                         'quantity',  # amount(s) sold from particular generators
                         'pc',
                         'deal', # all generators deals with this particular vpp together, it says "who is selling" i.e. me
                         'percent_req',  # how much of the particular request was filled by my resources
                         'bids_saldo',  # positive if revenue negative if have to be bought
                         'bids_gen_cost',
                         'pcf',  # price increase factor, that could be modified according to the vpp settings in json
                         'h_mid',
                         'sim_cases',  # quantity of the similar cases chosen from the memory before benchmark
                         'my_excess',  # matrix of excess generators from opf1
                         'exc_cost_range',  # the cheapest generation cost of excess generators
                         't',  # simple global time
                         'minute_t',  # time in minutes of the day, weekday and month
                         'week_t',
                         'month_t',
                         'po_wf',  # forecast numbers of the prospective opponents (all)
                         'res_inst',    # it is constant for particular vpp, but can be used to derive "weight"
                                        # of vpp in negotiation due to more or less RES
                         'av_weather',  # factors based on installed power and forecasts
                         'ton']]  # topology of negotiation

        memory.to_pickle(path_save + "temp_ln_" + str(data_names_dict[self.name]) + ".pkl")
        self.log_info("My learning memory updated (with exploration negotiation)")

        if memory_update and self.name in vpp_exploit:
            updated_memory_path = path_save + "updated_memory_ln_" + str(data_names_dict[self.name]) + ".pkl"
            if not os.path.isfile(updated_memory_path):  # make empty file and append
                initial_history = pd.read_pickle(path_dir_history + "memory_ln_" + str(data_names_dict[self.name]) + ".pkl")
                updated_memory = initial_history.append(to_append, ignore_index=True)
                updated_memory.to_pickle(updated_memory_path)
                updated_memory.to_csv(path_save + "updated_memory_ln_" + str(data_names_dict[self.name]) + "_view.csv")
            else:  # read existing file and append
                updated_memory = pd.read_pickle(updated_memory_path)
                updated_memory = updated_memory.append(to_append, ignore_index=True)
                updated_memory.to_pickle(updated_memory_path)
                updated_memory.to_csv(path_save + "updated_memory_ln_" + str(data_names_dict[self.name]) + "_view.csv")
            self.log_info("My learning memory (updated_memory) updated with current (exploitation) negotiation.")

            # memory should be updated with the esf mp_factors etc.
            # offset should make processing only for the new records
            self.prepare_memory(global_time, updated_memory_path, True)

        return

    def similarity(self, t, mode):
        """
        Calculates price increase factor according to the existing memory and assumption at the end of the function.
        :return: pcf
        """

        features_weights = {"mem_requests": 0.15,
                            "minute_t": 0.25,
                            "mem_week_t": 0.2,
                            "month_t": 0.1,
                            "mem_av_weather": 0.3}
        my_idx = data_names_dict[self.name]

        ### Prepare initial memory - change values in cells for the needs (only once)
        # choose only one if there are more negotiation for the same timestep! (e.g. the highest revenue one)
        path_initial_memory = path_dir_history + "memory_ln_" + str(my_idx) + ".pkl"
        updated_memory_path = path_save + "updated_memory_ln_" + str(data_names_dict[self.name]) + ".pkl"

        if not os.path.isfile(path_initial_memory):  # if memory file has not been prepared so far
            self.prepare_memory(t, path_initial_memory)
            self.log_info("Initial memory prepared and saved. Marginal prices (their pcfs) estimated.")
            fmem = pd.read_pickle(path_initial_memory)
        else:
            self.log_info("Initial memory and marginal prices pcfs have been already prepared before.")
            if not os.path.isfile(updated_memory_path):
                initial_history = pd.read_pickle(path_initial_memory)
                initial_history.to_pickle(updated_memory_path)
                initial_history.to_csv(path_save + "updated_memory_ln_" + str(data_names_dict[self.name]) + "_view.csv")
                fmem = pd.read_pickle(updated_memory_path)
            else:
                fmem = pd.read_pickle(updated_memory_path)

        # print("TAIL: ")
        # print(fmem.tail(5))

        ### Calculate similarity - prepare other data then memory
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
            res_now_power_all = 0  # like below but for
            max_res_weather_all = 0  # sums installed (max) power of the RESs in all prospective opponents (to calculate what's the RES weight of single opponent agent)
            for vpp_idx in po_idx:
                vpp_file = self.load_data(data_paths[vpp_idx])

                ppc0 = cases[vpp_file['case']]()
                ppc0 = self.time_modifications(ppc0, vpp_idx, t)
                max_generation0 = copy.deepcopy(ppc0['gen'][:, 8])  # this one is as a vector including slack bus

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

                res_weather = 0  # current sum of RES(weather dependant) production
                max_res_weather = 0  # installed sum of RES(weather dependant)
                max_other = 0  # installed sum of non-RES(weather dependant)
                installed_power = np.sum(stack[:, 0])  # total installed power RES(weather dependant) and non-RES(weather dependant)
                for s in stack:
                    if s[2] in weather_dependent_sources:
                        res_weather += s[1]
                        max_res_weather += s[0]
                    else:
                        max_other += s[0]

                # installed_power = np.round(np.sum(max_generation0[1:]), 4)
                res_inst = max_res_weather / installed_power
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
                # "mem_av_weather": np.round(np.abs(fmem['mem_av_weather'].max() - fmem['mem_av_weather'].min()), 4)}

            ### calculate similarity for each tuple in the memory:
            # make new column for similarity with zeros:
            fmem['sim'] = 0.0
            fmem['h_mid'] = 0.0

            # loop for each tuple in memory
            # fmem = fmem.drop(fmem.index[[range(2, 333)]])

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

            #################################################################
            ### Choosing the best option and price for the proposal #########
            #################################################################

            # 0) save original memory to file
            fmem.to_csv(path_save + "_temp_pre-sorting_memory.csv")

            # 0.1) load mp beliefs:
            updated_memory_path = path_save + "updated_memory_ln_" + str(data_names_dict[self.name]) + ".pkl"
            if update_mp_belief and os.path.isfile(updated_memory_path):
                updated_memory = pd.read_pickle(updated_memory_path)
                mp_factor_dict = updated_memory.iloc[-1]['mp_belief']
                # print(mp_factor_dict)
                mp_factor_table = pd.DataFrame.from_dict(mp_factor_dict)
                # print(mp_factor_table)
                mp_factor_table = mp_factor_table.loc[mp_factor_table['mp_factor_avg'] > mp_belief_treshold]
                mp_factor_table = mp_factor_table.sort_values(by=['mp_factor_avg'], ascending=False)
                mp_limits = np.round(mp_factor_table['pcfs_upper_h'].tolist(), 4)
                print("mp_factor_table: ")
                print(mp_factor_table)
                print("mp_limits (updated memory): " + str(mp_limits))

            else:
                mp_factor_table = pd.read_pickle(path_dir_history + "mp_belief_ln_" + str(my_idx) + ".pkl")
                mp_factor_table = mp_factor_table.loc[mp_factor_table['mp_factor_avg'] > mp_belief_treshold]
                mp_factor_table = mp_factor_table.sort_values(by=['mp_factor_avg'], ascending=False)
                mp_limits = np.round(mp_factor_table['pcfs_upper_h'].tolist(), 4)
                print("mp_factor_table: ")
                print(mp_factor_table)
                print("mp_limits (original memory): " + str(mp_limits))

            self.get_attr("selection").update({"mp_limits": mp_limits})

            ### Before calculating price, failures should be checked.
            # if X in a row for the hypothesis with high belief, we should start to decrease

            fmem_fail = fmem.tail(multiple_failure)
            print(fmem_fail[['t', 'sim', 'pcf', 'bids_saldo', 'mp_factor']])
            fmem_fail = fmem_fail.loc[fmem_fail['bids_saldo'] == 0]
            cond1 = False
            cond2 = False
            cond3 = False
            cond4 = False
            p = 0
            forced_price_decrease = 0

            print("fmem_fail: ")
            print(fmem_fail[['t', 'sim', 'pcf', 'bids_saldo', 'mp_factor']])

            if len(fmem_fail) == multiple_failure:
                cond1 = True

                f_dec = fmem.iloc[-1]['sim_cases']["forced_decrease"]
                print("f_dec: " + str(f_dec))
                if f_dec >= 1 and f_dec < multiple_failure:
                    self.get_attr("selection").update({"forced_decrease": f_dec + 1})
                    pcf_avg = fmem_fail.iloc[-1]['pcf'] - 1
                    return pcf_avg

                if fmem_fail.iloc[-1]['pcf'] == fmem_fail.iloc[-2]['pcf'] and fmem_fail.iloc[-2]['pcf'] == fmem_fail.iloc[-3]['pcf']:
                    p = fmem_fail.iloc[-1]['pcf']
                    cond2 = True
                    print("p: " + str(p))
                    print(mp_limits)
                if fmem_fail.iloc[-1]['bids_saldo'] == 0 and fmem_fail.iloc[-2]['bids_saldo'] == 0 and fmem_fail.iloc[-3]['bids_saldo'] == 0:
                    cond3 = True
                if np.any(np.isin(np.array(mp_limits), p)): # p + 1 to convert price to hypothesis
                    cond4 = True
            print(cond1)
            print(cond2)
            print(cond3)
            # print(cond4)
            if cond1 and cond2 and cond3:# and cond4:
                self.log_info("Multi-failure nagotiation failure (with high believed prices): "+str(multiple_failure)+"x!. Need to decrease the offers regardless of estimations.")
                self.get_attr("selection").update({"forced_decrease": 1})  # save decrease to memory for the next period
                pcf_avg = fmem_fail.iloc[-1]['pcf'] - 1

                return pcf_avg


            # 1) exclude unsuccessful ones
            select = fmem.index[fmem['bids_saldo'] == 0].tolist()
            fmem_mod = fmem.drop(fmem.index[select])

            print("2.1.1 condition:")
            fmem_mod = fmem_mod.loc[fmem_mod['mp_factor'] >= mp_factor_treshold_in_selection]
            print(fmem_mod[['t', 'sim', 'pcf', 'bids_saldo', 'mp_factor']])

            # 2) select the ones with similarity more than treshold
            fmem_mod = fmem_mod.loc[fmem_mod['sim'] > similarity_treshold]
            print(fmem_mod[['t', 'sim', 'pcf', 'bids_saldo', 'mp_factor']])

            # 2.1) order the chosen ones by
            fmem_mod = fmem_mod.sort_values(by=[order_by], ascending=False)
            self.get_attr("selection").update({deficit_agent: fmem_mod.shape[0]})

            # 3) select top X in in bids_saldo
            if not top_selection_quantity == 0:
                fmem_mod = fmem_mod.head(top_selection_quantity)
                print("SIM HEAD (top " + str(top_selection_quantity) + "): ")
                print(fmem_mod[['t', 'sim', 'pcf', 'mp_factor']])

            # 3.1) select only those where mp_factor >0
            # print("3.1 condition:")
            # fmem_mod = fmem_mod.loc[fmem_mod['mp_factor'] >= mp_factor_treshold_in_selection]
            # print(fmem_mod)

            print("SIM HEAD, mp_factor>"+str(mp_factor_treshold_in_selection)+": ")
            print(fmem_mod[['t', 'sim', 'pcf', 'bids_saldo', 'mp_factor']])
            print(fmem_mod.shape[0])

            self.get_attr("selection").update({"over_mpfactor": fmem_mod.shape[0]})

            # 4) calculate average of pcfs of all selected cases with that similarity
            if fmem_mod.empty:
                hmid_avg = 0
            else:
                ## needs to be checked which of the selected ones are normal, which forced_decrease (then pcf+1 if forced)
                fmem_mod = fmem_mod.reset_index()
                for index, row in fmem_mod.iterrows():
                    if 'forced_decrease' in row['sim_cases']:  # original history has those cells empty
                        if row['sim_cases']['forced_decrease'] > 0:
                            row['h_mid'] = np.round(row['pcf'] + self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase'][2] /2, 4)
                        else:
                            row['h_mid'] = np.round(row['pcf'] - self.load_data(data_paths[data_names_dict[self.name]])[
                                'pc_matrix_price_absolute_increase'][2] / 2, 4)
                    else:
                        row['h_mid'] = np.round(row['pcf'] - self.load_data(data_paths[data_names_dict[self.name]])[
                            'pc_matrix_price_absolute_increase'][2] / 2, 4)
                    fmem_mod.iloc[index] = row

                print("the table updated in case of chose forced_decrease rows (pcf+1):")
                print(fmem_mod[['t', 'sim', 'pcf', 'h_mid', 'bids_saldo', 'mp_factor']])

                pcfs = fmem_mod['pcf'].tolist()
                pcf_avg = np.round(np.sum(pcfs)/len(pcfs), 4)

                hmids = fmem_mod['h_mid'].tolist()
                hmid_avg = np.round(np.sum(hmids)/len(hmids), 4)

            # print("Average pcf from chosen list: " + str(pcf_avg))
            print("Average hmid from chosen list: " + str(hmid_avg))
            self.get_attr("selection").update({"raw_hmid_avg": hmid_avg})

            # 5.1) exclude the chosen ones (before averaging) that refer directly to other MPs (with high probability, high mp_factor)
            if use_pcf_hmid_exclude:
                print("\nhmid_avg is going to be modified in order to exclude rows refering to other MPs beliefs (origina hmid_avg): " + str(hmid_avg))
                print(fmem_mod[['t', 'sim', 'pcf', 'h_mid', 'bids_saldo', 'mp_factor']])

                hmid_exclude = fmem_mod['h_mid'].tolist()
                # print(hmid_exclude)  # before exclude
                # pcfs_exclude = fmem_mod['pcf'].tolist()

                # min distance to beliefs
                dist = 99999
                hypos_mid = np.round(mp_limits - self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase'][2]/2, 4)
                print("hypos_mid i.e. the mid values of hypothesis mp_limits: " + str(hypos_mid))
                for h_mid in hypos_mid:
                    d = np.abs(h_mid - hmid_avg)
                    if d < dist:
                        dist = d
                        min_dist_h_mid = h_mid

                self.log_info("MP belief (its middle value) closest to the initial h_mid average (" + str(hmid_avg) + "): " + str(min_dist_h_mid) + ". \nOther MP beliefs (mid) ("+str(np.setdiff1d(hypos_mid, min_dist_h_mid))+") excluded in calculation of the new average.")
                self.get_attr("selection").update({"closest_mid": min_dist_h_mid})

                for p in np.setdiff1d(hypos_mid, min_dist_h_mid):  # delete all the others, beside the minimum distance one
                    hmid_exclude = np.delete(hmid_exclude, np.where(hmid_exclude == p))
                self.log_info("The mid values left: " + str(hmid_exclude))

                if not len(hmid_exclude) == 0:  # if there are some left numbers afer excluding
                    hmid_avg_exclude = np.round(np.sum(hmid_exclude)/len(hmid_exclude), 4)
                else:
                    hmid_avg_exclude = hmid_avg
                    print("No left values! The original hmid_avg stays!")

                hmid_lower_boundry = self.value_in_hypothesis(hmid_avg_exclude, "lower")
                print("New average of middle values (after excluded): " + str(hmid_avg_exclude) + ". Lower boundry of hypothesis including that price: " + str(hmid_lower_boundry))
                print("Thus, this price (the lower boundry of hypothesis) is taken as the price to offer: " + str(hmid_lower_boundry))
                hmid_avg = hmid_avg_exclude
                self.get_attr("selection").update({"hmid_avg_exclude": hmid_avg_exclude})
                # self.get_attr("selection").update({"pcf_exclude": self.which_pcf_hypothesis(pcf_avg_exclude)})

            # 5) check the belief about the marginal prices
            if not do_not_exceed_mp_belief:
                print("Rages around the belief hypothesis limits DO NOT matter! ")
            if do_not_exceed_mp_belief:
                print("Rages around the belief hypothesis limits DO matter! ")
                print("hmid_avg before considering MP limits: " + str(hmid_avg))

                ranges = False
                hmin_of_ranges = 99999
                for h_mid in np.sort(hypos_mid):
                    if exceeding_or_vicinity: # see the settings description
                        mpl_down = np.round(h_mid - self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase'][2] / 2)  # gives the lower boundry of the hypothesis - it is the limit
                    else:
                        mpl_down = np.round(h_mid - self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase'][2] / 2 - mp_belief_range, 4)  # lower boundry like above minus the range

                    mpl_up = np.round(h_mid + self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase'][2] / 2 + mp_belief_range, 4)  # upper limit of the hypothesis (already the mp_limit) plus the range
                    print("Consideration limits (settings range is "+str(mp_belief_range)+"): " + str(mpl_down) + "; " + str(mpl_up))

                    if hmid_avg >= mpl_down and hmid_avg <= mpl_up:
                        ranges = True
                        if price_increase_policy == 1:
                            pcf_resolution = np.round(self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_increase_factor'][0]-1, 4)
                        if price_increase_policy == 2:
                            pcf_resolution = np.round(self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase'][2], 4)
                        print("hmid_avg ("+str(hmid_avg)+") in the range of considered MP limits!")
                        hmin_of_ranges = mpl_down

                if ranges:
                    final_price = np.round(hmin_of_ranges + mp_belief_range, 4)
                    print("hmid_avg after considering MP limits: " + str(hmin_of_ranges) + "\n\n")
                else:
                    print("hmid_avg not within the reanges\n\n")
                    final_price = hmid_avg

                self.get_attr('requests')[r].update({'final_price_hlow': final_price})

        self.get_attr("selection").update({"forced_decrease": 0})
        self.get_attr("selection").update({"final_price_hlow": final_price})
        return final_price

    def value_in_hypothesis(self, price, mode):
        """
        :return: returns either lower boundry or middle value or upper boundry of the hypothesis according to a price
        """
        data = self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase']
        vector = np.arange(data[0], data[1]+data[2], data[2])
        # print(vector)

        pcf_ranges = np.zeros((int(len(vector)-1), 4))
        pcf_ranges[0, 1] = data[0] - 1.0
        pcf_ranges[-1, 2] = data[1]
        # print(pcf_ranges)

        idxs = np.arange(0, len(vector)-1)
        # print(idxs)
        pcf_ranges[:, 0] = idxs
        pcf_ranges[:, 1] = np.arange(data[0], data[1], data[2])
        pcf_ranges[:, 2] = np.arange(data[0], data[1], data[2]) + data[2]
        pcf_ranges[:, 3] = np.arange(data[0], data[1], data[2]) + data[2]/2.0

        if mode == "idx":
            col = 0
        if mode == "lower":
            col = 1
        if mode == "upper":
            col = 2
        if mode == "middle":
            col = 3

        a = price > pcf_ranges[:, 1]
        b = price <= pcf_ranges[:, 2]
        # print(a)
        # print(b)
        # print(col)
        # print(pcf_ranges)
        chosen_value = int(pcf_ranges[a * b, col])

        # print("value_in_hypothesis: " + str(pcf_ranges))

        return chosen_value

    def prepare_memory(self, t, path_memory, update=False):  # offset used if not initial memory needs to be prepared

        my_idx = data_names_dict[self.name]

        # Load history
        path_pickle = path_dir_history + "temp_ln_" + str(my_idx) + ".pkl"
        if update:
            path_pickle = path_memory
        learn_memory = pd.read_pickle(path_pickle)
        mem_len = learn_memory.shape[0]

        # learn_memory = learn_memory.iloc[0:100, :]  # selection if necessary for example if only the first week as learning
        learn_memory_mod = copy.deepcopy(learn_memory)

        min_pcf = self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_increase_factor'][0]
        print("MIN_PCF=" + str(min_pcf) + ". If different then in the learning memory then should be adjusted !!!")

        gens_from_ppct = self.get_attr("opf1_ppct")["gen"][:, 0]  # idxes of gens from ppct
        gens_max_from_ppct = self.get_attr("opf1_ppct")["gen"][:, 8]  # idxes of gens from ppct (intalled power)

        if not update:
            # new columns:
            learn_memory_mod['mem_requests'] = np.array(learn_memory_mod.with_idx_req.tolist())[:, 1]
            learn_memory_mod['mem_requesters'] = np.array(learn_memory_mod.with_idx_req.tolist())[:, 0]
            learn_memory_mod['mem_week_t'] = [x + 1 for x in learn_memory_mod.week_t.tolist()]

            # week and month are ready in original version
            learn_memory_mod['mem_av_weather'] = ""

            learn_memory_mod['codf'] = ""   # !!! this should be per-my-generator! - it refers to my gens so should be divided
            learn_memory_mod['esf'] = ""    # it is ok like that (?) - only one for the environment i.e. other opponents
            learn_memory_mod['mp_factor'] = ""  # should be also divided into generators and thus into particular prices - refers to my own generators

            # additional factors:
            # 1) if percent_request increases
            # 2) if we sell anyway 100% of our excess, it is not MP
            learn_memory_mod['percent_of_excess_sold'] = ""
            # 3) if the decrease of request and decrease of deal are the same, ration ~1, it is not due to MP
            learn_memory_mod['cor2cod'] = ""  # change_of_request_to_change_of_deal

            # ---- factors per generator: excess, deal, deal_price, change of excess / change of deal ----

            # create empty colums for single generators
            for gen_idx in gens_from_ppct:
                if gen_idx == 0:
                    continue

                learn_memory_mod["g"+str(int(gen_idx))+"_excess"] = ""
                learn_memory_mod["g"+str(int(gen_idx))+"_deal"] = ""
                learn_memory_mod["g"+str(int(gen_idx))+"_offer_price"] = ""
                learn_memory_mod["g"+str(int(gen_idx))+"_coe2cod"] = ""
                learn_memory_mod["g"+str(int(gen_idx))+"_codrf"] = ""       # change of relative deal - like codf but per generator, percentage change

                learn_memory_mod["g" + str(int(gen_idx)) + "_mpf"] = ""     # mp factor per generator

            learn_memory_mod["mgen"] = ""
            learn_memory_mod["mp"] = ""
            learn_memory_mod["mp_belief"] = ""

        # DEACTIVATED create empty columns for all prospective opponents i.e. all vpps but myself:
        # for po_idx in data_names:
        #     if data_names == self.name:
        #         continue
        #     learn_memory_mod["mem_av_weather_"+str(po_idx)] = ""

        #####################################
        ### Calculations #################### if update, calculations should be done only for updated values (last row)
        #####################################

        # Extract values per my generators to columns:
        for index, row in learn_memory_mod.iterrows():
            if update:
                if not index == mem_len-1:
                    continue
            for gen_idx in gens_from_ppct:

                if gen_idx == 0:
                    continue
                col1_name = "g" + str(int(gen_idx)) + "_excess"
                col2_name = "g" + str(int(gen_idx)) + "_deal"
                col3_name = "g" + str(int(gen_idx)) + "_offer_price"

                if index != 0 and row['pcf'] != min_pcf:
                    my_excess_now = np.array(learn_memory_mod.iloc[index]['my_excess'])
                    col1 = np.round(my_excess_now[my_excess_now[:, 0] == gen_idx, 1], 4)
                    if col1.size == 0:
                        col1 = 0
                    my_deal_now = np.array(learn_memory_mod.iloc[index]['deal'])

                    if my_deal_now.size == 1:  # if no deals at all
                        col2 = 0
                        # pc = np.array(learn_memory_mod.iloc[index]['pc'])[0]['all']
                        # find = pc[pc[:, 0] == gen_idx, 2]
                        # if find.size == 0:
                        #     col3 = 0
                        # else:
                        col3 = 0
                            # col3 = pc[pc[:, 0] == gen_idx, 2][0]

                    else:  # if no deal of this generator in deals
                        col2 = np.round(-1 * my_deal_now[my_deal_now[:, 1] == gen_idx, 2], 4)
                        col3 = np.round(my_deal_now[my_deal_now[:, 1] == gen_idx, 3], 4)
                        if col2.size == 0:
                            col2 = 0
                        if col3.size == 0:
                            col3 = 0

                    row[col1_name] = float(col1)
                    row[col2_name] = float(col2)
                    row[col3_name] = float(col3)

                else:
                    row[col1_name] = 0
                    row[col2_name] = 0
                    row[col3_name] = 0

            learn_memory_mod.iloc[index] = row

        # Change of deal factors etc per generator!
        for index, row in learn_memory_mod.iterrows():
            if update:
                if not index == mem_len-1:
                    continue
            if index != 0 and row['pcf'] != min_pcf:
                for gen_idx in gens_from_ppct:
                    if gen_idx == 0:
                        continue
                    col1_name = "g" + str(int(gen_idx)) + "_codrf"
                    col2_name = "g" + str(int(gen_idx)) + "_coe2cod"

                    codaf = round(learn_memory_mod.iloc[index-1]["g" + str(int(gen_idx)) + "_deal"] -
                              learn_memory_mod.iloc[index]["g" + str(int(gen_idx)) + "_deal"], 4)
                    previous_deal = learn_memory_mod.iloc[index-1]["g" + str(int(gen_idx)) + "_deal"]
                    if previous_deal == 0:
                        codrf = 0
                    else:
                        codrf = round(codaf / gens_max_from_ppct[int(gen_idx)], 4)  # relatively to installed power of this generator
                        # codrf = round(codaf / learn_memory_mod.iloc[index-1]["g" + str(int(gen_idx)) + "_deal"], 4)  # relatively to the previous deal, depreciated
                    if codaf == 0:
                        coe2cod = -1
                    else:
                        coe2cod = (learn_memory_mod.iloc[index-1]["g" + str(int(gen_idx)) + "_excess"] -
                                learn_memory_mod.iloc[index]["g" + str(int(gen_idx)) + "_excess"]) / codaf # change of excess to change of deal

                    row[col1_name] = float(codrf)
                    row[col2_name] = float(coe2cod)

                learn_memory_mod.iloc[index] = row

        # ---- Calculating the potential renewable power of all opponents together in Watts ----
        for index, row in learn_memory_mod.iterrows():
            if update:
                if not index == mem_len-1:
                    continue

            res_now_power_all = 0
            av_weather = row.loc['av_weather']
            for vpp_idx in av_weather.keys():
                vpp_file = self.load_data(data_paths[vpp_idx])
                ppc0 = cases[vpp_file['case']]()
                ppc0 = self.time_modifications(ppc0, vpp_idx, t)
                installed_power = np.round(np.sum(ppc0['gen'][1:, 8]), 4)

                res_inst = row['res_inst'][vpp_idx]
                av_weather_onevpp = row['av_weather'][vpp_idx]
                res_now_power = installed_power * res_inst * av_weather_onevpp
                # DEACTIVATED ########## additional weather rows of av_weather for each opponent
                # col_name = "mem_av_weather_" + str(data_names[int(vpp_idx)])
                # row[col_name] = np.round(res_now_power, 4)
                ##########
                res_now_power_all += res_now_power

            row['mem_av_weather'] = res_now_power_all
            learn_memory_mod.iloc[index] = row

        ##########################33
        #### Calculate marginal price factors (mp_factor) and estimate the marginal prices factors/probabilites, and other stuff per generator
        ############################

        for index, row in learn_memory_mod.iterrows():
            if update:
                if not index == mem_len-1:
                    continue
            if index != 0 and row['pcf'] != min_pcf:

                # change of (successful) deal factor:
                codf = -1*(learn_memory_mod.iloc[index]['percent_req'] - learn_memory_mod.iloc[index-1]['percent_req'])

                my_excess_sum = np.sum(np.array(learn_memory_mod.iloc[index]['my_excess'])[:, 1])
                percent_of_excess_sold = np.abs(learn_memory_mod.iloc[index]['percent_req']*learn_memory_mod.iloc[index]['mem_requests']/my_excess_sum)

                cor2cod = np.abs((learn_memory_mod.iloc[index]['mem_requests']-learn_memory_mod.iloc[index-1]['mem_requests'])/  # change of request to change of deal
                                 (learn_memory_mod.iloc[index]['percent_req']*learn_memory_mod.iloc[index]['mem_requests']-
                                  learn_memory_mod.iloc[index-1]['percent_req']*learn_memory_mod.iloc[index-1]['mem_requests']))

                # environment similarity factor (discount of codf due to the change of environment):
                esf = self.environment_similarity_factor(learn_memory_mod, index)
                esf2 = esf*esf*esf
                mp_factor = np.round(codf * esf2, 4)
                mp_factor = np.abs(mp_factor)

                if ("forced_decrease" not in row['sim_cases']) or (row['sim_cases']["forced_decrease"] == 0):  # conditions designed earlier for mp_factor estimations in normal, increasing conditions
                    if codf < 0:
                        mp_factor = -1
                    if percent_of_excess_sold > 0.99 and percent_of_excess_sold < 1.01:
                        mp_factor = -1
                    if cor2cod > 0.99 and cor2cod < 1.01:
                        mp_factor = -1
                # else:  # in case we are under forced increase of the price, do not modify the mp_factor

                row['codf'] = codf
                row['esf'] = esf2
                row['cor2cod'] = cor2cod
                row['percent_of_excess_sold'] = percent_of_excess_sold

            else:
                mp_factor = -1
                # codf = -1
                # esf2 = -1
                # cor2cod = -1
                # percent_of_excess_sold = -1

            row['mp_factor'] = mp_factor
            learn_memory_mod.iloc[index] = row

        # per-generator part
        for index, row in learn_memory_mod.iterrows():
            if update:
                if not index == mem_len-1:
                    continue
            mpf_test = 0
            mgen = ""
            mp = ""

            for gen_idx in gens_from_ppct:

                if gen_idx == 0:
                    continue
                col1_name = "g" + str(int(gen_idx)) + "_mpf"

                g_excess = learn_memory_mod.iloc[index]["g" + str(int(gen_idx)) + "_excess"]
                g_deal = learn_memory_mod.iloc[index]["g" + str(int(gen_idx)) + "_deal"]
                g_offer_price = learn_memory_mod.iloc[index]["g" + str(int(gen_idx)) + "_offer_price"]
                g_coe2cod = learn_memory_mod.iloc[index]["g" + str(int(gen_idx)) + "_coe2cod"]

                if g_coe2cod == "":
                    g_coe2cod = 0
                g_codrf = learn_memory_mod.iloc[index]["g" + str(int(gen_idx)) + "_codrf"]
                if g_codrf == "":
                    g_codrf = 0

                if index != 0 and row['pcf'] != min_pcf:

                    esf = self.environment_similarity_factor(learn_memory_mod, index)
                    esf2 = esf * esf
                    mpf0 = np.round(g_codrf * esf2, 4)

                    if mpf0 < 0:
                        mpf0 = -1
                    if g_coe2cod > 0.99 and g_coe2cod < 1.01:
                        mpf0 = -1
                else:
                    mpf0 = 0

                row[col1_name] = mpf0

                if mpf0 > mpf_test:  # in order to find the maximum generator g_codrf in this timestamp
                    mpf_test = mpf0
                    mgen = gen_idx
                    mp = g_offer_price

            if mpf_test > 0:
                row["mgen"] = mgen
                row["mp"] = mp

            learn_memory_mod.iloc[index] = row

        ###############################################
        ### Make a belief table based on mp_factors ### # also needs to be updated !!
        ###############################################

        if not update:
            mp_factors_allsum = np.sum(learn_memory_mod[learn_memory_mod['mp_factor'] > 0]['mp_factor'].tolist())
            pcfs_list_memory = learn_memory_mod['pcf'].unique()  # all pcfs 3:20
            self.log_info("PCFs from memory: " + str(pcfs_list_memory))
            sett = self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase']
            pcfs_settings = np.arange(sett[0]+sett[2], sett[1]+sett[2], sett[2])  # should be 4:20 - upper boundries
            self.log_info("PCFs from settings (hypoth's lower boundries): " + str(pcfs_settings))
            # pcfs_list = learn_memory_mod[learn_memory_mod['mp_factor'] > 0]['pcf'].unique()  # only pcf where mp_factor is >0 in whole set
            pcfs_list = np.intersect1d(np.array(pcfs_list_memory), pcfs_settings)
            self.log_info("I am taking the intersection: " + str(pcfs_list))
            pcfs = []
            mp_factor_avg = []
            for pcf in pcfs_list:
                mp_factor_forpcf = learn_memory_mod[learn_memory_mod['pcf'] == pcf]
                mp_factor_forpcf_positive = mp_factor_forpcf[mp_factor_forpcf['mp_factor'] > 0]
                # print("mp_factor_forpcf_positive: " + str(mp_factor_forpcf_positive))
                mp_factor_forpcf_sum = np.sum(mp_factor_forpcf_positive['mp_factor'])
                # print("mp_factor_forpcf_sum: " + str(mp_factor_forpcf_sum))
                mp_factor_weight = np.round((mp_factor_forpcf_sum / mp_factors_allsum), 4)
                # print("mp_factor_weight: " + str(mp_factor_weight))
                # print("---------------------------- ")

                pcfs.append(pcf)
                mp_factor_avg.append(mp_factor_weight)

            data_dict = {'pcfs_upper_h': pcfs, 'mp_factor_avg': mp_factor_avg}
            # update the mp_beliefs to the last row of the prepared memory (from exploration)
            learn_memory_mod['mp_belief'].iloc[[-1]] = [data_dict]

        else:  # if update
            last_mp_belief = learn_memory_mod.iloc[-2]['mp_belief']  # -1 is current row
            # print("last_mp_belief: " + str(last_mp_belief))
            if update_mp_belief:
                # print("last mp_factor: " + str(learn_memory_mod.iloc[-1]['mp_factor']))
                if learn_memory_mod.iloc[-1]['mp_factor'] <= 0:  # if so, then no change - rewrite the previous one
                    learn_memory_mod['mp_belief'].iloc[[-1]] = [last_mp_belief]
                else:  # otherwise make whole BL based update
                    mpf = learn_memory_mod.iloc[-1]['mp_factor']  # my row
                    pri = learn_memory_mod.iloc[-1]['pcf']  # my row
                    P_H_dict = learn_memory_mod.iloc[-2]['mp_belief']  # previous row

                    a = self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase'][0]
                    b = self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase'][1]
                    c = self.load_data(data_paths[data_names_dict[self.name]])['pc_matrix_price_absolute_increase'][2]
                    thetaH = np.arange(a, b+c, c)
                    # print(thetaH)

                    bias = -0.5
                    if learn_memory_mod.iloc[-1]['sim_cases']['forced_decrease'] > 0:
                        bias = 0.5
                        print("BL update with forced decrease. This implies pcf being the lower boundry of a hypothesis, therefore pri value in the function increased to the upper boundry.")

                    # pri_mod is just to convert lower boundry to middle value. To focus belief on the hypothesis instead of on two hyp. if pri is on the boundry
                    # pri_mod = c/2.0

                    print("Triggering P_He_constrained for pri: " + str(pri) + " with bias: " + str(bias) + ". mp_factor=" + str(mpf))
                    print(P_H_dict)
                    P_He = self.P_He_constrained(P_H_dict, pri + bias, mpf, thetaH, pow, multi, theta_constraints, c)
                    print("== back to main ===> new P_He: ")
                    print(P_He)

                    P_He_dict = copy.deepcopy(P_H_dict)
                    P_He_dict['mp_factor_avg'] = P_He.tolist()
                    # print(P_H_dict)
                    # print(P_He_dict)

                    learn_memory_mod['mp_belief'].iloc[[-1]] = [P_He_dict]

                # print("after ASDASD: " + str(learn_memory_mod.iloc[[-3,-2,-1]][['t', 'mp_belief']]))

        # save to pickle and csv - both original and updated memory
        if update:
            updated_memory_path = path_save + "updated_memory_ln_" + str(data_names_dict[self.name]) + ".pkl"
            learn_memory_mod.to_pickle(updated_memory_path)
            learn_memory_mod.to_csv(path_save + "updated_memory_ln_" + str(data_names_dict[self.name]) + "_view.csv")
            return
        else:
            learn_memory_mod.to_pickle(path_memory)
            learn_memory_mod.to_csv(path_dir_history + "memory_ln_full_" + str(
                my_idx) + "_view.csv")


        # Save mp_belief to the separate file in the originial history folder - old. Now I save to prepared memory
        mpf_frame = pd.DataFrame(data=data_dict)
        mpf_frame.to_pickle(path_dir_history + "mp_belief_ln_" + str(my_idx) + ".pkl")
        mpf_frame.to_csv(path_dir_history + "mp_belief_ln_" + str(my_idx) + "_view.csv")


        ### Select only one for each timestep i.e. when more timesteps exist
        selection_idx = []
        t_vector = learn_memory_mod['t'].unique()
        if t_vector.size == learn_memory_mod['t'].size:
            print("No reduction of the memory necessary, becase no repetitions for the same timestamp.")
        else:
            for tt in t_vector:
                a = learn_memory_mod.loc[learn_memory_mod['t'] == tt]
                max_bidsaldo_idx = a['bids_saldo'].idxmax()
                if learn_memory_mod.iloc[max_bidsaldo_idx]['bids_saldo'] == 0:
                    max_bidsaldo_idx += 1
                selection_idx.append(max_bidsaldo_idx)
            learn_memory_mod = learn_memory_mod.iloc[selection_idx]
            learn_memory_mod = learn_memory_mod.reset_index()

            # save to pickle and csv - REDUCED memory
            learn_memory_mod.to_pickle(path_memory)
            learn_memory_mod.to_csv(path_dir_history + "memory_ln_reduced_" + str(my_idx) + "_view.csv") # change it as the path_initial_memory!!!!!!!!!!
        return


    def P_He_constrained(self, P_Hd, pri, mp_factor, thetaH, pow, multi, theta_constraints, prices_step):
        """

        :param pri: price relative increase, event
        :param mpf: mp_factor, event
        :param thetaH: range of hypothesis in absolute price values
        :param pow: variable for the function translating mp_factor to parameters for beta-distribution
        :param multi: variable for the function translating mp_factor to parameters for beta-distribution
        :param theta_constraints_idx: constraints for the range around the event price, to limit the influence of on event on other marginal prices
        :return:
        """
        # print("Triggering P_He_constrained")
        # print(mp_factor)
        print("\n\n")
        print(pri)
        # print(thetaH)  # 3:1:20

        P_H = np.array(P_Hd['mp_factor_avg'])

        # fake values
        # pri = 17.20
        # mp_factor = 0.8999

        n_hypoth = len(thetaH) - 1  # 17
        thetaH_idx = np.arange(0, n_hypoth)  # 0...16
        thetaH_idx_ranges = np.transpose(np.concatenate((np.array([thetaH_idx]), np.array([thetaH[0:-1]]), np.array([thetaH[0:-1]+prices_step])), axis=0))
        print("thetaH_idx_ranges")
        print(thetaH_idx_ranges)
        # print(thetaH_idx_ranges.shape)
        resolution_normalized = 1/n_hypoth
        theta = np.divide(thetaH_idx, n_hypoth)

        min_probability = resolution_normalized/10
        P_H[P_H < min_probability] = min_probability
        P_H = np.round(P_H / np.sum(P_H), 4)

        # print(P_H)
        # print("--- in function ---")
        # print(n_hypoth)
        # print(thetaH_idx)
        # print(resolution_normalized)
        # print(theta)

        cont_res = 0.001
        # B norm definition

        if np.isin((pri - theta_constraints), thetaH) and np.isin((pri + theta_constraints), thetaH):
            decim = 0
        else:
            decim = 1
        n_hypoth_constr = 2*theta_constraints/prices_step + decim

        # print("n_hypoth_constr: " + str(n_hypoth_constr))
        #
        # print("\n")
        # print(thetaH)
        # print(pri)
        # print(P_Hd)
        #
        # print("P_H: " + str(P_H))
        # print("pri: " + str(pri))
        # print(thetaH == pri)
        # print("mpf: " + str(mp_factor))

        # row beta-distribution part
        theta01 = np.arange(0, 1+cont_res, cont_res)
        alpha = multi * (mp_factor ** pow)
        # print("alpha01: " + str(alpha))
        if alpha < 1:
            P_He = P_H
            P_He[P_He < min_probability] = min_probability
            P_He = np.round(P_He / np.sum(P_He), 4)
            return P_He
        beta = (alpha - 1 + 2*0.5 - 0.5*alpha)/0.5
        # print("beta01: " + str(beta))
        B01 = self.B_norm(alpha, beta, theta01, cont_res)
        # print(B01)
        cont_probability = (theta01**(alpha-1) * (1-theta01)**(beta-1)) / B01
        # print("should be 1:" + str(np.sum(cont_probability*cont_res)))

        pri_ranges = np.zeros((int(n_hypoth_constr), 2))
        # print("pri_ranges0: " + str(pri_ranges))
        pri_start = pri-theta_constraints
        pri_ranges[0, 0] = pri_start
        pri_max = pri + theta_constraints
        pri_ranges[-1, -1] = pri_max
        thetaHext = np.sort(np.concatenate([thetaH, np.array([pri_start])]))

        # print(thetaHext)
        thetaHext_greater = thetaHext[thetaHext > pri_start]
        # print(thetaHext_greater)
        # print("pri_ranges: " + str(pri_ranges))
        # print("pri_max: " + str(pri_max))
        # print("\n")
        for nh in range(0, int(n_hypoth_constr)):
            # print(nh)
            if thetaHext_greater[nh] < pri_max:
                pri_ranges[nh, 1] = thetaHext_greater[nh]
                pri_ranges[nh+1, 0] = thetaHext_greater[nh]
                # print("pri_ranges loop: " + str(pri_ranges))
            else:
                break

        print("pri_ranges: " + str(pri_ranges))
        ranges = pri_ranges - pri_start
        ranges_norm = ranges / ranges[-1, -1]

        # print("ranges_norm: " + str(ranges_norm))

        P_eH_constr = np.empty(int(n_hypoth_constr))
        # print(P_eH_constr)
        for nh in range(0, int(n_hypoth_constr)):
            # print(nh)
            start = np.round(ranges_norm[nh, 0] + cont_res, 4)
            stop = np.round(ranges_norm[nh, 1], 4)
            # print(start)
            # print(stop)

            theta_prob = np.arange(start, stop+10e-6, cont_res)
            # print("prob-1: " + str(((1.0**(alpha-1)) * ((1-1.0)**(beta-1))) / B01))
            prob = ((theta_prob**(alpha-1.0)) * ((1.0-theta_prob)**(beta-1.0))) / B01
            prob = np.nan_to_num(prob)
            # print("B01: " + str(B01))
            # print("prob1: " + str(prob))

            # print("prob2: " + str(prob*cont_res))
            # print("prob3: " + str(np.sum(prob*cont_res)))
            P_eH_constr[nh] = np.sum(prob*cont_res)

        print("P_eH_constr: " + str(P_eH_constr))
        # print("Should be one: " + str(np.sum(P_eH_constr)))

        a = pri > thetaH_idx_ranges[:, 1]
        b = pri <= thetaH_idx_ranges[:, 2]
        middle_idx = thetaH_idx_ranges[a*b, 0]

        a = pri_start > thetaH_idx_ranges[:, 1]
        b = pri_start <= thetaH_idx_ranges[:, 2]
        down_idx = thetaH_idx_ranges[a * b, 0]

        a = pri_max > thetaH_idx_ranges[:, 1]
        b = pri_max <= thetaH_idx_ranges[:, 2]
        up_idx = thetaH_idx_ranges[a * b, 0]

        thetaH_idx_constr = np.arange(down_idx, up_idx+1)
        # print(P_H)
        print("Indexes of the hypoths, thetaH_idx_constr: ")
        print(thetaH_idx_constr)

        other_ones_idx = np.setdiff1d(thetaH_idx, thetaH_idx_constr)
        sum_other_ones = np.sum(P_H[other_ones_idx.astype(int)])

        P_H_constr = P_H[thetaH_idx_constr.astype(int)]
        eps_sum = 0

        # print(P_eH_constr)
        # print(P_H_constr)

        for i in range(0, int(n_hypoth_constr)):
            a = P_eH_constr[i] * P_H_constr[i]
            eps_sum = eps_sum + a
        # print("eps_sum: " + str(eps_sum))

        P_He_constr = copy.deepcopy(P_H_constr)
        for i in range(0, int(n_hypoth_constr)):
            # print(i)
            # print(P_H_constr[i] * P_eH_constr[i] / eps_sum)
            P_He_constr[i] = P_H_constr[i] * P_eH_constr[i] / eps_sum

        P_He_constr = P_He_constr/np.sum(P_He_constr)

        print("P_He_constr: " + str(P_He_constr))
        P_He_constr = P_He_constr * (1-sum_other_ones)
        print("P_He_constr rescaled according to (1-sum of other ones): " + str(P_He_constr))

        P_He_constr_put = copy.deepcopy(P_H)
        P_He_constr_put[thetaH_idx_constr.astype(int)] = P_He_constr
        # print("first P_He_constr_put: " + str(P_He_constr_put))
        # print("min_probability: " + str(min_probability))

        P_He_constr_put[P_He_constr_put < min_probability] = min_probability
        # print("second P_He_constr_put: " + str(P_He_constr_put))
        # print("sum: " + str(np.sum(P_He_constr_put)))
        P_He_constr_put = np.round(P_He_constr_put / np.sum(P_He_constr_put), 4)

        # print("final P_He_constr_put: " + str(P_He_constr_put))

        return P_He_constr_put


    def B_norm(self, alpha, beta, theta, const_res):
        return np.sum(theta ** (alpha-1) * (1-theta) ** (beta-1))*const_res


    def environment_similarity_factor(self, learn_memory_mod, index):
        """
        Function should calculate the "difference in the environment" between the t timestep and the t-1 timestep.
        The environment features are taken the same one like in the similarity calculations for price increase factor
        :param t:
        :return:
        """

        features_weights_esim = {"mem_requests": 0.0,
                                 "minute_t": 0.05,
                                 "mem_week_t": 0.0,
                                 "month_t": 0.0,
                                 "mem_av_weather": 0.95}

        ##### this part redundant >>>
        my_idx = data_names_dict[self.name]
        for r in range(int(self.get_attr('n_requests'))):
            request_value = self.get_attr("requests")[r]['value']
            deficit_agent = self.get_attr("requests")[r]['vpp_name']
            prospective_opponents_idx = np.where(np.array(adj_matrix[data_names_dict[deficit_agent]]) == True)
            po_idx = []
            for i in prospective_opponents_idx[0]:
                if i == my_idx or i == data_names_dict[deficit_agent]:
                    continue
                po_idx.append(int(i))

            features_ranges_esim  = {
                "mem_requests": np.round(np.abs(learn_memory_mod['mem_requests'].max() - learn_memory_mod['mem_requests'].min()), 4),
                "minute_t": 24 * 60 / 2,
                "mem_week_t": 1,
                "month_t": 12 / 2,
                "mem_av_weather": np.round(np.abs(learn_memory_mod['mem_av_weather'].max() - learn_memory_mod['mem_av_weather'].min()), 4)}

        ##### <<< this part redundant (but shorter)

            features_now = {
                "mem_requests": learn_memory_mod.iloc[index]['mem_requests'],
                "minute_t": learn_memory_mod.iloc[index]['minute_t'],
                "mem_week_t": learn_memory_mod.iloc[index]['mem_week_t'],
                "month_t": learn_memory_mod.iloc[index]['month_t'],
                "mem_av_weather": learn_memory_mod.iloc[index]['mem_av_weather']}

            features_before = {
                "mem_requests": learn_memory_mod.iloc[index-1]['mem_requests'],
                "minute_t": learn_memory_mod.iloc[index-1]['minute_t'],
                "mem_week_t": learn_memory_mod.iloc[index-1]['mem_week_t'],
                "month_t": learn_memory_mod.iloc[index-1]['month_t'],
                "mem_av_weather": learn_memory_mod.iloc[index-1]['mem_av_weather']}

            # print(features_now)
            # print(features_before)

            ### calculate similarity for each tuple in the memory:

            sim_sum = 0
            for label in features_weights_esim.keys():
                now = features_now[label]
                mem = features_before[label]
                weight = features_weights_esim[label]
                ran = features_ranges_esim[label]

                # print("label: " + str(label))
                # print("now: " + str(now))
                # print("mem: " + str(mem))
                # print("weight: " + str(weight))
                # print("range: " + str(ran))

                if label == "minute_t":
                    diff = min(min(now, mem) + 60 * 24 - max(now, mem), abs(now - mem))
                elif label == "mem_week_t":
                    if (now in [1, 2, 3, 4, 5] and mem in [6, 7]) or (now in [6, 7] and mem in [1, 2, 3, 4, 5]):
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

            return sim_sum


    def time_modifications(self, ppc0, vpp_idx, t):
        # if vpp_idx == 3 and t >= 2114:
        #     ppc0['gencost'][3, 4] = 13

        return ppc0