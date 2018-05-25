from osbrain import Agent
import json
from settings_3busML import *
#from utilities import system_state_update_and_balance
import time
from pprint import pprint as pp
import copy
import sys

from pypower.api import *
from pypower_mod.rundcopf_noprint import rundcopf
from pypower_mod.rundcpf_noprint import rundcpf


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

    def runopf1(self, t):
        """
        This should be internal PF in order to define excess/deficit.
        Updates the system ppc at time t according to data and runs opf.
        If excess, derive price curve.
        OPF1b: Check the feasibility and profitability of selling those price curves either to other VPPs or to DSO.
        :param mpc_t:
        :param t:
        :param data:
        :return: balance needed at PCC, max possible excess, objf with subtracted virtual slack cost
        """
        data = self.load_data(data_paths[data_names_dict[self.name]])
        ppc0 = cases[data['case']]()
        ppc_t = copy.deepcopy(ppc0)

        max_generation = data['max_generation']
        fixed_load = data['fixed_load']
        price = data['price']
        slack_idx = data['slack_idx']
        generation_type = np.array(data['generation_type'])

        ppc_t['bus'][:, 2] = fixed_load[t]
        ppc_t['gen'][:, 8] = max_generation[t]
        ppc_t['gencost'][:, 4] = price[t]

        res = rundcopf(ppc_t, ppoption(VERBOSE=opf1_verbose))
        if opf1_prinpf == True: printpf(res)

        if res['success'] == 1:
            self.log_info("I have successfully run the OPF1.")
            self.set_attr(opf1_resgen=res['gen'])

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
                                    'pc_matrix': False})

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
                # pc_matrix = np.ndarray.tolist(sorted_nonzero_squeezed)
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
                                    # 'objf_greentodso': objf_greentodso, # this comes from opf_e3 for "todso" case
                                    # 'pc_matrix': np.matrix(pc_matrix_incr), # price curves derived in opf_e2
                                    'exc_matrix': np.array(exc_matrix)})
            return True
        else:
            self.log_info("OPF1 does not converge. STOP.")
            sys.exit()

    def runopf_e2(self, exc_matrix, t):
        """
        Excess ageents calculations, before sending the price curves.
        This include verification of transmitting the excess to the other vpps (only to the ones that send requests),
        but also to DSO.
        """

        data = self.load_data(data_paths[data_names_dict[self.name]])
        generation_type = np.array(data['generation_type'])

        # build price curve
        pc_matrix_incr = copy.deepcopy(exc_matrix)
        pc_matrix_incr[2, :] *= pc_matrix_price_increase_factor
        pc_matrix_incr = np.matrix(np.round(pc_matrix_incr, 4))

        self.log_info("Final pc matrix for requesters (i.e. exc_matrix increased): " + str(pc_matrix_incr))
        self.set_attr(opfe2={'pc_matrix': np.array(pc_matrix_incr)})

        # calculate prospective revenue if green energy sold to DSO
        c1 = generation_type[exc_matrix[0, :].astype(int)]  # check gen types of the ones in pc_matrix
        c2 = np.isin(c1, green_sources)  # choose only green ones
        c3 = np.reshape(c2, np.squeeze(c2).shape)
        exc_matrix_green = exc_matrix[:, c3]

        if exc_matrix_green.size > 0:
            self.set_attr(opfe2={'exc_matrix_green': exc_matrix_green})
        else:
            self.log_info("No green excess that could be sold to DSO.")
            self.set_attr(opfe2={'exc_matrix_green': False})

        ######### OPFs for technical constraints of exporting power not for costs!
        low_price = 0.1
        high_price = 1000

        # load, update, modify data
        data = self.load_data(data_paths[data_names_dict[self.name]])
        ppc0 = cases[data['case']]()
        ppc_t = copy.deepcopy(ppc0)
        origin_opf1_resgen = self.get_attr('opf1_resgen')

        print('bus \ value \ pmax \ pmin (original opf1)')
        print(np.round(origin_opf1_resgen[:, [0, 1, 8, 9]], 4))

        max_generation = data['max_generation']
        fixed_load = data['fixed_load']
        price = data['price']
        slack_idx = data['slack_idx']
        ppc_t['bus'][:, 2] = fixed_load[t]  # from data - they're not controlled
        ppc_t['gen'][0:, 8] = 0  # virtual slack generator removed
        ppc_t['gen'][0:, 9] = 0
        ppc_t['gen'][1:, 8] = np.round(origin_opf1_resgen[1:, 1] * (1+relax_e2), 4)  # from OPF1, without slack
        ppc_t['gen'][1:, 9] = np.round(origin_opf1_resgen[1:, 1] * (1-relax_e2), 4)  # both bounds
        ppc_t['gencost'][:, 4] = price[t]  # from data

        # exporting green (feasible) energy (e.g. option of selling to DSO)
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
        ppc_tg['branch'][:, [5, 6, 7]] = ppc_tg['branch'][:, [5, 6, 7]] * (1+relax_e2)

        # pp(ppc_tg['gen'][:, [0, 8, 9]])
        # pp(ppc_tg['bus'][:, [1, 2, 3]])
        # pp(ppc_tg['gencost'][:, 4])
        # pp(ppc_tg['branch'][:, [5, 6, 7]])
        # pp(ppc_tg)

        res_g = rundcopf(ppc_tg, ppoption(VERBOSE=opf1_verbose, PDIPM_GRADTOL=PDIPM_GRADTOL_mod))

        if opfe2_prinpf:
            # printpf(res_g)
            print('bus \ value \ pmax \ pmin \ price (if whole available excess)')
            show = np.array(
                np.concatenate((np.matrix(res_g['gen'][:, [0, 1, 8, 9]]), np.matrix(ppc_tg['gencost'][:, 4]).T), axis=1))
            print(np.round(show, 4))
            print("Objective function: " + str(res_g['f']))
        if res_g['success']:
            self.log_info('Feasible OPFe2 with green resources available to DSO .')
            f1 = True
        else:
            self.log_info('NOT feasible OPFe2 with green resources available to DSO.')
            f1 = False

        # exporting whole (feasible) excess through PCC
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

        res = rundcopf(ppc_t, ppoption(VERBOSE=opf1_verbose, PDIPM_GRADTOL=PDIPM_GRADTOL_mod))
        if opfe2_prinpf:
            # printpf(res)
            print('bus \ value \ pmax \ pmin \ price (if whole available excess)')
            show = np.array(np.concatenate((np.matrix(res['gen'][:, [0, 1, 8, 9]]), np.matrix(ppc_t['gencost'][:, 4]).T), axis=1))
            print(np.round(show, 4))
            print("Objective function: "+str(res['f']))
        if res['success']:
            self.log_info('Feasible OPFe2 with overall excess available.')
            f2 = True
        else:
            self.log_info('NOT feasible OPFe2 with overall excess available.')
            f2 = False

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
        # load raw data
        data = self.load_data(data_paths[data_names_dict[self.name]])
        ppc0 = cases[data['case']]()
        ppc_t = copy.deepcopy(ppc0)

        # modify according to prospective accepted bids: loop through my generators (excluding slack)
        origin_opf1_resgen = self.get_attr('opf1_resgen')

        # load and modify according to current time and current OPF1 results!!
        max_generation = data['max_generation']
        fixed_load = data['fixed_load']
        price = data['price']
        slack_idx = data['slack_idx']
        ppc_t['bus'][:, 2] = fixed_load[t]  # from data - they're not controlled
        ppc_t['gen'][1:, 8] = np.round(origin_opf1_resgen[1:, 1], 4)  # from OPF1, without slack
        ppc_t['gen'][1:, 9] = np.round(origin_opf1_resgen[1:, 1], 4)  # both bounds
        ppc_t['gencost'][:, 4] = price[t]  # from data

        for gen_idx in np.unique(all_bids_mod[:, 2]):  # loop through the generators from bids (1,2,3) there should be no slack - as constraints for OPF of same Pmin and Pmax

            c1 = np.where(all_bids_mod[:, 2] == gen_idx)[0]  # rows where gen
            p_bid = np.round(np.sum(all_bids_mod[c1, 3]), 4)  # total value of bidded power for generator

            c2 = np.where(origin_opf1_resgen[:, 0] == gen_idx)[0]
            ppc_t['gen'][c2, [8, 9]] = np.array([0, 0])  # make the value 0 before modification
            ppc_t['gen'][c2, [8, 9]] += np.round(origin_opf1_resgen[c2, 1], 4)  # add value from opf1, to Pmax,Pmin

            c3 = np.where(ppc_t['gen'][:, 0] == gen_idx)[0]
            ppc_t['gen'][c3, [8, 9]] += np.round(p_bid, 4)  # add value from bidding, to Pmax,Pmin

        # modify the load at the pcc i.e. slack bus id:0 - same formulation for PF and OPF
        bids_sum = np.round(np.sum(all_bids_mod[:, 3]), 4)
        ppc_t['bus'][0, 2] += bids_sum

        # with bids updated, verify the power flow if feasible - must be opf in order to include e.g. thermal limits
        res = rundcopf(ppc_t, ppoption(VERBOSE=opf1_verbose))

        if opfe3_prinpf:
            printpf(res)

        # calculation of the costs: objective function minus revenue from selling
        bids_revenue = 0
        for bid in all_bids_mod:
            bids_revenue += bid[3] * bid[4]

        costs = np.round(res['f'] - bids_revenue, 4)  # costs of vpp as costs of operation (slackcost for E = 0) - revenue from bids

        self.set_attr(opf_e3={'objf_bidsrevenue': costs})

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

        all_pc = []
        for mem in memory:
            if mem['value'] == False:
                continue
            else:
                for gn in range(0, mem['price_curve'].shape[1]):
                    all_pc.append([data_names_dict[mem["vpp_name"]],
                                   mem["price_curve"][0, gn],
                                   mem["price_curve"][1, gn],
                                   mem["price_curve"][2, gn]])
        sorted_pc = sorted(all_pc, key=lambda price: price[3])
        bids = []
        for pc in sorted_pc:
            pc_vpp_idx = pc[0]
            pc_gen_idx = pc[1]
            pc_maxval = float(pc[2])
            pc_price = float(pc[3])
            if need != 0 and need > pc_maxval:
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
        all = np.unique(np.array(b[:, 0]))
        are = np.array(np.matrix(bids)[:, 0])
        missing = np.setdiff1d(all, are)
        for m in missing:
            empty_bid = np.array([[m, 0, 0, 0]])
            bids = np.concatenate((bids, empty_bid), axis=0)
        return bids  # list: [vpp_idx, gen_idx, bidgen_value, gen_price]

    def set_consensus_if_norequest(self):
        """
        This is called in case an excess agent does not have any requests from deficit agents in the deficit loop.
        :return:
        """

        time.sleep(0.1)
        if self.get_attr('n_requests') == 0:
            self.log_info('I am E and I nobody wants to buy from me (n_requests=' + str(self.get_attr('n_requests')) +
                          '). I set consensus.')
            self.set_attr(consensus=True)

    def deficit_opf(self, price_curves):
        """
        After a deficit agent receives all price curves, it should define bids through running internal opf
        as defined in this function.
        :param price_curves
        :return: bids that are sent back to the excess agents of interest
        """

    def sys_octave_test(self, mpc):
        res = octave.rundcopf(mpc)
        return res['success']

    def sys_pypower_test(self, ppc):
        r = rundcopf(ppc)
        return r['success']

    def bids_alignment1(self, mypc0, all_bids0):

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
        # print(all_bids)
        all_bids_mod = copy.deepcopy(all_bids)
        all_bids_mod[:, 3] = 0  # all bids values set to 0

        for pc in mypc0:

            gen_id = pc[0]
            gen_pmax = pc[1]
            bid_1gen = all_bids0[all_bids0[:, 2] == gen_id]
            sum_bids_1gen = sum(bid_1gen[:, 3])

            if gen_pmax < sum_bids_1gen:  # if resource exceeded for 1 generator, then share the exces
                self.log_info("Modification of bids of over-bidded gens... ")
                for bid_msg in self.get_attr('iteration_memory_bid'):
                    vpp_idx = data_names_dict[bid_msg['vpp_name']]
                    bid0 = np.array(bid_msg['bid'])
                    bids_sum = sum(bid0[:, 2])  # =request value in other words. =25

                    mod = bids_sum / all_bids_sum  # 25/55 - waga tej vpp

                    c1 = np.where(all_bids_mod[:, 0] == vpp_idx)[0]  # raws where vpp
                    c2 = np.where(all_bids_mod[:, 2] == gen_id)[0]  # raws where gens
                    c3 = np.intersect1d(c1, c2)

                    all_bids_mod[c3, 3] = np.round(mod * gen_pmax, 4)  # 25/55 * 20 = 9.1

            else:
                self.log_info("Filling bids with the other generators' excess... ")
                for bid_msg in self.get_attr('iteration_memory_bid'):
                    vpp_idx = data_names_dict[bid_msg['vpp_name']]
                    bid0 = np.array(bid_msg['bid'])
                    bids_sum = sum(bid0[:, 2])  # =request value in other words. =25

                    c1 = np.where(all_bids_mod[:, 0] == vpp_idx)[0]
                    bidded_sofar = sum(all_bids_mod[c1, 3])  # = 9.1
                    c2 = np.where(all_bids_mod[:, 2] == gen_id)[0]
                    c3 = np.intersect1d(c1, c2)
                    #if not c3:  # i.e. there was no original bid for that generator (e.g. the first one was sufficient)
                    all_bids_mod[c3, 3] = round(bids_sum - bidded_sofar, 4)

            # final check if the bids does not exceed excess
            c2 = np.where(all_bids_mod[:, 2] == gen_id)[0]
            if all_bids_mod[c2, 3].any() > gen_pmax:
                self.log_info("Wrong bid modification: some gen bids exceed the available excess!! STOP")
                sys.exit()

        return all_bids_mod