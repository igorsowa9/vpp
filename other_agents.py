from osbrain import Agent
import json
from settings import *
#from utilities import system_state_update_and_balance
import time
from pprint import pprint as pp
from oct2py import octave
import copy

octave.addpath('/home/iso/PycharmProjects/vpp/matpow_cases')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0/t')

from pypower.api import *
from case5_vpp import case5_vpp
from rundcopf_noprint import rundcopf


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

    def powerbalance_at(self, t):
        """
        This should be internal PF in order to define excess/deficit.
        :param time:
        :return: It returns the excess/deficit at the PCC, cost without balancing costs, excess
        """
        data = self.load_data(data_paths[data_names_dict[self.name]])
        ppc = cases[data['case']]()
        power_balance, _, _ = self.system_state_update_and_balance(copy.deepcopy(ppc), t, data)
        return power_balance

    def system_state_update_and_balance(self, mpc_t, t, data):
        """
        Updates the system mpc at time t according to data
        :param mpc_t:
        :param t:
        :param data:
        :return:
        """
        generation = data['max_generation']
        load = data['fixed_load']
        price = data['price']
        slack_idx = data['slack_idx']

        mpc_t['bus'][:, 2] = load[t]
        mpc_t['gen'][:, 8] = generation[t]
        mpc_t['gencost'][:, 4] = price[t]

        res = rundcopf(mpc_t, ppoption(VERBOSE=1))
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


    def current_price(self, time):
        """
        Loads price of own resources / create price curve, at a time.
        :param time:
        :return: price value / price curve
        """
        json_data = self.load_data(data_paths[data_names_dict[self.name]])
        return json_data["price"][time]

    def new_opf(self):
        """
        After a deficit agent receives all price curves, it should define bids through running internal opf
        as defined in this function.
        :return: bids that are sent back to the excess agents of interest
        """
        # opf asa system is integrated
        memory = self.get_attr('iteration_memory_pc')
        need = abs(self.get_attr('current_status')[1])

        memory_list = []
        for mem in memory:
            memory_list.append([data_names_dict[mem["vpp_name"]], mem["value"], mem["price"]])

        sorted_memory = sorted(memory_list, key=lambda price: price[2])
        print(sorted_memory)
        bids = []
        for pc in sorted_memory:
            pc_vpp_idx = pc[0]
            pc_maxval = float(pc[1])
            pc_price = float(pc[2])
            if need > pc_maxval:
                bids.append([pc_vpp_idx, pc_price, pc_maxval])
                need = need - pc_maxval
            else:
                bids.append([pc_vpp_idx, pc_price, need])
        return bids

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