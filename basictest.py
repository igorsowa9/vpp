from osbrain import Agent
import json
from settings import *
import time
from pprint import pprint as pp
from settings import data_names, data_names_dict, data_paths, vpp_n, ts_n, adj_matrix

import time
import sys
from osbrain import run_agent
from osbrain import run_nameserver
from pprint import pprint as pp
import copy

from settings import data_names, data_names_dict, data_paths, vpp_n, ts_n, adj_matrix, system_status, \
    small_wait, price_increase_factor, cases
from other_agents import VPP_ext_agent
from utilities import system_consensus_check, erase_iteration_memory, erase_timestep_memory


from oct2py import octave

octave.addpath('/home/iso/PycharmProjects/vpp/matpow_cases')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0/t')

from case5_vpp import case5_vpp


# - load basic topology
# - update the current generation constraints (weather), fixed loads (curves loads) or flexible loads constraints
#   (but usually they are fixed according to contracts with clients)
# - run first opf for status definition with only own resources -> after this requests distribution
# update bid offers in own system i.e. simulate what would be if you accept them, i.e. run OPF:
#   * to be modified:
#       - load at slack bus as negative generator
#       - price of that load (i.e. negative generator)


#   - is revenue increased then?
#   - LATER: influence of mediating to increase the revenue...?
#   - EXTENSION: maximize the revenue within the bid offer


def system_state_update_and_balance(mpc_t, t, data):
    max_generation = data['max_generation']
    fixed_load = data['fixed_load']
    price = data['price']
    slack_idx = data['slack_idx']

    mpc_t['bus'][:, 2] = fixed_load[t]
    mpc_t['gen'][:, 8] = max_generation[t]
    mpc_t['gencost'][:, 4] = price[t]

    res = octave.rundcopf(mpc_t, octave.mpoption('out.all', 1))
    power_balance = res['gen'][slack_idx, 1]
    objf_noslackcost = res['f'] - power_balance * mpc_t['gencost'][slack_idx][4]

    return power_balance, objf_noslackcost, res['f']


ns = run_nameserver()

a = run_agent('tester', base=VPP_ext_agent)
data = a.load_data("data/vpp1-case5.json")

case_name = data['case']

ppc = case5_vpp()
print("returns: ", a.sys_pypower_test(ppc))


sys.exit()

mpc = cases[case_name]()
a.sys_octave_test(mpc)
sys.exit()

balance, _, _ = system_state_update_and_balance(copy.deepcopy(mpc), 0, data)

print(balance)
