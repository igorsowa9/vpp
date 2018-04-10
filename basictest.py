from osbrain import Agent
import json
import settings
import time
from pprint import pprint as pp
from settings import data_names, data_names_dict, data_paths, vpp_n, ts_n, adj_matrix, print_data

import time
import sys
from osbrain import run_agent
from osbrain import run_nameserver
from pprint import pprint as pp
import copy

from settings import data_names, data_names_dict, data_paths, vpp_n, ts_n, adj_matrix, print_data, system_status, \
    small_wait, price_increase_factor
from other_agents import VPP_ext_agent
from utilities import system_consensus_check, erase_iteration_memory, erase_timestep_memory


from oct2py import octave

octave.addpath('/home/iso/PycharmProjects/vpp/matpow_cases')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0/t')

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
    generation = data['generation']
    load = data['load']
    price = data['price']
    slack_idx = data['slack_idx']

    mpc_t['bus'][:, 2] = load[t]
    mpc_t['gen'][:, 8] = generation[t]
    mpc_t['gencost'][:, 4] = price[t]

    res = octave.rundcopf(mpc_t, octave.mpoption('out.all', 1))
    power_balance = res['gen'][slack_idx, 1]

    return power_balance


ns = run_nameserver()

a = run_agent('tester', base=VPP_ext_agent)
data = a.load_data("data/vpp1-case5.json")

mpc = octave.case5_vpp()
mpopt = octave.mpoption('out.all', 1)

gens = system_state_update_and_balance(copy.deepcopy(mpc), 1, data)

print(gens)
sys.exit()

r = octave.rundcopf(mpc, mpopt)
print('SUCCESS?: ', r['success'])


def upload_bid_to_mpc(mpc0, bids):
    pass