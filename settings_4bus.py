import numpy as np
from time import gmtime, strftime
from data.vpp4bus.case5_vpp1 import case5_vpp1
from data.vpp4bus.case4_vpp2 import case4_vpp2
from data.vpp4bus.case4_vpp3 import case4_vpp3
from data.vpp4bus.case4_vpp4 import case4_vpp4

np.set_printoptions(suppress=True)

ts_0 = 7*int(60/5*24)#3861
constant_environment = False
ts_n = 7*int(60/5*24)-1  # number of timestamps of whole simulation

start_datetime = "01/09/2017 00:00"  # start of the __file!__ then ts_0 already introduces the offset!
# if you want to determine the prices based on the memory

# explit means that you want to determine price increase factor by the similarity method,
# and if you want to include the negotiation from already exploit (i.e. the most optimal) in the next negotiation already
exploit_mode = True
update_during_exploit = True  # it is stored at current folder, not at original history floder path_dir_history

# pc_matrix_price_increase_factor (1) vs. pc_matrix_price_absolute_increase (2)
# dumb VPPs always have price increase factor policy, regardless of this choice
price_increase_policy = 2

# pcf_avg modification (similarity() ) based on the belief in the memory
do_not_exceed_mp_belief = True  # during the derivation of deal pay attention to the mp_factors from the mp_belief in the history folder
mp_belief_treshold = 0.04  # minimum treshold of marginal price belief in order to take that price under consideration in mp_belief
mp_belief_range = 2.0  # absolute range of vicinity of the mp prices to consider in bids derivation
exceeding_or_vicinity = False  # modify in case of a pcf exceeding the probable MP or modify if the pcf is only in the vicinity of the pcf (i.e. also lower, within the range)
# rather vicinity!!!

update_mp_belief = False  # not yet there at all, update during negotiation according to BL

# the path to the folder where the exploration results are saved, ALSO: belief about the marginal price is saved in that folder:
# path_dir_history = '/home/iso/Desktop/vpp_some_results/2018_0830_1544_week1_multi_oneshot_10/'
# path_dir_history = '/home/iso/Desktop/vpp_some_results/2018_1201_0721_week2_multi_oneshot_10/'
# path_dir_history = '/home/iso/Desktop/vpp_some_results/2018_0822_1555_week1-2/'
path_dir_history = '/home/iso/Desktop/vpp_some_results/2019_0207_1401_history__week1_oneshot_pri4_2_38/'
# path_dir_history = '/home/iso/Desktop/vpp_some_results/2018_1206_1323_week1_multi_oneshot_1/'
# path_dir_history = '/home/iso/Desktop/vpp_some_results/2018_1206_1620_week1and2_explore_oneshot/'
# path_dir_history = '/home/iso/Desktop/vpp_some_results/2018_1207_1458_test_history/'

# if set to more then 1, during the exploration, the learning agent has more than one try to propose an offer with the
# exactly same conditions of environment, i.e. explore the opponent / environemnt better
explore_multi_oneshot = 1

# if you want to save the files to the .CSV and the plots to PDF instead of showing them
tocsv = True
pdf = True

directory_tail = "_week2_exploit_with_update"
# directory_tail = "_test"

path_save = '/home/iso/Desktop/vpp_some_results/' + strftime("%Y_%m%d_%H%M", gmtime()) + directory_tail + '/'

# maximum rounds of the negotiations between the agents
max_iteration = 1
negotiation = True  # if False, then only opf1 and requests

# vpps that lear/memorize the history of negotiation
vpp_learn = [0, 1, 2, 3]

# vpps that utilize the memory to exploit
vpp_exploit = ['vpp3']
similarity_treshold = 0.60
top_selection_quantity = 100
order_by = 'sim'
mp_factor_treshold_in_selection = 0.3  # treshold value that are selected based on mp_factor before the average is calculated

data_names = ["vpp1", "vpp2", "vpp3", "vpp4"]
data_names_dict = {"vpp1": 0, "vpp2": 1, "vpp3": 2, "vpp4": 3}
vpp_n = len(data_names)

data_paths = ["data/vpp4bus/vpp1-case5.json",
              "data/vpp4bus/vpp2-case4.json",
              "data/vpp4bus/vpp3-case4.json",
              "data/vpp4bus/vpp4-case4.json"]

cases = {'case5_vpp1': case5_vpp1,
         'case4_vpp2': case4_vpp2,
         'case4_vpp3': case4_vpp3,
         'case4_vpp4': case4_vpp4}

system_status = np.zeros([ts_n, vpp_n])

adj_matrix = [[True, True, False, False],
              [True, True, True, True],
              [False, True, True, False],
              [False, True, False, True]]

small_wait = 0.1  # waiting time to separate some steps, for testing
iteration_wait = 0.1

# size of all the figures that are saved to the pdfs
figsizeH = 24
figsizeL = 12

# the excess agents increase the original production prices when publishing the price curve, increase by 10%
# when the excess is sold for the original prices, there is no benefit for excess agents since:
# production_cost - bids_revenue = 0
dso_green_price_increase_factor = 1.05

opf1_verbose = 0
opf1_prinpf = False
opfe2_prinpf = False
opfe3_prinpf = False


# number of time periods that you check in the memory before the current timestamp in order to determine if the power
# increase factor should be modified for exploitation or not. I.e. if successful negotiation happened in the previous #
# of the timestamps, then change the modificator of the price to the next one.
max_ts_range_for_price_modification = 2


relax_e2 = 0.01  # relaxation of constraints in opf_e2
relax_e3 = 0.01  # relaxation of constraints in opf_e3
# modification of convergence condition, check:
# http://rwl.github.io/PYPOWER/api/pypower.dcopf_solver-pysrc.html
#         feastol = ppopt['PDIPM_FEASTOL']
#         gradtol = ppopt['PDIPM_GRADTOL']
#         comptol = ppopt['PDIPM_COMPTOL']
#         costtol = ppopt['PDIPM_COSTTOL']
#         max_it  = ppopt['PDIPM_MAX_IT']
#         max_red = ppopt['SCPDIPM_RED_IT']
PDIPM_GRADTOL_mod = 5*1e-6

# ASSUMPTIONS:
# slack bus as the first one with idx 0 (some simplification, non universalities in the code, e.g. in PC building)
# only linear cost in gencost matrix, no offset.
# deficit agents can buy only the amount of their deficit power, even if the excess of the neighbour might be cheaper,
#   than their own resources (deficit agents do not run opf when they receive PCs, but just take the cheapest resources)
#   from the neighbours' excess, according to simple sorting i.e. starting from the cheapest
#   - but if so, where is that effort for ML then...?
# only linear cost of generation assumed (i.e. fixed prices), with no offset

# Exchange with VPPs/DSO:
# (cost of power from DSO is the price of the virtual generator at slack bus (idx 0))
#
# * Excess:
#   - price curves offered to other VPPs are with prices of generation_cost * pc_matrix_price_increase_factor
#   - green energy (PV, wind, biogas) can be always sold to DSO for the price of generation cost + X% e.g. 5%
#   - gray energy (coal, atom, gas) should controlled i.e. reduced generation etc. or can be sold for lower price to DSO
#                                                                                        (no scenarios for that for now)
#   - the rest of excess (after making bids) can be sold to DSO with similar assumptions
# * Deficit:
#   - power can be bought from excess VPPs for the price in the price curve matrix
#   - if the power is bought from DSO, there is a fixed price (i.e. price of virtual generator at slack bus)
#       - contracts with DSO might be different for different VPPs (that could be learned by the other VPPs too)

# 7.09.18 - the multiple requests ignored for now

green_sources = [1, 2, 3, 4, 5]
weather_dependent_sources = [1, 2, 3]
grey_sources = [11, 12, 13, 14, 15]

# Generation types:
# 0 DSO

# 1 pv - green, intermittent
# 2 onshore wind - green, intermittent
# 3 offshore wind - green, intermittent

# 4 bio gas - green, controllable
# 5 hydro - green, controllable

# 11 black coal - gray, controllable
# 12 lignite - gray, controllable
# 13 atom - gray, controllable
# 14 natural gas - gray, controllable

# OPFs for technical constraints of exporting power not for costs!
low_price = 0.1
high_price = 1000

# Loading data from the files - the paths defined in the VPP files
TIMESTAMP = 0
FORECASTED = 1
MEASURED = 2

# offset for all loads and max generations from json files
gen_mod_offset = 0
load_mod_offset = 0
