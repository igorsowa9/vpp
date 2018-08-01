import numpy as np
from data.vpp4bus.case5_vpp1 import case5_vpp1
from data.vpp4bus.case4_vpp2 import case4_vpp2
from data.vpp4bus.case4_vpp3 import case4_vpp3
from data.vpp4bus.case4_vpp4 import case4_vpp4
import time, datetime

np.set_printoptions(suppress=True)

ts_0 = 120#int(60/5*0)
ts_n = 5#int(60/5*24)  # number of timestamps of whole simulation

start_datetime = "01/09/2017 00:00"

max_iteration = 10
negotiation = True  # if False, then only opf1 and requests

vpp_learn = [2]
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

small_wait = 0.0  # waiting time to separate some steps, for testing
iteration_wait = 0.1

# size of all the figures that are saved to the pdfs
figsizeH = 12
figsizeL = 12

# the excess agents increase the original production prices when publishing the price curve, increase by 10%
# when the excess is sold for the original prices, there is no benefit for excess agents since:
# production_cost - bids_revenue = 0
dso_green_price_increase_factor = 1.05

opf1_verbose = 0
opf1_prinpf = True
opfe3_prinpf = False
opfe2_prinpf = False

relax_e2 = 0.000  # relaxation of constraints in opf_e2
relax_e3 = 0.000  # relaxation of constraints in opf_e3
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
# deficit agents can buy only the amount of their deficit power, even if the excess of the neighbour migh be cheaper,
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

green_sources = [1, 2, 3, 4, 5]
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

######### OPFs for technical constraints of exporting power not for costs!
low_price = 0.1
high_price = 1000

# Loading data from the files - the paths defined in the VPP files
TIMESTAMP = 0
FORECASTED = 1
MEASURED = 2

# offset for all loads and max generations from json files
gen_mod_offset = 0
load_mod_offset = 0
