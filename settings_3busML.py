import numpy as np
from case5_vpp import case5_vpp
from case4_vpp import case4_vpp

np.set_printoptions(suppress=True)

ts_n = 3  # number of timestamps of whole simulation

data_names = ["vpp1", "vpp2", "vpp3", "vpp4"]
data_names_dict = {"vpp1": 0, "vpp2": 1, "vpp3": 2, "vpp4": 3}
data_paths = ["data/vpp3bus/vpp1-case5.json", "data/vpp3bus/vpp2-case4.json", "data/vpp3bus/vpp3-case4.json", "data/vpp3bus/vpp4-case4.json"]
vpp_n = len(data_names)

system_status = np.zeros([ts_n, vpp_n])

adj_matrix = [[True, True, True, True],
              [True, True, True, True],
              [True, True, True, False],
              [True, True, False, True]]

small_wait = 0.3  # waiting time to separate some steps, for testing
price_increase_factor = 6.0

# the excess agents increase the original production prices when publishing the price curve, increase by 10%
# when the excess is sold for the original prices, there is no benefit for excess agents since:
# production_cost - bids_revenue = 0
pc_matrix_price_increase_factor = 1.1
dso_green_price_increase_factor = 1.05

cases = {'case5': case5_vpp,
         'case4': case4_vpp}

opf1_verbose = 0
opf1_prinpf = False
opfe3_prinpf = False
opfe2_prinpf = True

relax_e2 = 0.01  # relaxation of constraints in opf_e2
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
