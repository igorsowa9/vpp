import numpy as np
from oct2py import octave
from case5_vpp import case5_vpp

octave.addpath('/home/iso/PycharmProjects/vpp/matpow_cases')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0/t')

ts_n = 3  # number of timestamps of whole simulation

data_names = ["vpp1", "vpp2", "vpp3", "vpp4"]
data_names_dict = {"vpp1": 0, "vpp2": 1, "vpp3": 2, "vpp4": 3}
data_paths = ["data/vpp1-case5.json", "data/vpp2-case5.json", "data/vpp3-case5.json", "data/vpp4-case5.json"]
vpp_n = len(data_names)

system_status = np.zeros([ts_n, vpp_n])

adj_matrix = [[True, True, False, True],
              [True, True, True, False],
              [False, True, True, False],
              [True, False, False, True]]

small_wait = 0.3  # waiting time to separate some steps, for testing
price_increase_factor = 6.0

cases = {'case5': case5_vpp,
         'case4gs': False}

# cases = {'case5': octave.case5_vpp,
#          'case4gs': octave.case4gs_vpp}
