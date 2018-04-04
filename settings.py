import numpy as np
import matplotlib.pyplot as plt
import json

ts_n = 10  # number of timestamps of whole simulation

data_names = ["vpp1", "vpp2", "vpp3", "vpp4"]
data_names_dict = {"vpp1": 0, "vpp2": 1, "vpp3": 2, "vpp4": 3}
data_paths = ["data/vpp1.json", "data/vpp2.json", "data/vpp3.json", "data/vpp4.json"]
vpp_n = len(data_names)

system_status = np.zeros([ts_n, vpp_n])

adj_matrix = [[True, True, False, True],
              [True, True, True, False],
              [False, True, True, False],
              [True, False, False, True]]

small_wait = 0.3  # waiting time to separate some steps, for testing

def print_data():
    gen_sum = np.zeros(ts_n)
    load_sum = np.zeros(ts_n)

    plt.figure(1)
    for vi in range(vpp_n):
        with open("data/" + data_names[vi] + ".json", 'r') as f:
            ar = json.load(f)

        plt.subplot(vi + 221)
        gen = plt.plot(ar["generation"])
        load = plt.plot(ar["load"])

        plt.setp(gen, 'color', 'b', 'linewidth', 2.0)
        plt.setp(load, 'color', 'r', 'linewidth', 2.0)

        plt.ylabel('gen/load aggr. power ' + data_names[vi])

        gen_sum = np.sum([gen_sum, np.array([ar["generation"]])], axis=0)
        load_sum = np.sum([load_sum, np.array([ar["load"]])], axis=0)

    plt.figure(2)  # sum of all laods generations etc.
    gen = plt.plot(gen_sum[0])
    load = plt.plot(load_sum[0])
    plt.setp(gen, 'color', 'b', 'linewidth', 2.0)
    plt.setp(load, 'color', 'r', 'linewidth', 2.0)
    plt.ylabel('total gen/load power of the system')

    plt.show()
