import time
import json
import numpy as np
import sys
from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent

import matplotlib.pyplot as plt

data = ["vpp1", "vpp2", "vpp3", "vpp4"]
vpp_n = len(data)
ts_n = 10
# connections for negotiations defined (squere)
adj_matrix = np.array([[True, True, True, False],
                          [True, True, False, True],
                          [True, False, True, True],
                          [False, True, True, True]], dtype=bool)

def print_data():

    gen_sum = np.zeros(ts_n)
    load_sum = np.zeros(ts_n)

    plt.figure(1)
    for vi in range(vpp_n):
        with open("data/"+data[vi]+".json", 'r') as f:
            ar = json.load(f)

        plt.subplot(vi + 221)
        gen = plt.plot(ar["generation"])
        load = plt.plot(ar["load"])

        plt.setp(gen, 'color', 'b', 'linewidth', 2.0)
        plt.setp(load, 'color', 'r', 'linewidth', 2.0)

        plt.ylabel('gen/load aggr. power '+data[vi])

        gen_sum = np.sum([gen_sum, np.array([ar["generation"]])], axis=0)
        load_sum = np.sum([load_sum, np.array([ar["load"]])], axis=0)

    plt.figure(2) # sum of all laods generations etc.
    print(gen_sum)
    gen = plt.plot(gen_sum[0])
    load = plt.plot(load_sum[0])
    plt.setp(gen, 'color', 'b', 'linewidth', 2.0)
    plt.setp(load, 'color', 'r', 'linewidth', 2.0)
    plt.ylabel('total gen/load power of the system')


def log(self, message):
    self.log_info('Log a: %s' % message)


if __name__ == '__main__':

    print_data()

    # initialization of the agents
    ns = run_nameserver()
    for vpp_idx in range(vpp_n):
        run_agent(data[vpp_idx])

    # Show agents registered in the name server
    for alias in ns.agents():
        print(alias)

    print("----------------------")

    # subscriptions to neighbours
    for vpp_idx in range(vpp_n):

        agent = ns.proxy(data[vpp_idx])
        addr = agent.bind('PUB', alias='main')

        for n in range(vpp_n):
            if n==vpp_idx:
                continue
            if adj_matrix[vpp_idx][n]==True:
                neighbour = ns.proxy(data[n])
                neighbour.connect(addr, handler={'a': log})

    # system time loop (10 timesteps)
    for t in range(ts_n):

        # each agent publishes deficit or exccess to the neighbours
        for vpp_idx in range(vpp_n):
            agent = ns.proxy(data[vpp_idx])

        # Send messages
        for i in range(6):
            time.sleep(1)
            message = 'Hello, %s!' % topic
            alice.send('main', message, topic=topic)



    ns.shutdown()

    plt.show()






