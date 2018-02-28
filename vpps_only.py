import time
import json
import numpy as np
import sys
from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent

import matplotlib.pyplot as plt

data_names = ["vpp1", "vpp2", "vpp3", "vpp4"]
data_paths = ["data/vpp1.json", "data/vpp2.json", "data/vpp3.json", "data/vpp4.json"]
vpp_n = len(data_names)
ts_n = 10
# connections for negotiations defined (squere)
adj_matrix = np.array([[True, True, True, False],
                       [True, True, False, True],
                       [True, False, True, True],
                       [False, True, True, True]], dtype=bool)


class VPP_ext_agent(Agent):
    def on_init(self):
        #self.bind('PUSH', alias='main')
        pass

    def load_data(self, path):
        with open(path, 'r') as f:
            arr = json.load(f)
        return arr

    def status_report(self, name):
        self.send('main', 'Hello, %s!' % name)

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

def request_log(self, message):
    self.log_info('Request log: %s' % message)


if __name__ == '__main__':

    print_data()

    # initialization of the agents
    ns = run_nameserver()
    for vpp_idx in range(vpp_n):
        agent = run_agent(data_names[vpp_idx], base=VPP_ext_agent)
        agent.set_attr(myname=str(data_names[vpp_idx]))
        print(agent.myname)

    # Show agents registered in the name server
    # for alias in ns.agents():
    #    print(alias)
    # print("----------------------")

    # subscriptions to neighbours only
    for vpp_idx in range(vpp_n):

        agent = ns.proxy(data_names[vpp_idx])
        addr = agent.bind('PUB', alias='main') # this agent will publish to neighbours

        for n in range(vpp_n):
            if n == vpp_idx:
                continue
            if adj_matrix[vpp_idx][n] == True:
                neighbour = ns.proxy(data_names[n])
                neighbour.connect(addr, handler={'request_topic': request_log}) # only neighbours connect to the agent

    # system time loop (10 timesteps for now)
    for t in range(ts_n):

        for vpp_idx in range(vpp_n):
            agent = ns.proxy(data_names[vpp_idx])

            # each agent checks the internal resources (from data or from internal agent (OPF))
            json_data = agent.load_data(data_paths[vpp_idx])

            # each deficit agent publishes deficit to the neighbours only
            if json_data["generation"][t] - json_data["load"][t] > 0:
                agent.log_info("I am excess")
            elif json_data["generation"][t] - json_data["load"][t] < 0:
                agent.log_info("I am deficit. I'll publish requests to neighbours.")

                message_request = 'I am '+ ns.agents()[vpp_idx] +'. I need support!'
                agent.send('main', message_request, topic='request_topic')

            else:
                agent.log_info("I am balanced")

            # each asked and interested excess agent publishes the price curve, based on prices from data
            # each

        time.sleep(1)

    ns.shutdown()

    plt.show()
