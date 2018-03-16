import time
import json
import numpy as np
import sys
from osbrain import run_agent
from osbrain import run_nameserver
from pprint import pprint as pp
from osbrain import Agent
import Pyro4

from settings import data_names, data_names_dict, data_paths, vpp_n, ts_n, adj_matrix, print_data
from ext_agents import VPP_ext_agent


global_time = 0


def global_time_set(new_time):
    global global_time
    global_time = new_time

    for alias in ns.agents():
        a = ns.proxy(alias)
        a.set_attr(agent_time=global_time)
    print("--- all time variables set to: " + str(new_time) + " ---")


def request_handler(self, message):  # Excess' reaction for a request from deficit agent (price curve or rejection)
    self.log_info('Request received from: ' + str(message[0]) +
                  ' Value: ' + str(message[1]) +
                  ' Other content: ' + str(message[2]))
    self.set_attr(n_requests=self.get_attr('n_requests')+1)  # counts number received requests

    # now, reply of the price curve should be triggered
    myaddr = self.bind('PUSH', alias='price_curve_reply')
    ns.proxy(message[0]).connect(myaddr, handler=price_curve_handler)
    power_balance = self.current_balance(self.get_attr('agent_time'))
    self.set_attr(power_balance=power_balance)

    # check if is an excess now
    if power_balance > 0:
        self.log_info("I have " + str(power_balance) + " to sell. Sending price curve...")
        val = float(message[1]) if power_balance >= float(message[1]) else power_balance
        price = self.current_price(self.get_attr('agent_time'))
        price_curve_array = np.array([self.name, val, price, "I send the price curve"])
        self.send('price_curve_reply', price_curve_array)
    else:
        self.log_info("I cannot sell (D of B). Sending rejection...")
        price_curve_array = np.array([self.name, False, False, "Rejection. No price curve"])
        self.send('price_curve_reply', price_curve_array)


def price_curve_handler(self, message): # Deficit reaction for the received price curve from Excess
    self.log_info('Price curve received from: ' + str(message[0]) +
                  ' Possible quantity: ' + str(message[1]) +
                  ' Price: ' + str(message[2]) +
                  ' Other content: ' + str(message[3]))
    # save all the curves
    self.get_attr('iteration_memory_pc').append(message)
    #pp(self.get_attr('iteration_memory_pc'))

    # after receiving all the curves&rejections, need to run my own opf now, implement to own system, create the bids...
    if len(self.get_attr('iteration_memory_pc')) == sum(self.get_attr('adj'))-1:
        self.log_info('All price curves received (' + str(len(self.get_attr('iteration_memory_pc'))) +
                      '), need to run new opf, derive bids etc...')
        bids = self.new_opf()

        # send bids back to the price-curve senders
        for b in bids:
            to_whom = b[0]
            price = b[1]
            bid_value = b[2]
            bid_offer_array = np.array([self.name, price, bid_value, "That's a bid."])
            myaddr = self.bind('PUSH', alias='bid_offer')
            ns.proxy(to_whom).connect(myaddr, handler=bid_offer_handler)
            self.send('bid_offer', bid_offer_array)


def bid_offer_handler(self, message):
    self.log_info(message)
    self.get_attr('iteration_memory_bid').append(message)

    # gather all the bids, same number as number of requests, thus number of price curves sent etc.
    if len(self.get_attr('iteration_memory_bid')) == self.get_attr('n_requests'):
        # make the calculation or just accept in some cases
        vsum = 0
        for bid in self.get_attr('iteration_memory_bid'):
            vsum = vsum + float(bid[2])
        if vsum <= self.get_attr('power_balance'): # if sum of all is less then excess -> accept all
            self.log_info('Accept all bids')
        else:
            self.log_info('Need another negotiation iteration because sum of bid (' + vsum + ') > excess: (' + self.get_attr('power_balance') + ')')


def runOneTimestep():
    """
    Should include all iterations of negotiations.
    :return:
    """
    multi_consensus = False

    while not multi_consensus:

        # 1 iteration loop includes separate steps (loops):
        #   - D request loop
        #       when all are distributed then each E that receive a request evaluates own opf
        #   - E response with price curves to all interested
        #   - D evaluates all received curves (OPF) and send bids to E according to its OPF
        #   - E accept bid or refuses and if so then send a new curve ---> new iteration

        erase_iteration_memory()

        for vpp_idx in range(vpp_n):
            agent = ns.proxy(data_names[vpp_idx])

            # each agent checks the internal resources (from data or from internal agent (OPF))
            power_balance = agent.current_balance(global_time)

            # each deficit agent publishes deficit to the neighbours only
            if power_balance > 0:
                agent.log_info("I am excess")
                agent.set_attr(current_status=['E', power_balance])

            elif power_balance < 0:
                agent.log_info("I am deficit. I'll publish requests to neighbours.")
                agent.set_attr(current_status=['D', power_balance])

                # request in the following form: name, quantity, some content
                message_request = np.array([ns.agents()[vpp_idx], -1*power_balance, "I need help"])
                agent.send('main', message_request, topic='request_topic')

            else:
                agent.log_info("I am balanced")
                agent.set_attr(current_status=['B', power_balance])
                agent.set_attr(consensus=True)

        multi_consensus = True


def erase_iteration_memory():
    for vpp_idx in range(vpp_n):
        a = ns.proxy(data_names[vpp_idx])
        a.set_attr(iteration_memory_pc=[])
        a.set_attr(iteration_memory_bid=[])
        a.set_attr(n_requests=0)


if __name__ == '__main__':

    #print_data()

    ##### Initial Settings #####

    #  initialization of the agents
    ns = run_nameserver()
    for vpp_idx in range(vpp_n):
        agent = run_agent(data_names[vpp_idx], base=VPP_ext_agent)
        agent.set_attr(myname=str(data_names[vpp_idx]))
        agent.set_attr(adj=adj_matrix[vpp_idx])


    # for alias in ns.agents(): print(alias)

    # subscriptions to neighbours only
    for vpp_idx in range(vpp_n):
        agent = ns.proxy(data_names[vpp_idx])
        addr = agent.bind('PUB', alias='main') # this agent will publish to neighbours
        for n in range(vpp_n): # loop for requesting only the neighbours
            if n == vpp_idx:
                continue
            if adj_matrix[vpp_idx][n]:
                neighbour = ns.proxy(data_names[n])
                neighbour.connect(addr, handler={'request_topic': request_handler
                                                 }) # only neighbours connect to the agent

    # ##### RUN the simulation

    global_time_set(1)
    runOneTimestep()
    ns.shutdown()
    sys.exit()

    for t in range(ts_n):
        runOneTimestep()
        time.sleep(1)
        global_time_set(t)

    ns.shutdown()

