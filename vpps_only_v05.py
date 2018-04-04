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

global_time = 0

message_id_request = 1
message_id_price_curve = 2
message_id_bid_offer = 3
message_id_bid_accept = 4
message_id_final_answer = 5


def global_time_set(new_time):
    global global_time
    global_time = new_time

    for alias in ns.agents():
        a = ns.proxy(alias)
        a.set_attr(agent_time=global_time)
    print("--- all time variables set to: " + str(new_time) + " ---")


def forall_iteration_set(value, vpp_exclude):

    for alias in ns.agents():
        if alias == vpp_exclude:
            pass
        else:
            a = ns.proxy(alias)
            a.set_attr(n_iteration=value)
    print("--- all agents set new iteration value: " + str(value) + " ---")


def request_handler(self, message):  # Excesses' reaction for a request from deficit agent (price curve or rejection)
    from_vpp = message["vpp_name"]
    power_value = message["value"]
    self.log_info('Request received from: ' + str(from_vpp) + ' Value: ' + str(power_value))
    self.set_attr(n_requests=self.get_attr('n_requests')+1)  # counts number received requests
    self.get_attr('requests').append(message)
    return


def requests_execute(self, myname, requests):
    for req in requests:
        from_vpp = req["vpp_name"]
        # now, reply of the price curve should be triggered
        myaddr = self.bind('PUSH', alias='price_curve_reply')
        ns.proxy(from_vpp).connect(myaddr, handler=price_curve_handler)
        power_balance = self.current_balance(self.get_attr('agent_time'))
        self.set_attr(power_balance=power_balance)

        # check if is an excess now
        if power_balance > 0:
            # val = float(power_value) if power_balance >= float(power_value) else power_balance
            val = float(power_balance)  # send all available power
            price = float(self.current_price(self.get_attr('agent_time'))) + \
                    price_increase_factor*self.get_attr("n_iteration")
            self.log_info("I have " + str(power_balance) + " to sell. Sending price curve... "
                                                           "(val="+str(val)+", for price="+str(price)+")")
            price_curve_message = {"message_id": message_id_price_curve, "vpp_name": myname,
                                   "value": val, "price": price}
            self.send('price_curve_reply', price_curve_message)
        else:
            self.log_info("I cannot sell (I am D or B). Sending rejection...")
            price_curve_message = {"message_id": message_id_price_curve, "vpp_name": myname,
                                   "value": 0, "price": 0}
            self.send('price_curve_reply', price_curve_message)


def price_curve_handler(self, message):  # Deficit reaction for the received price curve from Excess
    from_vpp = message["vpp_name"]
    possible_quantity = message["value"]
    price = message["price"]

    self.log_info('Price curve received from: ' + from_vpp +
                  ' Possible quantity: ' + str(possible_quantity) +
                  ' Price: ' + str(price))
    # save all the curves
    self.get_attr('iteration_memory_pc').append(message)

    # after receiving all the curves&rejections, need to run my own opf now, implement to own system, create the bids...
    if len(self.get_attr('iteration_memory_pc')) == sum(self.get_attr('adj'))-1:
        self.log_info('All price curves received (from all neigh.) (' + str(len(self.get_attr('iteration_memory_pc'))) +
                      '), need to run new opf, derive bids etc...')
        bids = self.new_opf()  # bids come as lists
        # send bids back to the price-curve senders
        for b in bids:
            self.set_attr(n_bids=self.get_attr('n_bids') + 1)  # counts number bids I send
            vpp_idx_1bid = b[0]
            price = b[1]
            bid_value = b[2]
            bid_offer_message = {"message_id": message_id_bid_offer, "vpp_name": self.name,
                               "price": price, "value": bid_value}
            myaddr = self.bind('PUSH', alias='bid_offer')
            ns.proxy(data_names[vpp_idx_1bid]).connect(myaddr, handler=bid_offer_handler)
            self.send('bid_offer', bid_offer_message)


def bid_offer_handler(self, message):  # Exc react if they receive a bid from Def (based on the price curve sent before)
    self.log_info("Received bid from deficit - " + message['vpp_name'])
    self.get_attr('iteration_memory_bid').append(message)

    # gather all the bids, same number as number of requests, thus number of price curves sent etc.
    if len(self.get_attr('iteration_memory_bid')) == self.get_attr('n_requests'):
        # make the calculation or just accept in some cases

        vsum = 0
        for bid in self.get_attr('iteration_memory_bid'):
            vsum = vsum + float(bid["value"])

        if vsum <= self.get_attr('power_balance'): # if sum of all is less then excess -> accept and wait for reply
            self.log_info('I accept all bids (bids sum < my excess) and send accept messages.')

            # send request of reply to deficit bidders, but do not set consensus yet, only when final accept is received
            for bid in self.get_attr('iteration_memory_bid'):
                bid_answer_message = {'message_id': message_id_bid_accept, 'vpp_name': self.name,
                                      'value': float(bid['value']), 'price': float(bid['price']),
                                      'str': "That's an accept message for the bid."}

                myaddr = self.bind('PUSH', alias='bid_answer')
                ns.proxy(bid['vpp_name']).connect(myaddr, handler=bid_answer_handler)
                self.send('bid_answer', bid_answer_message)


        else:  # send refuse and new price curve (new iteration)
            self.log_info('Need another negotiation iteration because sum of bids (' + str(vsum) + ') > excess: (' +
                          str(self.get_attr('power_balance')) + ') - I increase the price and send new price curves '
                                                                '(return)...')
            n_i = copy.deepcopy(self.get_attr("n_iteration"))
            n_i = n_i + 1  # increase iteration (based on local value)
            self.set_attr(n_iteration=n_i)  # setting higher iteration value for the price increase only to this agent
            return  # break to start from the PC curve with increased iteration step


def bid_answer_handler(self, message):  # executed in Defs
    self.log_info("Bid answer received from: " + message['vpp_name'])
    if message['message_id'] == message_id_bid_accept:
        self.get_attr('iteration_memory_bid_accept').append(message)

    # gather all the bids accepts, same number as n_bids
    if len(self.get_attr('iteration_memory_bid_accept')) == self.get_attr('n_bids'):
        self.log_info("All my bids with accept answer. I send the final confirmation (normal PUSH reply"
                      " and set mydeals)")

        for bid in self.get_attr('iteration_memory_bid_accept'):
            myaddr = self.bind('PUSH', alias='bid_final_confirm')
            bid_final_accept_message = {"message_id": message_id_final_answer, "vpp_name": self.name,
                                       "value": bid['value'], "price": bid['price']}
            ns.proxy(bid["vpp_name"]).connect(myaddr, handler=bid_final_confirm_handler)
            self.send('bid_final_confirm', bid_final_accept_message)

        mydeals = []
        for accepted_bid in self.get_attr('iteration_memory_bid_accept'):
            mydeals.append([accepted_bid['vpp_name'], accepted_bid['value'], accepted_bid['price']])
        self.set_attr(timestep_memory_mydeals=mydeals)
        self.set_attr(consensus=True)


def bid_final_confirm_handler(self, message):  # executed in Exc
    self.log_info("Final bid accept received from: " + message['vpp_name'])
    mybids = []
    for bid in self.get_attr('iteration_memory_bid'):
        mybids.append([bid["vpp_name"], -1 * float(bid['value']), float(bid['price'])])
    self.set_attr(timestep_memory_mydeals=mybids)
    self.set_attr(consensus=True)


def runOneTimestep():
    """
    Should include all iterations of negotiations.
     1 iteration includes steps:
     * D agents request (#1) loop including storing them, counting requests - to all adjacent;
        - E responses (#2) with PRICE CURVES to all interested, if any;
        - D evaluates all received price curves (through opf) and send BIDS (#3) to E according to its opf
        - E ACCEPTS or REFUSES bids (#4)and if so then terminates i.e. triggers new iteration
        - D sends final CONFIRMATION (#5) if all of its bids are accepted
    :return:
    """
    multi_consensus = False

    erase_timestep_memory(ns)

    print('--- Deficit agents initialization - making requests: ---')
    for vpp_idx in range(vpp_n):
        agent = ns.proxy(data_names[vpp_idx])
        power_balance = agent.current_balance(global_time)
        if power_balance < 0:
            agent.log_info("I am deficit. I'll publish requests to neighbours.")
            agent.set_attr(current_status=['D', power_balance])
            my_name = data_names[vpp_idx]
            message_request = {"message_id": message_id_request, "vpp_name": my_name,
                               "value": float(-1 * power_balance)}
            agent.send('main', message_request, topic='request_topic')

    time.sleep(small_wait)  # show gathered requests
    print("- Resulting requests: -")
    for vpp_idx in range(vpp_n):
        agent = ns.proxy(data_names[vpp_idx])
        print(str(data_names[vpp_idx]) + " (balance: " + str(agent.current_balance(global_time)) +
              ") (no. of requests: " + str(agent.get_attr('n_requests')) + ") : " + str(agent.get_attr('requests')))

    while not multi_consensus:

        erase_iteration_memory(ns)

        time.sleep(small_wait)
        print('--- Def loop: Execute requests ---')
        for vpp_idx in range(vpp_n):
            agent = ns.proxy(data_names[vpp_idx])
            if agent.get_attr('n_requests') > 0:
                requests_execute(agent, data_names[vpp_idx], agent.get_attr('requests'))

        time.sleep(small_wait)
        print('- Def loop: Consensus Check 1')
        system_consensus_check(ns, global_time)

        print('--- Excess and balanced agents loop: ---')
        for vpp_idx in range(vpp_n):
            agent = ns.proxy(data_names[vpp_idx])

            if agent.get_attr('consensus'):
                agent.log_info('I already have consensus from deficit loop.')
                continue

            if not agent.get_attr('consensus'):
                power_balance = agent.current_balance(global_time)
                if power_balance > 0:
                    agent.log_info("I am excess")
                    agent.set_attr(current_status=['E', power_balance])
                    agent.set_consensus_if_norequest()
                elif power_balance == 0:
                    agent.log_info("I am balanced")
                    agent.set_attr(current_status=['B', power_balance])
                    agent.set_attr(timestep_memory_mydeals=[])
                    agent.set_attr(consensus=True)
                elif power_balance < 0:
                    agent.log_info("I'm a deficit agent, shouldn't I be handled earlier...")

        time.sleep(small_wait)
        print('- Def loop: Consensus Check 2: ')
        system_consensus_check(ns, global_time)

        time.sleep(small_wait)
        multi_consensus = system_consensus_check(ns, global_time)


if __name__ == '__main__':

    #print_data()

    ##### Initial Settings #####
    ns = run_nameserver()

    for vpp_idx in range(vpp_n):
        agent = run_agent(data_names[vpp_idx], base=VPP_ext_agent)
        agent.set_attr(myname=str(data_names[vpp_idx]))
        agent.set_attr(adj=adj_matrix[vpp_idx])

    # subscriptions to neighbours
    for vpp_idx in range(vpp_n):
        agent = ns.proxy(data_names[vpp_idx])
        addr = agent.bind('PUB', alias='main') # this agent will publish to neighbours
        for n in range(vpp_n): # loop for requesting only the neighbours
            if n == vpp_idx:
                continue
            if adj_matrix[vpp_idx][n]:
                neighbour = ns.proxy(data_names[n])
                neighbour.connect(addr, handler={'request_topic'
                                                 : request_handler})  # only neighbours connect to the agent

    # ##### RUN the simulation

    ## TEST for one time only ###
    global_time_set(2)          #
    runOneTimestep()            #
    time.sleep(1)               #
    ns.shutdown()               #
    sys.exit()                  #
    #############################

    for t in range(3):

        time.sleep(1)
        global_time_set(t)
        runOneTimestep()

    time.sleep(1)
    ns.shutdown()

