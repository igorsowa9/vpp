import time
import sys
from osbrain import run_agent
from osbrain import run_nameserver

from archive.settings import data_names, vpp_n, adj_matrix
from other_agents import VPP_ext_agent

global_time = 0

message_id_request = 1
message_id_price_curve = 2
message_id_bid_offer = 3
message_id_bid_answer = 4
message_id_final_answer = 5

price_increase_factor = 4


def global_time_set(new_time):
    global global_time
    global_time = new_time

    for alias in ns.agents():
        a = ns.proxy(alias)
        a.set_attr(agent_time=global_time)
    print("--- all time variables set to: " + str(new_time) + " ---")


def request_handler(self, message):  # Excesses' reaction for a request from deficit agent (price curve or rejection)
    from_vpp = message["vpp_name"]
    power_value = message["value"]
    self.log_info('Request received from: ' + str(from_vpp) + ' Value: ' + str(power_value))
    self.set_attr(n_requests=self.get_attr('n_requests')+1)  # counts number received requests

    # now, reply of the price curve should be triggered
    myaddr = self.bind('PUSH', alias='price_curve_reply')
    ns.proxy(from_vpp).connect(myaddr, handler=price_curve_handler)
    power_balance = self.current_balance(self.get_attr('agent_time'))
    self.set_attr(power_balance=power_balance)

    # check if is an excess now
    if power_balance > 0:
        val = float(power_value) if power_balance >= float(power_value) else power_balance
        price = float(self.current_price(self.get_attr('agent_time')))
        self.log_info("I have " + str(power_balance) + " to sell. Sending price curve... "
                                                       "(val="+str(val)+", for price="+str(price)+")")
        price_curve_message = {"message_id": message_id_price_curve, "vpp_name": self.name,
                               "value": val, "price": price}
        self.send('price_curve_reply', price_curve_message)
    else:
        self.log_info("I cannot sell (I am D or B). Sending rejection...")
        price_curve_message = {"message_id": message_id_price_curve, "vpp_name": self.name,
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
        bids = self.new_opf()
        # send bids back to the price-curve senders
        for b in bids:
            self.set_attr(n_bids=self.get_attr('n_bids') + 1)  # counts number bids I send
            vpp_idx_1bid = b[0]
            price = b[1]
            bid_value = b[2]
            bid_offer_message = {"message_id": message_id_bid_offer, "vpp_name": self.name,
                               "price": price, "bid_value": bid_value}
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
            vsum = vsum + float(bid["bid_value"])

        if vsum <= self.get_attr('power_balance'): # if sum of all is less then excess -> accept all
            self.log_info('Accept all bids - send accept messages.')

            # send accept to deficit bidders, but do not set consensus yet, only when final accept is received
            for bid in self.get_attr('iteration_memory_bid'):
                bid_answer_message = {'message_id': message_id_bid_answer, 'vpp_name': self.name,
                                      'bid_value': float(bid['bid_value']), 'bid_price': float(bid['price']),
                                      'str': "That's an accept message for the bid."}

                myaddr = self.bind('PUSH', alias='bid_answer')
                ns.proxy(bid["vpp_name"]).connect(myaddr, handler=bid_accept_handler)
                self.send('bid_answer', bid_answer_message)

        else:  # do not send accept message but start another iteration instead
            self.log_info('Need another negotiation iteration because sum of bids (' + str(vsum) + ') > excess: (' +
                          str(self.get_attr('power_balance')) + ') - I increase the price and send new price curves...')

            # 1) increase iteration counter
            # 2) send new, increased price curves to the Defs that sent the bids


            temp = self.get_attr("timestep_memory_n_iter") + 1
            self.set_attr(timestep_memory_n_iter=temp)

            ###########################
            ###########################
            ###########################


            # erase_iteration_memory()

            for bid in self.get_attr('iteration_memory_bid'):  # loop to send new price curves

                power_value = bid["bid_value"]
                power_balance = self.get_attr('power_balance')
                val = float(power_value) if power_balance >= float(power_value) else power_balance
                price = float(self.current_price(self.get_attr('agent_time'))) + \
                        price_increase_factor*self.get_attr("timestep_memory_n_iter")
                self.log_info("I have " + str(power_balance) + " to sell. Sending INCREASED price curve... "
                                                               "(val=" + str(val) + ", for price=" + str(price) + ")")
                price_curve_message = {"message_id": message_id_price_curve, "vpp_name": self.name,
                                       "value": val, "price": price}
                myaddr = self.bind('PUSH', alias='price_curve_reply')
                ns.proxy(bid["vpp_name"]).connect(myaddr, handler=price_curve_handler)
                self.send('price_curve_reply', price_curve_message)


def bid_accept_handler(self, message):  # executed in Defs
    self.log_info("Bid accept received from: " + message['vpp_name'])
    self.get_attr('iteration_memory_bid_accept').append(message)

    # gather all the bids accepts, same number as number
    if len(self.get_attr('iteration_memory_bid_accept')) == self.get_attr('n_bids'):
        self.log_info("All bids accepted. I send the final accept.")
        # if all bids are accepted, then confirm to the Excess all deals
        myaddr = self.bind('PUSH', alias='bid_accept_answer')
        for bid in self.get_attr('iteration_memory_bid_accept'):
            bid_final_accept_message = {"message_id": message_id_final_answer, "vpp_name": self.name,
                                       "value": bid['value'], "price": bid['price']}
            ns.proxy(bid["vpp_name"]).connect(myaddr, handler=bid_final_accept_handler)
            self.send('bid_accept_answer', bid_final_accept_message)

        mydeals = []
        for accepted_bid in self.get_attr('iteration_memory_bid_accept'):
            mydeals.append([accepted_bid['vpp_name'], accepted_bid['bid_value'], accepted_bid['bid_price']])
        self.set_attr(timestep_memory_mydeals=mydeals)
        self.set_attr(consensus=True)


def bid_final_accept_handler(self,message):  # executed in Exc
    self.log_info("Final bid accept received from: " + message['vpp_name'])
    mybids = []
    for bid in self.get_attr('iteration_memory_bid'):
        mybids.append([bid["vpp_name"], -1 * float(bid['bid_value']), float(bid['price'])])
    self.set_attr(timestep_memory_mydeals=mybids)
    self.set_attr(consensus=True)


def erase_iteration_memory():
    for vpp_idx in range(vpp_n):
        a = ns.proxy(data_names[vpp_idx])
        a.set_attr(iteration_memory_pc=[])
        a.set_attr(iteration_memory_bid=[])
        a.set_attr(iteration_memory_bid_accept=[])
        a.set_attr(n_requests=0)
        a.set_attr(n_bids=0)
        a.set_attr(consensus=False)


def erase_timestep_memory():
    for vpp_idx in range(vpp_n):
        a = ns.proxy(data_names[vpp_idx])
        a.set_attr(timestep_memory_mydeals=[])
        a.set_attr(timestep_memory_n_iter=0)


def runOneTimestep():
    """
    Should include all iterations of negotiations.
    :return:
    """
    multi_consensus = False

    erase_timestep_memory()

    while not multi_consensus:

        # 1 iteration loop includes separate steps (loops):
        #   - D request loop
        #       when all are distributed then each E that receive a request evaluates own opf
        #   - E response with price curves to all interested
        #   - D evaluates all received curves (OPF) and send bids to E according to its OPF
        #   - E accept bid or refuses and if so then send a new curve ---> new iteration

        erase_iteration_memory()
        print('--- Deficit agents loop: ---')
        for vpp_idx in range(vpp_n):
            agent = ns.proxy(data_names[vpp_idx])

            # each agent checks the internal resources (from data or from internal agent (OPF))
            power_balance = agent.current_balance(global_time)
            if power_balance < 0:
                agent.log_info("I am deficit. I'll publish requests to neighbours.")
                agent.set_attr(current_status=['D', power_balance])
                my_name = data_names[vpp_idx]
                # request in the following form: name, quantity, some content
                message_request = {"message_id": message_id_request, "vpp_name": my_name,
                                   "value": float(-1 * power_balance)}
                agent.send('main', message_request, topic='request_topic')
        time.sleep(1)
        print('--- Excess and balanced agents loop: ---')
        for vpp_idx in range(vpp_n):
            agent = ns.proxy(data_names[vpp_idx])

            # each agent checks the internal resources (from data or from internal agent (OPF))
            power_balance = agent.current_balance(global_time)
            # each deficit agent publishes deficit to the neighbours only
            if power_balance > 0:
                agent.log_info("I am excess")
                agent.set_attr(current_status=['E', power_balance])
                agent.set_consensus_if_norequest()
            elif power_balance == 0:
                agent.log_info("I am balanced")
                agent.set_attr(current_status=['B', power_balance])
                agent.set_attr(timestep_memory_mydeals=[])
                agent.set_attr(consensus=True)

        #  check the consensus status
        time.sleep(0.5)

        multi_consensus = system_consensus_check()


def system_consensus_check():
        n_consensus = 0
        for alias in ns.agents():
            a = ns.proxy(alias)
            if a.get_attr('consensus')==True:
                n_consensus = n_consensus + 1

        if n_consensus == vpp_n:
            print("CONSENSUS reached for time: ", global_time)
            print("Deals: ")
            for alias in ns.agents():
                a = ns.proxy(alias)
                print(alias + " deals (with?, value buy+/-sell, price): ", a.get_attr('timestep_memory_mydeals'))

            return True
        else:
            print("CONSENSUS NOT reached for time: ", global_time)
            return False


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

