import time
import sys
import osbrain
from osbrain import run_agent
from osbrain import run_nameserver
from pprint import pprint as pp
import copy
from settings_4bus import *
from other_agents import VPP_ext_agent
from utilities import *

global_time = ts_0

message_id_request = 1
message_id_price_curve = 2
message_id_bid_offer = 3
message_id_bid_accept = 4
message_id_bid_accept_modify = 41
message_id_final_answer = 5


def global_ns():
    global ns
    ns = run_nameserver()


def global_time_set(new_time, ns):
    global global_time
    global_time = new_time

    for alias in ns.agents():
        a = ns.proxy(alias)
        a.set_attr(agent_time=global_time)
    print("--- all time variables set to: " + str(new_time) + " ---")


def each_iteration_set(agent_name, it):
    for alias in ns.agents():
        if alias == agent_name:
            continue
        a = ns.proxy(alias)
        a.set_attr(n_iteration=int(it))
    print("--- all n_iterations set to: " + str(it) + " ---")


def request_handler(self, message):
    """
    Excesses' gathering of requests
    """
    from_vpp = message["vpp_name"]
    power_value = message["value"]
    self.log_info('Request received from: ' + str(from_vpp) + ' Value: ' + str(power_value))
    self.set_attr(n_requests=self.get_attr('n_requests')+1)  # counts number received requests (Exc)
    self.get_attr('requests').append(message)
    return


def requests_execute(self, myname, requests):
    """
    An Excess agent that receive some requests answer either NO or with price curves.
    This includes running opf1.
    Additionally the opf1 is extended to check the feasibility of price curves (and profitability)
    :param self:
    :param myname:
    :param requests: e.g. {'from_vpp': 'vpp1', 'power_value': 25.0}
    :return:
    """

    for req in requests:
        from_vpp = req["vpp_name"]
        myaddr = self.bind('PUSH', alias='price_curve_reply')
        ns.proxy(from_vpp).connect(myaddr, handler=price_curve_handler)

    opf1 = self.get_attr('opf1')  # download opf1 results

    if opf1['power_balance'] == 0 and opf1['max_excess'] > 0:  # max_excess > 0 i.e. if excess agent

        val = float(opf1['max_excess'])  # max_excess
        if not self.get_attr('opfe2'):
            self.runopf_e2(global_time)  # make price curves based on the excess matrix

        self.log_info("Excess agent request execution. My current iteration number for request execution: " + str(self.get_attr("n_iteration")))
        self.log_info("I have " + str(opf1['max_excess']) + " to sell. Sending price curve... (total excess="
                      + str(val) + ", with price curves matrix about generators)")

        # price_curve should be for current iteration
        n_iteration = self.get_attr("n_iteration")
        price_curves_dict = copy.deepcopy(self.get_attr("pc_memory_exc")[n_iteration])

        if price_curves_dict == {}:
            self.log_info(
                "No my excess PCs for current iteration. I copy the previous iteration step and execute requests.")
            price_curves_dict = copy.deepcopy(self.get_attr("pc_memory_exc")[n_iteration - 1])
            temp = copy.deepcopy(self.get_attr("pc_memory_exc"))
            temp[n_iteration] = temp[n_iteration-1]
            self.set_attr(pc_memory_exc=temp)

        self.log_info("price_curves_dict for this iteration: " + str(self.get_attr('pc_memory_exc')[n_iteration]))

        if 'all' in price_curves_dict.keys():  # single price curve
            price_curve = price_curves_dict["all"]
            self.set_attr(iteration_memory_my_pc=np.array(price_curve))

            price_curve_message = {"message_id": message_id_price_curve, "vpp_name": myname,
                                   "value": val, "price_curve": price_curve}
            for req in requests:
                from_vpp = req["vpp_name"]
                myaddr = self.bind('PUSH', alias='price_curve_reply')
                ns.proxy(from_vpp).connect(myaddr, handler=price_curve_handler)
                self.send('price_curve_reply', price_curve_message)

        else:  # particular price curves
            price_curve = np.array([]).reshape(0, 3)
            for vpp_name, pc in price_curves_dict.items():  # make all PCs together
                price_curve = np.concatenate((price_curve, pc))
            self.set_attr(iteration_memory_my_pc=np.array(price_curve))

            for req in requests:
                from_vpp = req["vpp_name"]
                myaddr = self.bind('PUSH', alias='price_curve_reply')
                ns.proxy(from_vpp).connect(myaddr, handler=price_curve_handler)

                pc = price_curves_dict[from_vpp]
                val = np.round(np.sum(pc[:, 1]), 4)
                price_curve_message = {"message_id": message_id_price_curve, "vpp_name": myname,
                                       "value": val, "price_curve": pc}
                self.send('price_curve_reply', price_curve_message)

    else:
        self.log_info("Deficit agent request execution. My current iteration number for request execution: " + str(
            self.get_attr("n_iteration")))
        self.log_info("I cannot sell (I am D or B). Sending rejections...")
        for req in requests:
            from_vpp = req["vpp_name"]
            myaddr = self.bind('PUSH', alias='price_curve_reply')
            ns.proxy(from_vpp).connect(myaddr, handler=price_curve_handler)
            price_curve_message = {"message_id": message_id_price_curve, "vpp_name": myname,
                                   "value": False, "price_curve": False}
            self.send('price_curve_reply', price_curve_message)


def price_curve_handler(self, message):
    """
    Deficit reaction for the received price curve from Excess. Bids creation and sending to the Exc (also empty bids)
    """
    from_vpp = message["vpp_name"]
    possible_quantity = message["value"]
    price_curve = message["price_curve"]

    # save all the curves
    self.get_attr('iteration_memory_received_pc').append(message)

    # save nonzero curves to pc memory def for this iteration
    n_iteration = self.get_attr("n_iteration")
    if message['price_curve'] is not False:
        self.get_attr('pc_memory_def')[n_iteration].update({message['vpp_name']: np.array(message['price_curve'])})

    self.log_info('Price curve received from (my n_iter='+str(self.get_attr('n_iteration'))+'): ' + from_vpp +
                  ' Possible total quantity: ' + str(possible_quantity) +
                  ' Price curve matrix: ' + str(price_curve) + '\nSo far received: ' +
                  str(len(self.get_attr('iteration_memory_received_pc'))) + '/' + str(sum(self.get_attr('adj'))-1))

    # after receiving all the curves&rejections, need to run my own opf (OPF2) now, implement to own system,
    # create the bids...

    if len(self.get_attr('iteration_memory_received_pc')) == sum(self.get_attr('adj'))-1:
        # print("test004:" + str(self.get_attr('pc_memory_def')))
        self.log_info('All price curves/or empty received (from all neigh.) (' + str(len(self.get_attr('iteration_memory_received_pc'))) +
                      '), need to run new opf, derive bids etc...')
        # pp(self.get_attr('iteration_memory_received_pc'))
        # pp(self.get_attr('pc_memory_def'))
        self.runopf_d2()  # bids come as multi-list: [vpp_idx, gen_idx, bidgen_value, gen_price],[...]
        bids = self.get_attr("opfd2")['bids']

        for vi in range(0, vpp_n):
            bid = bids[np.where(bids[:, 0] == vi), :][0]  # take bids for only one vpp (might be bids for multiple gens)
            if len(bid) > 0:  # send bids back to the price-curve senders, but counts only bids>0
                if sum(bid[:, 2]) > 0:
                    self.set_attr(n_bids=self.get_attr('n_bids') + 1)  # counts number bids>0 I send
                vpp_idx_1bid = int(bid[0][0])  # id of the vpp where I send
                bid_offer_message = {"message_id": message_id_bid_offer, "vpp_name": self.name, "bid": bid}
                myaddr = self.bind('PUSH', alias='bid_offer')
                ns.proxy(data_names[vpp_idx_1bid]).connect(myaddr, handler=bid_offer_handler)
                self.send('bid_offer', bid_offer_message)


def bid_offer_handler(self, message):
    """
    Exc react if they receive a bid from Def (bids ofc based on the price curve sent before)
    """
    self.get_attr('iteration_memory_bid').append(message)
    if np.sum(message["bid"][:, 2]) > 0:
        self.log_info("Received bid>0 matrix from deficit (" + str(len(self.get_attr('iteration_memory_bid')))+"/"
                      + str(self.get_attr('n_requests'))+") - " + message['vpp_name'] + ": " + str(message['bid']))
    else:
        self.log_info("Received empty bid=0 from deficit (" + str(len(self.get_attr('iteration_memory_bid'))) + "/"
                      + str(self.get_attr('n_requests')) + ") - " + message['vpp_name'] + ": " + str(message['bid']))

    # gather all the bids, same number as number of requests, thus number of price curves sent etc.
    if len(self.get_attr('iteration_memory_bid')) == self.get_attr('n_requests'):
        self.log_info("All bids offers (also empty) received from deficit agents (" + str(len(self.get_attr('iteration_memory_bid')))
                      + "/" + str(self.get_attr('n_requests')))
        # make the calculation or just accept in some cases:
        all_bids = []
        for bid_message in self.get_attr('iteration_memory_bid'):
            for bid_message_gen in bid_message['bid']:
                all_bids.append(np.append(data_names_dict[bid_message['vpp_name']], bid_message_gen))

        # all_bids_np = np.array(all_bids)
        all_bids = np.matrix(all_bids)
        c1 = np.array(all_bids[:, 3] != 0)
        all_bids_nz = np.array(all_bids[c1[:, 0], :])
        self.log_info("My all non-zero bids from deficit vpps ("+str(len(all_bids_nz))+"): " + str(all_bids_nz))
        self.log_info("I compare with n_request ("+str(self.get_attr('n_requests')) +
                      ") and introduce n_bidoffers number, excluding vpps with empty bids. n_bidoffers="+str(len(np.unique(all_bids_nz[:, 0]))))
        self.set_attr(n_bidoffers=len(np.unique(all_bids_nz[:, 0])))
        if self.get_attr('n_bidoffers') == 0:
            self.log_info("There are no bids for me now. I set consensus")
            self.set_attr(consensus=True)
            return

        all_bids_sum = np.round(sum(all_bids_nz[:, 3]), 4)  # sum all the bid power (vppidx, genidx, power, price)

        if all_bids_sum <= self.get_attr('opf1')['max_excess']:  # if sum of all is less then excess
            self.log_info('I have sufficient generation to accept all bids (bids total sum ('+str(all_bids_sum)+') <= ('+str(self.get_attr('opf1')['max_excess'])+
                          ') my whole excess), but I have to check according to available generators...')
            # need to share between the gens if necessary.
            mypc0 = self.get_attr('iteration_memory_my_pc')  # current price curves
            # checking if any of bid sum / generator do not exceed available generator powers
            count = 0
            for pc in mypc0:
                gen_id = pc[0]
                bid_1gen = all_bids_nz[all_bids_nz[:, 2] == gen_id]
                if np.sum(bid_1gen[:, 3]) > np.sum(mypc0[mypc0[:, 0] == gen_id, 1]):
                    count = count + 1
                    self.log_info("Need sharing between the gens i.e. modify the bids")

            # If all bid-per-gen sums are less then available gen powers then send accept
            # but do not set consensus yet, only when final accept is received
            if count == 0:
                feasibility = self.runopf_e3(all_bids_nz, global_time)

                if feasibility:
                    self.log_info('pf_e3 (1): feasibility check with the prepared bids: ' + str(feasibility) +
                                  ' . Own original costs (opf1): ' + str(self.get_attr('opf1')['objf']) +
                                  ' . Costs if sold to DSO (opf2): ' + str(self.get_attr('opfe2')['objf_greentodso']) +
                                  ' . Costs with bids revenue (opfe3-bid revenue): ' + str(self.get_attr('opfe3')['objf_inclbidsrevenue']))
                else:
                    self.log_info('Unfeasibility in opf_e3 (' + str(self.name) + ')! Stop.')
                    sys.exit()

                for vi in range(0, vpp_n):
                    bid = all_bids_nz[np.where(all_bids_nz[:, 0] == vi), :][0]  # take bids for only one vpp (might be bids for multiple gens)
                    if len(bid) > 0:  # send bids back to the price-curve senders, but counts only bids>0
                        vpp_idx_1bid = int(bid[0][0])  # id of the vpp where I send
                        bid_answer_message = {'message_id': message_id_bid_accept, 'vpp_name': self.name,
                                              'bid': np.array(bid[:, 1:]),
                                              'str': "That's an accept message for the bid."}
                        self.log_info("I send bid_answer_message to "+str(data_names[vpp_idx_1bid])+".")
                        myaddr = self.bind('PUSH', alias='bid_answer')
                        ns.proxy(data_names[vpp_idx_1bid]).connect(myaddr, handler=bid_answer_handler)
                        self.send('bid_answer', bid_answer_message)

            else:  # i.e. if some further strategy such as simple alignment is necessary

                all_bids_mod, new_pc = self.bids_alignment1(mypc0, all_bids_nz)

                # opf should be checked if the transport of such a power is possible to the respective deficit vpps through the respective PCCs:
                feasibility = self.runopf_e3(all_bids_mod, global_time)

                if feasibility:
                    self.log_info('pf_e3 (2): feasibility check with the prepared bids: ' + str(feasibility) +
                                  ' . Own original costs (opf1): ' + str(self.get_attr('opf1')['objf']) +
                                  ' . Costs if sold to DSO (opf1): ' + str(self.get_attr('opfe2')['objf_greentodso']) +
                                  ' . Costs with bids revenue (opfe3-bid revenue): ' + str(self.get_attr('opfe3')['objf_inclbidsrevenue']))

                    ######## build new price curve according to modified bids, increase iteration, save and break
                    n_i = copy.deepcopy(self.get_attr("n_iteration"))
                    n_i = n_i + 1
                    self.set_attr(n_iteration=n_i)

                    # make new PC (for particular VPPs) according to aligned bids for the next iteration
                    exc_matrix = self.get_attr('opf1')['exc_matrix']

                    for vpp_idx in np.unique(all_bids_mod[:, 0]):
                        pc = all_bids_mod[all_bids_mod[:, 0] == vpp_idx, :]
                        self.get_attr('pc_memory_exc')[int(n_i)].update({data_names[int(vpp_idx)]: np.array(pc[:, 2:])})

                    self.log_info("Alignment: PC matrix for requesters according to alignment: "
                                  + str(self.get_attr('pc_memory_exc')[n_i]) + '\nNeed another negotiation with those new, '
                                  'modified price curves (tailored for each vpp)')
                    self.log_info("I BREAK iteration and increase for the other. My n_iteration becomes: " + str(self.get_attr('n_iteration')))
                    # increase the iteration for all
                    each_iteration_set(self.myname, self.get_attr("n_iteration"))
                    return
                else:
                    self.log_info('Unfeasibility in opf_e3! STOP.')
                    sys.exit()

        else:  # not enought excess - increase n_iteration and return to new iteration
            self.log_info('Need another negotiation iteration because sum of bids (' + str(all_bids_sum) + ') > excess: (' +
                          str(self.get_attr('opf1')['max_excess']) + ') - I increase the price and send new price curves '
                                                                '(return)...')

            n_i = copy.deepcopy(self.get_attr("n_iteration"))
            n_i = n_i + 1  # increase iteration (based on local value)
            self.set_attr(n_iteration=n_i)  # setting higher iteration value for the price increase only to this agent
            self.log_info("I BREAK iteration and increase for the other. My n_iteration becomes: " + str(self.get_attr('n_iteration')))

            # increase the iteration for all
            each_iteration_set(self.myname, self.get_attr("n_iteration"))
            return  # break to start from the PC curve with increased iteration step


def bid_answer_handler(self, message):
    """
    executed in Defs. After receiving answer about the bids (accept, modified accept, refused), the opf should be done
    in order to see of anything has improved:
        - for example in case of modified answer, the proposed bids might be too bad...
        - should be implemented into original price curves and evaluated again
    """
    if message['message_id'] == message_id_bid_accept:
        self.log_info("Bid answer accept received from: " + message['vpp_name'])
        self.get_attr('iteration_memory_bid_accept').append(message)

    else:
        self.log_info("Unhandled message type received. STOP")
        sys.exit()

    # gather all the bids accepts, same number as n_bids
    if len(self.get_attr('iteration_memory_bid_accept')) == self.get_attr('n_bids'):
        self.log_info("All my bids with accept answer. First I should check feasibility and if objf improved. "
                      "Then, I send the final confirmation (normal PUSH reply) and set mydeals")

        for bid in self.get_attr('iteration_memory_bid_accept'):
            myaddr = self.bind('PUSH', alias='bid_final_confirm')
            bid_final_accept_message = {"message_id": message_id_final_answer, "vpp_name": self.name, "bid": bid}

            ns.proxy(bid["vpp_name"]).connect(myaddr, handler=bid_final_confirm_handler)
            self.send('bid_final_confirm', bid_final_accept_message)

        self.runopf_d3()  # sets mydeals for DEFs and calculates the costs After for Deficit agents


def bid_final_confirm_handler(self, message):
    """
    executed in Exc
    """

    if message['message_id'] == message_id_final_answer:
        to_save = message['bid']
        to_save['vpp_name_buyer'] = message['vpp_name']
        self.get_attr('iteration_memory_bid_finalanswer').append(to_save)

    self.log_info("Final bid accept received from: " + message['vpp_name'] + ". So far: " +
                  str(len(self.get_attr('iteration_memory_bid_finalanswer'))) + "/" + str(self.get_attr('n_bidoffers')))

    if len(self.get_attr('iteration_memory_bid_finalanswer')) == self.get_attr('n_bidoffers'):
        mydeals = []
        for bid in self.get_attr('iteration_memory_bid_finalanswer'):
            bid['bid'][:, 2] = -1 * bid['bid'][:, 2]  # change value to negative ("I SELL...") for clarity
            mydeals.append([bid["vpp_name_buyer"], bid['bid']])

        self.set_attr(timestep_memory_mydeals=mydeals)
        self.set_attr(consensus=True)


def runOneTimestep():

    ##### Initial Settings #####
    global_ns()
    global_time_set(t, ns)

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
        agent.runopf1(global_time)

    ######## Save OPF1 to opf1_history ########
    ###########################################
    ###########################################
    # saving into opf1_history:
    # - general results of opf1: excess/deficit value, objf, objf_noslackcost,
    # - generators: results and constraints (dependent on the weather)
    # - loads: results (and constraints)
    opf1_save_balcost = np.zeros((vpp_n, 4))
    VPP_MAXEXC = 0   # max excess value
    VPP_PBAL = 1   # value to balance - deficit
    OBJF = 2   # objective function value from opf1
    OBJF_NODSO = 3  # objective function if no dso buying costs

    bnmax = 1
    for alias in ns.agents():
        a = ns.proxy(alias)
        bn = np.array(a.load_data(data_paths[data_names_dict[alias]])['bus_n'])
        if bn > bnmax:
            bnmax = bn

    opf1_save_genload = np.zeros((vpp_n, bnmax, 4))
    opf1_save_prices = np.zeros((vpp_n, bnmax))
    LOAD_FIX = 0  # max excess value
    GEN_RES = 1  # value to balance - deficit
    GEN_UP = 2  # objective function value from opf1
    GEN_LOW = 3  # objective function if no dso buying costs

    for alias in ns.agents():
        a = ns.proxy(alias)
        vpp_idx = data_names_dict[alias]

        opf1_save_balcost[vpp_idx, VPP_MAXEXC] = a.get_attr('opf1')['max_excess']
        opf1_save_balcost[vpp_idx, VPP_PBAL] = a.get_attr('opf1')['power_balance']
        opf1_save_balcost[vpp_idx, OBJF] = a.get_attr('opf1')['objf']
        opf1_save_balcost[vpp_idx, OBJF_NODSO] = a.get_attr('opf1')['objf_noslackcost']

        ppc_t = a.get_attr("opf1_ppct")
        res = a.get_attr("opf1_res")

        bus_n = ppc_t['bus'].shape[0]

        for b in range(bus_n):
            opf1_save_genload[vpp_idx, b, LOAD_FIX] = round(ppc_t['bus'][b, 2], 4)
            opf1_save_genload[vpp_idx, b, GEN_RES] = round(res['gen'][b, 1], 4)  # resulting powers
            opf1_save_genload[vpp_idx, b, GEN_UP] = round(ppc_t['gen'][b, 8], 4)  # modified upper constraint
            opf1_save_genload[vpp_idx, b, GEN_LOW] = round(ppc_t['gen'][b, 9], 4)  # modified lower constraint

            opf1_save_prices[vpp_idx, b] = ppc_t['gencost'][b, 4]

    save_opf1_history(global_time, opf1_save_balcost, opf1_save_genload, opf1_save_prices)

    ###########################################
    ###########################################

    if negotiation:
        for vpp_idx in  range(vpp_n):
            agent = ns.proxy(data_names[vpp_idx])
            opf1 = agent.get_attr('opf1')
            if opf1['power_balance'] < 0:
                agent.log_info("I am deficit. I'll publish requests to neighbours.")
                my_name = data_names[vpp_idx]
                message_request = {"message_id": message_id_request, "vpp_name": my_name,
                                   "value": float(-1 * opf1['power_balance'])}
                agent.send('main', message_request, topic='request_topic')

        time.sleep(small_wait)  # show gathered requests
        print('Adjacency matrix: ' + str(adj_matrix))
        print("\n\n#######################")
        print("# Resulting requests: #")
        print("#######################")
        for vpp_idx in range(vpp_n):
            agent = ns.proxy(data_names[vpp_idx])
            print(str(data_names[vpp_idx]) + ":\n(to balance, max exc, objf noslack: " + str(agent.get_attr('opf1')) +
                  ") \n(no. of received requests: " + str(agent.get_attr('n_requests')) + ") : \n" + str(agent.get_attr('requests')) + "\n")
        print("#######################")
        print("#######################")
        print("#######################\n\n")

        while not multi_consensus:

            print('\n\n\n\nNEW ITERATION LOOP')

            erase_iteration_memory(ns)

            time.sleep(iteration_wait)
            print('--- Def loop: Execute requests ---')
            for vpp_idx in range(vpp_n):
                agent = ns.proxy(data_names[vpp_idx])
                if agent.get_attr('n_requests') > 0:
                    requests_execute(agent, data_names[vpp_idx], agent.get_attr('requests'))

            time.sleep(small_wait)
            print('- Def loop: Consensus Check 1')
            if system_consensus_check(ns, global_time):
                break

            print('\n--- Excess and balanced agents loop: ---')
            for vpp_idx in range(vpp_n):
                agent = ns.proxy(data_names[vpp_idx])

                if agent.get_attr('consensus'):
                    agent.log_info('I already have consensus from deficit loop.')
                    continue

                if not agent.get_attr('consensus'):
                    if agent.get_attr('opf1'):
                        opf1 = agent.get_attr('opf1')
                    else:
                        agent.runopf1(agent.get_attr('agent_time'))
                        opf1 = agent.get_attr('opf1')

                    if opf1['power_balance'] == 0 and opf1['max_excess'] > 0:
                        agent.set_consensus_if_norequest()
                    elif opf1['power_balance'] == 0 and opf1['max_excess'] == 0:
                        agent.log_info("I am balanced. I set consensus.")
                        agent.set_attr(timestep_memory_mydeals=[])
                        agent.set_attr(consensus=True)
                    elif opf1['power_balance'] < 0 and opf1['max_excess'] == False:
                        agent.log_info("I'm a deficit agent, shouldn't I be handled earlier...")

            time.sleep(small_wait)
            multi_consensus = system_consensus_check(ns, global_time)

        ######## Save OPF3 to opf3_history ########
        ###########################################
        ###########################################
        # saving into opf1_history:
        # - general results of opf1: excess/deficit value, objf, objf_noslackcost,
        # - generators: results and constraints (dependent on the weather)
        # - loads: results (and constraints)

        VPP_OBJF_AFTER = 0  # costs after negotiation

        opfe3_history = np.zeros((vpp_n, 1))

        for alias in ns.agents():
            a = ns.proxy(alias)
            vpp_idx = data_names_dict[alias]
            if a.get_attr("opfe3"):
                objf_inclbidsrevenue = a.get_attr("opfe3")['objf_inclbidsrevenue']
                opfe3_history[vpp_idx, VPP_OBJF_AFTER] = np.round(objf_inclbidsrevenue, 4)
            elif a.get_attr("opfd3"):
                total_withbids = np.round(a.get_attr('opfd3')['buybids_cost'] + a.get_attr('opf1')['objf_noslackcost'], 4)
                opfe3_history[vpp_idx, VPP_OBJF_AFTER] = total_withbids
            else:
                opfe3_history[vpp_idx, VPP_OBJF_AFTER] = np.round(a.get_attr('opf1')['objf'], 4)

        save_opfe3_history(global_time, opfe3_history)

        """
        Previously in main
        """

        ns.shutdown()


if __name__ == '__main__':

    # SETTINGS OSBRAIN
    # osbrain.config['TRANSPORT'] = 'tcp'
    # ##### RUN the simulation

    erase_learning_memory(vpp_learn)

    if not constant_environment:
        timeline = range(ts_0, ts_0+ts_n)
        if explore_multi_oneshot > 1:
            timeline = [i for i in range(ts_0, ts_0 + ts_n) for _ in range(explore_multi_oneshot)]
    else:
        timeline = ts_0*np.ones(ts_n)
        timeline = timeline.astype(int).tolist()

    for t in timeline:
        time.sleep(small_wait)
        runOneTimestep()

    time.sleep(small_wait)

    save_learning_memory(tocsv)
    show_results_history(pdf)
