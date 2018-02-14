import time
import json
from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent

global_data = {'VPP':[('VPP1'),('VPP2'),('VPP3')],'DSO':True,'Ext':[('VPP1_Ext'),('VPP2_Ext'),('VPP3_Ext')]}

class DSOClass(Agent):
    def on_init(self):
        self.bind('REP', alias='repIfRequest', handler=self.reply_procedure)


    def pullFromExt(agent, message):
        agent.log_info('Received: %s' % message)


    def reply_procedure(self, message):
        self.log_info("I have received: " + str(
            message) + " from my other Ext. Now I should answer!")

        return 'This is an answer from ' + self.name + ' about the request of value: ' + str(message)



class VPPClassExt(Agent):
    def on_init(self):
        self.bind('PUSH', alias='extPushToInt')
        self.bind('REP', alias='repIfRequest', handler=self.reply_procedure)


    def pullFromInt(agent, message):
        agent.log_warning('Received: ' + str(message))
        if message < 0:
            # load knowledge about other parties - equal sensitivity
            vpp_amount = len(global_data['VPP'])
            if global_data['DSO'] == True:
                all_parties = vpp_amount + 1

            # run sensitivity analysis in order to send questions to other entities
            requests_pu = agent.sensitivity(all_parties)

            requests = [x * message for x in requests_pu]
            agent.log_info("I'll send request: "+str(requests))

            recipients = global_data['VPP']
            my_idx = global_data['Ext'].index(agent.name)  # =0
            recipients[my_idx] = ''
            print(str(recipients))

            for str0 in global_data['VPP']:
                if agent.name == global_data['Ext'][my_idx]: # =VPP1_Ext
                    continue
                str1 = 'agent.send('', request['+requests[my_idx]+'])'
                exec(str1)

            self.send('1to2', requests[1])
            self.send('1to3', requests[2])
            self.send('1toD', requests[3])


    def sensitivity(self, all_parties): # for now just equally between parties
        p = all_parties-1 # minus the one that requests
        v = [1/p]*all_parties
        v[global_data['Ext'].index(self.name)] = 0 # wyzeruj wartosc dla siebie
        return v


    def reply_procedure(self, message):
        # here what happens when Ext asked by other Ext/DSO about the resources
        # - asks its Int if the resources ar available
        # - if yes, confirms to request giver
        # - if not, informs about maximal current resources that can provide

        self.log_info("I have received: "+str(message)+" from my other Ext. Now I should send request/question to my internal agent!")
        #self.send('extPushtoInt', message)

        return 'This is an answer from '+self.name+' about the request of value: ' + str(message)


    def requestOthers(self, values): 
        pass # publishes the public request to other VPPs/DSO if the info about
             # violation comes from Internal agent


    def askInt(self, content):
        # sends the request to internal agent in order to ask about possible
        # ancillary services i.e. when a request from another party comes
        pass


    

class VPPClassInt(Agent):
    def on_init(self):
            # - establishes communication with its External agent (request-reply)
            # - establishes communication with the DGs if they are agents or
            # with the database/file the fake results of optimization data
            # - it monitors the internal state

        self.bind('PUSH', alias='intPushToExt')
        self.check_connection() # now it also loads all data
        self.curr_step = 1 # internal step counter of the agent


    def pullFromExt(agent, message):
        agent.log_info('Received: %s' % message)


    def periodical(self, start_ts, step): # for testing, timestamp from 1 to 10
        self.log_info('Current value for (timestamp='+str(self.curr_step)+'): '+
              str(self.net[self.curr_step-1]))

        n = self.net[self.curr_step-1]
        if n<0:
            self.send('intPushToExt', n) # request to its external agent

        if self.curr_step == self.max_step:
            self.kill()
        self.curr_step  = self.curr_step + 1 # incrementing of internal counter by step


    def check_connection(self): # from file or from database (...)
        file_path = 'data/data_' + self.name + '.json'
        with open(file_path) as data_file:    
            self.load_data = json.load(data_file)
        print('data ready (from file) for', self.name)
        data = self.load_data
        generation = data['generation']
        load = data['load']
        load = [i * (-1) for i in load]
        self.net = [x + y for x, y in zip(generation, load)]
        self.max_step = len(self.net)


    def live_read_from_db(self):
        # runs the optimization when all internal data are gathered
        pass

    def send_request(self):
        pass # sends info to External agent if there is a violation

