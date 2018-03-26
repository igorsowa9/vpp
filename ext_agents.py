from osbrain import Agent
import json
import settings
import time
from pprint import pprint as pp
from settings import data_names, data_names_dict, data_paths, vpp_n, ts_n, adj_matrix, print_data

#from vpps_only_v03 import system_consensus_check


class VPP_ext_agent(Agent):

    def on_init(self):
        #self.data_names = data_names
        #self.data_names_dict = data_names_dict
        #self.data_paths = data_paths
        #self.adj_matrix = adj_matrix
        pass

    def load_data(self, path):
        with open(path, 'r') as f:
            arr = json.load(f)
        return arr

    def current_balance(self, time):
        json_data = self.load_data(data_paths[data_names_dict[self.name]])
        return float(json_data["generation"][time] - json_data["load"][time])

    def current_price(self, time):
        json_data = self.load_data(data_paths[data_names_dict[self.name]])
        return json_data["price"][time]

    def new_opf(self):
        # opf asa system is integrated
        memory = self.get_attr('iteration_memory_pc')
        need = abs(self.get_attr('current_status')[1])

        memory_list = []
        for mem in memory:
            memory_list.append([data_names_dict[mem["vpp_name"]], mem["value"], mem["price"]])

        sorted_memory = sorted(memory_list, key=lambda price: price[2])

        bids = []
        for pc in sorted_memory:
            pc_vpp_idx = pc[0]
            pc_maxval = float(pc[1])
            pc_price = float(pc[2])
            if need > pc_maxval:
                bids.append([pc_vpp_idx, pc_price, pc_maxval])
                need = need - pc_maxval
            else:
                bids.append([pc_vpp_idx, pc_price, need])
        return bids


    def set_consensus_if_norequest(self):

        time.sleep(0.1)
        if self.get_attr('n_requests') == 0:
            self.log_info('I am E and I nobody wants to buy from me (n_requests=' + str(self.get_attr('n_requests')) +
                          '). I set consensus.')
            self.set_attr(consensus=True)
