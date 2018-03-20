from osbrain import Agent
import json
import settings
from pprint import pprint as pp
from settings import data_names, data_names_dict, data_paths, vpp_n, ts_n, adj_matrix, print_data


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

        sorted_memory = sorted(memory, key=lambda price: price[3])

        # {"message_id": message_id_price_curve, "vpp_name": self.name,
        #  "value": 0, "price": 0}
        #########################################3
        ######################################
        ####################################

        bid = []
        for pc in sorted_memory:
            pc_vpp_idx = pc[1]
            pc_max = float(pc[2])
            pc_price = float(pc[3])
            if need > pc_max:
                bid.append([pc_vpp_idx, pc_price, pc_max])
                need = need - pc_max
            else:
                bid.append([pc_vpp_idx, pc_price, need])
        return bid
