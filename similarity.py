import pandas as pd
import numpy as np
import sys
from pprint import pprint as pp
import copy
from settings_4bus import *
from utilities import *
from datetime import datetime, timedelta

vpp_idx = 2

path_save = '/home/iso/Desktop/vpp_some_results/2018_0822_1555_week1-2/'
path_pickle = path_save + "temp_ln_" + str(vpp_idx) + ".pkl"

f = pd.read_pickle(path_pickle)
# f = f.drop(['Unnamed: 0'], axis=1)

print(f.index)
print(f.columns)

learn_memory = f.iloc[0:333, :]  # only the first week as learning
learn_memory_mod = copy.deepcopy(learn_memory)

labels_for_similarity = {"with_idx_req": 0.15,
                         "minute_t": 0.2,
                         "week_t": 0.15,
                         "month_t": 0.2,
                         "av_weather": 0.3}

### change values in cells for the needs - prepare memory
learn_memory_mod['with_idx_req'] = learn_memory_mod.with_idx_req.apply(lambda x: x[1])
learn_memory_mod['week_t'] = learn_memory_mod.week_t.apply(lambda x: x+1)

for index, row in learn_memory_mod.iterrows():

    res_now_power_all = 0  # potencial renewable power of all opponents together in Watts
    av_weather = row.loc['av_weather']
    for vpp_idx in av_weather.keys():
        vpp_file = load_jsonfile(data_paths[vpp_idx])
        ppc0 = cases[vpp_file['case']]()
        installed_power = np.round(np.sum(ppc0['gen'][1:, 8]), 4)

        res_inst = row['res_inst'][vpp_idx]
        av_weather_onevpp = row['av_weather'][vpp_idx]
        res_now_power = installed_power * res_inst * av_weather_onevpp
        res_now_power_all += res_now_power

    row['av_weather'] = res_now_power_all
    learn_memory_mod.iloc[index] = row

# print(learn_memory_mod.loc[:, ['with_idx_req', 'week_t', 'av_weather']])

### calculate range of values values necessary for similarity
label_ranges = {"with_idx_req": np.round(np.abs(learn_memory_mod['with_idx_req'].max()-learn_memory_mod['with_idx_req'].min()), 4),
                "minute_t": 24*60 / 2,
                "week_t": 7 / 2,
                "month_t": 12 / 2,
                "av_weather": np.round(np.abs(learn_memory_mod['av_weather'].max()-learn_memory_mod['av_weather'].min()), 4)}


### calculate the similarity of the current request
present_set = f.iloc[333:, :]  # in reality each row comes separately in real-time

# this comes from the negotiating agent
request_value = present_set.loc[333, 'with_idx_req'][1]
print(request_value)

# weather data (necessary for the similarity calculation) should be downloaded from the public data:
start_date = datetime.strptime(start_datetime, "%d/%m/%Y %H:%M")
global_delta = timedelta(minutes=global_time * 5)
current_time = start_date + global_delta

minute_t_now = int(current_time.hour * 60 + current_time.minute)
week_t_now = int(current_time.weekday())
month_t_now = int(current_time.month)

# define prospective opponents in case of such a deal
myself = 'vpp3'
deficit_agent = 'vpp2'
prospective_opponents_idx = np.where(np.array(adj_matrix[data_names_dict[myself]]) == True)
po_idx = []
for i in prospective_opponents_idx[0]:
    if i == data_names_dict[myself] or i == data_names_dict[deficit_agent]:
        continue
    po_idx.append(int(i))

for vpp_idx in po_idx:
    vpp_file = self.load_data(data_paths[vpp_idx])

    ppc0 = cases[vpp_file['case']]()
    max_generation0 = copy.deepcopy(ppc0['gen'][:, 8])

    forecast_max_generation_factor = np.zeros(vpp_file['bus_n'])
    forecast_max_generation = np.zeros(vpp_file['bus_n'])
    for idx in range(vpp_file['bus_n']):
        gen_path = vpp_file['generation_profiles_paths'][idx]
        if not gen_path == "":
            d = self.load_data(gen_path)
            forecast_max_generation_factor[idx] = d[global_time][
                                                      MEASURED] + gen_mod_offset  # should be FORECASTED but for now 100% accuracy
            forecast_max_generation[idx] = np.round(forecast_max_generation_factor[idx] * max_generation0[idx], 4)
        po_wf.update({vpp_idx: forecast_max_generation})

    generation_type = np.array(self.load_data(data_paths[vpp_idx])['generation_type'])

    stack = np.vstack((max_generation0, forecast_max_generation, generation_type))
    stack = stack[:, 1:].T

    # print("stack: " + str(stack))

    res_weather = 0
    max_res_weather = 0
    max_other = 0
    max = np.sum(stack[:, 0])
    for s in stack:
        if s[2] in weather_dependent_sources:
            res_weather += s[1]
            max_res_weather += s[0]
        else:
            max_other += s[0]
    res_installed_power_factor.update({vpp_idx: round(max_res_weather / max, 4)})
    av_weather_factor.update({vpp_idx: round(res_weather / max_res_weather, 4)})

sys.exit()


pp(labels_for_similarity)
print("\n\n")

index = f.index
columns = f.columns
values = f.values

mem = f.loc[102][:]
now = f.loc[333][:]

sys.exit()
for label in labels_for_similarity.keys():
    print(str(label) + "-mem: " + str(mem[label]))
    print(str(label) + "-now: " + str(now[label]))

    n = now[label]
    m = mem[label]

    weight = labels_for_similarity[label]
    difference = abs(n - m)/label_range[label]
    weight * (1 - difference)



sys.exit()

# for label in labels_for_similarity.keys():
#     print(label)

a = f.iloc[5, 1]
print(a)