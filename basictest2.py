import numpy as np
import pandas as pd

currmem = pd.read_pickle("/home/iso/Desktop/vpp_some_results/2018_0808_1604_7days_with_negotiation_allsaving/temp_ln_2.pkl")
t = 151

t_range = np.arange(t - 3, t)

fulfilled = 0  # check if the time exist and if some conditions are fulfilled
for test_t in t_range:
    if 't' in currmem.columns:
        test_row = currmem.loc[currmem['t'] == test_t]
        if not test_row.empty:
            if (currmem['t'] == test_t).any() and test_row.iloc[0]['success'] == 1:
                fulfilled += 1