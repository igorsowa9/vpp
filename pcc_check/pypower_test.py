from pypower.api import *
from vpp1_fromOPF_e2_all import vpp1_fromOPF_e2_all
import numpy as np
from pprint import pprint as pp

ppc = vpp1_fromOPF_e2_all()

r = rundcopf(ppc)

print(np.round(r['gen'][:, [0, 1, 8, 9]], 4))