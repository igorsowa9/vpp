from pypower.api import *
from fromOPF_e3 import fromOPF_e3
import numpy as np
from pprint import pprint as pp

ppc = fromOPF_e3()

r = rundcopf(ppc)

print(np.round(r['gen'][:, [0, 1, 8, 9]], 4))