from pypower.api import *
from case5_vpp import case5_vpp

ppc = case5_vpp()
r = rundcopf(ppc)

#print(ppc)