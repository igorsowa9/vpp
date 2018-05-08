from pypower.api import *
from case4_vpp import case4_vpp
from pypower_mod.rundcopf_noprint import rundcopf

ppc = case4_vpp()
opt = ppoption(VERBOSE=1)
r = rundcopf(ppc, opt)

printpf(r)
