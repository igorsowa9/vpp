from pypower.api import *
from data.vpp4bus.case4_vpp3 import case4_vpp
from pypower_mod.rundcopf_noprint import rundcopf

ppc = case4_vpp()
opt = ppoption(VERBOSE=1)
r = rundcopf(ppc, opt)

printpf(r)
