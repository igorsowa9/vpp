from pypower.api import *
from case5_vpp import case5_vpp
from rundcopf_noprint import rundcopf

from other_agents import VPP_ext_agent



ppc = case5_vpp()
opt = ppoption(VERBOSE=1)
r = rundcopf(ppc, opt)
