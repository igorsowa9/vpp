from oct2py import octave

octave.addpath('/home/iso/PycharmProjects/vpp/matpow_cases')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0')
octave.addpath('/home/iso/PycharmProjects/vpp/matpower6.0/t')

# load basic topology
# update the current generation constraints (weather), fixed loads (curves loads) or flexible loads constraints
#   (but usually they are fixed according to contracts with clients)
# update bid offers in own system i.e. simulate what would be if you accept them, i.e. run OPF:
#   * to be modified:
#       - load at slack bus as negative generator
#       - price of that load (i.e. negative generator)


#   - is revenue increased then?
#   - LATER: influence of mediating to increase the revenue...?
#   - EXTENSION: maximize the revenue within the bid offer


ppc = octave.case5_vpp()
mpopt = octave.mpoption('out.all', 1)
r = octave.rundcopf(ppc, mpopt)
print('SUCCESS?: ', r['success'])


def upload_bid_to_mpc(mpc0, bids):
