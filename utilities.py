from settings import *

def system_consensus_check(ns, global_time):
    n_consensus = 0
    for alias in ns.agents():
        a = ns.proxy(alias)
        if a.get_attr('consensus') == True:
            n_consensus = n_consensus + 1

    if n_consensus == vpp_n:
        print("- Multi-consensus reached (" + str(n_consensus) + "/" + str(vpp_n) + ") for time: ", global_time)
        print("----- Deals: -----")
        for alias in ns.agents():
            a = ns.proxy(alias)
            print(alias + " deals (with?, value buy+/-sell, price): ", a.get_attr('timestep_memory_mydeals'))

        return True
    else:
        print("- Multi-consensus NOT reached (" + str(n_consensus) + "/" + str(vpp_n) + ") for time: ", global_time)
        return False


def erase_iteration_memory(ns):
    print('--- iteration M erase ---')
    for vpp_idx in range(vpp_n):
        a = ns.proxy(data_names[vpp_idx])
        a.set_attr(iteration_memory_pc=[])
        a.set_attr(iteration_memory_bid=[])
        a.set_attr(iteration_memory_bid_accept=[])
        a.set_attr(n_bids=0)


def erase_timestep_memory(ns):
    print('--- timestamp M erase ---')
    for vpp_idx in range(vpp_n):
        a = ns.proxy(data_names[vpp_idx])
        a.set_attr(timestep_memory_mydeals=[])
        a.set_attr(n_iteration=0)
        a.set_attr(n_requests=0)
        a.set_attr(consensus=False)
        a.set_attr(requests=[])
