from numpy import array

def case4_vpp3():

    ppc = {"version": '2'}

    ppc["baseMVA"] = 100.0

    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [0, 3, 0,   0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [1, 1, 170, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [2, 1, 125, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [3, 1, 80,  0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [0, 0, 0, 0, 0, 1, 100, 1, 1e4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 100, 1, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # fixed
        [2, 0, 0, 0, 0, 1, 100, 1, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 1, 100, 1, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # PV
    ])

    ## branch data
    #fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [0, 1, 0.01008,	0.0504,	0.1025,  0, 0, 0, 0, 0, 1, -360, 360],
        [0, 2, 0.00744,	0.0372,	0.0775,  0, 0, 0, 0, 0, 1, -360, 360],
        [1, 3, 0.00744,	0.0372,	0.0775,  0, 0, 0, 0, 0, 1, -360, 360],
        [2, 3, 0.01272,	0.0636,	0.1275,  0, 0, 0, 0, 0, 1, -360, 360]

    ])

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 0, 0, 2, 100, 0],  # slack-export
        [2, 0, 0, 2, 12,   0],  # decreased from 14 / 12
        [2, 0, 0, 2, 0,   0],
        [2, 0, 0, 2, 9,  0]
    ])

    return ppc
