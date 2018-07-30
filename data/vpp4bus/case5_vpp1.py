from numpy import array

def case5_vpp1():

    ppc = {"version": '2'}

    ppc["baseMVA"] = 100.0

    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [0, 3, 0,  0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [1, 1, 25, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],  # g1
        [2, 1, 30, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],  # g4
        [3, 1, 40, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],  # h1
        [4, 1, 65, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9]   # l1
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [0, 0, 0, 0, 0, 1, 100, 1, 1e4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 100, 1, 25,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # PV with forecast
        [2, 0, 0, 0, 0, 1, 100, 1, 60,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # onshore with forecast
        [3, 0, 0, 0, 0, 1, 100, 1, 40,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # offshore with forecast
        [4, 0, 0, 0, 0, 1, 100, 1, 130,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # -
    ])

    ## branch data
    #fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [0, 1, 0.00281,	0.0281,	0.00712, 0, 0, 0, 0, 0, 1, -360, 360],
        [0, 3, 0.00304,	0.0304,	0.00658, 0, 0, 0, 0, 0, 1, -360, 360],
        [0, 4, 0.00064,	0.0064,	0.03126, 0, 0, 0, 0, 0, 1, -360, 360],
        [1, 2, 0.00108,	0.0108,	0.01852, 0, 0, 0, 0, 0, 1, -360, 360],
        [2, 3, 0.00297,	0.0297,	0.00674, 0, 0, 0, 0, 0, 1, -360, 360],
        [3, 4, 0.00297,	0.0297,	0.00674, 24, 24, 24, 0, 0, 1, -360, 360]
    ])

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 0, 0, 2, 200, 0],
        [2, 0, 0, 2, 4,   0],
        [2, 0, 0, 2, 21,  0],
        [2, 0, 0, 2, 32,  0],
        [2, 0, 0, 2, 103, 0]
    ])

    return ppc
