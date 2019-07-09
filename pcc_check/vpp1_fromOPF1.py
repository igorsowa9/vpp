"""File created: 2019-05-31 18:02:19"""

from numpy import array
def vpp1_fromOPF1():
	ppc = {"version": '2'}
	ppc["baseMVA"] = 100.0
	ppc["bus"] = array([
		[0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[1.0, 1.0, 9.7875, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[2.0, 1.0, 11.745000000000001, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[3.0, 1.0, 5.427999999999999, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[4.0, 1.0, 11.0175, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9]
	])

	ppc["gen"] = array([
		[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 10000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 4.2075000000000005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 1.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 330.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	])

	ppc["branch"] = array([
		[0.0, 1.0, 0.00281, 0.0281, 0.00712, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[0.0, 3.0, 0.00304, 0.0304, 0.00658, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[0.0, 4.0, 0.00064, 0.0064, 0.03126, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[1.0, 2.0, 0.00108, 0.0108, 0.01852, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[2.0, 3.0, 0.00297, 0.0297, 0.00674, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[3.0, 4.0, 0.00297, 0.0297, 0.00674, 24.0, 24.0, 24.0, 0.0, 0.0, 1.0, -360.0, 360.0]
	])

	ppc["gencost"] = array([
		[2, 0, 0, 2, 100, 0],
		[2, 0, 0, 2, 4, 0],
		[2, 0, 0, 2, 21, 0],
		[2, 0, 0, 2, 32, 0],
		[2, 0, 0, 2, 93, 0]
	])


	return ppc

