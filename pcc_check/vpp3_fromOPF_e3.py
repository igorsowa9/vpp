"""File created: 2018-09-06 14:45:21"""

from numpy import array
def vpp3_fromOPF_e3():
	ppc = {"version": '2'}
	ppc["baseMVA"] = 100.0
	ppc["bus"] = array([
		[0.0, 3.0, 2.895, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[1.0, 1.0, 22.355, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[2.0, 1.0, 14.95, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[3.0, 1.0, 31.6, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9]
	])

	ppc["gen"] = array([
		[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 10000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 71.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	])

	ppc["branch"] = array([
		[0.0, 1.0, 0.01008, 0.0504, 0.1025, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[0.0, 2.0, 0.00744, 0.0372, 0.0775, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[1.0, 3.0, 0.00744, 0.0372, 0.0775, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[2.0, 3.0, 0.01272, 0.0636, 0.1275, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0]
	])

	ppc["gencost"] = array([
		[2, 0, 0, 2, 100, 0],
		[2, 0, 0, 2, 17, 0],
		[2, 0, 0, 2, 0, 0],
		[2, 0, 0, 2, 9, 0]
	])


	return ppc

