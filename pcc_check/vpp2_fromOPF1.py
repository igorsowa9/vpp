"""File created: 2018-08-23 15:39:30"""

from numpy import array
def vpp2_fromOPF1():
	ppc = {"version": '2'}
	ppc["baseMVA"] = 100.0
	ppc["bus"] = array([
		[0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[1.0, 1.0, 38.325, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[2.0, 1.0, 99.645, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[3.0, 1.0, 40.158, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9]
	])

	ppc["gen"] = array([
		[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 10000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 150.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 1.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	])

	ppc["branch"] = array([
		[0.0, 1.0, 0.01008, 0.0504, 0.1025, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[0.0, 2.0, 0.00744, 0.0372, 0.0775, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[1.0, 3.0, 0.00744, 0.0372, 0.0775, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[2.0, 3.0, 0.01272, 0.0636, 0.1275, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0]
	])

	ppc["gencost"] = array([
		[2, 0, 0, 2, 100, 0],
		[2, 0, 0, 2, 29, 0],
		[2, 0, 0, 2, 7, 0],
		[2, 0, 0, 2, 11, 0]
	])


	return ppc

