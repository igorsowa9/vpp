"""File created: 2018-08-22 15:49:17"""

from numpy import array
def vpp1_fromOPF_e2_all():
	ppc = {"version": '2'}
	ppc["baseMVA"] = 100.0
	ppc["bus"] = array([
		[0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[1.0, 1.0, 9.9325, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[2.0, 1.0, 11.919, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[3.0, 1.0, 5.224, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9],
		[4.0, 1.0, 7.572500000000001, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 230.0, 1.0, 1.1, 0.9]
	])

	ppc["gen"] = array([
		[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 1e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 7.8351, 7.6799, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 27.1594, 26.6216, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 1e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 1e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 65.2535, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 1e-06, -65.2535, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 7.49, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 1e-06, -7.49, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 330.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 1.0, 1e-06, -330.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	])

	ppc["branch"] = array([
		[0.0, 1.0, 0.00281, 0.0281, 0.00712, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[0.0, 3.0, 0.00304, 0.0304, 0.00658, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[0.0, 4.0, 0.00064, 0.0064, 0.03126, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[1.0, 2.0, 0.00108, 0.0108, 0.01852, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[2.0, 3.0, 0.00297, 0.0297, 0.00674, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -360.0, 360.0],
		[3.0, 4.0, 0.00297, 0.0297, 0.00674, 24.240000000000002, 24.240000000000002, 24.240000000000002, 0.0, 0.0, 1.0, -360.0, 360.0]
	])

	ppc["gencost"] = array([
		[2.0, 0.0, 0.0, 2.0, 100.0, 0.0],
		[2.0, 0.0, 0.0, 2.0, 4.0, 0.0],
		[2.0, 0.0, 0.0, 2.0, 21.0, 0.0],
		[2.0, 0.0, 0.0, 2.0, 32.0, 0.0],
		[2.0, 0.0, 0.0, 2.0, 93.0, 0.0],
		[2.0, 0.0, 0.0, 2.0, 0.1, 0.0],
		[2.0, 0.0, 0.0, 2.0, 1000.0, 0.0],
		[2.0, 0.0, 0.0, 2.0, 0.1, 0.0],
		[2.0, 0.0, 0.0, 2.0, 1000.0, 0.0],
		[2.0, 0.0, 0.0, 2.0, 0.1, 0.0],
		[2.0, 0.0, 0.0, 2.0, 1000.0, 0.0]
	])


	return ppc

