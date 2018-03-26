import cmath
import sys

d = 1
s = 50*1j

a = -3.26e-394*cmath.exp(-8.0*d*s)*(6.2e361*cmath.exp(7.0*d*s))

print(a)

# a = (-3.26e-394*math.exp(-8.0*d*s)*(6.2e361*math.exp(7.0*d*s) - math.exp(8.0*d*s)*(7e404 - 5.73e404i)))*s^34