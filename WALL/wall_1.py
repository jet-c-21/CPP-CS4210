from decimal import *

a = 0.2
b = -1
c = 0.1

d = Decimal(str(a)) * Decimal(str(b)) * Decimal(str(c))

print(float(d), type(d))

print(-0.0 == 0)