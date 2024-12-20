import math
import sys

n = int(sys.argv[1])
f = sys.argv[2]
fout = open(f, "w")

print(n, file=fout)
print(100000, file=fout)
print(40, file=fout)
print(0.0001, file=fout)
print(n * n / 2500, file=fout)
print(0.1, file=fout)  # kappa

for i in range(n):
    t = (i + 1) / (n + 1)
    print(
        f"{math.exp(-(t - 0.3) ** 2 / 0.02) + math.exp(-(t - 0.8) ** 2 / 0.0162) * 0.75}",
        file=fout,
    )
fout.close()
