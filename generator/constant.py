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
    print(f"{0.5}", file=fout)
fout.close()
