import sys

outfile1 = sys.argv[1]
outfile2 = sys.argv[2]

f1 = open(outfile1, "r")
n1 = int(f1.readline())
step1 = int(f1.readline())
width1 = float(f1.readline())

f2 = open(outfile2, "r")
n2 = int(f2.readline())
step2 = int(f2.readline())
width2 = float(f2.readline())

if n1 != n2 or step1 != step2 or width1 != width2:
    sys.stdout.write("Wrong Answer\n")
    sys.exit(1)

result1 = f1.read().split()
result2 = f2.read().split()

for i in range((step1 + 1) * n1):
    if abs(float(result1[i]) - float(result2[i])) > 1e-5:
        sys.stdout.write("Wrong Answer\n")
        sys.stderr.write(f"at step {i // n1}, {i % n1} element\n")
        sys.stderr.write(f"Error = {abs(float(result1[i]) - float(result2[i]))}\n")
        sys.stderr.write(f"number1 = {float(result1[i])}, number2 = {float(result2[i])}\n")
        sys.exit(1)

sys.stderr.write("Accepted\n")
