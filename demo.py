import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import sys

demofile = sys.argv[1]
outfile = sys.argv[2]
rate = int(sys.argv[3])

fig = plt.figure()

f = open(demofile, "r")
n = int(f.readline())
step = int(f.readline())
width = float(f.readline())
result = f.read().split()
f.close()

result_np = np.zeros((step + 1, n + 2))
for i in range(step + 1):
    for j in range(n):
        result_np[i][j + 1] = float(result[i * n + j])

x = np.linspace(0, 1, n + 2) * width
u = result_np[0]
(plot,) = plt.plot(x, u)


def update(t):
    u = result_np[rate * t]
    plot.set_ydata(u)
    return (plot,)


anim = FuncAnimation(fig, update, frames=(step + 1) // rate, blit=True)
anim.save(outfile, fps=30)
# plt.show()
# plt.savefig(outfile)
