import matplotlib.pyplot as plt
import scipy.stats as scs
import numpy as np

fig, ax = plt.subplots(1, 1)
x = np.linspace(scs.norm.ppf(0.0001), scs.norm.ppf(0.9999), 10000)

ax.plot(x, scs.norm.pdf(x), 'g-', lw=2)
rv = scs.norm(loc=0, scale=0.5)
ax.plot(x, rv.pdf(x), 'b-', lw=5)

plt.show()