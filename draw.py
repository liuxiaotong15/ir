import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.txt')

y1 = list(data[0]/np.amax(data[0]))
y2 = list(data[1]/np.amax(data[1]))

x = list(range(len(y1)))

plt.plot(x, y2)
plt.plot(x, y1)

plt.show()
