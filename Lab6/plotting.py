import numpy as np
import matplotlib.pyplot as plt


data1 = np.genfromtxt("init.csv", delimiter=",")
data2 = np.genfromtxt("finalTemperatures.csv", delimiter=",")
# plt.imshow(data1, cmap='hot', interpolation='nearest')
# plt.imshow(data2, cmap='hot', interpolation='nearest')

fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].imshow(data1, cmap='hot', interpolation='nearest')
axs[1].imshow(data2, cmap='hot', interpolation='nearest')

plt.show()
