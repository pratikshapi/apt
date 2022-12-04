import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("test1", delimiter=",")
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.show()