import numpy as np

a = np.ones(3)
print(a)
print(a[None, :], a[None, :].shape) # [[]]
print(a[:, None], a[:, None].shape) # [[], [], []]
