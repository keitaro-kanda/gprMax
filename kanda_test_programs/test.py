import numpy as np
from scipy import linalg

a = np.array([[0., -1.], [1., 0.]])
print(a)
print(linalg.eig(a, left=False, right=True)[1])
print(linalg.eigvals(a))

b = np.array([2., 1.])

print(a@b)