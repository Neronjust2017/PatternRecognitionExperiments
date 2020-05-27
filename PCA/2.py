import numpy as np
a = np.array([48.08181865, 19.09195919,  6.62368557,  4.31294935])
s = a.sum()
np.set_printoptions(precision=4)
c = 0
for i in range(len(a)):
    c += a[i]
    print(np.array(c / s))