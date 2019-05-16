from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
mat = loadmat('matlab/surface1d_poison_xent.mat')['mat']
# mat = np.log10(mat)
# plt.hist(mat.ravel(), 100)
# plt.plot(mat[26,:])
# plt.show()

centers = np.argmin(mat, axis=1)
threshes = np.inter



plt.hist(center)
plt.show()


