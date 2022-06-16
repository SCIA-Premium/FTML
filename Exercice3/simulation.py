import numpy as np
import pandas as pd
from numpy.random import normal

sigma2 = 3
n = 1000000
d = 3
X = [np.random.randint(10000, size=n)] * d
noise = normal(0,size=n, scale=np.sqrt(sigma2))

ols_y = 12.34*X[0] + 3.67*X[1] + 4.92*X[2] + 69.2
real_y = ols_y + noise

def noise_variance_estimator(real_y, y):
    res = 0
    for i in range(len(real_y)):
        res += (real_y[i] - y[i])**2
    return res/(n - (d+1))

res = noise_variance_estimator(ols_y, real_y)

print("The approximated noise variance is:",res)
print("The real variance is:", sigma2)