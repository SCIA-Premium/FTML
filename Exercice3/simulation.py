import numpy as np
import pandas as pd
from numpy.random import normal

# Here we define our main variable 
sigma2 = 3
n = 1000000
d = 3
X = [np.random.randint(10000, size=n)] * d
noise = normal(0, size=n, scale=np.sqrt(sigma2))

ols_y = 12.34*X[0] + 3.67*X[1] + 4.92*X[2] + 69.2
real_y = ols_y + noise

noised_var = 0
for i in range(len(ols_y)):
    noised_var += (ols_y[i] - real_y[i])**2
noised_var = noised_var/(n - (d+1))

print("The approximated noise variance is:", noised_var)
print("The real variance is:", sigma2)