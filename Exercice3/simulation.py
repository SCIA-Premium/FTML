import numpy as np
import pandas as pd
from numpy.random import normal

# Here we define our main variable 
sigma2 = 3
n = 1000000
d = 3
X = [np.random.randint(10000, size=n)] * d
noise = normal(0, size=n, scale=np.sqrt(sigma2))

ols = 12.34*X[0] + 3.67*X[1] + 4.92*X[2] + 69.2
real = ols + noise

noised_var = 0
for i in range(len(ols)):
    noised_var += (ols[i] - real[i])**2
noised_var = noised_var/(n - (d+1))


print("\n\nThe expected value of the expression in step 6 was Sigma square.")

print("\nHere in the step7, we had to produce a numerical simulation that estimate the previous result and check if it was consistent :\n")
print("Our approximated simulation of the variance is:", noised_var)
print("The real variance is:", sigma2, "\n\n")