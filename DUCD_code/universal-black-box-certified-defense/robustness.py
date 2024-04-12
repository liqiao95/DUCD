import numpy as np

def calculate_robustness(Acc_sigma, sigma_candidates, R_range):
    robustness = 0.0

    for R in R_range:
        max_acc = -np.inf
        for sigma in sigma_candidates:
            acc = Acc_sigma[sigma][R]
            if acc > max_acc:
                max_acc = acc

        robustness += max_acc

    return robustness
sigmas = [0.25, 0.5,0.75, 1.0]
import numpy as np
from scipy.integrate import quad

def integrand(x, alpha, beta):
    return x**2 * np.exp(-np.abs(x / alpha)**beta)

def variance(alpha, beta):
    result, _ = quad(integrand, 0, np.inf, args=(alpha, beta))
    return result - 1

# 使用二分法求解方差为1的适当 alpha 值
def find_alpha(beta):
    lower_bound = 0.01
    upper_bound = 10.0
    precision = 0.001

    while upper_bound - lower_bound > precision:
        mid = (lower_bound + upper_bound) / 2
        if variance(mid, beta) > 0:
            lower_bound = mid
        else:
            upper_bound = mid

    return (lower_bound + upper_bound) / 2

beta = 0.5
var = -1.0

alpha = find_alpha(beta)
print(f"The alpha value for beta={beta} and variance={var} is: {alpha}")

