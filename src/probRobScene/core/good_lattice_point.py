import time

import numpy as np
from itertools import combinations, permutations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def good_lattice_point(s: int, n: int) -> np.ndarray:
    rps = relative_primes(n)[1:]
    if len(rps) + 1 < s:
        raise ValueError(f"Not enough relative primes of n: {n} for factor size s: {s}")

    rp_combos = np.array(list(combinations(rps, s - 1)))
    rp_combos = np.pad(rp_combos, ((0, 0), (1, 0)), constant_values=1.0)

    normalized_designs = [design_from_genvec(gen_vec, n, endpoint_normalize) for gen_vec in rp_combos]

    best_design = min(normalized_designs, key=centred_l2_discrepancy)

    return best_design


def centred_l2_discrepancy(X: np.ndarray) -> float:
    n: int = X.shape[0]
    s: int = X.shape[1]

    sum_prod_1 = np.sum(np.prod(1 + 0.5 * np.abs(X - 0.5) - 0.5 * (X - 0.5) ** 2, axis=1), axis=0)
    sum_prod_2 = 0.0

    Xik_diffs = 0.5 * np.abs(X[:, np.newaxis] - X[np.newaxis, :])
    X_centered = 0.5 * np.abs(X - 0.5)
    X_ik_centre_sums = X_centered[:, np.newaxis] + X_centered[np.newaxis, :]

    sum_prod_2 += np.sum(np.prod(1.0 + X_ik_centre_sums - Xik_diffs, axis=2))

    return ((13 / 12) ** s - (2 / n) * sum_prod_1 + (1.0 / n ** 2) * sum_prod_2) ** 0.5


def centred_l2_discrepancy_slow(X: np.ndarray) -> float:
    n: int = X.shape[0]
    s: int = X.shape[1]

    sum_prod_1 = np.sum(np.prod(1 + 0.5 * np.abs(X - 0.5) - 0.5 * (X - 0.5) ** 2, axis=1), axis=0)
    sum_prod_2 = 0.0

    for i in range(n):
        for k in range(n):
            p = 1.0
            for j in range(s):
                p *= 1.0 + 0.5 * np.abs(X[i, j] - 0.5) + 0.5 * np.abs(X[k, j] - 0.5) - 0.5 * np.abs(X[i, j] - X[k, j])
            sum_prod_2 += p

    return ((13 / 12) ** s - (2 / n) * sum_prod_1 + (1.0 / n ** 2) * sum_prod_2) ** 0.5


def left_normalize(X: np.ndarray, n: int) -> np.ndarray:
    return (X - 1.0) / n


def centred_normalize(X: np.ndarray, n: int) -> np.ndarray:
    return (X - 0.5) / n


def endpoint_normalize(X: np.ndarray, n: int) -> np.ndarray:
    return (X - 1.0) / (n - 1.0)


def miss_endpoint_normalize(X: np.ndarray, n: int) -> np.ndarray:
    return X / (n + 1)


def relative_primes(n):
    assert n > 1
    nums = np.array(range(1, n))
    gcds = np.gcd(nums, n)
    return nums[gcds == 1]


def design_from_genvec(gen_vec: np.ndarray, n: int, normalizing_func=None) -> np.ndarray:
    multiples = np.arange(1, n + 1)
    potential_design = np.mod(np.outer(multiples, gen_vec), n)
    potential_design[potential_design == 0] = n

    if normalizing_func is not None:
        return normalizing_func(potential_design, n)
    return potential_design


if __name__ == "__main__":
    uniform_design = good_lattice_point(2, 31)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(uniform_design[:, 0], uniform_design[:, 1])
    plt.show()