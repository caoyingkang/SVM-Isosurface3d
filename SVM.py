#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2018-11-14
# @Author: Yingkang Cao

from scipy import linalg
import numpy as np
from collections import defaultdict
from math import exp
import os
import argparse
import config

data_file = os.path.join("data", config.data_file_name)
eps = config.eps
num_iter = 0


def Kernel(x):
    return exp(-linalg.norm(x) ** 2 / (2 * sigma ** 2))


def checkKKT(_yf, _alpha, _E, _skip):
    KKT = True
    pick_i = -1
    # choose pick_i: outer loop
    tmp = 0.0
    for i, (yfi, alphai) in enumerate(zip(_yf, _alpha)):
        if eps < alphai < C - eps and not (1 - eps <= yfi <= 1 + eps):
            KKT = False
            if i in _skip and len(_skip[i]) == n - 1:
                continue
            elif abs(yfi - 1) > tmp:
                tmp = abs(yfi - 1)
                pick_i = i
    # choose pick_i: inner loop
    if pick_i == -1:
        for i, (yfi, alphai) in enumerate(zip(_yf, _alpha)):
            if (alphai <= eps and not yfi > 1 + eps) or (
                alphai >= C - eps and not yfi < 1 - eps
            ):
                KKT = False
                if i in _skip and len(_skip[i]) == n - 1:
                    continue
                else:
                    pick_i = i
                    break
    if pick_i == -1:
        return KKT, None, None
    # choose pick_j
    Ei = _E[pick_i]
    tmp = 0.0
    pick_j = -1
    for j, Ej in enumerate(_E):
        if j == pick_i or j in _skip[pick_i]:
            continue
        if abs(Ej - Ei) > tmp:
            tmp = abs(Ej - Ei)
            pick_j = j
    assert pick_j != -1, "pick_i >= 0 , pick_j == -1 !"
    return KKT, pick_i, pick_j


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sigma",
        type=float,
        default=config.sigma_deflt,
        help="parameter in RBF Kernel (float, >0, default={})".format(
            config.sigma_deflt),
    )
    parser.add_argument(
        "-c",
        "--C",
        type=float,
        default=config.C_deflt,
        help="penalty term (float, >0, default={})".format(
            config.C_deflt),
    )
    args = parser.parse_args()
    sigma = args.sigma
    C = args.C
    assert sigma > 0, "sigma should be positive!"
    assert C > 0, "C should be positive!"

    # read data
    X = []
    y = []
    with open(data_file, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) > 0:
                curr_x = list(map(float, line[:-1]))
                X.append(np.array(curr_x))
                y.append(1 if line[-1] == "1" else -1)
    X = np.array(X)
    y = np.array(y)
    n = X.shape[0]

    # get K matrix
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        K[i, i] = Kernel(np.zeros(n))
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = Kernel(X[i] - X[j])

    # initializa parameters
    alpha = np.zeros(n, dtype=float)
    b = 0
    f = np.dot(K, alpha * y) + b
    E = f - y
    skip = defaultdict(set)

    # SMO
    while True:
        # check KKT, decide which two alpha's to update
        getKKT, idx1, idx2 = checkKKT(y * f, alpha, E, skip)
        if idx1 is None:
            print("obtain KKT?", getKKT)
            break

        # calculate alpha_2_new, and see if (idx1, idx2) is good
        alpha_1_old, alpha_2_old = alpha[idx1], alpha[idx2]
        K11, K22, K12 = K[idx1, idx1], K[idx2, idx2], K[idx1, idx2]
        y1, y2 = y[idx1], y[idx2]
        E1, E2 = E[idx1], E[idx2]
        if y1 == y2:
            L = max(0, alpha_1_old + alpha_2_old - C)
            H = min(C, alpha_1_old + alpha_2_old)
        else:
            L = max(0, alpha_2_old - alpha_1_old)
            H = min(C, C - alpha_1_old + alpha_2_old)
        alpha_2_new = alpha_2_old + y2 * (E1 - E2) / (K11 + K22 - 2 * K12)
        alpha_2_new = max(alpha_2_new, L)
        alpha_2_new = min(alpha_2_new, H)

        if abs(alpha_2_new - alpha_2_old) < eps:
            skip[idx1].add(idx2)
            continue

        # do one iteration, renew alpha and b
        num_iter += 1
        print("obtain KKT?", getKKT)
        skip.clear()
        alpha_1_new = alpha_1_old + y1 * y2 * (alpha_2_old - alpha_2_new)
        alpha[idx1] = alpha_1_new
        alpha[idx2] = alpha_2_new
        b_new_1 = (
            b - E1
            - y1 * K11 * (alpha_1_new - alpha_1_old)
            - y2 * K12 * (alpha_2_new - alpha_2_old)
        )
        b_new_2 = (
            b - E2
            - y1 * K12 * (alpha_1_new - alpha_1_old)
            - y2 * K22 * (alpha_2_new - alpha_2_old)
        )
        if eps < alpha_1_new < C - eps:
            b = b_new_1
        elif eps < alpha_2_new < C - eps:
            b = b_new_2
        else:
            b = (b_new_1 + b_new_2) / 2

        # renew f, E
        f = np.dot(K, alpha * y) + b
        E = f - y
        prec = np.sum(y * f > 0) / n
        print("iteration #{}\tprecision:{:.2f}%\n".format(num_iter, prec * 100))

    # show result
    print("")
    print("sigma={}, C={}".format(sigma, C))
    print("alpha=")
    print(alpha)

    # save ndarrays for plotting: refer to "SVM_isosurface.py"
    base_dir = "tmp"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    np.save(os.path.join(base_dir, "X.npy"), X)
    np.save(os.path.join(base_dir, "y.npy"), y)
    np.save(os.path.join(base_dir, "alpha.npy"), alpha)
    np.save(os.path.join(base_dir, "b.npy"), b)
    np.save(os.path.join(base_dir, "sigma.npy"), sigma)
