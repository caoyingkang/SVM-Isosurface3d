#!/usr/bin/python2
# -*- coding: utf-8 -*-
# @Time: 2018-11-14
# @Author: Yingkang Cao

from math import exp
import numpy as np
from scipy import linalg
from mayavi import mlab
import os
import config

plt_margin = (config.plt_d1_margin,
              config.plt_d2_margin,
              config.plt_d3_margin)


def Kernel(x):
    return exp(-linalg.norm(x) ** 2 / (2 * sigma ** 2))


if __name__ == "__main__":
    # load ndarrays
    base_dir = "tmp"
    X = np.load(os.path.join(base_dir, "X.npy"))
    y = np.load(os.path.join(base_dir, "y.npy"))
    alpha = np.load(os.path.join(base_dir, "alpha.npy"))
    b = np.load(os.path.join(base_dir, "b.npy"))
    sigma = np.load(os.path.join(base_dir, "sigma.npy"))

    # plot scatter points
    X0, X1 = X[y == -1], X[y == 1]
    mlab.points3d(X0[:, 0], X0[:, 1], X0[:, 2],
                  scale_factor=0.15, color=(1, 0, 0))
    mlab.points3d(X1[:, 0], X1[:, 1], X1[:, 2],
                  scale_factor=0.15, color=(0, 0, 1))

    # plot separating surface
    x_d1_min, x_d2_min, x_d3_min = X.min(axis=0) - plt_margin
    x_d1_max, x_d2_max, x_d3_max = X.max(axis=0) + plt_margin
    x_d1_plot, x_d2_plot, x_d3_plot = np.mgrid[
        x_d1_min:x_d1_max:20j, x_d2_min:x_d2_max:20j, x_d3_min:x_d3_max:20j
    ]

    @np.vectorize
    def func(_x1, _x2, _x3):
        curr_x = np.array([_x1, _x2, _x3])
        curr_k = np.array([Kernel(curr_x - xi) for xi in X])
        return np.dot(alpha * y, curr_k) + b

    src = mlab.pipeline.scalar_field(
        x_d1_plot, x_d2_plot, x_d3_plot, func(x_d1_plot, x_d2_plot, x_d3_plot)
    )
    mlab.pipeline.iso_surface(src, contours=[-1, 0, 1], opacity=0.3)
    mlab.xlabel("x1")
    mlab.ylabel("x2")
    mlab.zlabel("x3")
    mlab.scalarbar(label_fmt="%.1f")
    mlab.outline()
    mlab.show()
