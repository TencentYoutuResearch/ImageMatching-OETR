#!/usr/bin/env python
"""
@File    :   visualization.py
@Time    :   2021/06/18 18:35:34
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import random

import numpy as np
from matplotlib import pyplot as plt


def plot_hpatches_mAA(errors, methods, names, n_i=52, n_v=56):
    colors = [
        '#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        for i in range(len(methods))
    ]

    plt_lim = [1, len(methods)]
    plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

    plt.rc('axes', titlesize=25)
    plt.rc('axes', labelsize=25)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    ls = '-'
    for method, name, color in zip(methods, names, colors):
        i_err, v_err, _ = errors[method]
        plt.plot(
            plt_rng,
            [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng],
            color=color,
            ls=ls,
            linewidth=3,
            label=name,
        )
    plt.title('Overall')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylabel('MMA')
    plt.ylim([0, 1])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend()

    plt.subplot(1, 3, 2)
    for method, name, color in zip(methods, names, colors):
        i_err, v_err, _ = errors[method]
        plt.plot(
            plt_rng,
            [i_err[thr] / (n_i * 5) for thr in plt_rng],
            color=color,
            ls=ls,
            linewidth=3,
            label=name,
        )
    plt.title('Illumination')
    plt.xlabel('threshold [px]')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylim([0, 1])
    plt.gca().axes.set_yticklabels([])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.subplot(1, 3, 3)
    for method, name, color in zip(methods, names, colors):
        i_err, v_err, _ = errors[method]
        plt.plot(
            plt_rng,
            [v_err[thr] / (n_v * 5) for thr in plt_rng],
            color=color,
            ls=ls,
            linewidth=3,
            label=name,
        )
    plt.title('Viewpoint')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylim([0, 1])
    plt.gca().axes.set_yticklabels([])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.savefig('hseq.pdf', bbox_inches='tight', dpi=300)
