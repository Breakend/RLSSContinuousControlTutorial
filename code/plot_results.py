import argparse
import time
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
from numpy.random import choice


def multiple_plot(average_vals_list, std_dev_list, traj_list, other_labels, env_name, smoothing_window=5, no_show=False, ignore_std=False, limit=None, extra_lines=None):
    fig = plt.figure(figsize=(15, 10))
    colors = ["k", "red", "blue", "green", "magenta", "cyan", "brown", "purple"]
    color_index = 0
    ax = plt.subplot() # Defines ax variable by creating an empty plot

    # Set the tick labels font
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(22)

    index = 0
    for average_vals, std_dev, label, trajs in zip(average_vals_list, std_dev_list, other_labels[:len(average_vals_list)], traj_list):
        index += 1
        rewards_smoothed_1 = pd.Series(average_vals).rolling(smoothing_window, min_periods=smoothing_window).mean()[:limit]
        if limit is None:
            limit = len(rewards_smoothed_1)
        rewards_smoothed_1 = rewards_smoothed_1[:limit]
        std_dev = std_dev[:limit]

        fill_color = colors[color_index]#choice(colors, 1)
        color_index += 1
        cum_rwd_1, = plt.plot(range(len(rewards_smoothed_1)), rewards_smoothed_1, label=label, color=fill_color[0])
        if not ignore_std:
            plt.fill_between(range(len(rewards_smoothed_1)), rewards_smoothed_1 + std_dev,   rewards_smoothed_1 - std_dev, alpha=0.3, edgecolor=fill_color, facecolor=fill_color)

    if extra_lines:
        for lin in extra_lines:
            plt.plot(range(len(rewards_smoothed_1)), np.repeat(lin, len(rewards_smoothed_1)), linestyle='-.', color = colors[color_index], linewidth=2.5, label=other_labels[index])
            color_index += 1
            index += 1

    axis_font = {'fontname':'Arial', 'size':'28'}
    #plt.legend(loc='upper left', prop={'size' : 16})
    plt.legend(loc='lower right', prop={'size' : 16})
    plt.xlabel("Iterations", **axis_font)
    plt.ylabel("Average Return", **axis_font)
    plt.title("%s Environment"% env_name, **axis_font)

    if no_show:
        fig.savefig('%s.png' % env_name, dpi=fig.dpi)
    else:
        plt.show()

    return fig


parser = argparse.ArgumentParser()
parser.add_argument("paths_to_progress_csvs", nargs="+", help="All the csvs")
parser.add_argument("env_name")
parser.add_argument("--save", action="store_true")
parser.add_argument("--ignore_std", action="store_true")
parser.add_argument('--labels', nargs='+', help='List of labels to go along with the paths', required=False)
parser.add_argument('--smoothing_window', default=5, type=int)
parser.add_argument('--limit', default=None, type=int)
parser.add_argument('--extra_lines', nargs="+", type=float)

args = parser.parse_args()

avg_rets = []
std_dev_rets = []
trajs = []

for o in args.paths_to_progress_csvs:
    data = pd.read_csv(o)
    avg_ret = np.array(data["AverageReturn"])
    std_dev_ret = np.array(data["StdReturn"])
    if "NumTrajs" in data:
        trajs.append(np.cumsum(np.array(data["NumTrajs"])))
    else:
        trajs.append(np.cumsum(np.array([25]*len(data["AverageReturn"]))))
    avg_rets.append(avg_ret)
    std_dev_rets.append(std_dev_ret)

multiple_plot(avg_rets, std_dev_rets, trajs, args.labels, args.env_name, smoothing_window=args.smoothing_window, no_show=args.save, ignore_std=args.ignore_std, limit=args.limit, extra_lines=args.extra_lines)
