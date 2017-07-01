import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from itertools import cycle

from numpy import genfromtxt




def single_plot(average_vals, std_dev, env_name, smoothing_window=5, no_show=False, ignore_std=False):
    fig = plt.figure(figsize=(15, 10))
    rewards_smoothed_1 = pd.Series(average_vals).rolling(smoothing_window, min_periods=smoothing_window).mean()#[:200]

    cum_rwd_1, = plt.plot(range(len(rewards_smoothed_1)), rewards_smoothed_1, label="Unified On-Policy and Off-Policy DDPG")
    if not ignore_std:
        plt.fill_between(range(len(rewards_smoothed_1)), rewards_smoothed_1 + std_dev,   rewards_smoothed_1 - std_dev, alpha=0.3, edgecolor='blue', facecolor='blue')

    plt.legend(handles=[cum_rwd_1])
    plt.xlabel("Epsiode")
    plt.ylabel("Average Return")
    plt.title("%s Environment"% env_name)

    if no_show:
        fig.savefig('%s.png' % env_name, dpi=fig.dpi)
    else:
        plt.show()

    return fig


def multiple_plot(average_vals_list, std_dev_list, other_labels, env_name, smoothing_window=5, no_show=False, ignore_std=False, limit=None):
    fig = plt.figure(figsize=(15, 10))
    colors = cycle(["aqua", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon", "navy", "olive", "purple", "red", "silver", "teal", "yellow"])

    for average_vals, std_dev, label in zip(average_vals_list, std_dev_list, other_labels):
        if limit is None:
            limit = len(rewards_smoothed_1)
        rewards_smoothed_1 = pd.Series(average_vals).rolling(smoothing_window, min_periods=smoothing_window).mean()[:limit]

        cum_rwd_1, = plt.plot(range(len(rewards_smoothed_1)), rewards_smoothed_1, label=label)[:limit]
        if not ignore_std:
            plt.fill_between(range(len(rewards_smoothed_1)), rewards_smoothed_1 + std_dev,   rewards_smoothed_1 - std_dev, alpha=0.3, edgecolor='blue', facecolor=next(colors))

    plt.legend(loc='lower right')
    plt.xlabel("Epsiode")
    plt.ylabel("Average Return")
    plt.title("%s Environment"% env_name)

    if no_show:
        fig.savefig('%s.png' % env_name, dpi=fig.dpi)
    else:
        plt.show()

    return fig

# def multipe_plot(stats1, stats2, smoothing_window=50, noshow=False):
#
#     fig = plt.figure(figsize=(30, 20))
#     rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
#
#     rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
#
#     cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="DDPG")
#     plt.fill_between( eps, rewards_smoothed_1 + ddpg_walker_std_return,   rewards_smoothed_1 - ddpg_walker_std_return, alpha=0.3, edgecolor='blue', facecolor='blue')
#
#     cum_rwd_2, = plt.plot(eps2, rewards_smoothed_2, label="Unified DDPG")
#     plt.fill_between( eps2, rewards_smoothed_2 + unified_ddpg_walker_std_return,   rewards_smoothed_2 - unified_ddpg_walker_std_return, alpha=0.3, edgecolor='blue', facecolor='red')
#
#     plt.legend(handles=[cum_rwd_1, cum_rwd_2])
#     plt.xlabel("Epsiode")
#     plt.ylabel("Average Return")
#     plt.title("Walker Environment")
#
#     plt.show()
#
#     return fig






import argparse
parser = argparse.ArgumentParser()
# example python plot_results.py data/progress.csv "CartPole" --other_paths not_gated/progress.csv --labels "Unified On-Policy and Off-Policy DDPG - Random Sigma" "Unified On-Policy and Off-Policy DDPG - Learned Gated Sigma" --save
parser.add_argument("path_to_progress_csv")
parser.add_argument("env_name")
parser.add_argument("--save", action="store_true")
parser.add_argument("--ignore_std", action="store_true")
parser.add_argument('--other_paths', nargs='+', help='List of other progress csv files', required=False)
parser.add_argument('--labels', nargs='+', help='List of labels to go along with the paths', required=False)
parser.add_argument('--smoothing_window', default=5, type=int)
parser.add_argument('--limit', default=None, type=int)

args = parser.parse_args()

data = pd.read_csv(args.path_to_progress_csv)

avg_ret = np.array(data["AverageReturn"])
std_dev_ret = np.array(data["StdReturn"])

# "Unified On-Policy and Off-Policy DDPG"

if args.other_paths:
    avg_rets = [avg_ret]
    std_dev_rets = [std_dev_ret]
    for o in args.other_paths:
        data = pd.read_csv(o)
        avg_ret = np.array(data["AverageReturn"])
        std_dev_ret = np.array(data["StdReturn"])
        avg_rets.append(avg_ret)
        std_dev_rets.append(std_dev_ret)
    multiple_plot(avg_rets, std_dev_rets, args.labels, args.env_name, smoothing_window=args.smoothing_window, no_show=args.save, ignore_std=args.ignore_std, limit=args.limit)
else:
    single_plot(avg_ret, std_dev_ret, args.env_name, no_show=args.save, smoothing_window=args.smoothing_window, ignore_std=args.ignore_std)
