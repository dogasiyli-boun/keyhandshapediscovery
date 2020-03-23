import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from helperFuncs import pad_array, setPandasDisplayOpts

def stack_fig_disp(result_mat, save_fold_name):
    mins = np.nanmin(result_mat[:, :-1], axis=1).ravel()
    maxes = np.nanmax(result_mat[:, :-1], axis=1).ravel()
    means = np.nanmean(result_mat[:, :-1], axis=1).ravel()
    std = np.nanstd(result_mat[:, :-1], axis=1).ravel()

    # create stacked errorbars:
    fig = plt.clf()
    plt.errorbar(np.arange(len(mins)), means, std, fmt='ok', lw=4)
    plt.errorbar(np.arange(len(mins)), means, [means - mins, maxes - means], fmt='.r', ecolor='green', lw=1)
    plt.xlabel("userIDs")
    plt.ylabel("accuracy")
    plt.xlim(-1, len(mins))
    plt.ylim(np.min([mins]), np.max([maxes]))
    plt.xticks(np.arange(0, len(mins)), ["u2", "u3", "u4", "u5", "u6", "u7"], rotation=20)
    plt.savefig(os.path.join(save_fold_name, "errBar.png"), bbox_inches='tight')
    return fig

def pdf_bar_plot_users(result_mmm, save_fold_name):
    ax = result_mmm.plot.bar(rot=90, color=['r', 'g', 'b'])
    ax.get_legend().set_bbox_to_anchor((1.22, 1.02))
    fig = ax.get_figure()
    fig.savefig(os.path.join(save_fold_name, "triBar.png"), bbox_inches='tight')

def plot_acc_eval(acc_list, key_list, title_str):
    setPandasDisplayOpts()
    arr_ep = np.min(pad_array(acc_list["ep"]), axis=0).squeeze()
    arr_key = np.asarray(acc_list[key_list])
    arr_min = np.nanmin(pad_array(arr_key), axis=0).squeeze()
    arr_max = np.nanmax(pad_array(arr_key), axis=0).squeeze()
    arr_mean = np.nanmean(pad_array(arr_key), axis=0).squeeze()
    fig, ax = plt.subplots(1, figsize=(15, 8), dpi=80)
    ax.set_title(title_str)
    ax.plot(arr_ep, arr_min, lw=2, label='min accuracy', color='red')
    ax.plot(arr_ep, arr_mean, lw=2, label='mean accuracy', color='green', ls='--')
    ax.plot(arr_ep, arr_max, lw=2, label='max accuracy', color='blue')
    ax.fill_between(arr_ep, arr_min, arr_max, facecolor='yellow', alpha=0.5, label='min-max range')
    ax.legend(loc='lower right')
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    ax.grid()
    return fig