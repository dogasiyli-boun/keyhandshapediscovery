import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
from helperFuncs import pad_array, setPandasDisplayOpts, sortVec
import imageio
import skimage
from cycler import cycler

def stack_fig_disp(result_mat, save_fold_name, save_fig_name, title_str, top_bot_lim=0.003):
    mins = np.nanmin(result_mat[:, :-1], axis=1).ravel()
    maxes = np.nanmax(result_mat[:, :-1], axis=1).ravel()
    means = np.nanmean(result_mat[:, :-1], axis=1).ravel()
    std = np.nanstd(result_mat[:, :-1], axis=1).ravel()

    usr_cnt = len(mins)
    x_ticks = np.linspace(0.0, 1.0, num=usr_cnt + 2)
    x_lim = (x_ticks[0], x_ticks[-1])
    x_ticks = x_ticks[1:usr_cnt+1]

    # create stacked errorbars:
    fig = plt.clf()
    plt.errorbar(x_ticks, means, std, fmt='oy', lw=8, zorder=0)
    plt.errorbar(x_ticks, means, [means - mins, maxes - means], fmt='.b', ecolor='green', lw=2, zorder=5)
    for u in range(usr_cnt):
        plt.scatter(x_ticks[u] * np.ones(result_mat[u, :].shape), result_mat[u, :], marker='*', c='m', lw=3, zorder=10)
    plt.xlabel("Test User IDs", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.xlim(x_lim)
    plt.ylim(np.nanmin([mins])-top_bot_lim, top_bot_lim+np.nanmax([maxes]))
    plt.xticks(x_ticks, ["u2", "u3", "u4", "u5", "u6", "u7"], rotation=20, fontsize=16)
    plt.yticks(fontsize=16)
    #plt.title(title_str, fontsize=20)
    plt.savefig(os.path.join(save_fold_name, save_fig_name), bbox_inches='tight')
    return fig

def pdf_bar_plot_users(result_mmm, save_fold_name, save_fig_name, title_str):
    ax = result_mmm.plot.bar(rot=90, color=['r', 'g', 'b'])
    ax.get_legend().set_bbox_to_anchor((1.22, 1.02))
    ax.set_title(title_str)
    fig = ax.get_figure()
    fig.savefig(os.path.join(save_fold_name, save_fig_name), bbox_inches='tight')

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

def plot_acc_range_for_user(usr_list, user_id, title_str):
    setPandasDisplayOpts()

    print("plotting title = ", title_str)
    user_id_str = "u" + str(user_id)
    acc_list = usr_list[user_id_str]

    arr_ep = np.min(pad_array(acc_list["ep"]), axis=0).squeeze()

    tr_arr = np.asarray(acc_list["tr"])
    tr_min = np.nanmin(pad_array(tr_arr), axis=0).squeeze()
    tr_max = np.nanmax(pad_array(tr_arr), axis=0).squeeze()
    # tr_mean = np.nanmean(pad_array(tr_arr), axis=0).squeeze()

    va_arr = np.asarray(acc_list["va"])
    va_min = np.nanmin(pad_array(va_arr), axis=0).squeeze()
    va_max = np.nanmax(pad_array(va_arr), axis=0).squeeze()
    # va_mean = np.nanmean(pad_array(va_arr), axis=0).squeeze()

    te_arr = pad_array(np.asarray(acc_list["te"]))
    te_min = np.nanmin(te_arr, axis=0).squeeze()
    te_max = np.nanmax(te_arr, axis=0).squeeze()
    te_mean = np.nanmean(te_arr, axis=0).squeeze()
    if te_arr.shape[0]>1:
        te_max_per_cv = np.nanmax(te_arr, axis=1)
        idx = np.unravel_index(np.argmax(te_max_per_cv), te_max_per_cv.shape)
        val = te_max_per_cv[idx]
        te_max_vec = te_arr[idx].squeeze().reshape(arr_ep.shape)
    else:
        te_max_vec = te_arr.squeeze().reshape(arr_ep.shape)

    fig, ax = plt.subplots(1, figsize=(15, 8), dpi=80)
    ax.set_title(title_str)

    if tr_arr.shape[0] > 1:
        ax.fill_between(arr_ep, tr_min, tr_max, facecolor='yellow', alpha=0.5, label='train-range', zorder=0)
    else:
        ax.plot(arr_ep, tr_arr.squeeze().reshape(arr_ep.shape), lw=2, label='train-single', color='yellow', ls='-', zorder=0)

    if va_arr.shape[0] > 1:
        ax.fill_between(arr_ep, va_min, va_max, facecolor='blue', alpha=0.5, label='validation-range', zorder=5)
    else:
        ax.plot(arr_ep, va_arr.squeeze().reshape(arr_ep.shape), lw=2, label='validation-single', color='blue', ls='-', zorder=5)

    ax.plot(arr_ep, te_mean, lw=2, label='test-mean', color='green', ls='--', zorder=15)
    if te_arr.shape[0] > 1:
        ax.fill_between(arr_ep, te_min, te_max, facecolor='green', alpha=0.5, label='test-range', zorder=10)
        ax.plot(arr_ep, te_max_vec, lw=4, label='test-best-peak', color='#FFA500', ls='-', zorder=13)

    ax.legend(loc='lower right')
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    ax.grid(which='minor', alpha=2)
    ax.grid(which='major', alpha=10)
    return fig

def get_khs_im(name, folder_list):
    im = []
    for fold in folder_list:
        fnamefull = os.path.join(fold, name + ".png")
        if os.path.isfile(fnamefull):
            # im = plt.imread(fnamefull).copy()
            im = imageio.imread(fnamefull)
            im = skimage.img_as_float(im)
            return im
    return im

#  https://stackoverflow.com/questions/44246650/automating-bar-charts-plotting-to-show-country-flags-as-ticklabels/44264051#44264051
def offset_image(coord, ax, img, zoom=0.1):
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax
    ab = AnnotationBbox(im, (coord, 0),  xybox=(0., -16.), frameon=False,
                        xycoords='data',  boxcoords="offset points", pad=0)

    ax.add_artist(ab)

def bar_cnt_plot(nos=11, zoom=0.1, step_xticks = 2, title_str="", fontsize_cnt=8):
    folder_list = ["/home/doga/Desktop/desktopShit/khsImages/bothHands", "/home/doga/Desktop/desktopShit/khsImages/singleHand"]
    labelnames_csv_filename = "/home/doga/DataFolder/sup/data/nos" + str(nos) + "_labels.csv"
    labelNames = list(pd.read_csv(labelnames_csv_filename, sep="*")['khsName'].values.flatten())
    labelCounts = list(pd.read_csv(labelnames_csv_filename, sep="*")['total'].values.flatten())
    labelCounts, idx = sortVec(np.asarray(labelCounts))
    labelNames = [labelNames[i] for i in idx[0]]

    labelPerc = labelCounts*100 / np.sum(labelCounts)
    print(labelNames)
    print(labelCounts)
    x_tick_int = np.arange(start=0, stop=step_xticks*len(labelNames), step=step_xticks)
    print(x_tick_int)

    fig, ax = plt.subplots(dpi=480)  # figsize=(step_xticks*len(labelNames)*4, 200)
    ax.bar(x_tick_int, labelCounts, width=step_xticks*.75, align="center")
    ax.set_xticks(x_tick_int)
    plt.xticks(rotation=90)
    ax.set_xticklabels(labelNames)
    plt.ylim(0, np.max(labelCounts)+500)

    label_color_loop = ['b', 'm', 'g', 'k']
    ax.tick_params(axis='x', which='major', pad=20)
    ax.set_title(title_str)

    for i, c in enumerate(labelNames):
        im = get_khs_im(labelNames[i], folder_list)
        if len(im) > 0:
            offset_image(i*step_xticks, ax, im, zoom)
        else:
            print("no image for  - {:s}".format(labelNames[i]))
        row_int = labelCounts[i]
        col_int = i*step_xticks
        color_cur = label_color_loop[np.mod(i, 4)]
        ax.text(y=row_int, x=i*step_xticks,
                s=str(row_int) + "(%{:.1f})".format(labelPerc[i]),
                va='bottom', ha='left', rotation=45,
                color=color_cur, fontsize=fontsize_cnt)
        ax.get_xticklabels()[i].set_color(color_cur)
        ax.get_xticklabels()[i].set_fontsize(fontsize_cnt)

    plt.show()
    fig.savefig("/home/doga/DataFolder/sup/data/khs_cnt_bar_" + str(nos) + ".png", bbox_inches='tight')
    plt.close('all')

def confusion_plot(saveConfFigFileName):


    if saveConfFigFileName != '':
        saveConfFigFileName = saveConfFigFileName.replace(".", "_ccm(" + confCalcMethod + ").")
        saveConfFigFileName = os.path.join(predictResultFold, saveConfFigFileName)

    #iterID = -1
    #normalizeByAxis = -1
    #add2XLabel = ''
    #add2YLabel = ''
    #funcH.plotConfMat(_confMat, labelNames, addCntXTicks=False, addCntYTicks=False, tickSize=10,
    #                  saveFileName=saveConfFigFileName, iterID=iterID,
    #                  normalizeByAxis=normalizeByAxis, add2XLabel=add2XLabel, add2YLabel=add2YLabel)
    fig, ax = funcH.plot_confusion_matrix(conf_mat=_confMat,
                                          colorbar=False,
                                          show_absolute=True,
                                          show_normed=True,
                                          class_names=labelNames,
                                          saveConfFigFileName=saveConfFigFileName,
                                          figMulCnt=figMulCnt,
                                          confusionTreshold=confusionTreshold)
