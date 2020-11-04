import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import os
from PIL import Image

def plot_vars(X, contain_str, ylabel,
              save_file_name=None,
              title_str="", plot_label_add_str="",
              line_color=None,
              create_plot=True, ax=None, figsize=(10, 3), dpi=80,
              legend_loc='upper right',
              max_epoch=None):
    if create_plot:
        fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
        ax.set_title(title_str)
    drw_color = line_color
    for par in X if isinstance(X, dict) else X.files:
        if par.__contains__(contain_str):
            if max_epoch is None:
                disp_epoch = len(X[par])
            else:
                disp_epoch = np.minimum(len(X[par]), max_epoch)
            plot_x = np.asarray(list(range(0, disp_epoch)))
            if line_color is None:
                drw_color = 0.5 + 0.5 * np.random.rand(1, 3).squeeze()
            ax.plot(plot_x, X[par][:disp_epoch], lw=2, label=plot_label_add_str + par, color=drw_color)

    ax.legend(loc=legend_loc)
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)
    ax.grid()
    if save_file_name is not None:
        plt.savefig(save_file_name)


def plot_compare(X_dict, title_str, contain_str, ylabel, figsize=(10, 3), dpi=80, max_epoch=None, legend_loc=None):
    fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
    ax.set_title(title_str)
    if legend_loc is None:
        legend_loc = 'upper right'
        if contain_str.__contains__('acc'):
            legend_loc = 'lower right'
    for key in X_dict:
        X = X_dict[key]
        plot_vars(X['X'], plot_label_add_str=X['file_name'], line_color=X['line_color'], contain_str=contain_str,
                  ylabel=ylabel, create_plot=False, ax=ax, legend_loc=legend_loc, max_epoch=max_epoch)

def plot_cf(cf_int, data_log_keys = ['tr_tr', 'tr_va', 'va', 'te'], k_loss_disp_list=None, max_act_ep=None, plot_cnt = 5,
            select_id_type='linspace',
            experiments_folder='/mnt/USB_HDD_1TB/GitHub/keyhandshapediscovery/',
            exp_base_name='output_sae_k256_is64_cf', ae_f_name_base='ae_ft_sae_k256_is64.npy'):
    ae_fold_name = os.path.join(experiments_folder, exp_base_name + str(cf_int).zfill(2))
    ae_f_name = os.path.join(ae_fold_name, ae_f_name_base)
    vfz = np.load(ae_f_name, allow_pickle=True)
    loss_log_dict = {}
    n = 0
    for k in data_log_keys:
        loss_log_dict[k] = vfz.item().get(k)
        n = len(loss_log_dict[k])
        print(k, " - log is loaded with len: ", n)
    loss_key_list = [key for key in loss_log_dict[data_log_keys[0]][0].keys()]

    for k_loss in loss_key_list:
        if k_loss_disp_list is not None and k_loss not in k_loss_disp_list:
            print('skipping - ', k_loss)
            continue
        print(k_loss, ':')
        for k_data in data_log_keys:
            if k_loss in loss_log_dict[k_data][0]:
                los_vec_cur = [loss_log_dict[k_data][l][k_loss] for l in range(0, len(loss_log_dict[k_data]))]
                disp_epoch = len(loss_log_dict[k_data])
                los_vec_cur = [loss_log_dict[k_data][l][k_loss] for l in range(0, disp_epoch)]
                plot_x = np.asarray(list(range(0, disp_epoch)))
                fig, ax = plt.subplots(1, figsize=(10, 3), dpi=80)
                ax.set_title('cf(' + str(cf_int).zfill(2) + '), ' + k_data + '--' + k_loss)
                ax.plot(plot_x, los_vec_cur[:disp_epoch], lw=2, label=k_loss, color='black')
    max_act_ep = np.minimum(max_act_ep,n-1) if max_act_ep is not None else n-1
    if select_id_type=='last':
        subplot_ids = np.asarray(range(max_act_ep-plot_cnt-1,max_act_ep-1)).astype(int)
        subplot_ids = subplot_ids[subplot_ids>0]
        print(max_act_ep, plot_cnt, subplot_ids)
    else:
        subplot_ids = (np.linspace(0, max_act_ep, num=plot_cnt)).astype(int)
    plot_cnt = int(len(subplot_ids))
    try :
        for i in range (0, plot_cnt):
            s = subplot_ids[i]
            try:
                f_name = "btl_" + str(s).zfill(3) +"_tr_va"
                ful_file_name = os.path.join(ae_fold_name, f_name + "_.png")
                print(str(i)+"*"+ful_file_name+"*")
                img = Image.open(ful_file_name)
            except:
                f_name = "btl_" + str(s).zfill(3) +"_tr_te"
                ful_file_name = os.path.join(ae_fold_name, f_name + "_.png")
                print(str(i)+"*"+ful_file_name+"*")
                img = Image.open(ful_file_name)

            if i == 0:
                fig, ax = plt.subplots(1, plot_cnt, figsize=[30, 30], dpi=300)
            ax[i].set_title(f_name + ' activation')
            ax[i].imshow(img)
        plt.show()
    except Exception as e:
        print(e)

def plot_cf_compare(cf_int_arr, data_log_keys=['tr_va', 'va', 'te'],
                    mul_before_plot=None,
                    loss_key='reconstruction', title_add_front_str='',
                    max_act_ep=None,
                    legend_loc='upper right',
                    experiments_folder='/mnt/USB_HDD_1TB/GitHub/keyhandshapediscovery/',
                    exp_base_name='output_sae_k256_is64_cf', ae_f_name_base='ae_ft_sae_k256_is64.npy',
                    save_to_fold=None, save_as_png_name=None,
                    add_min_to_label=False, add_max_to_label=False,
                    z_fill_int=2):
    fig, ax = plt.subplots(1, figsize=(10, 5), dpi=300)
    ax.set_title(title_add_front_str + loss_key)

    if mul_before_plot is None or np.shape(cf_int_arr) != np.shape(mul_before_plot):
        mul_before_plot = np.ones(cf_int_arr.shape, dtype=float)
    else:
        mul_before_plot = np.asarray(mul_before_plot, dtype=float)
    cf_int_arr = np.asarray(cf_int_arr, dtype=int)

    tab_id = 10
    cmap = get_cmap('tab'+str(tab_id))
    color_id = 0
    for i in range(0, len(cf_int_arr)):
        cf_int = cf_int_arr[i]
        mul_plt = mul_before_plot[i]
        ae_fold_name = os.path.join(experiments_folder, exp_base_name + str(cf_int).zfill(z_fill_int))
        ae_f_name = os.path.join(ae_fold_name, ae_f_name_base)
        if not os.path.exists(ae_f_name):
            continue
        vfz = np.load(ae_f_name, allow_pickle=True)
        loss_log_dict = {}
        for k in data_log_keys:
            loss_log_dict[k] = vfz.item().get(k)
            if loss_log_dict[k] is None:
                print("cf("+str(cf_int)+") --> loss_log_dict"+str(k)+"] is none")
                continue
            n = len(loss_log_dict[k])
            print(str(cf_int), ', ', k, " - log is loaded with len: ", n)
        try:
            loss_key_exist_list = [key for key in loss_log_dict[data_log_keys[0]][0].keys()]
        except:
            continue

        if loss_key not in loss_key_exist_list:
            continue

        for k_data in data_log_keys:
            if loss_key in loss_log_dict[k_data][0]:
                vec_len = len(loss_log_dict[k_data])
                disp_epoch = np.min([vec_len, max_act_ep]) if max_act_ep is not None else vec_len
                los_vec_cur = [loss_log_dict[k_data][l][loss_key] for l in range(0, disp_epoch)]
                plot_x = np.asarray(list(range(0, disp_epoch)))
                label_str = str(cf_int) + '_' + k_data + '_' + loss_key + ('*{}'.format(mul_plt) if mul_plt != 1 else ' ')
                print(label_str, los_vec_cur[-3:], "\nmax({:4.2f}),min({:4.2f})".format(np.max(los_vec_cur), np.min(los_vec_cur)))
                if add_min_to_label:
                    label_str += "_min({:4.2f})".format(np.min(los_vec_cur[-int(vec_len/2):]))
                if add_max_to_label:
                    label_str += "_max({:4.2f}/{:4.2f})".format(np.max(los_vec_cur[:int(vec_len/2)]), np.max(los_vec_cur[-int(vec_len/2):]))
                color_float = (color_id+0.5)/tab_id
                rgba = cmap(color_float)
                print("color_id:", color_id, ", color_float:", color_float, ", rgba:", rgba)
                color_id = (color_id + 1) % tab_id

                ax.plot(plot_x, mul_plt*np.asarray(los_vec_cur[:disp_epoch]), lw=2, label=label_str, color=rgba)

    ax.legend(loc=legend_loc)
    if save_to_fold is not None and os.path.exists(save_to_fold):
        if save_as_png_name is None:
            save_as_png_name = title_add_front_str + loss_key + '.png'
        png_save_name = os.path.join(save_to_fold, save_as_png_name)
        fig.savefig(png_save_name, bbox_inches='tight', dpi=300)

def plot_cf_compare_list(cf_int_arr, data_log_keys, loss_key_list, title_add_front_str,
                         experiments_folder, exp_base_name, ae_f_name_base,
                         save_to_fold=None, max_act_ep=None):
    if save_to_fold is not None and not os.path.exists(save_to_fold):
        os.makedirs(save_to_fold)
    for lk in loss_key_list:
        for dlk in data_log_keys:
            add_min_to_label = not lk.__contains__('bottleneck')
            add_max_to_label = lk.__contains__('bottleneck')
            legend_loc = 'lower right' if add_max_to_label else 'upper right'
            plot_cf_compare(cf_int_arr=cf_int_arr,
                            data_log_keys=[dlk],
                            loss_key=lk,
                            title_add_front_str=title_add_front_str + dlk + '-',
                            max_act_ep=max_act_ep, legend_loc=legend_loc,
                            experiments_folder=experiments_folder,
                            exp_base_name=exp_base_name, ae_f_name_base=ae_f_name_base,
                            add_min_to_label=add_min_to_label, add_max_to_label=add_max_to_label,
                            save_to_fold=save_to_fold)

def get_hid_state_vec(hidStateID):
    hid_state_cnt_vec = [2048, 1024, 1024, 512, 512, 256, 256]
    if hidStateID == 1:
        hid_state_cnt_vec = [256, 256]
    elif hidStateID == 2:
        hid_state_cnt_vec = [512, 512]
    elif hidStateID == 3:
        hid_state_cnt_vec = [512, 512, 256, 256]
    elif hidStateID == 4:
        hid_state_cnt_vec = [1024, 512, 512, 256]
    elif hidStateID == 5:
        hid_state_cnt_vec = [64, 64, 64, 64]
    elif hidStateID == 6:
        hid_state_cnt_vec = [128, 128, 128, 128]
    elif hidStateID == 7:
        hid_state_cnt_vec = [256, 256, 256, 256]
    elif hidStateID == 8:
        hid_state_cnt_vec = [512, 512, 512, 512]
    elif hidStateID == 9:
        hid_state_cnt_vec = [128, 128]
    return hid_state_cnt_vec

def create_hidstate_dict(hid_state_cnt_vec, init_mode_vec = None, act_vec=None, verbose=0):
    hid_state_cnt = len(hid_state_cnt_vec)
    hidStatesDict = {}
    for i in range(hid_state_cnt):
        hid_state_id_str = str(i+1).zfill(2)
        hid_state_name = "hidStateDict_" + hid_state_id_str
        dim_str = str(hid_state_cnt_vec[i])
        dim_int = int(hid_state_cnt_vec[i])
        try:#if init_mode_vec is not None and len(init_mode_vec)>=i:
            initMode = init_mode_vec[i]
        except:#else:
            initMode = "kaiming_uniform_"
        try:#if act_vec is not None and len(act_vec)>=i:
            actStr = act_vec[i]
        except:#else:
            actStr = "relu"
        if verbose>0:
            print(hid_state_name, ' = {"dimOut": "', dim_str, '", "initMode": "', initMode ,'", "act": "', actStr,'"}')
        hidStatesDict[hid_state_id_str] = {"dimOut": dim_int, "initMode": initMode, "act": actStr}
    return hidStatesDict