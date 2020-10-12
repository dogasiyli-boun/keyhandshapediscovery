import matplotlib.pyplot as plt
import numpy as np

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