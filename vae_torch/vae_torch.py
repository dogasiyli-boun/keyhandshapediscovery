import torch
import vae_torch_model
from data_classes import khs_dataset
import numpy as np
import datetime
import os
import sys

def getFileList(dir2Search, startString="", endString="", sortList=False):
    fileList = [f for f in os.listdir(dir2Search) if f.startswith(startString) and
                                                     f.endswith(endString) and
                                                    os.path.isfile(os.path.join(dir2Search, f))]
    if sortList:
        fileList = np.sort(fileList)
    return fileList

def load_data(input_size, input_initial_resize, data_main_fold = "/home/doga/DataFolder/sup/data"):
    #"/media/doga/SSD258/DataPath/sup/data" #/home/doga/DataFolder/sup/data
    data_folder = os.path.join(data_main_fold, "data_XX_")
    X_tr = khs_dataset(root_dir=data_folder.replace("_XX_", "_tr"), is_train=True, input_size=input_size, input_initial_resize=input_initial_resize)
    X_va = khs_dataset(root_dir=data_folder.replace("_XX_", "_va"), is_train=False, input_size=input_size)
    X_te = khs_dataset(root_dir=data_folder.replace("_XX_", "_te"), is_train=False, input_size=input_size)
    return X_tr, X_va, X_te

def load_model_vars(model_name):
    if model_name == "ConvVAE_2":
        chn_sizes = [3, 32, 32, 16, 16, 16]
        kern_sizes = [5, 5, 5, 3, 3]
        hid_sizes = [16 * 9 * 9, 512]
    elif model_name == "ConvVAE":
        chn_sizes = [3, 32, 32, 16]
        kern_sizes = [5, 5, 5]
        hid_sizes = [16 * 24 * 24, 512]
    return chn_sizes, kern_sizes, hid_sizes

def get_model(updatedModelFile, model_name, input_size, chn_sizes, kern_sizes, hid_sizes, feat_size):
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(torch_device)
    print("Torch is in ", device, " mode.")
    if os.path.exists(updatedModelFile):
        _model_vae = torch.load(f=updatedModelFile)
    elif model_name == "ConvVAE":
        _model_vae = vae_torch_model.ConvVAE(input_size=input_size, chn_sizes=chn_sizes, kern_sizes=kern_sizes,
                                               hid_sizes=hid_sizes, feat_size=feat_size).to(device)
    elif model_name == "ConvVAE_2":
        _model_vae = vae_torch_model.ConvVAE_2(input_size=input_size, chn_sizes=chn_sizes, kern_sizes=kern_sizes,
                                               hid_sizes=hid_sizes, feat_size=feat_size).to(device)

    return _model_vae

def main(epochs = 20):
    #data related variables
    input_initial_resize = 80
    input_size = 64
    X_tr, X_va, X_te = load_data(input_size, input_initial_resize, data_main_fold="/home/doga/DataFolder/sup/data/conv_data")

    #learner related variables

    batch_size = 64
    #model related variables
    model_name = "ConvVAE"
    chn_sizes, kern_sizes, hid_sizes = load_model_vars(model_name)

    feat_size = 64

    #experiment related variables
    base_str = model_name + "_is" + str(input_size) + "_hs" + str(hid_sizes[0]) + "_fs" + str(feat_size)
    out_folder = 'output_' + base_str
    vae_f_name = os.path.join(out_folder, 'vae_ft_' + base_str + '.npz')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    updatedModelFile = os.path.join(out_folder, "model_" + model_name + "_is" + str(input_size) + "_hs" + str(hid_sizes[0]) + "_fs" + str(feat_size) + '.model')

    _model_vae = get_model(updatedModelFile, model_name, input_size, chn_sizes, kern_sizes, hid_sizes, feat_size)

    epoch_out_img_list = getFileList(out_folder, startString="output_val", endString=".png", sortList=False)
    ep_fr = len(epoch_out_img_list)

    if os.path.exists(vae_f_name):
        vfz = np.load(vae_f_name)
        tr_loss = list(vfz['tr_loss'])
        val_loss = list(vfz['val_loss'])
        tes_loss = list(vfz['tes_loss'])
        print("tr_loss is loaded with len", len(tr_loss))
        print("val_loss is loaded with len", len(val_loss))
        print("tes_loss is loaded with len", len(tes_loss))
    else:
        tr_loss = []
        val_loss = []
        tes_loss = []

    for epoch in range(ep_fr, epochs):
        print(f"Epoch {epoch+1} of {epochs}")

        train_epoch_loss = _model_vae.fit(X_tr, batch_size)
        val_epoch_loss = _model_vae.validate(X_va, epoch, batch_size, out_folder, out_name_add_str="val_")
        tes_epoch_loss = _model_vae.validate(X_te, epoch, batch_size, out_folder, out_name_add_str="tes_")

        tr_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        tes_loss.append(tes_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Tes Loss: {tes_epoch_loss:.4f}")
        print("tr last 5", tr_loss[-5:])
        print("val last 5", val_loss[-5:])
        print("tes last 5", tes_loss[-5:])

        print(datetime.datetime.now().strftime("%H:%M:%S"))

        torch.save(_model_vae, f=updatedModelFile)
        np.savez(vae_f_name, tr_loss=tr_loss, val_loss=val_loss, tes_loss=tes_loss)

if __name__ == '__main__':
    main(epochs=20)