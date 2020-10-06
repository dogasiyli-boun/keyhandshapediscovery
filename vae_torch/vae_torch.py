import torch
import vae_torch_model
from data_classes import khs_dataset
import numpy as np
import datetime
import os

def getFileList(dir2Search, startString="", endString="", sortList=False):
    fileList = [f for f in os.listdir(dir2Search) if f.startswith(startString) and
                                                     f.endswith(endString) and
                                                    os.path.isfile(os.path.join(dir2Search, f))]
    if sortList:
        fileList = np.sort(fileList)
    return fileList

def main():
    input_initial_resize = 80
    input_size = 64
    epochs = 500
    batch_size = 64
    chn_sizes = [3, 32, 32, 16, 16, 16]
    kern_sizes = [5, 5, 5, 3, 3]
    hid_sizes = [16 * 9 * 9, 512]
    feat_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_folder = "/home/doga/DataFolder/sup/data/data_te2_cv1_neuralNetHandImages_nos11_rs224_rs01/neuralNetHandImages_nos11_rs224_rs01_XX_"
    X_tr = khs_dataset(root_dir=data_folder.replace("_XX_", "_tr"), is_train=True, input_size=input_size, input_initial_resize=input_initial_resize)
    # X_va = khs_dataset(root_dir=data_folder.replace("_XX_", "_va"), is_train=False, input_size=input_size)
    X_te = khs_dataset(root_dir=data_folder.replace("_XX_", "_te"), is_train=False, input_size=input_size)

    out_folder = "output_C18_is" + str(input_size) + "_hs" + str(hid_sizes[0]) + "_fs" + str(feat_size)
    vae_f_name = 'vae_ft_C18_is' + str(input_size) + "_hs" + str(hid_sizes[0]) + '_fs' + str(feat_size) + '.npz'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    updatedModelFile = os.path.join(out_folder, "model_C18_is" + str(input_size) + "_hs" + str(hid_sizes[0]) + "_fs" + str(feat_size) + '.model')

    if os.path.exists(updatedModelFile):
        _model_vae = torch.load(f=updatedModelFile)
    else:
        _model_vae = vae_torch_model.ConvVAE_2(input_size=input_size, chn_sizes=chn_sizes, kern_sizes=kern_sizes,
                                               hid_sizes=hid_sizes, feat_size=feat_size).to(device)

    epoch_out_img_list = getFileList(out_folder, startString="output_", endString=".png", sortList=False)
    ep_fr = len(epoch_out_img_list)

    tr_loss = []
    val_loss = []
    for epoch in range(ep_fr, epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = _model_vae.fit(X_tr, batch_size)
        val_epoch_loss = _model_vae.validate(X_te, epoch, batch_size, out_folder)
        tr_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
        print("tr last 5", tr_loss[-5:])
        print("val last 5", val_loss[-5:])
        print(datetime.datetime.now().strftime("%H:%M:%S"))
        torch.save(_model_vae, f=updatedModelFile)
    np.savez(vae_f_name, tr_loss=tr_loss, val_loss=val_loss, allow_pickle=True)