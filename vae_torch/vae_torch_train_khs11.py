import torch
import torch.optim as optim
import argparse
import torch.nn as nn
import vae_torch_model
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import datetime
from torch.utils.data import Dataset
import os
from PIL import Image

input_initial_resize = 80
input_size = 64
#hid_sizes = [2048, 1024, 512]
#feat_size = 128

chn_sizes = [3, 32, 32, 16, 16, 16]
kern_sizes = [5, 5, 5, 3, 3]
hid_sizes = [16*9*9, 512]
feat_size = 64


out_folder = "output_C18_is" + str(input_size) + "_hs" + str(hid_sizes[0]) + "_fs" + str(feat_size)
vae_f_name = 'vae_ft_C18_is' + str(input_size) + "_hs" + str(hid_sizes[0]) + '_fs' + str(feat_size) + '.npz'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

train_data_transform = transforms.Compose([
    transforms.Resize(input_initial_resize),
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
valid_data_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])

def parseDataset_khs(root_dir):
    images = []
    labels = []
    ids = []

    khsList = os.listdir(root_dir)
    im_id = 0

    for khs_id in range(0,len(khsList)):
        khsName = khsList[khs_id]
        khsFolder = os.path.join(root_dir, khsName)
        if os.path.isdir(khsFolder)==0:
            continue
        image_list = os.listdir(khsFolder)
        imCnt = len(image_list)

        for imID in range(0, imCnt):#range(0,len(class_list)):
            images.append(os.path.join(khsFolder, image_list[imID]))
            labels.append(khs_id)
            ids.append(im_id)
            im_id = im_id + 1

    return images, labels, ids

class khs_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.datasetname = "hospdev"

        images, labels, ids = parseDataset_khs(root_dir)

        self.images = images
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        ids = self.ids[idx]
        sample = {'image': image, 'label': label, 'id': ids}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def _len_(self):
        return len(self.labels)

X_tr = khs_dataset(root_dir="/home/doga/DataFolder/sup/data/data_te2_cv1_neuralNetHandImages_nos11_rs224_rs01/neuralNetHandImages_nos11_rs224_rs01_tr", transform=train_data_transform)
X_va = khs_dataset(root_dir="/home/doga/DataFolder/sup/data/data_te2_cv1_neuralNetHandImages_nos11_rs224_rs01/neuralNetHandImages_nos11_rs224_rs01_va", transform=valid_data_transform)
X_te = khs_dataset(root_dir="/home/doga/DataFolder/sup/data/data_te2_cv1_neuralNetHandImages_nos11_rs224_rs01/neuralNetHandImages_nos11_rs224_rs01_te", transform=valid_data_transform)


# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=500, type=int, help='number of epochs to train the VAE for')
args = vars(parser.parse_args())

# learning parameters
epochs = args['epochs']
batch_size = 64
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = vae_torch_model.ConvVAE_2(input_size=input_size, chn_sizes=chn_sizes, kern_sizes=kern_sizes, hid_sizes=hid_sizes, feat_size=feat_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def fit(model, X_data):
    model.train()
    running_loss = 0.0
    batch = [b['image'] for b in X_data]
    fr = 0
    while (fr+batch_size<len(X_tr)):
        to = fr+batch_size
        #print("fr=", fr, ", to=", to)
        data = torch.stack(batch[fr:to], dim=0)
        data = data.to(device)
        #data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        fr = to

    train_loss = running_loss/len(X_data)
    return train_loss

def validate(model, X_vate):
    model.eval()
    running_loss = 0.0
    batch = [b['image'] for b in X_vate]
    batch_lb = [b['label'] for b in X_vate]
    uqlb, unid = np.unique(batch_lb, return_index=True)
    data_al = [batch[i] for i in unid]
    data_cn = len(unid)

    with torch.no_grad():
        fr = 0
        while (fr + batch_size < len(X_vate)):
            to = fr + batch_size
            data = torch.stack(batch[fr:to], dim=0)
            data = data.to(device)
            #data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            fr = to

    with torch.no_grad():
        # save the last batch input and output of every epoch
        data = torch.stack(data_al, dim=0)
        data = data.to(device)
        #data = data.view(data.size(0), -1)
        reconstruction, mu, logvar = model(data)
        both = torch.cat((data.view(data_cn, 3, input_size, input_size)[:data_cn],
                          reconstruction.view(data_cn, 3, input_size, input_size)[:data_cn]))
        f_name = out_folder + "/output_{:03d}.png".format(epoch)
        save_image(both.cpu(), f_name, nrow=data_cn)
    val_loss = running_loss/len(X_vate)
    return val_loss

tr_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, X_tr)
    val_epoch_loss = validate(model, X_te)
    tr_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
    print("tr last 5", tr_loss[-5:])
    print("val last 5", val_loss[-5:])
    print(datetime.datetime.now().strftime("%H:%M:%S"))
np.savez(vae_f_name, tr_loss=tr_loss, val_loss=val_loss, allow_pickle=True)