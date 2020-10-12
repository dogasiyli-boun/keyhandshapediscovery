#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from numpy import unique as np_unique
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score

# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self, input_size, hid_sizes, feat_size):
        super(LinearVAE, self).__init__()

        self.in_linear_size = input_size*input_size
        self.input_size = input_size
        self.hid_sizes = hid_sizes
        self.feat_size = feat_size

        # encoder
        self.enc1 = nn.Linear(in_features=self.in_linear_size, out_features=hid_sizes[0])
        self.enc2 = nn.Linear(in_features=hid_sizes[0], out_features=hid_sizes[1])
        self.enc3 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[2])
        self.enc4 = nn.Linear(in_features=hid_sizes[2], out_features=feat_size*2)

        # decoder
        self.dec1 = nn.Linear(in_features=feat_size, out_features=hid_sizes[2])
        self.dec2 = nn.Linear(in_features=hid_sizes[2], out_features=hid_sizes[1])
        self.dec3 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[0])
        self.dec4 = nn.Linear(in_features=hid_sizes[0], out_features=self.in_linear_size)

        print("input_size=", input_size)
        print("in_linear_size=", self.in_linear_size)
        print("hid_sizes=", hid_sizes)
        print("feat_size=", feat_size)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = self.enc4(x).view(-1, 2, self.feat_size)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction, mu, log_var

# define a Convolutional VAE
class ConvVAE(nn.Module):
    def __init__(self, input_size, chn_sizes, kern_sizes, hid_sizes, feat_size):
        super(ConvVAE, self).__init__()

        self.input_size = input_size  # 64
        self.chn_sizes = chn_sizes  # chn_sizes = [3, 32, 32, 16]
        self.kern_sizes = kern_sizes  # kern_sizes = [5, 5, 5]
        self.hid_sizes = hid_sizes  # hid_sizes = [9216(16*24*24), 512]
        self.feat_size = feat_size  # feat_size = 64
        self.bottle_neck_image_size = None

        # encoder
        self.L0_conv1 = nn.Conv2d(in_channels=chn_sizes[0], out_channels=chn_sizes[1], kernel_size=kern_sizes[0], stride=1, padding=0)
        self.L1_conv2 = nn.Conv2d(in_channels=chn_sizes[1], out_channels=chn_sizes[2], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L2_maxp1 = nn.MaxPool2d(kernel_size=2)
        self.L3_conv3 = nn.Conv2d(in_channels=chn_sizes[2], out_channels=chn_sizes[3], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L4_lenc1 = nn.Linear(in_features=hid_sizes[0], out_features=hid_sizes[1])
        self.L5_lenc2 = nn.Linear(in_features=hid_sizes[1], out_features=feat_size*2)

        # decoder
        self.L6_ldec2 = nn.Linear(in_features=feat_size, out_features=hid_sizes[1])
        self.L7_ldec1 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[0])
        self.L8_dcnv3 = nn.ConvTranspose2d(in_channels=chn_sizes[3], out_channels=chn_sizes[2], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L9_upsm1 = nn.Upsample(scale_factor=2)
        self.L10_dcnv2 = nn.ConvTranspose2d(in_channels=chn_sizes[2], out_channels=chn_sizes[1], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L11_dcnv1 = nn.ConvTranspose2d(in_channels=chn_sizes[1], out_channels=chn_sizes[0], kernel_size=kern_sizes[0], stride=1, padding=0)

        print("input_size=", self.input_size)
        print("chn_sizes=", self.chn_sizes)
        print("kern_sizes=", self.kern_sizes)
        print("hid_sizes=", self.hid_sizes)
        print("feat_size=", self.feat_size)

        lr = 0.0001
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss(reduction='sum')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def enc(self, x):
        # encoding
        x = F.relu(self.L0_conv1(x))
        x = F.relu(self.L1_conv2(x))
        x = self.L2_maxp1(x)
        x = F.relu(self.L3_conv3(x))
        if self.bottle_neck_image_size is None:
            self.bottle_neck_image_size = [x.size(1), x.size(2), x.size(3)]
        x = x.view(x.size(0), -1)
        x = F.relu(self.L4_lenc1(x))
        x = self.L5_lenc2(x)
        return x

    def dec(self, z):
        # decoding
        x = F.relu(self.L6_ldec2(z))
        x = F.relu(self.L7_ldec1(x))
        x = x.view(x.size(0), self.bottle_neck_image_size[0], self.bottle_neck_image_size[1], self.bottle_neck_image_size[2])
        x = F.relu(self.L8_dcnv3(x))
        x = self.L9_upsm1(x)
        x = F.relu(self.L10_dcnv2(x))
        reconstruction = torch.sigmoid(self.L11_dcnv1(x))
        return reconstruction

    def forward(self, x):
        x = self.enc(x)

        x = x.view(-1, 2, self.feat_size)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        reconstruction = self.dec(z)

        return reconstruction, mu, log_var

    def final_loss(self, bce_loss, mu, logvar):
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

    def fit(self, X_data, batch_size):
        self.train()
        running_loss = 0.0
        dloader = DataLoader(X_data, batch_size=batch_size, shuffle=True)
        #lab_vec = []
        for b in dloader:
            data = b['image']
            #labels = b['label']
            #lab_vec.append(labels)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self.forward(data)
            bce_loss = self.criterion(reconstruction, data)
            loss = self.final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        train_loss = running_loss/len(X_data)
        return train_loss

    def validate(self, X_vate, epoch, batch_size, out_folder, out_name_add_str=""):
        self.eval()
        running_loss = 0.0
        batch = [b['image'] for b in X_vate]
        batch_lb = [b['label'] for b in X_vate]
        uqlb, unid = np_unique(batch_lb, return_index=True)
        data_al = [batch[i] for i in unid]
        data_cn = len(unid)

        with torch.no_grad():
            fr = 0
            while (fr + batch_size < len(X_vate)):
                to = fr + batch_size
                data = torch.stack(batch[fr:to], dim=0)
                data = data.to(self.device)
                reconstruction, mu, logvar = self.forward(data)
                bce_loss = self.criterion(reconstruction, data)
                loss = self.final_loss(bce_loss, mu, logvar)
                running_loss += loss.item()
                fr = to

        with torch.no_grad():
            # save the last batch input and output of every epoch
            data = torch.stack(data_al, dim=0)
            data = data.to(self.device)
            #data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = self.forward(data)
            both = torch.cat((data.view(data_cn, 3, self.input_size, self.input_size)[:data_cn],
                              reconstruction.view(data_cn, 3, self.input_size, self.input_size)[:data_cn]))
            f_name = out_folder + "/output_" + out_name_add_str + "{:03d}.png".format(epoch)
            save_image(both.cpu(), f_name, nrow=data_cn)
        val_loss = running_loss/len(X_vate)
        return val_loss

    @staticmethod
    def feat_extract_ext(model, X_vate, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(model, str):
            model = torch.load(model, map_location=device)
        model.eval()
        dloader = DataLoader(X_vate, batch_size=batch_size, shuffle=False)

        mu_vec = []
        x_vec = []
        lab_vec = []
        with torch.no_grad():
            for b in dloader:
                data = b['image']
                labels = b['label']
                lab_vec.append(labels)

                x = data.to(device)
                # encode
                x = F.relu(model.L0_conv1(x))
                x = F.relu(model.L1_conv2(x))
                x = model.L2_maxp1(x)
                x = F.relu(model.L3_conv3(x))
                x = x.view(x.size(0), -1)
                x = F.relu(model.L4_lenc1(x))
                x = model.L5_lenc2(x)
                x_vec.append(x)

                x = x.view(-1, 2, model.feat_size)
                mu = x[:, 0, :]  # the first feature values as mean
                mu_vec.append(mu)

        mu_vec = np.asarray(torch.cat(mu_vec).to(torch.device('cpu')))
        x_vec = np.asarray(torch.cat(x_vec).to(torch.device('cpu')))
        lab_vec = np.asarray(torch.cat(lab_vec).to(torch.device('cpu')))
        #np.savez('?_data.npz', mu_vec=mu_vec, x_vec=x_vec, labsTr=batch_lb)
        return mu_vec, x_vec, lab_vec

    def feat_extract(self, X_vate, batch_size):
        return self.feat_extract_ext(self, X_vate, batch_size)

class ConvVAE_2(nn.Module):
    def __init__(self, input_size, chn_sizes, kern_sizes, hid_sizes, feat_size, droput_val=None):
        super(ConvVAE_2, self).__init__()

        self.input_size = input_size  # 64            0   1   2   3   4   5
        self.chn_sizes = chn_sizes  # chn_sizes =    [3, 32, 32, 16, 16, 16]
        self.kern_sizes = kern_sizes  # kern_sizes = [5,  5,  5,  3,  3]
        self.hid_sizes = hid_sizes  # hid_sizes =    [1600, 256]
        self.feat_size = feat_size  # feat_size = 64

        self.DropLayer = nn.Dropout(p=droput_val) if droput_val is not None else None

        # encoder
        self.L0_conv1 = nn.Conv2d(in_channels=chn_sizes[0], out_channels=chn_sizes[1], kernel_size=kern_sizes[0], stride=1, padding=0)
        self.L1_conv2 = nn.Conv2d(in_channels=chn_sizes[1], out_channels=chn_sizes[2], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L2_maxp1 = nn.MaxPool2d(kernel_size=2)
        self.L3_conv3 = nn.Conv2d(in_channels=chn_sizes[2], out_channels=chn_sizes[3], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L4_conv4 = nn.Conv2d(in_channels=chn_sizes[3], out_channels=chn_sizes[4], kernel_size=kern_sizes[3], stride=1, padding=0)
        self.L5_maxp2 = nn.MaxPool2d(kernel_size=2)
        self.L6_conv5 = nn.Conv2d(in_channels=chn_sizes[4], out_channels=chn_sizes[5], kernel_size=kern_sizes[4], stride=1, padding=0)
        self.L7_lenc1 = nn.Linear(in_features=hid_sizes[0], out_features=hid_sizes[1])
        self.L8_lenc2 = nn.Linear(in_features=hid_sizes[1], out_features=feat_size*2)

        # decoder
        self.L8_ldec2 = nn.Linear(in_features=feat_size, out_features=hid_sizes[1])
        self.L7_ldec1 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[0])
        self.L6_dcnv5 = nn.ConvTranspose2d(in_channels=chn_sizes[5], out_channels=chn_sizes[4], kernel_size=kern_sizes[4], stride=1, padding=0)
        self.L5_upsm1 = nn.Upsample(scale_factor=2)
        self.L4_dcnv4 = nn.ConvTranspose2d(in_channels=chn_sizes[4], out_channels=chn_sizes[3], kernel_size=kern_sizes[3], stride=1, padding=0)
        self.L3_dcnv3 = nn.ConvTranspose2d(in_channels=chn_sizes[3], out_channels=chn_sizes[2], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L2_upsm1 = nn.Upsample(scale_factor=2)
        self.L1_dcnv2 = nn.ConvTranspose2d(in_channels=chn_sizes[2], out_channels=chn_sizes[1], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L0_dcnv1 = nn.ConvTranspose2d(in_channels=chn_sizes[1], out_channels=chn_sizes[0], kernel_size=kern_sizes[0], stride=1, padding=0)

        print("input_size=", self.input_size)
        print("chn_sizes=", self.chn_sizes)
        print("kern_sizes=", self.kern_sizes)
        print("hid_sizes=", self.hid_sizes)
        print("feat_size=", self.feat_size)

        lr = 0.0001
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss(reduction='sum')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def enc(self, x):
        # encoding
        x = F.relu(self.L0_conv1(x))
        x = F.relu(self.L1_conv2(x))
        if self.DropLayer is not None:
            x = self.DropLayer(x)
        x = self.L2_maxp1(x)
        x = F.relu(self.L3_conv3(x))
        x = F.relu(self.L4_conv4(x))
        if self.DropLayer is not None:
            x = self.DropLayer(x)
        x = self.L5_maxp2(x)
        x = F.relu(self.L6_conv5(x))
        rp0, rp1, rp2 = x.size(1), x.size(2), x.size(3)
        x = x.view(x.size(0), -1)
        x = F.relu(self.L7_lenc1(x))
        if self.DropLayer is not None:
            x = self.DropLayer(x)
        x = self.L8_lenc2(x)
        return x, rp0, rp1, rp2

    def dec(self, x, rp0, rp1, rp2):
        # decoding
        x = F.relu(self.L7_ldec1(x))
        x = x.view(x.size(0), rp0, rp1, rp2)
        x = F.relu(self.L6_dcnv5(x))
        x = self.L5_upsm1(x)
        x = F.relu(self.L4_dcnv4(x))
        x = F.relu(self.L3_dcnv3(x))
        x = self.L2_upsm1(x)
        x = F.relu(self.L1_dcnv2(x))
        reconstruction = torch.sigmoid(self.L0_dcnv1(x))
        return reconstruction

    def forward(self, x):
        x, rp0, rp1, rp2 = self.enc(x)

        x = x.view(-1, 2, self.feat_size)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        x = F.relu(self.L8_ldec2(z))

        reconstruction = self.dec(x, rp0, rp1, rp2)

        return reconstruction, mu, log_var

    def final_loss(self, bce_loss, mu, logvar):
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

    def fit(self, X_data, batch_size):
        self.train()
        running_loss = 0.0
        dloader = DataLoader(X_data, batch_size=batch_size, shuffle=True)
        #lab_vec = []
        for b in dloader:
            data = b['image']
            #labels = b['label']
            #lab_vec.append(labels)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self.forward(data)
            bce_loss = self.criterion(reconstruction, data)
            loss = self.final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        train_loss = running_loss/len(X_data)
        return train_loss

    def validate(self, X_vate, epoch, batch_size, out_folder):
        self.eval()
        running_loss = 0.0
        batch = [b['image'] for b in X_vate]
        batch_lb = [b['label'] for b in X_vate]
        uqlb, unid = np_unique(batch_lb, return_index=True)
        data_al = [batch[i] for i in unid]
        data_cn = len(unid)

        with torch.no_grad():
            fr = 0
            while (fr + batch_size < len(X_vate)):
                to = fr + batch_size
                data = torch.stack(batch[fr:to], dim=0)
                data = data.to(self.device)
                reconstruction, mu, logvar = self.forward(data)
                bce_loss = self.criterion(reconstruction, data)
                loss = self.final_loss(bce_loss, mu, logvar)
                running_loss += loss.item()
                fr = to

        with torch.no_grad():
            # save the last batch input and output of every epoch
            data = torch.stack(data_al, dim=0)
            data = data.to(self.device)
            #data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = self.forward(data)
            both = torch.cat((data.view(data_cn, 3, self.input_size, self.input_size)[:data_cn],
                              reconstruction.view(data_cn, 3, self.input_size, self.input_size)[:data_cn]))
            f_name = out_folder + "/output_{:03d}.png".format(epoch)
            save_image(both.cpu(), f_name, nrow=data_cn)
        val_loss = running_loss/len(X_vate)
        return val_loss

    @staticmethod
    def feat_extract_ext(model, X_vate, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(model, str):
            model = torch.load(model, map_location=device)
        model.eval()
        dloader = DataLoader(X_vate, batch_size=batch_size, shuffle=False)

        mu_vec = []
        x_vec = []
        lab_vec = []
        with torch.no_grad():
            for b in dloader:
                data = b['image']
                labels = b['label']
                lab_vec.append(labels)

                x = data.to(device)
                # encode
                x = F.relu(model.L0_conv1(x))
                x = F.relu(model.L1_conv2(x))
                x = model.L2_maxp1(x)
                x = F.relu(model.L3_conv3(x))
                x = F.relu(model.L4_conv4(x))
                x = model.L5_maxp2(x)
                x = F.relu(model.L6_conv5(x))
                x = x.view(x.size(0), -1)
                x = F.relu(model.L7_lenc1(x))
                x = model.L8_lenc2(x)

                x_vec.append(x)

                x = x.view(-1, 2, model.feat_size)
                mu = x[:, 0, :]  # the first feature values as mean
                mu_vec.append(mu)

        mu_vec = np.asarray(torch.cat(mu_vec).to(torch.device('cpu')))
        x_vec = np.asarray(torch.cat(x_vec).to(torch.device('cpu')))
        lab_vec = np.asarray(torch.cat(lab_vec).to(torch.device('cpu')))
        #np.savez('?_data.npz', mu_vec=mu_vec, x_vec=x_vec, labsTr=batch_lb)
        return mu_vec, x_vec, lab_vec

class ConvVAE_MultiTask(nn.Module):
    def __init__(self,
                 input_size,
                 chn_sizes, kern_sizes, hid_sizes, feat_size,
                 class_count, apply_classification_task=True,
                 random_seed= 0,
                 update_weights_method=None):
        super(ConvVAE_MultiTask, self).__init__()

        self.input_size = input_size  # 64
        self.chn_sizes = chn_sizes  # chn_sizes = [3, 32, 32, 16]
        self.kern_sizes = kern_sizes  # kern_sizes = [5, 5, 5]
        self.hid_sizes = hid_sizes  # hid_sizes = [9216(16*24*24), 512]
        self.feat_size = feat_size  # feat_size = 64
        self.bottle_neck_image_size = None
        self.class_count = class_count  # class_count=27
        self.apply_classification_task = apply_classification_task
        self.random_seed = random_seed
        self.update_weights_method = update_weights_method

        self.apply_random_seed()

        # encoder
        self.L0_conv1 = nn.Conv2d(in_channels=chn_sizes[0], out_channels=chn_sizes[1], kernel_size=kern_sizes[0], stride=1, padding=0)
        self.L1_conv2 = nn.Conv2d(in_channels=chn_sizes[1], out_channels=chn_sizes[2], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L2_maxp1 = nn.MaxPool2d(kernel_size=2)
        self.L3_conv3 = nn.Conv2d(in_channels=chn_sizes[2], out_channels=chn_sizes[3], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L4_lenc1 = nn.Linear(in_features=hid_sizes[0], out_features=hid_sizes[1])
        self.L5_lenc2 = nn.Linear(in_features=hid_sizes[1], out_features=feat_size*2)

        # decoder
        self.L6_ldec2 = nn.Linear(in_features=feat_size, out_features=hid_sizes[1])
        self.L7_ldec1 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[0])
        self.L8_dcnv3 = nn.ConvTranspose2d(in_channels=chn_sizes[3], out_channels=chn_sizes[2], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L9_upsm1 = nn.Upsample(scale_factor=2)
        self.L10_dcnv2 = nn.ConvTranspose2d(in_channels=chn_sizes[2], out_channels=chn_sizes[1], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L11_dcnv1 = nn.ConvTranspose2d(in_channels=chn_sizes[1], out_channels=chn_sizes[0], kernel_size=kern_sizes[0], stride=1, padding=0)

        # classifier
        if self.apply_classification_task:
            self.L8_lcls1 = nn.Linear(in_features=feat_size*2, out_features=hid_sizes[1])
            self.L9_lcls2 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[0])
            self.L10_sofm = nn.Linear(in_features=hid_sizes[0], out_features=self.class_count)

        print("input_size=", self.input_size)
        print("chn_sizes=", self.chn_sizes)
        print("kern_sizes=", self.kern_sizes)
        print("hid_sizes=", self.hid_sizes)
        print("feat_size=", self.feat_size)

        if self.update_weights_method is not None:
            self.update_weight_list = ["BCE", "KLD"]
        else:
            self.update_weight_list = None

        if self.apply_classification_task:
            print("class_count=", self.class_count)
            if self.update_weights_method is not None:
                self.update_weight_list.append("CLS")

        if self.update_weights_method is not None:
            self.update_weight_init = 1/len(self.update_weight_list)
            self.update_weight_min = 1/(len(self.update_weight_list)+1)

        lr = 0.0001
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_BCE = nn.BCELoss(reduction='sum')
        if self.apply_classification_task:
            self.loss_CLS = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_epoch_params = {
            "wBCE": 1/3,
            "wCLS": 1/3,
            "wKLD": 1/3,
            "sumBCE": None,
            "sumCLS": None,
            "sumKLD": None,
        }
        self.loss_prev_epoch = None

    def update_weights(self):
        if self.update_weights_method is None or self.loss_prev_epoch is None:
            return

        w_k_new_sum = 0
        for k in self.update_weight_list:
            sum_k_cur = self.loss_epoch_params["sum" + k]
            sum_k_prv = self.loss_prev_epoch["sum" + k]
            if sum_k_prv is None:
                print("skip update of weights - first epoch")
                return
            w_k_cur = self.loss_epoch_params["w" + k]
            print("w", k, "_cur=", w_k_cur)

            dif_k = sum_k_cur - sum_k_prv
            dif_k_p = dif_k/sum_k_cur

            w_k_new = np.maximum(w_k_cur * (1 + dif_k_p), self.update_weight_min)
            w_k_new_sum += w_k_new

            self.loss_epoch_params["w" + k] = w_k_new

        for k in self.update_weight_list:
            self.loss_epoch_params["w" + k] /= w_k_new_sum
            print("w", k, "_new=", self.loss_epoch_params["w" + k])

    def apply_random_seed(self):
        np.random.seed(self.random_seed )
        torch.manual_seed(self.random_seed )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def enc(self, x):
        # encoding
        x = F.relu(self.L0_conv1(x))
        x = F.relu(self.L1_conv2(x))
        x = self.L2_maxp1(x)
        x = F.relu(self.L3_conv3(x))
        if self.bottle_neck_image_size is None:
            self.bottle_neck_image_size = [x.size(1), x.size(2), x.size(3)]
        x = x.view(x.size(0), -1)
        x = F.relu(self.L4_lenc1(x))
        x = self.L5_lenc2(x)
        return x

    def dec(self, z):
        # decoding
        x = F.relu(self.L6_ldec2(z))
        x = F.relu(self.L7_ldec1(x))
        x = x.view(x.size(0), self.bottle_neck_image_size[0], self.bottle_neck_image_size[1], self.bottle_neck_image_size[2])
        x = F.relu(self.L8_dcnv3(x))
        x = self.L9_upsm1(x)
        x = F.relu(self.L10_dcnv2(x))
        reconstruction = torch.sigmoid(self.L11_dcnv1(x))
        return reconstruction

    def predict(self, x):
        x = F.relu(self.L8_lcls1(x))
        x = F.relu(self.L9_lcls2(x))
        x = self.L10_sofm(x)
        _, preds = torch.max(x, 1)
        return x, preds

    def forward(self, x):
        x = self.enc(x)

        if self.apply_classification_task:
            xProb, preds = self.predict(x)
        else:
            xProb, preds = None, None

        x = x.view(-1, 2, self.feat_size)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        reconstruction = self.dec(z)

        return reconstruction, mu, log_var, xProb, preds

    def final_loss(self, xProb, labels, reconstruction, data, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """

        if self.apply_classification_task:
            CLS = self.loss_CLS(xProb, labels)
        else:
            CLS=0
        BCE = self.loss_BCE(reconstruction, data)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        self.loss_epoch_params["sumBCE"] += BCE.item()
        if self.apply_classification_task:
            self.loss_epoch_params["sumCLS"] += CLS.item()
        self.loss_epoch_params["sumKLD"] += KLD.item()
        return self.loss_epoch_params["wBCE"]*BCE + self.loss_epoch_params["wKLD"]*KLD + self.loss_epoch_params["wCLS"]*CLS

    def fit(self, X_data, batch_size):
        self.train()
        self.apply_random_seed()
        running_loss = 0.0
        dloader = DataLoader(X_data, batch_size=batch_size, shuffle=True)
        lab_vec = []
        pred_vec = []

        self.loss_prev_epoch = self.loss_epoch_params.copy()

        self.loss_epoch_params["sumBCE"] = 0
        self.loss_epoch_params["sumCLS"] = 0
        self.loss_epoch_params["sumKLD"] = 0

        for b in dloader:
            data = b['image']
            labels = b['label']

            data = data.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            reconstruction, mu, logvar, xProb, preds = self.forward(data)

            lab_vec.append(labels)
            if self.apply_classification_task:
                pred_vec.append(preds)

            loss = self.final_loss(xProb, labels, reconstruction, data, mu, logvar)

            running_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        lab_vec = np.asarray(torch.cat(lab_vec).to(torch.device('cpu')))
        if self.apply_classification_task:
            pred_vec = np.asarray(torch.cat(pred_vec).to(torch.device('cpu')))
            acc = accuracy_score(lab_vec, pred_vec)
        else:
            acc = 0

        self.update_weights()

        n = len(X_data)
        loss_acc_dict = {
            "loss": running_loss/n,
            "acc": acc,
            "wBCE": self.loss_epoch_params["wBCE"],
            "wCLS": self.loss_epoch_params["wCLS"],
            "wKLD": self.loss_epoch_params["wKLD"],
            "sumBCE": self.loss_epoch_params["sumBCE"]/n,
            "sumCLS": self.loss_epoch_params["sumCLS"]/n,
            "sumKLD": self.loss_epoch_params["sumKLD"]/n,
        }

        return loss_acc_dict

    def validate(self, X_vate, epoch, batch_size, out_folder, out_name_add_str=""):
        self.eval()
        self.apply_random_seed()

        running_loss = 0.0
        batch = [b['image'] for b in X_vate]
        batch_lb = [b['label'] for b in X_vate]
        uqlb, unid = np_unique(batch_lb, return_index=True)
        data_al = [batch[i] for i in unid]
        data_cn = len(unid)
        pred_vec = []

        self.loss_epoch_params["sumBCE"] = 0
        self.loss_epoch_params["sumCLS"] = 0
        self.loss_epoch_params["sumKLD"] = 0
        with torch.no_grad():
            fr = 0
            while (fr < len(X_vate)):
                to = fr + batch_size
                if to > len(X_vate):
                    to = len(X_vate)

                data = torch.stack(batch[fr:to], dim=0)
                labels = torch.tensor(batch_lb[fr:to])

                data = data.to(self.device)
                labels = labels.to(self.device)

                reconstruction, mu, logvar, xProb, preds = self.forward(data)

                if self.apply_classification_task:
                    pred_vec.append(preds)

                loss = self.final_loss(xProb, labels, reconstruction, data, mu, logvar)

                running_loss += loss.item()
                fr = to

        labels = np.asarray(batch_lb)
        if self.apply_classification_task:
            pred_vec = np.asarray(torch.cat(pred_vec).to(torch.device('cpu')))
            acc = accuracy_score(labels, pred_vec)
        else:
            acc = 0

        with torch.no_grad():
            # save the last batch input and output of every epoch
            data = torch.stack(data_al, dim=0)
            data = data.to(self.device)
            reconstruction, _, _, _, _ = self.forward(data)
            both = torch.cat((data.view(data_cn, 3, self.input_size, self.input_size)[:data_cn],
                              reconstruction.view(data_cn, 3, self.input_size, self.input_size)[:data_cn]))
            f_name = out_folder + "/output_" + out_name_add_str + "{:03d}.png".format(epoch)
            save_image(both.cpu(), f_name, nrow=data_cn)

        n = len(X_vate)
        loss_acc_dict = {
            "loss": running_loss/n,
            "acc": acc,
            "wBCE": self.loss_epoch_params["wBCE"],
            "wCLS": self.loss_epoch_params["wCLS"],
            "wKLD": self.loss_epoch_params["wKLD"],
            "sumBCE": self.loss_epoch_params["sumBCE"]/n,
            "sumCLS": self.loss_epoch_params["sumCLS"]/n,
            "sumKLD": self.loss_epoch_params["sumKLD"]/n,
        }
        return loss_acc_dict

    @staticmethod
    def feat_extract_ext(model, X_vate, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(model, str):
            model = torch.load(model, map_location=device)
        model.eval()
        model.apply_random_seed()
        dloader = DataLoader(X_vate, batch_size=batch_size, shuffle=False)

        mu_vec = []
        x_vec = []
        lab_vec = []
        pred_vec = []
        with torch.no_grad():
            for b in dloader:
                data = b['image']
                labels = b['label']
                lab_vec.append(labels)

                x = data.to(device)
                # encode
                x = F.relu(model.L0_conv1(x))
                x = F.relu(model.L1_conv2(x))
                x = model.L2_maxp1(x)
                x = F.relu(model.L3_conv3(x))
                x = x.view(x.size(0), -1)
                x = F.relu(model.L4_lenc1(x))
                x = model.L5_lenc2(x)
                x_vec.append(x)

                if model.apply_classification_task:
                    _, preds = model.predict(x)
                    pred_vec.append(preds)

                x = x.view(-1, 2, model.feat_size)
                mu = x[:, 0, :]  # the first feature values as mean
                mu_vec.append(mu)

        mu_vec = np.asarray(torch.cat(mu_vec).to(torch.device('cpu')))
        x_vec = np.asarray(torch.cat(x_vec).to(torch.device('cpu')))
        lab_vec = np.asarray(torch.cat(lab_vec).to(torch.device('cpu')))
        if model.apply_classification_task:
            pred_vec = np.asarray(torch.cat(pred_vec).to(torch.device('cpu')))
        #np.savez('?_data.npz', mu_vec=mu_vec, x_vec=x_vec, labsTr=lab_vec, predsTr=pred_vec)
        return mu_vec, x_vec, lab_vec, pred_vec

    def feat_extract(self, X_vate, batch_size):
        return self.feat_extract_ext(self, X_vate, batch_size)
