import os
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.datasets import cifar10
from helperFuncs import getVariableByComputerName, createDirIfNotExist, download_file

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    y = np.squeeze(y)
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    y_names = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
               5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
    return x, y, y_names

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    return x, y

def load_mnist_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_test
    y = y_test
    x = np.divide(x, 255.)
    x = x.reshape((x.shape[0], -1))
    return x, y

def load_fashion():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    y_names = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
               5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
    return x, y, y_names

def load_har(data_path = os.path.join(getVariableByComputerName("n2d_experiments"), 'har', 'data')):
    # load this dataset this way ??
    # https://pypi.org/project/kcc2020-tutorial-HAR-dataset/
    # entire_dataset = load_har_all()
    createDirIfNotExist(data_path)
    fold_train = os.path.join(data_path, 'train')
    fold_test = os.path.join(data_path, 'test')
    createDirIfNotExist(fold_train)
    createDirIfNotExist(fold_test)
    fname_train_x = os.path.join(fold_train, 'X_train.txt')
    fname_train_y = os.path.join(fold_train, 'y_train.txt')
    fname_test_x = os.path.join(fold_test, 'X_test.txt')
    fname_test_y = os.path.join(fold_test, 'y_test.txt')

    # https://github.com/mollybostic/cleaning-data-assignment/tree/master/UCI%20HAR%20Dataset
    # for windows = https://sourceforge.net/projects/gnuwin32/files/wget/1.11.4-1/wget-1.11.4-1-setup.exe/download
    # https://stackoverflow.com/questions/29113456/wget-not-recognized-as-internal-or-external-command

    link_adr_path = 'https://raw.githubusercontent.com/mollybostic/cleaning-data-assignment/master/UCI%20HAR%20Dataset/<trte>/<Xy>_<trte>.txt'
    if not os.path.isfile(fname_train_x):
        print('downloading X_train.txt(66.0MB)')
        download_file(link_adr_path.replace("<trte>", "train").replace("<Xy>", "X"), save2path=fold_train, savefilename='X_train.txt')
        #os.system("wget --no-verbose 'https://raw.githubusercontent.com/mollybostic/cleaning-data-assignment/master/UCI HAR Dataset/train/X_train.txt' -P %s" % fold_train)
        print('downloading y_train.txt(14.7kB)')
        download_file(link_adr_path.replace("<trte>", "train").replace("<Xy>", "y"), save2path=fold_train, savefilename='y_train.txt')
        #os.system("wget --no-verbose 'https://raw.githubusercontent.com/mollybostic/cleaning-data-assignment/master/UCI HAR Dataset/train/y_train.txt' -P %s" % fold_train)
        print('downloading X_test.txt(26.5MB)')
        download_file(link_adr_path.replace("<trte>", "test").replace("<Xy>", "X"), save2path=fold_test, savefilename='X_test.txt')
        #os.system("wget --no-verbose 'https://raw.githubusercontent.com/mollybostic/cleaning-data-assignment/master/UCI HAR Dataset/test/X_test.txt' -P %s" % fold_test)
        print('downloading y_test.txt(5.9kB)')
        download_file(link_adr_path.replace("<trte>", "test").replace("<Xy>", "y"), save2path=fold_test, savefilename='y_test.txt')
        #os.system("wget --no-verbose 'https://raw.githubusercontent.com/mollybostic/cleaning-data-assignment/master/UCI HAR Dataset/test/y_test.txt' -P %s" % fold_test)

    x_train = pd.read_csv(fname_train_x, sep=r'\s+', header=None)
    y_train = pd.read_csv(fname_train_y, header=None)
    x_test = pd.read_csv(fname_test_x, sep=r'\s+', header=None)
    y_test = pd.read_csv(fname_test_y, header=None)
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    # # labels start at 1 so..
    y = y - 1
    y = y.reshape((y.size,))
    y_names = {0: 'Walking', 1: 'Upstairs', 2: 'Downstairs', 3: 'Sitting', 4: 'Standing', 5: 'Laying', }
    os.error("not implemented")
    return x, y, y_names

def load_usps(data_path = os.path.join(getVariableByComputerName("n2d_experiments"), 'usps', 'data')):
    createDirIfNotExist(data_path)

    file_name_tr = os.path.join(data_path, 'usps_train.jf')
    file_name_te = os.path.join(data_path, 'usps_test.jf')
    link_adr_path = 'https://raw.githubusercontent.com/cvjena/ITAL/master/data/usps_<trte>.jf'
    if not os.path.exists(file_name_tr):
        download_file(link_adr_path.replace("<trte>", "train"), save2path=data_path, savefilename='usps_train.jf')
        #os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s' % data_path)
        download_file(link_adr_path.replace("<trte>", "test"), save2path=data_path, savefilename='usps_test.jf')
        #os.system('wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s' % data_path)

    with open(file_name_tr) as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(file_name_te) as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64')
    y = np.concatenate((labels_train, labels_test))
    print('USPS samples', x.shape)
    return x, y

def load_pendigits(data_path = os.path.join(getVariableByComputerName("n2d_experiments"), 'pendigits', 'data')):
    createDirIfNotExist(data_path)
    file_name_tr = os.path.join(data_path, 'pendigits.tra')
    file_name_te = os.path.join(data_path, 'pendigits.tes')
    link_adr_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits<file_ending>'
    if not os.path.exists(file_name_tr):
        os.makedirs(data_path,  exist_ok=True)
        download_file(link_adr_path.replace("<file_ending>", ".tra"), save2path=data_path, savefilename='pendigits.tra')
        #os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra -P %s' % data_path)
        download_file(link_adr_path.replace("<file_ending>", ".tes"), save2path=data_path, savefilename='pendigits.tes')
        #os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes -P %s' % data_path)
        download_file(link_adr_path.replace("<file_ending>", ".names"), save2path=data_path, savefilename='pendigits.names')
        #os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.names -P %s' % data_path)

    # load training data
    with open(file_name_tr) as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_train, labels_train = data[:, :-1], data[:, -1]

    # load testing data
    with open(file_name_te) as file:
        data = file.readlines()
    data = [list(map(float, line.split(','))) for line in data]
    data = np.array(data).astype(np.float32)
    data_test, labels_test = data[:, :-1], data[:, -1]

    x = np.concatenate((data_train, data_test)).astype('float32')
    y = np.concatenate((labels_train, labels_test))
    x /= 100.
    y = y.astype('int')
    return x, y
