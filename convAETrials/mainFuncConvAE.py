# Import all the required Libraries
import matplotlib.pyplot as plt
import dataLoaderConvAE as dataLoader
import modelLoaderConvAE as modelLoader
import numpy as np
import helperFuncs as funcH
import os
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

runForMNIST = False
if runForMNIST :
    train_images, train_labels, test_images, test_labels = dataLoader.loadMNISTData()
    # train_images --> ndarray of size (60000,28,28,1)
    # train_labels --> ndarray of size (60000,1)
    # test_images --> ndarray of size (10000,28,28,1)
    # test_labels --> ndarray of size (10000,1)
    trainFromScratch = False
    if trainFromScratch:
        ae = modelLoader.modelLoad_MNIST()
        # compile it using adam optimizer
        ae.compile(optimizer="adam", loss="mse")
        #Train it by providing training images
        ae.fit(train_images, train_images, epochs=2)
        modelLoader.saveModel(ae, "model_tex")
    else:
        ae = modelLoader.loadModel("model_tex")

    prediction = ae.predict(train_images[0:199,:,:,:], verbose=1, batch_size=100)
    x =prediction[0].reshape(28,28)
    plt.imshow(x)
    plt.show()
else:
    exp_name = 'cnnAE'
    results_dir = funcH.getVariableByComputerName('results_dir')
    outdir = os.path.join(results_dir, 'results', exp_name)

    csv_name = os.path.join(results_dir, 'epochs') + os.sep + exp_name + '.csv'
    model_name = os.path.join(results_dir, 'models') + os.sep + exp_name + '.h5'

    funcH.createDirIfNotExist(os.path.join(results_dir, 'epochs'))
    funcH.createDirIfNotExist(os.path.join(results_dir, 'models'))
    funcH.createDirIfNotExist(outdir)

    checkpointer = ModelCheckpoint(filepath=model_name, verbose=0, save_best_only=False, period=1)
    csv_logger = CSVLogger(csv_name, append=True, separator=';')
    #ES = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=0, mode='auto')
    #callbacks = [csv_logger, ES, checkpointer]

    feat_set, labels_all, detailedLabels_all = dataLoader.loadData_nnVidImages('/home/dg/DataPath/bdData')
    non_zero_labels = labels_all[np.where(labels_all)]

    ae = modelLoader.modelLoad_KHS()
    ae_tester = modelLoader.modelLoad_KHS_Clusters()
    ae.compile(optimizer="adam", loss="mse")

    for i in range(200):
        ae.fit(feat_set, feat_set, batch_size=128, epochs=1,callbacks=[csv_logger,checkpointer])
        #prediction = ae.predict(feat_set[0:199,:,:,:], verbose=1, batch_size=100)
        #x =prediction[0].reshape(224,224)
        #plt.imshow(x)
        #plt.show()

        ae_tester.load_weights(model_name, by_name=True)
        cluster_posteriors = np.transpose(ae_tester.predict(feat_set))
        predicted_labels = np.argmax(cluster_posteriors,axis=0)
        non_zero_predictions = predicted_labels[np.where(labels_all)]
        np.savez(outdir + os.sep + 'cnnAE_' + str(i).zfill(3) + '.npz', predictions=predicted_labels, labels=labels_all)

        nmi_cur, acc_cur = funcH.get_NMI_Acc(non_zero_labels, non_zero_predictions)
        nmi_and_acc_file_name = outdir + os.sep + 'cnnAE_' + '_nmi_acc.txt'

        f = open(nmi_and_acc_file_name, 'a+')
        f.write('i=' + str(i) + ' NMI=' + str(nmi_cur) + ' ACC=' + str(acc_cur)+'\n')
        f.close()
        print(' i =', i, ' NMI = ', nmi_cur, ' ACC= ', acc_cur, '\n')



