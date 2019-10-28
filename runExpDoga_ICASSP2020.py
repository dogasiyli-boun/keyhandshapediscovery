import os

dataToUse = 'hog'
trainMode = 'cosae' #trainMode = {'sae','cosae','corsa'}
for corr_randMode in {1}:
    for posterior_dim in {64, 128, 256}:#32
        for weight_of_regularizer in {0.5, 1.0}:
            runString = "python train_SAE_HOG.py " + str(posterior_dim) + " " + str(weight_of_regularizer) + " " + str(trainMode) + " " + str(corr_randMode) + " " + str(dataToUse)
            print(runString)
            os.system(runString)
