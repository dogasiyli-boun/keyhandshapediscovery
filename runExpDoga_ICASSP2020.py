import os

applyCorr = 2
for corr_randMode in {0, 1}:
    for posterior_dim in {32, 64, 128, 256}:
        for weight_of_regularizer in {0.2, 0.5}:
            runString = "python train_SAE_HOG.py " + str(posterior_dim) + " " + str(weight_of_regularizer) + " " + str(applyCorr) + " " + str(corr_randMode)
            print(runString)
            os.system(runString)
