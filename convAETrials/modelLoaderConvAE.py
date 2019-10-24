from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D

def modelLoad_MNIST():
    #ENCODER
    inp = Input((28, 28,1))
    e = Conv2D(32, (3, 3), activation='relu', name='conv01')(inp)
    e = MaxPooling2D((2, 2))(e)
    e = Conv2D(64, (3, 3), activation='relu', name='conv02')(e)
    e = MaxPooling2D((2, 2))(e)
    e = Conv2D(64, (3, 3), activation='relu', name='conv03')(e)
    l = Flatten(name='flat01')(e)
    l = Dense(49, activation='softmax', name='dense01')(l)

    #DECODER
    d = Reshape((7,7,1), name='flat2img')(l)
    d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same', name='deconv01')(d)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same', name='deconv02')(d)
    d = BatchNormalization()(d)
    d = Conv2DTranspose(32,(3, 3), activation='relu', padding='same', name='deconv03')(d)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decodeLayer')(d)
    ae = Model(inp, decoded)
    ae.summary()
    return ae

def modelLoad_KHS():
    #ENCODER
    inp = Input((224, 224, 1), name='input01')
    e = Conv2D(32, (11, 11), strides=3, activation='relu', name='conv01')(inp)
    e = MaxPooling2D((2, 2), name='maxPool01')(e)
    e = Conv2D(64, (5, 5), strides=2, activation='relu', name='conv02')(e)
    e = MaxPooling2D((2, 2), name='maxPool02')(e)
    e = Conv2D(64, (3, 3), activation='relu', name='conv03')(e)
    l = Flatten(name='flat01')(e)
    l = Dense(196, activation='softmax', name='dense01')(l)

    # #DECODER
    d = Reshape((14,14,1), name='flat2img')(l)
    d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same', name='deconv01')(d)
    d = BatchNormalization(name='batchNorm01')(d)
    d = Conv2DTranspose(64,(5, 5), strides=4, activation='relu', padding='same', name='deconv02')(d)
    d = BatchNormalization(name='batchNorm02')(d)
    d = Conv2DTranspose(32,(11, 11), strides=2, activation='relu', padding='same', name='deconv03')(d)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decodeLayer')(d)
    ae = Model(inp, decoded)
    ae.summary()
    return ae

def modelLoad_KHS_Clusters():
    #ENCODER
    inp = Input((224, 224, 1), name='input01')
    e = Conv2D(32, (11, 11), strides=3, activation='relu', name='conv01')(inp)
    e = MaxPooling2D((2, 2), name='maxPool01')(e)
    e = Conv2D(64, (5, 5), strides=2, activation='relu', name='conv02')(e)
    e = MaxPooling2D((2, 2), name='maxPool02')(e)
    e = Conv2D(64, (3, 3), activation='relu', name='conv03')(e)
    l = Flatten(name='flat01')(e)
    l = Dense(196, activation='softmax', name='dense01')(l)
    ae = Model(inp, l)
    ae.summary()
    return ae

#saveModel(ae, "model_tex")
def saveModel(modelToSave, saveFileName):
    # IF you want to save the model
    model_json = modelToSave.to_json()
    with open(saveFileName + ".json", "w") as json_file:
        json_file.write(model_json)

    modelToSave.save_weights(saveFileName + ".h5")
    print("Saved model to (", saveFileName, ")")

#loadModel(loadFileName)
def loadModel(loadFileName):
    json_file = open(loadFileName + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(loadFileName + ".h5")
    print("Loaded model from disk")
    return loaded_model