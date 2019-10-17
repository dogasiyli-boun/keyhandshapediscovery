from keras import backend as K

def neg_l2_reg(activation, weight_of_regularizer):
    return -weight_of_regularizer*K.mean(K.square(activation))

def neg_l2_reg2(activation, weight_of_regularizer):
    return -weight_of_regularizer*(K.square(activation))

def penalized_loss(l2_val):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true)) + K.mean(l2_val)
    return loss

def only_sparsity_loss(l2_val):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true))*0 + K.mean(l2_val)
    return loss
