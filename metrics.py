import keras.backend  as K
from  backend import greater_equal

def binary_accuracy_with_threshold(y_true, y_pred, y_threshold):
    '''
    :param y_true: binary tensor of shape nb_samples, input_dim or nb_samples, time_steps, input_dim
    :param y_pred: prediction tensor of the same shape of y_true
    :param y_threshold: threshold tensor, only if the y_pred is no less than the threshold, predict 1; otherwise 0
    :return: binary accuracy, a scalar
    '''
    ndim_threshold = K.ndim(y_threshold) #ndim_threshold = 0 or 1
    if ndim_threshold == 1: # expand dims to match the shape of y_true
        ndim_y =  K.ndim(y_true) # ndim_y = 2  nb_samples, input_dim; or 3:
        if ndim_y == 2:
            y_threshold = K.expand_dims(y_threshold,0)
        elif ndim_y == 3:
            y_threshold = K.expand_dims(y_threshold, 0)
            y_threshold = K.expand_dims(y_threshold, 0)
    y_pred =greater_equal( y_pred , y_threshold)
    return K.mean(K.equal(y_true, y_pred))