'''
Created on Aug 1, 2016

@author: lxh5147
'''
import keras.backend as K
from keras.backend.common import  _EPSILON

def categorical_crossentropy_ex(y_true, y_pred):
    '''
    Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return K.sum(K.categorical_crossentropy(y_pred, y_true))


def binary_crossentropy_ex(y_true, y_pred):
    '''
    Returns a scalar of the average binary crossentropy.
    '''
    return K.mean(K.binary_crossentropy(y_pred, y_true))

def weighted_binary_crossentropy_ex(y_true, y_pred):
    '''
    y_true: binary tensor
    y_pred: prediction probability
    '''
    pos = y_true * K.log(y_pred)
    neg = (1 - y_true) * K.log(1 - y_pred)
    weighted_pos = K.sum(pos) / (_EPSILON + K.sum(y_true))
    weighted_neg = K.sum(neg) / (_EPSILON + K.sum(1 - y_true))
    return -(weighted_pos + weighted_neg)
