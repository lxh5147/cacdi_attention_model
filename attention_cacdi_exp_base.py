import keras.backend as K
from metrics import  binary_accuracy_with_threshold

def build_binary_accuracy_with_threshold_func(y_shape):
    threshold = K.placeholder(ndim=0)
    y_pred = K.placeholder(y_shape)
    y_true = K.placeholder(y_shape)
    acc = binary_accuracy_with_threshold(y_true, y_pred, threshold)
    return K.function(inputs=[y_true, y_pred, threshold], outputs=[acc])

def fine_tune_threshold(y_pred_val, y_true_val, threshold_step_size):
    # compile a function to compute the acc
    binary_accuracy_with_threshold_func = build_binary_accuracy_with_threshold_func(y_pred_val.shape)
    # now run this function multiple times to find the optimal threshold
    threshold_val = .0
    threshold_val_best = .0
    acc_val_best = -1.0
    while threshold_val  < 1.0:
        acc_val = binary_accuracy_with_threshold_func([y_true_val, y_pred_val, threshold_val])[0]
        if acc_val > acc_val_best:
            acc_val_best = acc_val
            threshold_val_best = threshold_val
        threshold_val += threshold_step_size
    return threshold_val_best, acc_val_best

def evaluate_with_threshold(y_pred_val, y_true_val, threshold_val):
    binary_accuracy_with_threshold_func = build_binary_accuracy_with_threshold_func(y_pred_val.shape)
    return binary_accuracy_with_threshold_func([y_true_val, y_pred_val, threshold_val])[0]

def predict_with_threshold(y_pred_val, threshold_val):
    return (y_pred_val >= threshold_val).astype(float)
