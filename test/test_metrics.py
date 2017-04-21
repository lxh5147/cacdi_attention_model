import unittest

from metrics import binary_accuracy_with_threshold
import keras.backend as K
from keras.layers import Input
import numpy as np

class MetricsTest(unittest.TestCase):

    def test_binary_accuracy_with_threshold_(self):
        y_true = Input((2,))
        y_pred = Input((2,))
        threshold = K.placeholder((2,))
        acc = binary_accuracy_with_threshold(y_true,y_pred,threshold)
        self.assertEqual(K.ndim(acc), 0)
        binary_accuracy_with_threshold_func = K.function(inputs=[y_true,y_pred,threshold], outputs=[acc])
        acc_val=binary_accuracy_with_threshold_func([np.array([[0,1],[1,0]]),np.array([[0.2,0.6],[0.3,0.1]]),np.array([0.25,0.4])])[0]
        self.assertEqual(round(acc_val,2), 1.00,"acc_val")

        #works on a single threshold
        threshold = K.placeholder(ndim=0)
        acc = binary_accuracy_with_threshold(y_true, y_pred, threshold)
        binary_accuracy_with_threshold_func = K.function(inputs=[y_true, y_pred, threshold], outputs=[acc])
        acc_val = binary_accuracy_with_threshold_func(
            [np.array([[0, 1], [1, 0]]), np.array([[0.2, 0.6], [0.3, 0.1]]), 0.5])[0]
        self.assertEqual(round(acc_val, 2), 0.75, "acc_val")

        # works on 3 dimension inputs
        y_true = Input((None,2))
        y_pred = Input((None,2))
        threshold = K.placeholder((2,))
        acc = binary_accuracy_with_threshold(y_true,y_pred,threshold)
        self.assertEqual(K.ndim(acc), 0)
        binary_accuracy_with_threshold_func = K.function(inputs=[y_true,y_pred,threshold], outputs=[acc])
        acc_val=binary_accuracy_with_threshold_func([np.array([[[0,1]],[[1,0]]]),np.array([[[0.2,0.6]],[[0.3,0.1]]]),np.array([0.25,0.4])])[0]
        self.assertEqual(round(acc_val,2), 1.00,"acc_val")



if __name__ == '__main__':
    unittest.main()
