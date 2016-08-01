'''
Created on Aug 1, 2016

@author: lxh5147
'''
import unittest
from objectives import weighted_binary_crossentropy_ex
from keras.layers import Input
import keras.backend as K

class ObjectivesTest(unittest.TestCase):


    def test_weighted_binary_crossentropy_ex(self):
        y_true = Input(shape = (3, 2))
        y_pred = Input(shape = (3, 2))
        score = weighted_binary_crossentropy_ex(y_true = y_true, y_pred = y_pred)
        self.assertEqual(0, K.ndim(score), "score")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
