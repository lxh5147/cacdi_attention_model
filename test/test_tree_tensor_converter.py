import unittest

from tree_tensor_converter import  (_get_dim, get_sequence_inputs)
import numpy as np

class TreeTensorConverterTest(unittest.TestCase):

    def test_get_dim(self):
        self.assertAlmostEqual(_get_dim(1),0 )
        self.assertAlmostEqual(_get_dim([1,2,3]), 1)
        self.assertAlmostEqual(_get_dim([[1, 2, 3],[4,5]]), 2)
        self.assertAlmostEqual(_get_dim(np.array([3,4])), 1)

    def test_get_sequence_inputs(self):
        mask_value = -1
        r1=[[0,2],[0,3,2,0]]
        r2 =[[3,4,5]]
        tree=[r1,r2] # has 2 reviews: reviews, sentences, words
        inputs = get_sequence_inputs(tree, mask_value=mask_value)
        input1=np.array([[0,2,-1,-1],[0,3,2,0],[3,4,5,-1]])
        input0=np.array([[0,1],[2,-1]])
        self.assertEqual(len(inputs),2)
        self.assertTrue(np.array_equal(inputs[0],input0))
        self.assertTrue(np.array_equal(inputs[1], input1))

if __name__ == '__main__':
    unittest.main()
