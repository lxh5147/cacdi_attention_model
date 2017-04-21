import unittest
import numpy as np
from attention_cacdi_exp_base import  fine_tune_threshold, evaluate_with_threshold, predict_with_threshold
class AttentionCACDIExpBaseTest(unittest.TestCase):
    def test_fine_tune_threshold(self):
        y_pred_val= np.array([[0.2, 0.6], [0.3, 0.1]])
        y_true_val = np.array([[0, 1], [1, 0]])
        threshold_step_size = 0.1
        threshold_val, acc_val= fine_tune_threshold(y_pred_val, y_true_val, threshold_step_size)
        self.assertEqual(round(threshold_val,2),0.30,"threshold_val")
        self.assertEqual(round(acc_val, 2), 1.00, "acc_val")

    def test_evaluate_with_threshold(self):
        y_pred_val = np.array([[0.2, 0.6], [0.3, 0.1]])
        y_true_val = np.array([[0, 1], [1, 0]])
        threshold_val = 0.3
        acc_val = evaluate_with_threshold(y_pred_val, y_true_val, threshold_val)
        self.assertEqual(round(acc_val, 2), 1.00, "acc_val")

    def test_predict_with_threshold(self):
        y_pred_val = np.array([[0.2, 0.6], [0.3, 0.1]])
        threshold_val = 0.3
        y_final_pred_val = predict_with_threshold(y_pred_val,threshold_val)
        y_true_val = np.array([[0, 1], [1, 0]])
        self.assertTrue(np.array_equal(y_final_pred_val,y_true_val),"y_final_pred_val")

if __name__ == '__main__':
    unittest.main()
