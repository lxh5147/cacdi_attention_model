'''
Created on Aug 1, 2016

@author: lxh5147
'''
import unittest
from callbacks import   EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript
import numpy as np

class ObjectivesTest(unittest.TestCase):
    def test_export(self):
        all_id = [np.array(['CHBG_H00056025406_20151012','CHBG_H00056025407_20151012','CHBG_H00056025408_20151012']),
                  np.array(['CHBG_H00056025409_20151012', 'CHBG_H00056025410_20151012', 'CHBG_H00056025411_20151012'])]
        all_pred = [np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6]]),
                    np.array([[0.11,0.22],[0.33,0.44],[0.55,0.66]])]
        all_true = [np.array([[0, 0], [0, 1], [1, 1]]),
                    np.array([[0, 0], [1, 1], [1, 1]])]

        pred_file = "pred.csv"
        true_file = "true.csv"
        EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript._export(all_id, all_pred, all_true,
                                                                               pred_file, true_file)
        threshold = EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript._fine_tune(pred_file, true_file)
        score = EarlyStoppingByAccAndSaveWeightsWithThirdPartyEvalautionScript._evaluate(threshold, pred_file, true_file)
        print(threshold)
        print(score)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
