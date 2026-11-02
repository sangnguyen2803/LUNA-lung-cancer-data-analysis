"""Check grouped splitting, validation-only tuning and stored-threshold inference."""
import unittest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from test_notebook_contracts import notebook_definition

BOOK = '04_f1_optimization.ipynb'


class RecordingEstimator(BaseEstimator):
    records = []

    def fit(self, X, y):
        self.training_ids = set(X.index)
        type(self).records.append(self.training_ids)
        return self

    def predict_proba(self, X):
        assert not self.training_ids.intersection(X.index), 'Validation rows were used to fit'
        positive = np.where(X['row_key'].to_numpy() % 2 == 1, 0.8, 0.2)
        return np.column_stack([1 - positive, positive])


class F1OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.threshold = notebook_definition(BOOK, 'choose_threshold', {'np': np})

    def test_threshold_optimizes_mean_validation_fold_f1(self):
        threshold, score, _ = self.threshold(
            [0, 0, 1, 1], [.1, .2, .3, .4], [0, 1, 0, 1], [.2, .3, .5],
        )
        self.assertAlmostEqual(threshold, .3)
        self.assertAlmostEqual(score, 1.0)

    def test_tied_thresholds_prefer_default_cutoff(self):
        threshold, score, _ = self.threshold(
            [0, 0, 1, 1], [.1, .2, .8, .9], [0, 1, 0, 1], [.25, .5, .75],
        )
        self.assertAlmostEqual(threshold, .5)
        self.assertAlmostEqual(score, 1.0)

    def test_grouped_folds_keep_each_patient_together(self):
        split = notebook_definition(BOOK, 'grouped_splits', {'np': np, 'StratifiedGroupKFold': StratifiedGroupKFold})
        X = pd.DataFrame({'row_key': range(8)})
        groups = np.repeat(['a', 'b', 'c', 'd'], 2)
        folds = split(X, np.tile([0, 1], 4), groups, 2, 42)
        for train, validation in folds:
            self.assertFalse(set(groups[train]).intersection(groups[validation]))
        self.assertEqual(sorted(np.concatenate([validation for _, validation in folds]).tolist()), list(range(8)))

    def test_inner_tuning_uses_only_out_of_fold_predictions(self):
        RecordingEstimator.records = []
        X = pd.DataFrame({'row_key': range(8)}, index=range(100, 108))
        y = pd.Series(np.tile([0, 1], 4), index=X.index)
        folds = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
        evaluate = notebook_definition(BOOK, 'inner_candidate', {
            'np': np, 'clone': clone, 'threadpool_limits': threadpool_limits,
            'roc_auc_score': roc_auc_score, 'choose_threshold': self.threshold,
            'THRESHOLDS': np.array([.3, .5, .7]),
        })
        result = evaluate(RecordingEstimator(), X, y, folds)
        self.assertEqual(RecordingEstimator.records, [set(range(100, 104)), set(range(104, 108))])
        self.assertAlmostEqual(result['inner_mean_f1'], 1.0)

    def test_inference_uses_saved_threshold(self):
        class FixedScores:
            def predict_proba(self, X):
                return np.array([[.4, .6], [.2, .8]])
        predict = notebook_definition(BOOK, 'predict_feature_rows', {
            'np': np, 'pd': pd, 'threadpool_limits': threadpool_limits,
        })
        bundle = {'required_inputs': ['diameter_mm'], 'pipeline': FixedScores(),
                  'threshold': .7, 'label_mapping': {0: 'BENIGN', 1: 'MALIGN'}}
        result = predict(bundle, [{'diameter_mm': 3}, {'diameter_mm': 8}])
        self.assertEqual(result['predicted_label'].tolist(), ['BENIGN', 'MALIGN'])


if __name__ == '__main__':
    unittest.main()
