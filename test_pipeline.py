import ast
import atexit
from operator import methodcaller
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest

import nbformat
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent


def notebook_definition(notebook_name, definition_name, namespace):
    notebook = nbformat.read(ROOT / notebook_name, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == definition_name:
                module = ast.Module(body=[node], type_ignores=[])
                exec(compile(module, notebook_name, "exec"), namespace)
                return namespace[definition_name]
    raise AssertionError(f"Missing definition: {definition_name}")


class ImageContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.load_pair = notebook_definition("01_voi_radiomics.ipynb", "load_pair", {
            "sitk": sitk, "np": np, "ndimage": ndimage,
        })
        self.image = sitk.GetImageFromArray(np.full((8, 8, 8), -300, dtype=np.float32))
        self.image.SetSpacing((0.5, 0.5, 2.0))
        labels = np.zeros((8, 8, 8), dtype=np.uint8)
        labels[1:3, 1:3, 1:3] = 1
        labels[5:7, 5:7, 5:7] = 1
        self.mask = sitk.GetImageFromArray(labels)
        self.mask.CopyInformation(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def read_pair(self):
        image_path, mask_path = self.directory / "image.nii.gz", self.directory / "mask.nii.gz"
        sitk.WriteImage(self.image, str(image_path))
        sitk.WriteImage(self.mask, str(mask_path))
        return self.load_pair(image_path, mask_path)

    def test_all_components_preserved_and_physical_volume(self):
        _, mask, qc = self.read_pair()
        self.assertEqual(qc["components"], 2)
        self.assertEqual(qc["native_voxels"], 16)
        self.assertEqual(qc["native_volume_mm3"], 8.0)
        self.assertEqual(sitk.GetArrayFromImage(mask).sum(), 16)

    def test_geometry_mismatch_is_rejected(self):
        self.mask.SetOrigin((10.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "physical geometry mismatch"):
            self.read_pair()

    def test_unexpected_mask_label_is_rejected(self):
        self.mask = self.mask * 2
        with self.assertRaisesRegex(ValueError, "other than 0 and 1"):
            self.read_pair()

    def test_empty_mask_is_rejected(self):
        self.mask = self.mask * 0
        with self.assertRaisesRegex(ValueError, "fewer than eight"):
            self.read_pair()


class PredictionContractTests(unittest.TestCase):
    def setUp(self):
        self.predict = notebook_definition("02_malign_benign_modeling.ipynb", "predict_feature_rows", {"pd": pd, "np": np})
        frame = pd.DataFrame({"diameter_mm": [2, 4, 20, 25]})
        pipeline = Pipeline([("impute", SimpleImputer()), ("classifier", LogisticRegression())])
        pipeline.fit(frame, [0, 0, 1, 1])
        self.bundle = {"pipeline": pipeline, "required_inputs": ["diameter_mm"],
                       "threshold": 0.5, "label_mapping": {0: "BENIGN", 1: "MALIGN"}}

    def test_missing_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required input columns"):
            self.predict(self.bundle, [{"other_column": 3}])

    def test_missing_value_uses_fitted_imputer(self):
        result = self.predict(self.bundle, [{"diameter_mm": np.nan}])
        self.assertTrue(np.isfinite(result["score_malign"]).all())


class ThinCropRegressionTests(unittest.TestCase):
    def test_four_two_slice_crops_produce_finite_features(self):
        cases = ['LIDC-IDRI-0046_R_4', 'LIDC-IDRI-0185_R_4', 'LIDC-IDRI-0272_R_1', 'LIDC-IDRI-1003_R_3']
        if not all((ROOT / 'VOIs/VOIs/image' / (name + '.nii.gz')).exists() for name in cases):
            self.skipTest('Local dataset is required for thin-crop regression')
        program = """
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
extractor = featureextractor.RadiomicsFeatureExtractor('radiomics_params.yaml')
assert extractor.settings['interpolator'] == 'sitkLinear'
for name in ['LIDC-IDRI-0046_R_4', 'LIDC-IDRI-0185_R_4', 'LIDC-IDRI-0272_R_1', 'LIDC-IDRI-1003_R_3']:
    image = sitk.ReadImage('VOIs/VOIs/image/' + name + '.nii.gz', sitk.sitkFloat32)
    mask = sitk.ReadImage('VOIs/VOIs/nodule_mask/' + name + '.nii.gz', sitk.sitkUInt8)
    result = extractor.execute(image, mask, label=1)
    features = [float(value) for key, value in result.items() if key.startswith('original_')]
    assert len(features) == 107 and np.isfinite(features).all(), name
"""
        completed = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class WorkerContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        worker_path = self.directory / "worker.py"
        worker_path.write_text(
            "import sys,json,os,time\n"
            "for line in sys.stdin:\n"
            "    request=json.loads(line)\n"
            "    if request['image_path']=='crash': os._exit(7)\n"
            "    if request['image_path']=='stall': time.sleep(5)\n"
            "    print(json.dumps({'ok':True,'payload':{'features':{'value':1}}}),flush=True)\n",
            encoding="utf-8",
        )
        cls = notebook_definition("01_voi_radiomics.ipynb", "ExtractionWorker", {
            "OUT": self.directory, "WORKER_PATH": worker_path, "sys": sys, "json": __import__("json"),
            "queue": queue, "subprocess": subprocess, "threading": threading, "atexit": atexit,
        })
        self.worker = cls()

    def tearDown(self):
        self.worker.close()
        self.temp.cleanup()

    def test_native_exit_does_not_prevent_next_case(self):
        with self.assertRaisesRegex(RuntimeError, "exited with code 7"):
            self.worker.extract({"image_path": "crash", "mask_path": "unused"}, timeout=10)
        result = self.worker.extract({"image_path": "ok", "mask_path": "unused"}, timeout=10)
        self.assertEqual(result["features"]["value"], 1)

    def test_timeout_restarts_worker(self):
        with self.assertRaises(TimeoutError):
            self.worker.extract({"image_path": "stall", "mask_path": "unused"}, timeout=0.5)
        result = self.worker.extract({"image_path": "ok", "mask_path": "unused"}, timeout=10)
        self.assertEqual(result["features"]["value"], 1)


class EngineeredFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.engineer_pair = notebook_definition('03_feature_engineering_ablation.ipynb', 'engineer_pair', {
            'sitk': sitk, 'np': np,
        })
        labels = np.zeros((8, 8, 8), dtype=np.uint8)
        labels[1:3, 1:3, 1:3] = 1
        labels[5:7, 5:7, 5:7] = 1
        values = np.full(labels.shape, -1000, dtype=np.float32)
        values[1:3, 1:3, 1:3] = -100
        values[5:7, 5:7, 5:7] = -300
        self.image = sitk.GetImageFromArray(values)
        self.image.SetSpacing((0.5, 0.5, 2.0))
        self.mask = sitk.GetImageFromArray(labels)
        self.mask.CopyInformation(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def extract(self):
        image_path, mask_path = self.directory / 'image.nii.gz', self.directory / 'mask.nii.gz'
        sitk.WriteImage(self.image, str(image_path))
        sitk.WriteImage(self.mask, str(mask_path))
        return self.engineer_pair(image_path, mask_path)

    def test_feature_values_include_all_components_and_voxel_spacing(self):
        result = self.extract()
        expected = {
            'log_volume': np.log1p(8.0),
            'intensity_difference': 800.0,
            'inside_median': -200.0,
            'inside_iqr': 200.0,
            'mask_size_x_mm': 3.0,
            'mask_size_y_mm': 3.0,
            'mask_size_z_mm': 12.0,
            'mask_fill_ratio': 16 / 216,
        }
        self.assertEqual(set(result), set(expected))
        for feature in expected:
            self.assertAlmostEqual(result[feature], expected[feature], places=7, msg=feature)

    def test_orientation_does_not_change_scalar_features(self):
        baseline = self.extract()
        self.image.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))
        self.image.SetOrigin((30., 50., -10.))
        self.mask.CopyInformation(self.image)
        reoriented = self.extract()
        for feature in baseline:
            self.assertAlmostEqual(baseline[feature], reoriented[feature], places=7)

    def test_misaligned_mask_is_rejected(self):
        self.mask.SetOrigin((1., 0., 0.))
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            self.extract()


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


class AccuracyOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.threshold = notebook_definition(BOOK, 'choose_accuracy_threshold', {'np': np})

    def test_threshold_uses_mean_fold_accuracy(self):
        threshold, score = self.threshold(
            [0, 0, 1, 1, 1, 1, 1, 0], [.4, .4, .4, .4, .4, .4, .4, .1],
            [0, 0, 1, 1, 1, 1, 1, 1], [.3, .5],
        )
        self.assertAlmostEqual(threshold, .5)
        self.assertAlmostEqual(score, 7 / 12)

    def test_ties_prefer_default_threshold(self):
        threshold, score = self.threshold([0, 0, 1, 1], [.1, .2, .8, .9], [0, 1, 0, 1], [.25, .5, .75])
        self.assertAlmostEqual(threshold, .5)
        self.assertAlmostEqual(score, 1.0)

    def test_accuracy_tuning_uses_only_validation_scores(self):
        RecordingEstimator.records = []
        X = pd.DataFrame({'row_key': range(8)}, index=range(100, 108))
        y = pd.Series(np.tile([0, 1], 4), index=X.index)
        folds = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
        evaluate = notebook_definition(BOOK, 'accuracy_candidate', {
            'np': np, 'clone': clone, 'threadpool_limits': threadpool_limits,
            'roc_auc_score': roc_auc_score, 'choose_accuracy_threshold': self.threshold,
            'THRESHOLDS': np.array([.3, .5, .7]),
        })
        result = evaluate(RecordingEstimator(), X, y, folds)
        self.assertEqual(RecordingEstimator.records, [set(range(100, 104)), set(range(104, 108))])
        self.assertAlmostEqual(result['inner_accuracy'], 1.0)


class ImprovementModelTests(unittest.TestCase):
    def model(self, family):
        grouped = notebook_definition(BOOK, 'grouped_splits', {
            'np': np, 'StratifiedGroupKFold': StratifiedGroupKFold,
        })
        build = notebook_definition(BOOK, 'improvement_pipeline', {
            'accuracy_inputs': {'tabular': ['row_key']}, 'TAB_NUMERIC': ['row_key'],
            'SEED': 42, 'SMOKE': True, 'SVC': SVC, 'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'CalibratedClassifierCV': CalibratedClassifierCV, 'StackingClassifier': StackingClassifier,
            'grouped_splits': grouped,
            'make_pipeline': lambda numeric, classifier, percentile: Pipeline([
                ('impute', SimpleImputer()), ('classifier', classifier),
            ]),
        })
        X = pd.DataFrame({'row_key': range(8)})
        y = pd.Series(np.tile([0, 1], 4))
        groups = pd.Series(np.repeat(['a', 'b', 'c', 'd'], 2))
        params = {'C': 1.0, 'gamma': 'scale'} if family == 'svm' else {'C': 1.0, 'class_weight': None}
        spec = {'feature_set': 'tabular', 'family': family, 'params': params,
                'percentile': 100, 'reference': False}
        return build(spec, X, y, groups), groups

    def test_svm_calibration_keeps_patients_together(self):
        model, groups = self.model('svm')
        self.assertFalse(model.ensemble)
        for train, validation in model.cv:
            self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[validation]))

    def test_stacking_keeps_patients_together(self):
        model, groups = self.model('stacking')
        for train, validation in model.cv:
            self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[validation]))
        self.assertEqual(sorted(np.concatenate([validation for _, validation in model.cv]).tolist()), list(range(8)))


class NativeCategoryTests(unittest.TestCase):
    def setUp(self):
        build = notebook_definition(BOOK, 'native_pipeline', {
            'np': np, 'Pipeline': Pipeline, 'SimpleImputer': SimpleImputer,
            'FunctionTransformer': FunctionTransformer, 'ColumnTransformer': ColumnTransformer,
            'TAB_NUMERIC': ['diameter_mm'], 'TAB_CATEGORICAL': ['calcification', 'structure'],
            'SMOKE': True, 'SEED': 42, 'LGBMClassifier': lambda **kwargs: LogisticRegression(),
            'methodcaller': methodcaller,
        })
        self.pipeline = build({'reference': False, 'params': {}})
        self.preprocess = self.pipeline.named_steps['preprocess']
        self.training = pd.DataFrame({'diameter_mm': [1, 2, 3],
                                      'calcification': [1, 1, 2], 'structure': [1, 2, 2]})

    def test_integer_and_float_category_codes_match(self):
        self.preprocess.fit(self.training)
        integers = self.preprocess.transform(self.training)
        floats = self.preprocess.transform(self.training.astype(float))
        pd.testing.assert_frame_equal(integers, floats)
        self.assertEqual(integers['calcification'].tolist(), ['1.0', '1.0', '2.0'])

    def test_missing_categories_use_training_modes(self):
        self.preprocess.fit(self.training)
        validation = pd.DataFrame({'diameter_mm': [np.nan, 1000],
                                    'calcification': [np.nan, 99], 'structure': [np.nan, 99]})
        result = self.preprocess.transform(validation)
        self.assertEqual(result.loc[0, 'diameter_mm'], 2)
        self.assertEqual(result.loc[0, 'calcification'], '1.0')
        self.assertEqual(result.loc[0, 'structure'], '2.0')
        self.assertEqual(result.loc[1, 'calcification'], '99.0')

    def test_classifier_receives_category_columns(self):
        frame = self.preprocess.fit_transform(self.training)
        result = self.pipeline.named_steps['category'].fit_transform(frame)
        self.assertIsInstance(result['calcification'].dtype, pd.CategoricalDtype)
        self.assertIsInstance(result['structure'].dtype, pd.CategoricalDtype)
        self.assertEqual(result['calcification'].astype(str).tolist(), ['1.0', '1.0', '2.0'])


if __name__ == "__main__":
    unittest.main()
