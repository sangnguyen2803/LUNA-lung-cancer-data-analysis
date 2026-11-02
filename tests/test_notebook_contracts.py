"""Exercise failure boundaries in the actual notebook functions."""
import ast
import atexit
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
