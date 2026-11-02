"""Check physical and intensity definitions in notebook 03."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk
from test_notebook_contracts import notebook_definition


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


if __name__ == '__main__':
    unittest.main()
