"""Small analytical checks for supplemental evaluation pooling and GT strata."""
import unittest
import torch

from Modi01.evaluate import pixel_strata, summarize_strata


class EvaluationDiagnosticsTest(unittest.TestCase):
    def test_false_positive_and_missed_return_are_counted(self):
        t = lambda x: torch.tensor([[x]], dtype=torch.float32)
        row = pixel_strata(t([0.5, 0.8, 0.9]), t([1, 3, 2]), t([1, 0, 1]),
                           t([0.5, 0.8, 0]), t([1, 3, 0]), t([1, 1, 0]), scale=1)
        summary = summarize_strata([row])
        all_pixels = summary['all_pixels']
        self.assertEqual([all_pixels[k] for k in ('tp', 'fp', 'fn', 'tn')], [1, 1, 1, 0])
        self.assertAlmostEqual(all_pixels['return_f1'], 0.5)
        self.assertEqual(row['depth_edge']['pixels'], 2)
        self.assertEqual(row['range_0_10m']['pixels'], 2)
        self.assertIsNone(summary['range_50_80m']['depth_rmse_m'])

    def test_pooling_uses_pixel_counts_not_average_of_frame_rmse(self):
        a = {'x': dict(pixels=1, depth_sse=4, intensity_sse=0.25, tp=1, fp=0, fn=0, tn=0)}
        b = {'x': dict(pixels=3, depth_sse=0, intensity_sse=0, tp=1, fp=1, fn=1, tn=0)}
        pooled = summarize_strata([a, b])['x']
        self.assertEqual(pooled['depth_rmse_m'], 1)
        self.assertEqual(pooled['intensity_rmse'], 0.25)
        self.assertAlmostEqual(pooled['return_f1'], 2 / 3)


if __name__ == '__main__':
    unittest.main()
