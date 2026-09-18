"""Refinement must detect changes to field tensors and representation metadata."""
import unittest
from types import SimpleNamespace
import torch
from Modi01.refine import check_fixed_field


class FrozenFieldTest(unittest.TestCase):
    def test_tensor_and_extra_state_roundtrip(self):
        expected = {'field.weight': torch.tensor([1., 2.]), 'field._extra_state': {'representation': 'tied'}}
        current = {'field.weight': expected['field.weight'].clone(), 'field._extra_state': {'representation': 'tied'}}
        check_fixed_field(torch, SimpleNamespace(state_dict=lambda: current), expected)
        current['field.weight'][0] = 0
        with self.assertRaisesRegex(RuntimeError, 'field.weight'):
            check_fixed_field(torch, SimpleNamespace(state_dict=lambda: current), expected)

    def test_metadata_change_is_rejected(self):
        expected = {'field._extra_state': {'representation': 'tied'}}
        current = {'field._extra_state': {'representation': 'untied'}}
        with self.assertRaisesRegex(RuntimeError, 'field._extra_state'):
            check_fixed_field(torch, SimpleNamespace(state_dict=lambda: current), expected)


if __name__ == '__main__':
    unittest.main()
