import os
import unittest
from importlib.resources import files

from neuron_morphology.constants import AXON, BASAL_DENDRITE

from src.tree_comparison.tree_compare import compare_two_trees


class TestTreeCompare(unittest.TestCase):
    def setUp(self):
        tree1 = os.path.join(os.path.dirname(__file__), "test_data", "swc_dir", "601506507.swc")
        tree2 = os.path.join(os.path.dirname(__file__), "test_data", "ref_dir", "898703349.swc")
        self.valid_set_dict = os.path.join(
            str(files("tree_comparison") / "data"), "validSet_mouse_inh_viz_ctx.json"
        )

        self.test_cases = [
            {
                "params": {
                    "tree1": tree1,
                    "tree2": tree2,
                    "compartments": [AXON],
                    "simFunc": "convex",
                    "maxDepth": 2,
                    "orientation": 0,
                },
                "expected_distance": 6516.7242,
            },
            {
                "params": {
                    "tree1": tree1,
                    "tree2": tree2,
                    "compartments": [BASAL_DENDRITE],
                    "simFunc": "convex",
                    "maxDepth": 2,
                    "orientation": 0,
                },
                "expected_distance": 4340.9097,
            },
            {
                "params": {
                    "tree1": tree1,
                    "tree2": tree2,
                    "compartments": [BASAL_DENDRITE],
                    "simFunc": "convex",
                    "maxDepth": 2,
                    "orientation": 180,
                },
                "expected_distance": 4624.9772,
            },
            {
                "params": {
                    "tree1": tree1,
                    "tree2": tree2,
                    "compartments": [BASAL_DENDRITE],
                    "simFunc": "convex",
                    "maxDepth": 1,
                    "orientation": 0,
                },
                "expected_distance": 4362.4997,
            },
            {
                "params": {
                    "tree1": tree1,
                    "tree2": tree2,
                    "compartments": [BASAL_DENDRITE],
                    "simFunc": "length",
                    "maxDepth": 2,
                    "orientation": 0,
                },
                "expected_distance": 2921.645,
            },
            {
                "params": {
                    "tree1": tree1,
                    "tree2": tree2,
                    "compartments": [BASAL_DENDRITE],
                    "simFunc": "length",
                    "maxDepth": 1,
                    "orientation": 0,
                },
                "expected_distance": 3060.1586,
            },
            {
                "params": {
                    "tree1": tree1,
                    "tree2": tree1,
                    "compartments": [AXON],
                    "simFunc": "convex",
                    "maxDepth": 2,
                    "orientation": 0,
                },
                "expected_distance": 0,
            },
            {
                "params": {
                    "tree1": tree1,
                    "tree2": tree1,
                    "compartments": [BASAL_DENDRITE],
                    "simFunc": "length",
                    "maxDepth": 2,
                    "orientation": 0,
                },
                "expected_distance": 0,
            },
        ]

    def test_compare_two_trees(self):
        for case in self.test_cases:
            with self.subTest(**case["params"]):
                distance, *_ = compare_two_trees(
                    swc_file_1=case["params"]["tree1"],
                    swc_file_2=case["params"]["tree2"],
                    compartments=case["params"]["compartments"],
                    simFunc=case["params"]["simFunc"],
                    maxDepth=case["params"]["maxDepth"],
                    orientation=case["params"]["orientation"],
                    valid_set_dict=self.valid_set_dict,
                    partition_length=0.0005,
                    downsample_spacing=5.0,
                    relative_branch_check=False,
                    relative_branch_threshold=-1,
                )

                # Tree distance as expected?
                self.assertAlmostEqual(distance, case["expected_distance"], places=2)


if __name__ == "__main__":
    unittest.main()
