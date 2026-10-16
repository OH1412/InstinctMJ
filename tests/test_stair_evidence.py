"""CPU geometry tests without importing the simulator or allocating a GPU."""
import ast
from pathlib import Path
import unittest

import torch


source = Path(__file__).resolve().parents[1] / "src/instinct_mj/tasks/parkour/mdp/rewards.py"
node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "stair_tread_pixels")
namespace = {"torch": torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
detect = namespace["stair_tread_pixels"]


class StairEvidenceTest(unittest.TestCase):
    def scene(self):
        points = torch.zeros(1, 6, 16, 3)
        points[..., 0] = torch.arange(6).view(1, 6, 1) * .15
        points[..., 1] = torch.arange(16).view(1, 1, 16) * .04
        return points, torch.ones(1, 6, 16), torch.ones(1, 6, 16, dtype=torch.bool)

    def test_stairs_both_directions(self):
        for sign in (-1, 1):
            p, n, v = self.scene()
            p[..., 2] = sign * torch.tensor([0, 0, .05, .05, .10, .10]).view(1, 6, 1)
            self.assertTrue(detect(p, n, v).any())

    def test_flat_and_single_elevated_surface(self):
        p, n, v = self.scene()
        p[..., 2] = 1.
        self.assertFalse(detect(p, n, v).any())

    def test_narrow_slanted_rail_even_with_upward_normals(self):
        p, n, v = self.scene()
        p[..., 1] *= .05
        p[..., 2] = torch.arange(6).view(1, 6, 1) * .05
        self.assertFalse(detect(p, n, v).any())

    def test_vertical_poles(self):
        p, n, v = self.scene()
        n.zero_()
        self.assertFalse(detect(p, n, v).any())

    def test_ramp(self):
        p, n, v = self.scene()
        p[..., 2] = torch.arange(6).view(1, 6, 1) * .05
        n.fill_(.94)
        self.assertFalse(detect(p, n, v).any())


if __name__ == "__main__":
    unittest.main()
