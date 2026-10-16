import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch


class HeightProgressTest(unittest.TestCase):
    def test_signed_progress_and_lifecycle(self):
        path = Path(__file__).resolve().parents[1] / 'src/instinct_mj/tasks/parkour/mdp/rewards.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'corridor_target_height_exp')
        class Base:
            def __init__(self, env): pass
        ns = {'torch': torch, 'ManagerTermBase': Base,
              'SceneEntityCfg': lambda *a, **k: NS(name='robot', body_ids=[0, 1])}
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
        # Bodies 0/1 are hips; 2/3 are feet and must not affect height reward.
        pos = torch.zeros(3, 4, 3)
        cmd = NS(pos_command_w=torch.tensor([[0., 0., 1.], [0., 0., -1.], [0., 0., 1.]]), ordered_waypoint_count=torch.ones(3))
        env = NS(num_envs=3, device='cpu', common_step_counter=0,
                 episode_length_buf=torch.full((3,), 2), step_dt=.02,
                 scene={'robot': NS(data=NS(body_link_pos_w=pos))},
                 command_manager=NS(get_term=lambda name: cmd))
        term = ns[node.name](None, env)
        def run():
            return term(env, command_name='base_velocity', std=1., hip_height_offset=0.)
        self.assertTrue(torch.equal(run(), torch.zeros(3)))
        env.common_step_counter += 1
        pos[:, 2:, 2] = .2
        self.assertTrue(torch.equal(run(), torch.zeros(3)))
        env.common_step_counter += 1
        pos[0, :, 2] = .003
        pos[1, :, 2] = -.003
        reward = run().clone()
        self.assertTrue(torch.allclose(reward, torch.tensor([.5, .5, 0.]), atol=1e-5))
        self.assertTrue(torch.equal(run(), reward))
        env.common_step_counter += 1
        pos.zero_()
        self.assertTrue((run()[:2] < 0).all())
        term.reset(torch.tensor([0]))
        env.common_step_counter += 1
        self.assertEqual(run()[0], 0)
        cmd.pos_command_w[1, 2] = 2
        env.common_step_counter += 1
        self.assertEqual(run()[1], 0)
        pos[2, :, 2] = .96
        env.common_step_counter += 1
        self.assertGreater(run()[2], 0)
        self.assertTrue(term._completed[2])
        env.common_step_counter += 1
        self.assertEqual(run()[2], 0)
