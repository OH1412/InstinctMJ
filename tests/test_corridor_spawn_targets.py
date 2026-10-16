import unittest
from types import SimpleNamespace as NS

import torch

from instinct_mj.tasks.parkour.mdp.commands.pose_velocity_command import PoseVelocityCommand
from instinct_mj.tasks.parkour.mdp.events import reset_root_state_uniform_with_terrain_spawn


class CorridorSpawnTargetTest(unittest.TestCase):
    def term(self, count):
        n = count - 1
        targets = torch.zeros(count, 3)
        targets[:, 2] = torch.arange(count) * .5
        return NS(
            num_envs=n, device='cpu',
            _env=NS(_corridor_spawn_layer=torch.arange(n)),
            cfg=NS(target_dis_threshold=.4, corridor_target_layer_offsets=(1, 2), rel_standing_envs=0.),
            valid_targets=targets[None, None], pos_command_w=torch.zeros(n, 3),
            terrain=NS(terrain_levels=torch.zeros(n, dtype=torch.long), terrain_types=torch.zeros(n, dtype=torch.long)),
            ordered_waypoint_count=torch.full((n,), count), ordered_waypoints_w=targets[None].repeat(n, 1, 1),
            ordered_waypoint_index=torch.zeros(n, dtype=torch.long), command_counter=torch.zeros(n, dtype=torch.long),
            # All xpos values deliberately stale at ground, including upper resets.
            robot=NS(data=NS(root_link_pos_w=torch.tensor([[0., 0., .82]]).repeat(n, 1),
                             default_root_state=torch.tensor([[0., 0., .82]]).repeat(n, 1))),
            max_command_b=torch.zeros(n, 3), is_standing_env=torch.zeros(n, dtype=torch.bool),
            random_velocity_indices=torch.zeros(n, dtype=torch.bool),
            lin_vel_x_range=torch.tensor([[.45, .8]]).repeat(n, 1), lin_vel_y_range=torch.zeros(n, 2),
            ang_vel_z_range=torch.tensor([[-1., 1.]]).repeat(n, 1),
        )

    def test_stale_xpos_cannot_select_same_or_lower_platform(self):
        for count in (4, 6):
            term = self.term(count)
            for seed in range(30):
                torch.manual_seed(seed)
                PoseVelocityCommand._resample_command(term, torch.arange(term.num_envs))
                self.assertTrue((term.pos_command_w[:, 2] > torch.arange(term.num_envs) * .5).all())
                difference = term.ordered_waypoint_index - term._env._corridor_spawn_layer
                self.assertTrue(((difference == 1) | (difference == 2)).all())

    def test_missing_metadata_fails_instead_of_guessing(self):
        term = self.term(4)
        del term._env._corridor_spawn_layer
        with self.assertRaisesRegex(RuntimeError, 'recorded reset layer'):
            PoseVelocityCommand._resample_command(term, torch.arange(term.num_envs))

    def test_reset_records_height_matching_layer_and_partial_reset(self):
        term = self.term(6)
        n = term.num_envs
        sub = NS(ordered_waypoints_local=[[0., 0., z] for z in (.5, 1., 1.5, 2., 2.5)],
                 spawn_points_local=[[0., 0., z] for z in (.5, 1., 1.5, 2.)], spawn_yaws=[0.] * 4)
        captured = {}
        root = torch.zeros(n, 13)
        root[:, 2] = .82
        root[:, 3] = 1.
        robot = NS(data=NS(default_root_state=root),
                   write_root_link_pose_to_sim=lambda pose, **kw: captured.update(pose=pose),
                   write_root_link_velocity_to_sim=lambda *a, **kw: None)
        terrain = NS(terrain_levels=torch.zeros(n, dtype=torch.long), terrain_types=torch.zeros(n, dtype=torch.long),
                     terrain_generator=NS(get_subterrain_cfg=lambda levels, types: [sub] * len(levels)))
        class Scene(dict): pass
        scene = Scene(robot=robot, terrain=terrain)
        scene.env_origins = torch.zeros(n, 3)
        env = NS(num_envs=n, device='cpu', scene=scene)
        reset_root_state_uniform_with_terrain_spawn(env, None, {}, {}, corridor_spawn_ratio=1.)
        actual_layer = torch.round((captured['pose'][:, 2] - .82) / .5).long()
        self.assertTrue(torch.equal(env._corridor_spawn_layer, actual_layer))
        before = env._corridor_spawn_layer.clone()
        reset_root_state_uniform_with_terrain_spawn(env, torch.tensor([1]), {}, {}, corridor_spawn_ratio=0.)
        self.assertEqual(env._corridor_spawn_layer[1], 0)
        self.assertTrue(torch.equal(env._corridor_spawn_layer[[0, 2, 3, 4]], before[[0, 2, 3, 4]]))
