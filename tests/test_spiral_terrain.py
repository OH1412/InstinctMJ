import copy
import unittest

import mujoco
import numpy as np

import instinct_mj.tasks  # noqa: F401
from instinct_mj.tasks.registry import load_env_cfg


class SpiralTerrainTest(unittest.TestCase):
    def test_both_turn_directions_and_difficulties(self):
        cfg = load_env_cfg('Instinct-Parkour-Target-Amp-G1-D455-SpiralTerrain-v0')
        for name, sign in [('spiral_left', 1), ('spiral_right', -1)]:
            for difficulty in [0., .5, 1.]:
                c = copy.deepcopy(cfg.scene.terrain.terrain_generator.sub_terrains[name])
                c.size = (20., 20.)
                spec = mujoco.MjSpec()
                spec.worldbody.add_body(name='terrain')
                output = c.function(difficulty, spec, np.random.default_rng(42))
                model = spec.compile()
                # Floor is first, four perimeter walls last. The complete
                # stair/rail/landing footprint must be centered in this tile.
                lo = np.full(2, np.inf)
                hi = np.full(2, -np.inf)
                for geom in output.geometries[1:-4]:
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, np.asarray(geom.geom.quat))
                    extent = np.abs(rotation.reshape(3, 3)) @ np.asarray(geom.geom.size)
                    center = np.asarray(geom.geom.pos)
                    lo = np.minimum(lo, (center - extent)[:2])
                    hi = np.maximum(hi, (center + extent)[:2])
                np.testing.assert_allclose((lo + hi) / 2, [10., 10.], atol=1e-6)
                self.assertTrue((lo > .2).all() and (hi < 19.8).all())
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                self.assertEqual(len(c.spawn_yaws), 4)
                self.assertEqual(len(c.ordered_waypoints_local), 5)
                heights = np.asarray(c.ordered_waypoints_local)[:, 2]
                np.testing.assert_allclose(np.diff(heights), 10 * (.05 + .15 * difficulty))
                np.testing.assert_allclose(np.diff(np.unwrap(c.spawn_yaws)), sign * np.pi / 2)
                self.assertAlmostEqual(c.spawn_yaws[-1], 0.)
                # At torso height the path from landing center toward the
                # next stair must not be blocked by a landing side-wall.
                for point, yaw in zip(c.spawn_points_local, c.spawn_yaws):
                    start = np.asarray(output.origin) + np.asarray(point)
                    start[2] += .7
                    direction = np.array([np.cos(yaw), np.sin(yaw), 0.])
                    geom_id = np.array([-1], dtype=np.int32)
                    distance = mujoco.mj_ray(model, data, start, direction, None, True, -1, geom_id)
                    self.assertTrue(distance < 0 or distance > c.corridor_width / 2 + .05)

    def test_old_task_unchanged(self):
        cfg = load_env_cfg('Instinct-Parkour-Target-Amp-G1-D455-LegacyHmvTerrain-v0')
        c = cfg.scene.terrain.terrain_generator.sub_terrains['corridor_left']
        self.assertEqual(c.component_types, ['u_turn'])
        self.assertFalse(c.spiral_quarter_turn)
        self.assertEqual(c.segment_count, 2)
        self.assertEqual(c.proportion, 0.)

    def test_close_spawns_on_walkable_landings(self):
        for suffix in ['LegacyHmvTerrain', 'SpiralTerrain']:
            cfg = load_env_cfg(f'Instinct-Parkour-Target-Amp-G1-D455-{suffix}-v0')
            for name in ['corridor_left', 'corridor_right']:
                for difficulty in [0., 1.]:
                    c = copy.deepcopy(cfg.scene.terrain.terrain_generator.sub_terrains[name])
                    self.assertEqual(c.initial_spawn_backoff, .6)
                    self.assertEqual(c.platform_spawn_clearance, .5)
                    c.size = (20., 20.)
                    spec = mujoco.MjSpec()
                    spec.worldbody.add_body(name='terrain')
                    output = c.function(difficulty, spec, np.random.default_rng(42))
                    model = spec.compile()
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)
                    for point in c.spawn_points_local:
                        start = np.asarray(output.origin) + np.asarray(point) + [0., 0., .1]
                        geom_id = np.array([-1], dtype=np.int32)
                        distance = mujoco.mj_ray(model, data, start, np.array([0., 0., -1.]), None, True, -1, geom_id)
                        self.assertAlmostEqual(distance, .1, places=6)
