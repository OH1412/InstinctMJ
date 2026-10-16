"""CPU regressions against the inspected real Hiking checkpoint; no simulator launch.

HIKING_TEACHER_CHECKPOINT may point to a later compatible completed Teacher.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch

from instinct_mj.rl.visual_memory.gdu import GDUCell
from instinct_mj.rl.visual_memory.masking import DepthValidityHistory, HeadYawVisualMask, VisualMaskConfig
from instinct_mj.rl.visual_memory.policy import DeploymentCore, MemoryConfig, VisualMemoryPolicy
from instinct_mj.rl.visual_memory.runtime import load_teacher, restore_memory_checkpoint, save_memory_checkpoint
from instinct_mj.rl.visual_memory.training import MemoryWasabiPPO, supervised_update

ROOT = Path(__file__).resolve().parents[1]
TEACHER = Path(os.environ.get("HIKING_TEACHER_CHECKPOINT", str(ROOT / "logs/instinct_rl/g1_d455_parkour/"
    "2026-10-15_13-22-02_height7_stop_at_5cm_resume66000/model_67000.pt")))


class VisualMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if not TEACHER.exists():
            raise unittest.SkipTest("Set HIKING_TEACHER_CHECKPOINT to the real checkpoint")
        cls.fmt = json.loads((ROOT / "configs/visual_memory/g1_d455_inspected_layout.json").read_text())
        cls.fmt = {g: {k: tuple(v) for k, v in d.items()} for g, d in cls.fmt.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            cls.teacher, cls.normalizers, cls.checkpoint, cls.agent = load_teacher(TEACHER, cls.fmt, 31)

    def setUp(self):
        torch.manual_seed(7)
        self.student = VisualMemoryPolicy(self.teacher, MemoryConfig(hidden_dim=32, num_groups=4))
        self.student.initialize_runtime(2, "cpu")

    def observations(self, n=2, valid=True, start=False):
        raw = torch.rand(n, self.student.raw_dim)
        meta = torch.zeros(n, 7)
        meta[:, :2] = float(valid)
        meta[:, 2:6] = torch.randn(n, 4) * 0.1
        meta[:, 6] = float(start)
        return torch.cat((raw, meta), -1)

    def test_01_strict_real_checkpoint_and_inheritance(self):
        self.assertEqual(self.student.base.state_dict().keys(), self.checkpoint["model_state_dict"].keys())
        for k, v in self.student.base.state_dict().items():
            self.assertTrue(torch.equal(v, self.checkpoint["model_state_dict"][k]), k)
        self.assertEqual(self.student.encoded_dim, 928)
        self.assertEqual(self.student.context_dim, 800)
        self.assertEqual(self.student.visual_dim, 128)

    def test_02_initial_action_equivalence(self):
        obs = self.observations()
        with torch.no_grad():
            old = self.teacher.act_inference(obs[:, :self.student.raw_dim])
            new, _, _ = self.student.step(obs, self.student._hidden)
        torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_03_gdu_original_equation(self):
        cell = GDUCell(5, 8, 2, (0.5, 3.0))
        x, h = torch.randn(3, 5), torch.randn(3, 8)
        out, a = cell(x, h)
        joined = torch.cat((x, h), -1)
        torch.testing.assert_close(out, (1 - a) * h + a * cell.candidate(joined).tanh())

    def test_04_group_budgets_both_branches(self):
        cell = GDUCell(3, 16, 4, (0.2, 1.0, 2.0, 3.5))
        for scale in (1, 100, 1000):
            a = cell.gates(torch.randn(7, 19) * scale).reshape(7, 4, 4)
            torch.testing.assert_close(a.sum(-1), cell.budgets.expand(7, -1))
            self.assertTrue(((a >= 0) & (a <= 1)).all())

    def test_05_hidden_continuity(self):
        obs = self.observations()
        _, h1, _ = self.student.step(obs, self.student._hidden)
        _, h2, _ = self.student.step(obs, h1)
        self.assertFalse(torch.equal(h1, h2))

    def test_06_done_resets_only_one_env(self):
        self.student._hidden.fill_(1)
        self.student.reset(torch.tensor([True, False]))
        self.assertEqual(float(self.student._hidden[0].sum()), 0)
        self.assertEqual(float(self.student._hidden[1].sum()), 32)

    def test_07_mask_yaw_eligibility_and_time_budget(self):
        m = HeadYawVisualMask(VisualMaskConfig(trigger_rate_hz=1e6), 2, .02)
        ever = torch.zeros(2, dtype=torch.bool)
        for _ in range(400):
            valid = m.step(torch.tensor([0.0, 1.0]))
            ever |= ~valid
            self.assertTrue((m.mask_total_duration <= .5 * m.episode_elapsed_time + 1e-5).all())
        self.assertTrue(ever[0]); self.assertFalse(ever[1])

    def test_08_hysteresis_and_exit(self):
        m = HeadYawVisualMask(VisualMaskConfig(trigger_rate_hz=1e6), 1, .02)
        for _ in range(60):
            m.step(torch.zeros(1))
            if m.mask_active.item():
                break
        self.assertTrue(m.mask_active.item())
        m.step(torch.tensor([18 * torch.pi / 180]))
        self.assertTrue(m.front_view_state.item())
        self.assertTrue(m.step(torch.tensor([21 * torch.pi / 180])).item())
        self.assertFalse(m.mask_active.item())

    def test_09_mask_does_not_mutate_physical_or_full_data(self):
        obs = self.observations(valid=False)
        original = obs.clone()
        self.student.step(obs, self.student._hidden)
        self.assertTrue(torch.equal(original, obs))

    def test_10_invalid_rows_never_reach_cnn(self):
        calls = []
        handle = self.student.base.encoders.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
        self.student.step(self.observations(valid=False), self.student._hidden)
        self.assertEqual(calls, [])
        handle.remove()

    def test_11_teacher_student_no_visual_leakage(self):
        a = self.observations(valid=False)
        b = a.clone(); b[:, self.student.depth_slice] = torch.randn_like(b[:, self.student.depth_slice]) * 1e6
        aa, ha, _ = self.student.step(a, self.student._hidden)
        ab, hb, _ = self.student.step(b, self.student._hidden)
        torch.testing.assert_close(aa, ab, rtol=0, atol=0)
        torch.testing.assert_close(ha, hb, rtol=0, atol=0)

    def test_12_history_refill_and_delay(self):
        h = DepthValidityHistory(2)
        for _ in range(37): h.append(torch.ones(2, dtype=torch.bool))
        self.assertTrue(h.sample(torch.tensor([0, 1])).all())
        h.append(torch.tensor([False, True]))
        self.assertFalse(h.sample(torch.zeros(2))[0].all())
        for _ in range(35): h.append(torch.ones(2, dtype=torch.bool))
        self.assertFalse(h.sample(torch.zeros(2))[0].all())
        h.append(torch.ones(2, dtype=torch.bool))
        self.assertTrue(h.sample(torch.zeros(2)).all())
        self.assertFalse(h.sample(torch.ones(2))[0].all())

    def test_13_masked_memory_receives_proprio(self):
        a = self.observations(valid=False); b = a.clone(); b[:, :24] += 1
        _, ha, _ = self.student.step(a, self.student._hidden)
        _, hb, _ = self.student.step(b, self.student._hidden)
        self.assertFalse(torch.equal(ha, hb))

    def test_14_normal_memory_changes_body_actions_but_not_head(self):
        obs = self.observations()
        old, _, _ = self.student.step(obs, self.student._hidden)
        with torch.no_grad(): self.student.projection.weight.normal_(std=.01)
        new, _, _ = self.student.step(obs, self.student._hidden)
        self.assertFalse(torch.equal(old[:, :-2], new[:, :-2]))
        torch.testing.assert_close(old[:, -2:], new[:, -2:], rtol=0, atol=0)

    def test_15_normal_vision_gradients_after_projection_warmup(self):
        with torch.no_grad(): self.student.projection.weight.normal_(std=.01)
        out, _, _ = self.student.step(self.observations(), self.student._hidden)
        out[:, :-2].square().mean().backward()
        self.assertGreater(float(self.student.memory.cell.candidate.weight.grad.norm()), 0)
        self.assertGreater(float(self.student.fusion_gate[0].weight.grad.norm()), 0)

    def test_16_recovery_keeps_hidden(self):
        _, h, _ = self.student.step(self.observations(valid=False), self.student._hidden)
        _, recovered, _ = self.student.step(self.observations(valid=True), h)
        _, cold, _ = self.student.step(self.observations(valid=True), torch.zeros_like(h))
        self.assertFalse(torch.equal(recovered, cold))

    def make_ppo(self, steps=4):
        cfg = copy.deepcopy(self.agent["algorithm"])
        cfg.pop("class_name")
        cfg = {k: v for k, v in cfg.items() if v is not None}
        cfg.update(num_learning_epochs=1, num_mini_batches=1)
        with contextlib.redirect_stdout(io.StringIO()):
            alg = MemoryWasabiPPO(self.student, self.teacher, bptt_length=steps, device="cpu", **cfg)
            fmt = copy.deepcopy(self.fmt); fmt["policy"] = self.student.obs_segments
            alg.init_storage(2, steps, fmt, 31)
        alg.discriminator.load_state_dict(self.checkpoint["discriminator"], strict=True)
        return alg

    def fill_ppo(self, alg):
        critic_dim = sum(__import__('math').prod(x) for x in self.fmt['critic'].values())
        with torch.no_grad():
            for t in range(alg.storage.num_transitions_per_env):
                obs = self.observations(start=t==0)
                if t == 2: obs[0, -1] = 1
                critic = torch.rand(2, critic_dim)
                alg.storage.teacher_observations[t].copy_(obs[:, :self.student.raw_dim])
                alg.act(obs, critic)
                done = torch.tensor([t==1, False]).long()
                info = {'observations': {'amp_policy': torch.rand(2, 670), 'amp_reference': torch.rand(2, 670)},
                        'step': {}, 'time_outs': torch.zeros(2)}
                alg.process_env_step(torch.rand(2,1), done, info, obs, critic)
            alg.compute_returns(critic)

    def test_17_ppo_contiguous_sequence_and_saved_hidden(self):
        alg = self.make_ppo(); self.fill_ppo(alg)
        batch = next(alg.storage.recurrent_mini_batch_generator(1,1))
        self.assertEqual(batch.obs.shape[:2], (4,2))
        torch.testing.assert_close(batch.obs, alg.storage.observations)
        torch.testing.assert_close(batch.hidden_states.actor, alg.storage.saved_hidden_states.actor[0])

    def test_18_ppo_old_log_prob_and_done_alignment(self):
        alg = self.make_ppo(); self.fill_ppo(alg)
        batch = next(alg.storage.recurrent_mini_batch_generator(1,1))
        with torch.no_grad():
            self.student.act(batch.obs, masks=batch.masks, hidden_states=batch.hidden_states.actor)
            replay = self.student.get_actions_log_prob(batch.actions)
        torch.testing.assert_close(replay, batch.old_actions_log_prob.squeeze(-1), rtol=1e-5, atol=1e-4)
        self.assertTrue(batch.obs[2,0,-1].item())

    def test_19_20_synthetic_distillation_losses_decrease(self):
        optimizer = torch.optim.Adam([p for p in self.student.parameters() if p.requires_grad], lr=.003)
        raw = self.observations()
        full = raw[:, :self.student.raw_dim].unsqueeze(0).repeat(8,1,1)
        seq = raw.unsqueeze(0).repeat(8,1,1)
        seq[3:,:,self.student.raw_dim:self.student.raw_dim+2] = 0
        first = None
        for _ in range(20):
            loss = supervised_update(self.student,self.teacher,seq,full,self.student.initial_state(2),optimizer,
                                     feature_coef=1.0,normal_feature_weight=.01)
            if first is None: first=loss
        self.assertLess(loss['teacher_feature_loss'],first['teacher_feature_loss'])
        self.assertLess(loss['teacher_action_loss'],first['teacher_action_loss'])

    def test_21_student_autonomous_synthetic_feedback_loop(self):
        obs = self.observations(valid=False)
        for _ in range(20):
            with torch.no_grad(): action = self.student.act_inference(obs)
            obs[:, :3] += .01 * action[:, :3]
            self.assertTrue(torch.isfinite(action).all())
        self.assertIsNone(self.student._hidden.grad_fn)

    def test_22_checkpoint_roundtrip_and_schema_rejection(self):
        optimizer=torch.optim.Adam([p for p in self.student.parameters() if p.requires_grad])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'student.pt'
            save_memory_checkpoint(path,self.student,optimizer,7,{'stage':'distill'},TEACHER)
            out=restore_memory_checkpoint(path,self.student,optimizer)
            self.assertEqual(out['iteration'],7)
            with self.assertRaises(FileExistsError):
                save_memory_checkpoint(path,self.student,optimizer,7,{},TEACHER)
            other=VisualMemoryPolicy(self.teacher,MemoryConfig(hidden_dim=64,num_groups=4))
            with self.assertRaises(ValueError): restore_memory_checkpoint(path,other)

    def test_23_export_core_outputs_match_valid_and_masked(self):
        from instinct_mj.scripts.visual_memory import export_artifacts
        with tempfile.TemporaryDirectory() as d:
            core,example=export_artifacts(self.student,Path(d)/'export')
            with torch.no_grad():
                for valid in (0.,1.):
                    inp=list(example);inp[4]=torch.full_like(inp[4],valid);inp[5]=torch.randn_like(inp[5])
                    eager=DeploymentCore(self.student)(*inp); exported=core(*inp)
                    for a,b in zip(eager,exported): torch.testing.assert_close(a,b)

    def test_24_no_collection_autograd_accumulation(self):
        obs=self.observations()
        for _ in range(100):
            self.student.act_inference(obs)
            self.assertIsNone(self.student._hidden.grad_fn)

    def test_25_actual_wasabi_ppo_amp_update(self):
        alg=self.make_ppo();self.fill_ppo(alg)
        losses,stats=alg.update(0)
        for k,v in losses.items():self.assertTrue(torch.isfinite(torch.as_tensor(v)).all(),k)
        self.assertIn('teacher_action_loss',losses)
        self.assertIn('discriminator_loss',losses)
        self.assertIn('policy_kl',stats)

    def test_26_padding_does_not_change_hidden(self):
        seq=self.observations().unsqueeze(0).repeat(3,1,1)
        mask=torch.tensor([[True,True],[False,True],[False,True]])
        _,h,_=self.student.sequence(seq,self.student.initial_state(2),mask)
        _,h0,_=self.student.step(seq[0],self.student.initial_state(2))
        torch.testing.assert_close(h[0],h0[0])

    def test_27_all_cli_stages_on_synthetic_vector_environment(self):
        """Exercise real CLI loops/IO, not a MuJoCo performance evaluation."""
        from instinct_mj.scripts.visual_memory import parse_args, run
        fmt = self.fmt
        raw_dim = self.student.raw_dim

        class SyntheticVecEnv:
            def __init__(self):
                self.num_envs, self.num_actions, self.num_rewards, self.device = 2, 31, 1, torch.device('cpu')
                self.unwrapped = self
                self.step_dt = .02
                self.camera = SimpleNamespace(frame=torch.zeros(2,dtype=torch.long),
                    cfg=SimpleNamespace(data_histories={'distance_to_image_plane_noised':37},update_period=0.0))
                data = SimpleNamespace(joint_pos=torch.zeros(2,31),joint_vel=torch.zeros(2,31),
                    root_link_pos_w=torch.zeros(2,3),body_link_pos_w=torch.zeros(2,2,3))
                data.body_link_pos_w[:,:,2] = .035
                robot = SimpleNamespace(data=data,
                    find_joints=lambda names,**kw:([29,30],list(names)),
                    find_bodies=lambda names,**kw:([0,1],list(names)))
                self.scene={'robot':robot,'camera':self.camera}
                self.observation_manager=SimpleNamespace(get_term_cfg=lambda *a:SimpleNamespace(
                    params={'sensor_cfg':SimpleNamespace(name='camera'),'num_output_frames':8,'history_skip_frames':5},
                    func=SimpleNamespace(_num_delayed_frames=torch.zeros(2),history_skip_frames=5,num_output_frames=8)))
                self.command_manager=SimpleNamespace(get_term=lambda n:SimpleNamespace(
                    pos_command_w=torch.tensor([[1.,0.,0.]]*2),cfg=SimpleNamespace(target_dis_threshold=.2)))
                self.termination_manager=SimpleNamespace(active_terms=[])
                self.t=0
            def get_obs_format(self):return copy.deepcopy(fmt)
            def reset(self):
                self.t=0
                return torch.rand(2,raw_dim),{'observations':{'critic':torch.rand(2,raw_dim+24)}}
            def step(self,action):
                self.t+=1
                self.scene['robot'].data.root_link_pos_w[:,:2] += .001*action[:,:2]
                self.camera.frame+=1
                done=torch.tensor([self.t%32==0,self.t%48==0]).long()
                return torch.rand(2,raw_dim),torch.rand(2,1),done,{
                    'observations':{'critic':torch.rand(2,raw_dim+24),'amp_policy':torch.rand(2,670),
                                    'amp_reference':torch.rand(2,670)},'time_outs':done,'step':{}}
            def close(self):pass

        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            student_path=None
            for stage in ('distill','dagger','ppo','evaluate'):
                argv=['visual_memory','--stage',stage,'--teacher',str(TEACHER),'--output',d,
                      '--device','cpu','--num-envs','2','--iterations','1','--save-interval','1',
                      '--hidden-dim','32','--groups','4']
                if student_path:argv+=['--student',str(student_path)]
                # Real saved algorithm requests four minibatches; two fake
                # envs deliberately require a small algorithm config here.
                source_loader=__import__('instinct_mj.scripts.visual_memory',fromlist=['load_teacher']).load_teacher
                def loader(*a,**kw):
                    teacher,normalizers,data,cfg=source_loader(*a,**kw)
                    cfg['algorithm']['num_mini_batches']=1
                    cfg['algorithm']['num_learning_epochs']=1
                    return teacher,normalizers,data,cfg
                with patch('sys.argv',argv), patch('instinct_mj.scripts.visual_memory.build_environment',
                     side_effect=lambda args:(SyntheticVecEnv(),{})), patch('instinct_mj.scripts.visual_memory.load_teacher',loader):
                    run(parse_args())
                if stage!='evaluate':
                    student_path=next((Path(d)/stage).glob('*/student_1.pt'))
                    self.assertTrue(student_path.exists())
            self.assertTrue(list((Path(d)/'evaluate').glob('*/evaluation.json')))

    def test_28_different_visual_histories_same_masked_state(self):
        # Mechanistic history-dependence check, not a terrain success claim.
        with torch.no_grad(): self.student.projection.weight.normal_(std=.02)
        a = self.observations(); b = a.clone()
        a[:, self.student.depth_slice] = 0.1
        b[:, self.student.depth_slice] = 0.9
        _, ha, _ = self.student.step(a, self.student.initial_state(2))
        _, hb, _ = self.student.step(b, self.student.initial_state(2))
        masked = self.observations(valid=False)
        ma, _, _ = self.student.step(masked, ha)
        mb, _, _ = self.student.step(masked, hb)
        cold, _, _ = self.student.step(masked, self.student.initial_state(2))
        self.assertFalse(torch.equal(ma[:, :-2], mb[:, :-2]))
        self.assertFalse(torch.equal(ma[:, :-2], cold[:, :-2]))

    def test_29_default_256_dim_layout(self):
        model=VisualMemoryPolicy(self.teacher,MemoryConfig())
        self.assertEqual(model.memory.cell.hidden_dim,256)
        self.assertEqual(model.memory.cell.num_groups,16)
        self.assertEqual(model.memory.cell.candidate.in_features,1190)
        self.assertEqual(model.fusion_gate[0].in_features,1057)
        self.assertEqual(model.projection.out_features,128)

    def test_30_ppo_single_env_minibatch_no_time_broadcast(self):
        alg=self.make_ppo();self.fill_ppo(alg)
        batch=next(alg.storage.recurrent_mini_batch_generator(2,1))
        alg.current_learning_iteration=0
        losses,inter,stats=alg.compute_losses(batch)
        self.assertEqual(tuple(inter['ratio'].shape),(4,))
        self.assertTrue(torch.isfinite(losses['surrogate_loss']))

    def test_31_student_feedback_separate_mask_and_history_states(self):
        from instinct_mj.rl.visual_memory.feedback import feedback_state
        perception=SimpleNamespace(sensor_valid=torch.tensor([False,True,True]),
            frame_valid=torch.tensor([[False]*8,[False]+[True]*7,[True]*8]))
        self.assertEqual(feedback_state(perception,0)['status'],'Sensor Mask')
        self.assertEqual(feedback_state(perception,1)['status'],'CNN History Invalid')
        self.assertEqual(feedback_state(perception,2)['status'],'Visual Valid')
        self.assertFalse(feedback_state(perception,0)['student_visible'])
        self.assertFalse(feedback_state(perception,1)['student_visible'])
        self.assertTrue(feedback_state(perception,2)['student_visible'])

    def test_32_feedback_hides_stale_student_preserves_original_and_switches_env(self):
        from instinct_mj.rl.visual_memory.feedback import StudentCameraFeedback
        class Gui:
            def add_folder(self,*a,**kw):return contextlib.nullcontext()
            def add_html(self,content):return SimpleNamespace(content=content)
            def add_image(self,image,**kw):return SimpleNamespace(image=image.copy(),visible=True)
        depth=torch.full((2,8,18,32),.75)
        depth[1]=.25
        perception=SimpleNamespace(feedback_depth=depth,sensor_valid=torch.tensor([True,False]),
                                   frame_valid=torch.ones(2,8,dtype=torch.bool))
        panel=StudentCameraFeedback(SimpleNamespace(gui=Gui()),perception)
        panel.update(0)
        self.assertTrue(panel.student.visible)
        original0=panel.original.image.copy()
        panel.update(1)
        self.assertTrue(panel.original.visible)
        self.assertFalse(panel.student.visible)
        self.assertEqual(int(panel.student.image.sum()),0)
        self.assertFalse((original0==panel.original.image).all())
        self.assertIn('Robot 1',panel.status.content)
        perception.sensor_valid[1]=True
        perception.frame_valid[1,0]=False
        panel.update(1)
        self.assertIn('CNN History Invalid',panel.status.content)
        self.assertFalse(panel.student.visible)
        perception.frame_valid[1]=True
        panel.update(1)
        self.assertTrue(panel.student.visible)


if __name__ == '__main__':
    unittest.main()
