"""Continuous-sequence TBPTT and the installed WasabiPPO/GAE/AMP machinery."""
from __future__ import annotations

from collections import namedtuple

import torch
from torch import nn
from instinct_rl.algorithms.wasabi import WasabiPPO
from instinct_rl.modules.actor_critic_recurrent import ActorCriticHiddenState
from instinct_rl.storage.rollout_storage import RolloutStorage


MemoryMiniBatch = namedtuple("MemoryMiniBatch", list(RolloutStorage.MiniBatch._fields) + ["teacher_obs"])


def distillation_losses(mean, aux, teacher_mean, teacher_visual, feature_weight=1.0,
                        normal_feature_weight=0.01, transition_weight=0.25):
    action = (mean - teacher_mean.detach()).square().mean()
    valid = aux["window_valid"].squeeze(-1).bool()
    # The first restored window is still a transition, not a request to copy
    # every valid CNN feature indefinitely.
    restored = torch.zeros_like(valid)
    if valid.ndim == 2 and valid.shape[0] > 1:
        restored[1:] = valid[1:] & ~valid[:-1]
    weight = torch.where(valid, normal_feature_weight, 1.0)
    weight = torch.where(restored, transition_weight, weight)
    feature = ((aux["memory_feature"] - teacher_visual.detach()).square().mean(-1) * weight).mean()
    return action, feature * feature_weight


class ContinuousMemoryStorage(RolloutStorage):
    """Rectangular TBPTT chunks, no time permutation or padding.

    Episodes may end inside a chunk: the policy resets from stored start bits.
    Each chunk loads its own collected pre-action hidden state. All envs and
    final short chunks are retained even when batch sizes do not divide evenly.
    """
    def __init__(self, *args, teacher_obs_dim, bptt_length=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.bptt_length = bptt_length
        self.teacher_observations = self.observations.new_zeros(
            self.num_transitions_per_env, self.num_envs, teacher_obs_dim)

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=1):
        if self.saved_hidden_states is None:
            raise RuntimeError("Missing collected GDU states")
        for _ in range(num_epochs):
            for ids in torch.tensor_split(torch.randperm(self.num_envs, device=self.device), num_mini_batches):
                if not len(ids):
                    continue
                for start in range(0, self.step, self.bptt_length):
                    sel = slice(start, min(self.step, start + self.bptt_length))
                    h = self.saved_hidden_states
                    hidden = ActorCriticHiddenState(h.actor[start, :, ids].contiguous(),
                                                    h.critic[start, :, ids].contiguous())
                    obs = self.observations[sel][:, ids]
                    critic = obs if self.critic_observations is None else self.critic_observations[sel][:, ids]
                    values = [self.actions, self.values, self.advantages, self.returns,
                              self.actions_log_prob, self.mu, self.sigma]
                    yield MemoryMiniBatch(obs, critic, *(v[sel][:, ids] for v in values), hidden,
                                          torch.ones(obs.shape[:2], dtype=torch.bool, device=self.device),
                                          self.teacher_observations[sel][:, ids])


class MemoryWasabiPPO(WasabiPPO):
    """Reuse installed PPO clipping, GAE, timeout bootstrap and AMP discriminator.

    Only sequence storage and extra supervised losses are changed. Teacher is
    frozen and evaluated on full observations from the same collected states.
    """
    def __init__(self, actor_critic, teacher, *, bptt_length=64, action_coef=1.0,
                 feature_coef=0.1, normal_feature_weight=0.01, distill_decay=1.0, **kwargs):
        super().__init__(actor_critic, **kwargs)
        self.teacher = teacher.eval().requires_grad_(False)
        self.bptt_length = bptt_length
        self.teacher_action_loss_coef = action_coef
        self.teacher_feature_loss_coef = feature_coef
        self.normal_feature_weight = normal_feature_weight
        self.distill_decay = distill_decay

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        if num_transitions_per_env != self.bptt_length or num_envs % self.num_mini_batches:
            raise ValueError("Installed PPO reduction requires horizon == BPTT and envs divisible by minibatches")
        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)
        old = self.storage
        self.storage = ContinuousMemoryStorage(
            num_envs, num_transitions_per_env, old.obs_shape, old.critic_obs_shape, old.actions_shape,
            num_rewards=num_rewards, device=self.device, teacher_obs_dim=self.actor_critic.raw_dim,
            bptt_length=self.bptt_length,
        )

    def compute_losses(self, minibatch):
        # Original PPO squeezes all singleton log-prob dimensions. Flatten the
        # loss-side tensors to [T*B,...] so B=1 cannot broadcast across time.
        flat = minibatch._replace(**{name: getattr(minibatch, name).reshape(
            -1, getattr(minibatch, name).shape[-1]) for name in (
                "critic_obs", "actions", "values", "advantages", "returns",
                "old_actions_log_prob", "old_mu", "old_sigma")})
        losses, inter, stats = super().compute_losses(flat)
        with torch.no_grad():
            full = minibatch.teacher_obs.reshape(-1, minibatch.teacher_obs.shape[-1])
            target_mean = self.teacher.act_inference(full).reshape(*minibatch.teacher_obs.shape[:2], -1)
            target_visual = self.teacher.encoders(minibatch.teacher_obs)[..., self.actor_critic.visual_slice]
        action, feature = distillation_losses(
            self.actor_critic.action_mean.reshape_as(target_mean), self.actor_critic.last_aux, target_mean, target_visual,
            normal_feature_weight=self.normal_feature_weight,
        )
        decay = self.distill_decay ** self.current_learning_iteration
        losses["teacher_action_loss"] = action * decay
        losses["teacher_feature_loss"] = feature * decay
        with torch.no_grad():
            sigma = self.actor_critic.action_std
            mu = self.actor_critic.action_mean
            stats["policy_kl"] = (torch.log(sigma / flat.old_sigma)
                + (flat.old_sigma.square() + (flat.old_mu - mu).square())
                / (2 * sigma.square()) - 0.5).sum(-1).mean()
        aux = self.actor_critic.last_aux
        stats.update(gdu_hidden_norm=aux["hidden"].detach().norm(dim=-1).mean(),
                     fusion_gate=aux["fusion_gate"].detach().mean(),
                     memory_feature_norm=aux["memory_feature"].detach().norm(dim=-1).mean(),
                     gdu_gate_mean=aux["group_gate"].detach().mean(),
                     gdu_group_update_mean=aux["group_gate"].detach().reshape(
                         *aux["group_gate"].shape[:-1], self.actor_critic.cfg.num_groups, -1).sum(-1).mean())
        return losses, inter, stats

    def gradient_step(self, loss, average_stats):
        # Existing method handles distributed reductions and total clipping.
        super().gradient_step(loss, average_stats)
        grads = [p.grad.detach().square().sum() for p in self.actor_critic.memory.parameters() if p.grad is not None]
        if grads:
            average_stats["gdu_grad_norm"] += torch.stack(grads).sum().sqrt()


def supervised_update(student, teacher, observations, full_observations, initial_hidden, optimizer,
                      action_coef=1.0, feature_coef=0.1, normal_feature_weight=0.01):
    """One contiguous TBPTT update. No loss can access the Teacher in forward."""
    means, _, aux = student.sequence(observations, initial_hidden.detach())
    with torch.no_grad():
        full = full_observations.reshape(-1, full_observations.shape[-1])
        labels = teacher.act_inference(full).reshape(*full_observations.shape[:2], -1)
        features = teacher.encoders(full_observations)[..., student.visual_slice]
    action, feature = distillation_losses(means, aux, labels, features,
                                         normal_feature_weight=normal_feature_weight)
    loss = action_coef * action + feature_coef * feature
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    total_norm = nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    gradients = [p.grad.detach().square().sum() for p in student.memory.parameters() if p.grad is not None]
    optimizer.step()
    return {"teacher_action_loss": float(action.detach()), "teacher_feature_loss": float(feature.detach()),
            "actor_grad_norm": float(total_norm),
            "gdu_grad_norm": float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0,
            "gdu_hidden_norm": float(aux["hidden"].detach().norm(dim=-1).mean()),
            "gdu_gate_mean": float(aux["group_gate"].detach().mean()),
            "gdu_group_update_mean": float(aux["group_gate"].detach().reshape(
                *aux["group_gate"].shape[:-1], student.cfg.num_groups, -1).sum(-1).mean()),
            "fusion_gate": float(aux["fusion_gate"].detach().mean()),
            "memory_feature_norm": float(aux["memory_feature"].detach().norm(dim=-1).mean())}
