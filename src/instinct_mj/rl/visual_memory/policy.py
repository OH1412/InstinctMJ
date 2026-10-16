"""Keep the pretrained EncoderMoEActorCritic intact behind a GDU residual."""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.distributions import Normal
from instinct_rl.modules.actor_critic_recurrent import ActorCriticHiddenState

from .gdu import GDUMemory, MemoryProjection


@dataclass
class MemoryConfig:
    hidden_dim: int = 256
    num_groups: int = 16
    update_budget: float | tuple[float, ...] = 1.0
    fusion_hidden_dim: int = 128
    fusion_mode: str = "continuous"  # fallback_only is the E ablation
    freeze_head: bool = True
    head_action_count: int = 2


def segments_layout(segments):
    result, offset = {}, 0
    for name, shape in segments.items():
        size = math.prod(shape)
        result[name] = slice(offset, offset + size)
        offset += size
    return result, offset


class VisualMemoryPolicy(nn.Module):
    """Metadata is sensor-valid, window-valid, raw head q/dq (4), episode-start.

    Original policy observations and metadata remain distinct until the input
    boundary. Teacher labels are never accepted by this forward interface.
    """
    is_recurrent = True
    metadata_dim = 7

    def __init__(self, pretrained: nn.Module, cfg: MemoryConfig):
        super().__init__()
        self.base = copy.deepcopy(pretrained)
        self.cfg = cfg
        self.raw_segments = copy.deepcopy(pretrained.obs_segments)
        self.layout, self.raw_dim = segments_layout(self.raw_segments)
        self.depth_slice = self.layout["depth_image"]
        output = self.base.encoders.output_segment
        self.encoded_layout, self.encoded_dim = segments_layout(output)
        latent_names = [k for k in output if k.startswith("parallel_latent_")]
        if len(latent_names) != 1 or "depth_image" in output:
            raise ValueError("This adapter requires one depth-only encoder with input removed")
        self.visual_slice = self.encoded_layout[latent_names[0]]
        if self.visual_slice.stop != self.encoded_dim:
            raise ValueError("Depth latent must be last in the real ParallelLayer output")
        self.visual_dim = self.visual_slice.stop - self.visual_slice.start
        self.context_dim = self.visual_slice.start
        self.memory = GDUMemory(self.visual_dim + self.context_dim + 4 + 2,
                                cfg.hidden_dim, cfg.num_groups, cfg.update_budget)
        self.projection = MemoryProjection(cfg.hidden_dim, self.visual_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * self.visual_dim + self.context_dim + 1, cfg.fusion_hidden_dim), nn.ELU(),
            nn.Linear(cfg.fusion_hidden_dim, self.visual_dim), nn.Sigmoid(),
        )
        if cfg.fusion_mode not in ("continuous", "fallback_only"):
            raise ValueError("Unknown fusion ablation")
        # Preserve the existing head policy and its inputs. Memory learning
        # cannot alter head weights or send gradients through head actions.
        self.head_actor = copy.deepcopy(self.base.actor) if cfg.freeze_head else None
        self._hidden = None
        self.distribution = None
        self.last_aux = {}
        self.freeze_legacy()

    @property
    def obs_segments(self):
        return {**self.raw_segments, "visual_memory_metadata": (self.metadata_dim,)}

    def freeze_legacy(self, train_gate=False, train_experts=False, train_cnn=False):
        self.base.requires_grad_(False)
        if train_gate:
            self.base.actor.gate.requires_grad_(True)
        if train_experts:
            self.base.actor.experts.requires_grad_(True)
        if train_cnn:
            self.base.encoders.requires_grad_(True)
        if self.head_actor is not None:
            self.head_actor.requires_grad_(False)

    def initial_state(self, n, device=None):
        return self.memory.initial_state(n, device)

    def reset(self, dones=None):
        if self._hidden is None:
            return
        if dones is None:
            self._hidden = torch.zeros_like(self._hidden)
        else:
            self._hidden = torch.where(dones.bool().reshape(-1, 1), torch.zeros_like(self._hidden), self._hidden)

    def get_hidden_states(self):
        if self._hidden is None:
            raise RuntimeError("Call initialize_runtime before PPO rollout")
        return ActorCriticHiddenState(self._hidden.detach().unsqueeze(0),
                                      self._hidden.new_zeros(1, self._hidden.shape[0], 1))

    def initialize_runtime(self, n, device):
        self._hidden = self.initial_state(n, device)

    def encode_available(self, observations, window_valid):
        # Read only non-visual context from the input when its window is invalid.
        context = torch.cat([observations[..., self.layout[k]] for k in self.raw_segments
                             if k != "depth_image"], dim=-1)
        visual = observations.new_zeros(observations.shape[0], self.visual_dim)
        rows = window_valid.bool().nonzero(as_tuple=False).flatten()
        if rows.numel():
            encoded = self.base.encoders(observations.index_select(0, rows))
            visual = visual.index_copy(0, rows, encoded[:, self.visual_slice])
        return context, visual

    def core_step(self, context, visual, head_state, sensor_valid, window_valid, hidden, episode_start):
        v = window_valid.float().reshape(-1, 1)
        visual = torch.where(v.bool(), visual, torch.zeros_like(visual))
        memory_input = torch.cat((visual, context, head_state, sensor_valid.float().reshape(-1, 1), v), dim=-1)
        hidden, group_gate = self.memory(memory_input, hidden, episode_start)
        memory_feature = self.projection(hidden)
        gate = self.fusion_gate(torch.cat((visual, memory_feature, context, v), dim=-1))
        if self.cfg.fusion_mode == "fallback_only":
            gate = gate * 0
        weight = (1 - v) + v * gate
        fused = v * visual + weight * memory_feature
        mean = self.base.actor(torch.cat((context, fused), dim=-1))
        if self.head_actor is not None:
            # The existing head branch remains frozen and has no Teacher input.
            # Without valid vision it receives an explicit absent visual feature.
            head_mean = self.head_actor(torch.cat((context, visual), dim=-1)).detach()
            k = self.cfg.head_action_count
            mean = torch.cat((mean[..., :-k], head_mean[..., -k:]), dim=-1)
        return mean, hidden, {"memory_feature": memory_feature, "fusion_gate": gate,
                              "group_gate": group_gate, "hidden": hidden, "window_valid": v}

    def step(self, observation, hidden):
        raw = observation[:, :self.raw_dim]
        meta = observation[:, self.raw_dim:]
        if meta.shape[-1] != self.metadata_dim:
            raise ValueError("Memory metadata dimension mismatch")
        context, visual = self.encode_available(raw, meta[:, 1])
        return self.core_step(context, visual, meta[:, 2:6], meta[:, 0], meta[:, 1], hidden, meta[:, 6])

    def sequence(self, observations, hidden, padding_mask=None):
        means, aux = [], []
        for t in range(observations.shape[0]):
            mean, next_hidden, state = self.step(observations[t], hidden)
            if padding_mask is not None:
                keep = padding_mask[t].bool().unsqueeze(-1)
                hidden = torch.where(keep, next_hidden, hidden)
                mean = torch.where(keep, mean, torch.zeros_like(mean))
            else:
                hidden = next_hidden
            means.append(mean)
            aux.append(state)
        return torch.stack(means), hidden, {k: torch.stack([a[k] for a in aux]) for k in aux[0]}

    def act(self, observations, masks=None, hidden_states=None):
        if hidden_states is not None:
            mean, _, self.last_aux = self.sequence(observations, hidden_states.squeeze(0), masks)
            mean = mean.reshape(-1, mean.shape[-1])
        else:
            mean, hidden, self.last_aux = self.step(observations, self._hidden)
            # Collection never retains a graph across control cycles.
            self._hidden = hidden.detach()
        self.distribution = Normal(mean, self.base.std.expand_as(mean))
        return self.distribution.sample()

    def act_inference(self, observations):
        mean, hidden, self.last_aux = self.step(observations, self._hidden)
        self._hidden = hidden.detach()
        return mean

    def evaluate(self, observations, **kwargs):
        leading = observations.shape[:-1]
        # Installed MoeLayer has a batch-only einsum; keep this local adapter
        # rather than modifying the running framework's implementation.
        values = self.base.evaluate(observations.reshape(-1, observations.shape[-1]))
        return values.reshape(*leading, -1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions.reshape(-1, actions.shape[-1])).sum(-1).reshape(actions.shape[:-1])

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def clip_std(self, **kwargs):
        pass  # pretrained distribution frozen in stage one

    def architecture(self):
        return {"memory": asdict(self.cfg), "obs_segments": self.raw_segments,
                "raw_dim": self.raw_dim, "context_dim": self.context_dim,
                "visual_dim": self.visual_dim, "moe_input_dim": self.encoded_dim}


class DeploymentCore(nn.Module):
    """Exportable step after valid-window CNN; explicit previous/next hidden."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, context, visual, head, sensor_valid, window_valid, hidden, episode_start):
        action, hidden, _ = self.policy.core_step(context, visual, head, sensor_valid,
                                                 window_valid, hidden, episode_start)
        return action, hidden
