"""Viser feedback for the Student information channel, independent of physics."""
from __future__ import annotations

import numpy as np


def feedback_state(perception, env_idx):
    """Independent flags: sensor can recover while CNN history is still invalid."""
    sensor_mask = not bool(perception.sensor_valid[env_idx])
    history_invalid = not bool(perception.frame_valid[env_idx].all())
    visible = not sensor_mask and not history_invalid
    status = "Sensor Mask" if sensor_mask else "CNN History Invalid" if history_invalid else "Visual Valid"
    return {"sensor_mask": sensor_mask, "history_invalid": history_invalid,
            "student_visible": visible, "status": status}


class StudentCameraFeedback:
    def __init__(self, server, perception):
        self.server, self.perception = server, perception
        blank = np.zeros((18, 32, 3), dtype=np.uint8)
        with server.gui.add_folder("Student Camera Feedback"):
            self.status = server.gui.add_html("Waiting for policy observation…")
            self.original = server.gui.add_image(blank, label="Original Camera Feedback", format="png")
            self.student = server.gui.add_image(blank, label="Student Camera Feedback", format="png")
            self.student.visible = False

    def update(self, env_idx):
        depth = self.perception.feedback_depth
        if depth is None:
            self.original.visible = False
            self.student.visible = False
            self.status.content = "Waiting for policy observation…"
            return
        state = feedback_state(self.perception, env_idx)
        # Show the newest selected CNN-history frame, not a different render
        # camera with a potentially different resolution/extrinsic calibration.
        frame = depth[env_idx, -1].detach().cpu().numpy()
        gray = (np.nan_to_num(frame, nan=0.0).clip(0, 1) * 255).astype(np.uint8)
        image = np.repeat(np.repeat(np.repeat(gray[..., None], 3, axis=-1), 4, axis=0), 4, axis=1)
        self.original.image = image
        self.original.visible = True
        # Clear the old image before hiding, so it cannot remain as a stale
        # 'Student sees this' frame after masking or changing selected env.
        self.student.image = image.copy() if state["student_visible"] else np.zeros_like(image)
        self.student.visible = state["student_visible"]
        self.status.content = (
            f"<b>Robot {env_idx} — {state['status']}</b><br>"
            f"Sensor Mask: {'ON' if state['sensor_mask'] else 'OFF'}<br>"
            f"CNN History Invalid: {'ON' if state['history_invalid'] else 'OFF'}<br>"
            f"Student visual channel: {'available' if state['student_visible'] else 'hidden; memory fallback'}"
        )


def run_viser_preview(env, teacher, student, perception, *, max_steps, teacher_control=False):
    """No optimization; Teacher can control a perception-only preview safely."""
    import torch
    from mjlab.viewer import ViserPlayViewer
    from instinct_mj.scripts.instinct_rl.play import _ViewerEnvAdapter

    class MemoryViewerAdapter(_ViewerEnvAdapter):
        def step(self, actions):
            result = super().step(actions)
            done = result[2]
            student.reset(done)
            perception.reset_done(done)
            return result

        def reset(self):
            result = self._vec_env.reset()
            student.reset()
            perception.reset_done(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
            perception.feedback_depth = None
            return result

    class MemoryViewer(ViserPlayViewer):
        def setup(self):
            super().setup()
            self.student_feedback = StudentCameraFeedback(self._server, perception)
            self._server.gui.add_html(
                "<b>Controller: frozen Teacher; Student perception preview only</b>" if teacher_control else
                "<b>Controller: Student GDU policy</b>"
            )

        def sync_env_to_viewer(self):
            super().sync_env_to_viewer()
            self.student_feedback.update(self._scene.env_idx)

    @torch.no_grad()
    def policy(full_obs):
        packed, full = perception.prepare(full_obs)
        prediction = student.act_inference(packed)
        return teacher.act_inference(full) if teacher_control else prediction

    viewer = MemoryViewer(MemoryViewerAdapter(env), policy)
    viewer.run(num_steps=max_steps)
