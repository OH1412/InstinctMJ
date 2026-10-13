"""Play a retargeted G1 motion reference in a Viser viewer."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
import tyro
import viser
from mjviser import ViserMujocoScene

from instinct_mj.assets.unitree_g1 import G1_MJCF_PATH


def main(
    motion_file: str,
    model_file: str = G1_MJCF_PATH,
    host: str = "0.0.0.0",
    port: int = 8080,
    speed: float = 1.0,
    loop: bool = True,
) -> None:
    motion_path = Path(motion_file).expanduser().resolve()
    if not motion_path.exists():
        raise FileNotFoundError(motion_path)

    motion = np.load(motion_path, allow_pickle=True)
    frame_rate = float(motion["framerate"])
    joint_names = [str(name) for name in motion["joint_names"].tolist()]
    joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
    base_pos = np.asarray(motion["base_pos_w"], dtype=np.float64)
    base_quat = np.asarray(motion["base_quat_w"], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(Path(model_file).resolve()))
    data = mujoco.MjData(model)

    qpos_adrs: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"Joint from motion file not found in MJCF: {name}")
        qpos_adrs.append(int(model.jnt_qposadr[joint_id]))

    server = viser.ViserServer(host=host, port=port, label="G1 AMP motion")
    scene = ViserMujocoScene(server, model, num_envs=1)
    print(f"[INFO] Viser: http://localhost:{port}", flush=True)
    print(
        f"[INFO] Motion: {motion_path.name}, frames={len(joint_pos)}, "
        f"duration={len(joint_pos) / frame_rate:.2f}s, fps={frame_rate:g}",
        flush=True,
    )

    frame_dt = 1.0 / max(frame_rate * max(speed, 1e-3), 1e-3)
    frame = 0
    while True:
        data.qpos[:] = 0.0
        data.qpos[:3] = base_pos[frame]
        data.qpos[3:7] = base_quat[frame]
        for qpos_adr, value in zip(qpos_adrs, joint_pos[frame], strict=True):
            data.qpos[qpos_adr] = value
        mujoco.mj_forward(model, data)
        scene.update_from_mjdata(data)

        frame += 1
        if frame >= len(joint_pos):
            if not loop:
                break
            frame = 0
        time.sleep(frame_dt)


if __name__ == "__main__":
    tyro.cli(main)
