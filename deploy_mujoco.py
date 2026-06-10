import time
import os
import sys
import argparse
import threading
import numpy as np
import mujoco
import mujoco.viewer
import torch
import yaml
import pygame

# Add project root to path for importing rsl_rl modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from rsl_rl.modules import ActorCriticEncoder

ISAACLAB_JOINT_ORDER = [
    'left_hip_pitch_joint',      # 0 - 实际Isaac Sim顺序
    'right_hip_pitch_joint',     # 1
    'waist_yaw_joint',            # 2
    'left_hip_roll_joint',       # 3
    'right_hip_roll_joint',      # 4
    'waist_roll_joint',          # 5
    'left_hip_yaw_joint',        # 6
    'right_hip_yaw_joint',       # 7
    'waist_pitch_joint',         # 8
    'left_knee_joint',           # 9
    'right_knee_joint',          # 10
    'left_shoulder_pitch_joint', # 11
    'right_shoulder_pitch_joint',# 12
    'left_ankle_pitch_joint',    # 13
    'right_ankle_pitch_joint',   # 14
    'left_shoulder_roll_joint',  # 15
    'right_shoulder_roll_joint', # 16
    'left_ankle_roll_joint',     # 17
    'right_ankle_roll_joint',    # 18
    'left_shoulder_yaw_joint',   # 19
    'right_shoulder_yaw_joint',   # 20
    'left_elbow_joint',           # 21
    'right_elbow_joint',          # 22
    'left_wrist_roll_joint',     # 23
    'right_wrist_roll_joint',    # 24
    'left_wrist_pitch_joint',    # 25
    'right_wrist_pitch_joint',   # 26
    'left_wrist_yaw_joint',      # 27
    'right_wrist_yaw_joint',     # 28
]

MUJOCO_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Default joint pose from UNITREE_G1_29DOF_CFG (unitree.py init_state) — training reference.
TRAINING_DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    "waist_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "waist_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_shoulder_pitch_joint": 0.3,
    "right_shoulder_pitch_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.97,
    "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# PD gains from UNITREE_G1_29DOF_CFG actuators (unitree.py L420-477), per joint name.
# (stiffness Kp, damping Kd)
TRAINING_JOINT_PD = {
    # N7520-14.3: .*_hip_pitch_.*, .*_hip_yaw_.*, waist_yaw_joint
    "left_hip_pitch_joint": (100.0, 2.0),
    "right_hip_pitch_joint": (100.0, 2.0),
    "left_hip_yaw_joint": (100.0, 2.0),
    "right_hip_yaw_joint": (100.0, 2.0),
    "waist_yaw_joint": (200.0, 5.0),
    # N7520-22.5: .*_hip_roll_.*, .*_knee_.*
    "left_hip_roll_joint": (100.0, 2.0),
    "right_hip_roll_joint": (100.0, 2.0),
    "left_knee_joint": (150.0, 4.0),
    "right_knee_joint": (150.0, 4.0),
    # N5020-16: shoulder, elbow, wrist_roll, ankle, waist_roll/pitch
    "left_ankle_pitch_joint": (40.0, 2.0),
    "left_ankle_roll_joint": (40.0, 2.0),
    "right_ankle_pitch_joint": (40.0, 2.0),
    "right_ankle_roll_joint": (40.0, 2.0),
    "waist_roll_joint": (40.0, 5.0),
    "waist_pitch_joint": (40.0, 5.0),
    "left_shoulder_pitch_joint": (40.0, 10.0),
    "left_shoulder_roll_joint": (40.0, 10.0),
    "left_shoulder_yaw_joint": (40.0, 10.0),
    "left_elbow_joint": (40.0, 10.0),
    "left_wrist_roll_joint": (40.0, 10.0),
    "right_shoulder_pitch_joint": (40.0, 10.0),
    "right_shoulder_roll_joint": (40.0, 10.0),
    "right_shoulder_yaw_joint": (40.0, 10.0),
    "right_elbow_joint": (40.0, 10.0),
    "right_wrist_roll_joint": (40.0, 10.0),
    # W4010-25: wrist_pitch, wrist_yaw
    "left_wrist_pitch_joint": (40.0, 10.0),
    "left_wrist_yaw_joint": (40.0, 10.0),
    "right_wrist_pitch_joint": (40.0, 10.0),
    "right_wrist_yaw_joint": (40.0, 10.0),
}

# effort_limit_sim from same actuators (for tau clip)
TRAINING_JOINT_EFFORT = {
    "left_hip_pitch_joint": 88.0,
    "right_hip_pitch_joint": 88.0,
    "left_hip_yaw_joint": 88.0,
    "right_hip_yaw_joint": 88.0,
    "waist_yaw_joint": 88.0,
    "left_hip_roll_joint": 139.0,
    "right_hip_roll_joint": 139.0,
    "left_knee_joint": 139.0,
    "right_knee_joint": 139.0,
    "left_ankle_pitch_joint": 25.0,
    "left_ankle_roll_joint": 25.0,
    "right_ankle_pitch_joint": 25.0,
    "right_ankle_roll_joint": 25.0,
    "waist_roll_joint": 25.0,
    "waist_pitch_joint": 25.0,
    "left_shoulder_pitch_joint": 25.0,
    "left_shoulder_roll_joint": 25.0,
    "left_shoulder_yaw_joint": 25.0,
    "left_elbow_joint": 25.0,
    "left_wrist_roll_joint": 25.0,
    "right_shoulder_pitch_joint": 25.0,
    "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0,
    "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 25.0,
    "left_wrist_pitch_joint": 5.0,
    "left_wrist_yaw_joint": 5.0,
    "right_wrist_pitch_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}


def get_training_pd_gains_mujoco():
    """Kp/Kd arrays in MUJOCO_JOINT_ORDER from UNITREE_G1_29DOF_CFG."""
    kp = np.zeros(len(MUJOCO_JOINT_ORDER), dtype=np.float32)
    kd = np.zeros(len(MUJOCO_JOINT_ORDER), dtype=np.float32)
    for i, name in enumerate(MUJOCO_JOINT_ORDER):
        kp[i], kd[i] = TRAINING_JOINT_PD[name]
    return kp, kd


def get_training_effort_limits_mujoco():
    """Torque limits in MUJOCO_JOINT_ORDER from effort_limit_sim."""
    return np.array(
        [TRAINING_JOINT_EFFORT[name] for name in MUJOCO_JOINT_ORDER],
        dtype=np.float32,
    )


def get_training_default_angles_policy():
    """Default joint angles in IsaacLab policy order (matches joint_pos_rel)."""
    return np.array(
        [TRAINING_DEFAULT_JOINT_POS[name] for name in ISAACLAB_JOINT_ORDER],
        dtype=np.float32,
    )


def get_training_default_angles_mujoco():
    """Default joint angles in MuJoCo qpos order (for sim init and PD hold)."""
    return np.array(
        [TRAINING_DEFAULT_JOINT_POS[name] for name in MUJOCO_JOINT_ORDER],
        dtype=np.float32,
    )


def get_projected_gravity(mj_model, mj_data, body_name="pelvis"):
    """Gravity unit vector in body frame — same as Isaac Lab projected_gravity."""
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found for projected_gravity")
    rot = mj_data.xmat[body_id].reshape(3, 3)
    return (rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD controller for joint control."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def apply_axis_deadzone(value, deadzone):
    return 0.0 if abs(value) < deadzone else value


def gamepad_reader(joystick, cmd, cmd_lock, running, deadzone, cmd_gain):
    """Read gamepad inputs in a separate thread."""
    while running[0]:
        try:
            pygame.event.pump()
            left_y = -joystick.get_axis(1)
            left_x = -joystick.get_axis(0)
            right_x = -joystick.get_axis(3)

            left_y = apply_axis_deadzone(left_y, deadzone)
            left_x = apply_axis_deadzone(left_x, deadzone)
            right_x = apply_axis_deadzone(right_x, deadzone)

            new_cmd = np.array(
                [left_y * cmd_gain[0], left_x * cmd_gain[1], right_x * cmd_gain[2]],
                dtype=np.float32,
            )

            with cmd_lock:
                cmd[:] = new_cmd

            time.sleep(0.01)
        except Exception:
            time.sleep(0.01)


class ObservationHistoryBuffer:
    """Observation history buffer for stacking multiple frames."""
    def __init__(self, num_obs, history_steps):
        self.num_obs = num_obs
        self.history_steps = history_steps
        self.buffer = np.zeros((history_steps, num_obs), dtype=np.float32)
        self.initialized = False

    def reset(self, obs):
        """Reset buffer with current observation filling all history frames."""
        self.buffer[:] = obs
        self.initialized = True

    def insert(self, obs):
        """Insert new observation, shifting old observations back."""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs

    def get_stacked_obs(self):
        """Get stacked observation (from oldest to newest)."""
        return self.buffer.flatten()


class TerrainScanner:
    """Height scanner aligned with training elevation_map (torso_link, yaw frame)."""

    def __init__(
        self,
        grid_size=(33, 21),
        scan_size=(1.6, 1.0),
        ray_z_offset=20.0,
        ray_cutoff=30.0,
        torso_body_name="torso_link",
        robot_root_body_name="pelvis",
        height_clip=(-1.2, 0.0),
        exclude_robot=True,
    ):
        self.L, self.W = grid_size  # L=33 (x), W=21 (y)
        self.nx, self.ny = self.L, self.W
        self.scan_length, self.scan_width = scan_size
        self.ray_z_offset = ray_z_offset
        self.ray_cutoff = ray_cutoff
        self.torso_body_name = torso_body_name
        self.robot_root_body_name = robot_root_body_name
        self.height_clip = height_clip
        self.exclude_robot = exclude_robot
        self._robot_body_exclude = -1

        self.grid_x = np.linspace(-self.scan_length / 2, self.scan_length / 2, self.L)
        self.grid_y = np.linspace(-self.scan_width / 2, self.scan_width / 2, self.W)
        self.ray_hits_w = np.full((self.L * self.W, 3), np.nan, dtype=np.float32)

    def bind_model(self, mj_model):
        """Resolve body id for ray exclusion (must run after model load)."""
        if not self.exclude_robot:
            self._robot_body_exclude = -1
            self._robot_geom_ids = set()
            return
        root_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, self.robot_root_body_name
        )
        if root_id < 0:
            raise ValueError(f"Body '{self.robot_root_body_name}' not found for ray exclusion")
        self._robot_body_exclude = root_id
        robot_bodies = {root_id}
        for i in range(mj_model.nbody):
            if int(mj_model.body_parentid[i]) in robot_bodies:
                robot_bodies.add(i)
        self._robot_geom_ids = set()
        for i in range(mj_model.ngeom):
            if int(mj_model.geom_bodyid[i]) in robot_bodies:
                self._robot_geom_ids.add(i)

    def _yaw_rotation(self, quat_wxyz):
        qw, qx, qy, qz = quat_wxyz
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ], dtype=np.float64)

    def elevation_map(self, mj_model, mj_data):
        """
        Returns elevation map in shape (3, W, L) = (3, ny, nx),
        matching working AME sim2sim before (W,L,3) flatten.
        """
        torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.torso_body_name)
        if torso_id < 0:
            raise ValueError(f"Body '{self.torso_body_name}' not found in MuJoCo model")

        torso_pos = mj_data.xpos[torso_id].copy()
        torso_quat = mj_data.xquat[torso_id].copy()  # w, x, y, z
        R_yaw = self._yaw_rotation(torso_quat)

        elev = np.zeros((3, self.W, self.L), dtype=np.float32)
        hits_w = self.ray_hits_w
        ray_dir = np.array([[0.0], [0.0], [-1.0]], dtype=np.float64)
        geom_id = np.zeros((1, 1), dtype=np.int32)
        geom_id[0, 0] = -1

        for ix, x in enumerate(self.grid_x):
            for iy, y in enumerate(self.grid_y):
                ray_idx = iy * self.L + ix
                local_xy = np.array([x, y, 0.0], dtype=np.float64)
                world_xy = torso_pos + R_yaw @ local_xy
                ray_origin = np.array(
                    [[world_xy[0]], [world_xy[1]], [torso_pos[2] + self.ray_z_offset]],
                    dtype=np.float64,
                )

                hit_world = None
                search_origin = ray_origin.copy()
                for _ in range(32):
                    geom_id[0, 0] = -1
                    hit_dist = mujoco.mj_ray(
                        mj_model,
                        mj_data,
                        search_origin,
                        ray_dir,
                        None,
                        1,
                        -1,
                        geom_id,
                        None,
                    )
                    if hit_dist < 0:
                        break
                    if int(geom_id[0, 0]) not in self._robot_geom_ids:
                        hit_world = search_origin[:, 0] + ray_dir[:, 0] * hit_dist
                        break
                    search_origin[:, 0] = search_origin[:, 0] + ray_dir[:, 0] * (hit_dist + 1e-3)

                if hit_world is None:
                    hit_world = np.array([world_xy[0], world_xy[1], 0.0], dtype=np.float64)
                    hits_w[ray_idx] = np.nan
                else:
                    hits_w[ray_idx] = hit_world.astype(np.float32)

                rel_world = hit_world - torso_pos
                point_sensor = R_yaw.T @ rel_world
                point_sensor[2] = np.clip(point_sensor[2], self.height_clip[0], self.height_clip[1])
                elev[:, iy, ix] = point_sensor

        return elev


def draw_hit_markers(viewer, hits_w, radius=0.015, rgba=(0.1, 0.9, 0.2, 0.85)):
    """Draw small spheres at valid ray hit points in the MuJoCo viewer."""
    valid = np.isfinite(hits_w).all(axis=1)
    points = hits_w[valid]
    ngeom = min(len(points), viewer.user_scn.maxgeom)
    viewer.user_scn.ngeom = ngeom
    mat = np.eye(3).flatten()
    size = np.array([radius, 0.0, 0.0], dtype=np.float64)
    rgba_arr = np.array(rgba, dtype=np.float32)
    for i in range(ngeom):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=size,
            pos=points[i],
            mat=mat,
            rgba=rgba_arr,
        )


class AMEPolicyWrapper:
    """Flat-obs inference wrapper (same interface as working AME FSM)."""

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def act(self, obs_flat):
        obs_t = torch.from_numpy(obs_flat).float().reshape(1, -1).to(self.device)
        with torch.inference_mode():
            encoded_obs, _ = self.policy._encode_terrain(obs_t)
            return self.policy.actor(encoded_obs).detach().cpu().numpy().reshape(-1)


def build_proprio_obs(
    ang_vel, gravity_orientation, cmd, qj_policy, dqj_policy, default_angles_policy,
    last_action, num_actions, ang_vel_scale, dof_pos_scale, dof_vel_scale, cmd_scale,
):
    """Policy proprio (96): ang_vel, gravity, cmd, joint_pos, joint_vel, last_action."""
    proprio = np.zeros(96, dtype=np.float32)
    proprio[0:3] = ang_vel * ang_vel_scale
    proprio[3:6] = gravity_orientation
    proprio[6:9] = cmd * cmd_scale
    proprio[9:9 + num_actions] = (qj_policy - default_angles_policy) * dof_pos_scale
    proprio[9 + num_actions:9 + 2 * num_actions] = dqj_policy * dof_vel_scale
    proprio[9 + 2 * num_actions:9 + 3 * num_actions] = last_action
    return proprio


def build_policy_obs(proprio, elevation_map):
    """
    Concatenate proprio + elevation flatten.
    elevation_map: (3, W, L) -> permute to (W, L, 3) -> flatten (matches ActorCriticEncoder).
    """
    elev_wl3 = np.transpose(elevation_map, (1, 2, 0))  # (W, L, 3)
    elev_flat = elev_wl3.reshape(-1)
    return np.concatenate([proprio, elev_flat], dtype=np.float32)


def make_flat_elevation_map(L, W, scan_size=(1.6, 1.0), ground_z=-0.79):
    """
    Flat-ground elevation_map (3, W, L) matching training elevation_map layout.
    Each cell stores local (x, y, z) in torso yaw frame — NOT (0, 0, z_only).
    """
    scan_length, scan_width = scan_size
    grid_x = np.linspace(-scan_length / 2, scan_length / 2, L)
    grid_y = np.linspace(-scan_width / 2, scan_width / 2, W)
    elev = np.zeros((3, W, L), dtype=np.float32)
    for ix, x in enumerate(grid_x):
        for iy, y in enumerate(grid_y):
            elev[0, iy, ix] = x
            elev[1, iy, ix] = y
            elev[2, iy, ix] = ground_z
    return elev


def format_elevation_debug(elevation_map, policy_step, sim_time):
    """
    Pretty-print height scan fed to policy (local z in torso yaw frame, after clip).
    elevation_map layout: (3, W, L) with W=21 (y), L=33 (x).
    """
    ex, ey, ez = elevation_map[0], elevation_map[1], elevation_map[2]
    w, l = ez.shape
    lines = [
        f"--- elevation_map policy_step={policy_step} sim_t={sim_time:.3f}s ---",
        f"  local_x: mean={ex.mean():.4f} min={ex.min():.4f} max={ex.max():.4f}",
        f"  local_y: mean={ey.mean():.4f} min={ey.min():.4f} max={ey.max():.4f}",
        (
            f"  local_z (CNN height): mean={ez.mean():.4f} min={ez.min():.4f} max={ez.max():.4f} "
            f"(flat ground ~ -0.75..-0.85; if mean > -0.5 rays likely hit robot)"
        ),
        f"  grid local_z [rows=y 0..{w - 1}, cols=x 0..{l - 1}]:",
    ]
    for iy in range(w):
        row = " ".join(f"{v:7.3f}" for v in ez[iy, :])
        lines.append(f"    y[{iy:02d}] {row}")
    return "\n".join(lines)


def yaw_quat(quat):
    """Extract yaw rotation quaternion from full quaternion."""
    # quat: [w, x, y, z]
    yaw = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                      1 - 2 * (quat[2]**2 + quat[3]**2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Deploy G1 robot in MuJoCo with AME trained policy')
    parser.add_argument('--policy_path', type=str, default=None,
                       help='Direct path to policy checkpoint (e.g., pretrained/ame2.pt)')
    parser.add_argument('--config', type=str, default='g1_29dof.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no_terrain', action='store_true',
                       help='Disable terrain scanning (use flat ground assumption)')
    parser.add_argument('--verify_alignment', action='store_true',
                       help='Print observation/action alignment diagnostics on first policy step')
    parser.add_argument('--gamepad', action='store_true',
                       help='Use joystick for velocity commands (default: cmd_init / keyboard only)')
    parser.add_argument('--no_gamepad', action='store_true',
                       help='Deprecated alias; gamepad is off by default unless --gamepad is set')
    parser.add_argument('--print_elevation', action='store_true',
                       help='Print height-scan local_z grid each policy step (see yaml print_elevation_z)')
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()

    config_file = os.path.join(script_dir, args.config)

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

        # Process paths
        if "xml_path" in config:
            xml_path = config["xml_path"]
            if not os.path.isabs(xml_path):
                xml_path = os.path.join(script_dir, xml_path)
            config["xml_path"] = xml_path

        # Process policy_path
        if args.policy_path:
            policy_path = args.policy_path
            if not os.path.isabs(policy_path):
                policy_path = os.path.join(script_dir, policy_path)
            config["policy_path"] = policy_path
            print(f"Using command line policy path: {policy_path}")
        elif "policy_path" in config and config["policy_path"]:
            policy_path = config["policy_path"]
            if not os.path.isabs(policy_path):
                policy_path = os.path.join(script_dir, policy_path)
            config["policy_path"] = policy_path
            print(f"Using config policy path: {policy_path}")
        else:
            print("Error: No policy path provided! Use --policy_path")
            sys.exit(1)

        use_training_pd = bool(config.get("use_training_pd", True))
        if use_training_pd:
            kps, kds = get_training_pd_gains_mujoco()
            torque_limit = get_training_effort_limits_mujoco()
        else:
            kps = np.array(config["stiffness"], dtype=np.float32)
            kds = np.array(config["damping"], dtype=np.float32)
            torque_limit = np.array(config["torque_limit"], dtype=np.float32)
        default_joint_angles = np.array(config["default_joint_angles"], dtype=np.float32)
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        ang_vel_scale = float(config.get("ang_vel_scale", 0.2))
        dof_pos_scale = float(config.get("dof_pos_scale", 1.0))
        dof_vel_scale = float(config.get("dof_vel_scale", 0.05))
        action_scale = float(config["action_scale"])
        # Training ObsTerm has NO scale on velocity_commands; keep [1,1,1] for Isaac alignment.
        cmd_scale = np.array(config.get("cmd_scale", [1.0, 1.0, 1.0]), dtype=np.float32)
        startup_blend_steps = int(config.get("startup_blend_steps", 30))
        startup_cmd_zero_steps = int(config.get("startup_cmd_zero_steps", 20))
        policy_delay_steps = int(config.get("policy_delay_steps", 0))
        pd_scale = float(config.get("pd_scale", 1.0))
        if pd_scale != 1.0:
            kps = kps * pd_scale
            kds = kds * pd_scale
        use_gamepad = bool(config.get("use_gamepad", False)) or args.gamepad
        if args.no_gamepad:
            use_gamepad = False
        gamepad_deadzone = float(config.get("gamepad_deadzone", 0.15))
        gamepad_cmd_gain = np.array(
            config.get("gamepad_cmd_gain", [1.2, 0.6, 0.8]), dtype=np.float32
        )
        print_elevation_z = bool(config.get("print_elevation_z", False)) or args.print_elevation
        print_elevation_z_interval = int(config.get("print_elevation_z_interval", 1))
        print_elevation_z_first_n = int(config.get("print_elevation_z_first_n", 5))

    # AME model configuration
    map_scan_dim = config.get("map_scan_dim", (33, 21, 3))  # (L, W, coord_dim)
    L, W, coord_dim = map_scan_dim
    mha_dim = config.get("mha_dim", 64)
    num_heads = config.get("num_heads", 16)
    cnn_downsample = config.get("cnn_downsample", True)
    attach_global = config.get("attach_global", True)

    # Observation dimensions for AME
    # Actor proprioceptive obs: base_ang_vel(3) + projected_gravity(3) + commands(3) + joint_pos(29) + joint_vel(29) + actions(29) = 96
    # Critic proprioceptive obs: base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + commands(3) + joint_pos(29) + joint_vel(29) + actions(29) = 99
    actor_proprio_dim = config.get("actor_proprio_dim", 96)
    critic_proprio_dim = config.get("critic_proprio_dim", 99)

    # Full obs includes height_scan
    map_scan_size = L * W * coord_dim
    actor_single_obs_dim = actor_proprio_dim + map_scan_size  # 96 + 2079 = 2175
    critic_single_obs_dim = critic_proprio_dim + map_scan_size  # 99 + 2079 = 2178

    print(f"=" * 80)
    print(f"AME Model Configuration:")
    print(f"  Map scan dimension: {map_scan_dim}")
    print(f"  MHA dimension: {mha_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  CNN downsample: {cnn_downsample}")
    print(f"  Attach global: {attach_global}")
    print(f"  Actor proprio dim: {actor_proprio_dim}")
    print(f"  Critic proprio dim: {critic_proprio_dim}")
    print(f"  Map scan size: {map_scan_size}")
    print(f"  Actor single obs dim: {actor_single_obs_dim}")
    print(f"  Critic single obs dim: {critic_single_obs_dim}")
    print(f"=" * 80)
    print("PD gains (MUJOCO order, from unitree.py UNITREE_G1_29DOF_CFG):")
    print(f"  use_training_pd={use_training_pd}")
    print(f"  Kp: {kps.tolist()}")
    print(f"  Kd: {kds.tolist()}")
    print(f"  torque_limit: {torque_limit.tolist()}")

    cmd_lock = threading.Lock()
    running = [True]

    if use_gamepad and pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        pygame.event.pump()
        raw_axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
        print(f"Gamepad enabled: {joystick.get_name()}, raw axes={raw_axes}")
        print(f"  deadzone={gamepad_deadzone}, cmd_gain={gamepad_cmd_gain.tolist()}")
        gamepad_thread = threading.Thread(
            target=gamepad_reader,
            args=(joystick, cmd, cmd_lock, running, gamepad_deadzone, gamepad_cmd_gain),
            daemon=True,
        )
        gamepad_thread.start()
    else:
        if pygame.joystick.get_count() > 0 and not use_gamepad:
            js = pygame.joystick.Joystick(0)
            js.init()
            pygame.event.pump()
            n = js.get_numaxes()
            raw = [js.get_axis(i) for i in range(n)]
            print(
                f"Joystick detected ({js.get_name()}) but gamepad DISABLED — "
                f"cmd stays cmd_init={cmd.tolist()} (use --gamepad to enable)"
            )
            print(f"  (idle axes often sit at ±1.0; e.g. axis2={raw[2] if n > 2 else 'n/a'} "
                  f"would become yaw cmd {raw[2] * gamepad_cmd_gain[2]:.2f})")
        else:
            print(f"Velocity command fixed to cmd_init={cmd.tolist()} (no joystick)")

    num_actions = config["actions_dim"]
    last_action = np.zeros(num_actions, dtype=np.float32)
    startup_step = 0

    # Initialize terrain scanner
    terrain_scanner = None
    if not args.no_terrain:
        terrain_scanner = TerrainScanner(
            grid_size=(L, W),
            scan_size=(1.6, 1.0),
            ray_z_offset=20.0,
            height_clip=(-1.2, 0.0),
            torso_body_name="torso_link",
        )
        print(
            f"Terrain scanner initialized (exclude_robot={terrain_scanner.exclude_robot}, "
            f"root={terrain_scanner.robot_root_body_name})"
        )

    mj_model = mujoco.MjModel.from_xml_path(config["xml_path"])
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = config["simulation_dt"]
    if terrain_scanner is not None:
        terrain_scanner.bind_model(mj_model)
        if terrain_scanner.exclude_robot:
            print(
                f"Terrain rays exclude robot body id={terrain_scanner._robot_body_exclude} "
                f"({terrain_scanner.robot_root_body_name})"
            )

    # Print MuJoCo joint names
    mujoco_joint_names = []
    for i in range(mj_model.njnt):
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name and joint_name != "floating_base_joint":
            mujoco_joint_names.append(joint_name)

    # print("=" * 80)
    # print(f"MuJoCo joint order ({len(mujoco_joint_names)} joints):")
    # for i, joint_name in enumerate(mujoco_joint_names):
    #     print(f"  [{i:2d}] {joint_name}")
    # print("=" * 80)

    # Create IsaacLab to MuJoCo mapping
    isaaclab_to_mujoco = []
    for isaaclab_idx, joint_name in enumerate(ISAACLAB_JOINT_ORDER):
        if joint_name in MUJOCO_JOINT_ORDER:
            mujoco_idx = MUJOCO_JOINT_ORDER.index(joint_name)
            isaaclab_to_mujoco.append(mujoco_idx)
        else:
            print(f"Warning: IsaacLab joint {joint_name} not found in MuJoCo order")
            isaaclab_to_mujoco.append(-1)

    # print("IsaacLab to MuJoCo mapping:")
    # for i, mujoco_idx in enumerate(isaaclab_to_mujoco):
    #     if mujoco_idx >= 0:
    #         print(f"  IsaacLab[{i:2d}] {ISAACLAB_JOINT_ORDER[i]:30s} -> MuJoCo[{mujoco_idx:2d}] {MUJOCO_JOINT_ORDER[mujoco_idx]}")
    # print("=" * 80)

    counter = 0

    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[0:3] = [0.0, 0.0, 0.8]
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    num_joints_in_model = len(mujoco_joint_names)
    training_defaults_mujoco = get_training_default_angles_mujoco()
    num_joints_to_set = min(num_joints_in_model, len(training_defaults_mujoco))
    # Sim init must match training default pose (yaml arm defaults differ, e.g. elbow 1.31 vs 0.97).
    mj_data.qpos[7:7 + num_joints_to_set] = training_defaults_mujoco[:num_joints_to_set]
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # Load AME model
    if not config.get("policy_path") or not config["policy_path"]:
        raise ValueError("Must provide policy_path!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {config['policy_path']}")
    checkpoint = torch.load(config["policy_path"], map_location=device)

    # Extract model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Create model configuration
    # Note: We create FULL model (with critic) to match checkpoint structure,
    # but for sim2sim we only use the actor part for inference
    # IMPORTANT: obs_groups order must match training config (velocity_env_cfg_29dof.py)
    # Order: base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions, height_scan
    obs_groups = {
        "policy": ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions", "height_scan"],
        "critic": ["base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions", "height_scan"]
    }

    # Create dummy obs tensors with both actor and critic dimensions
    # Order must match obs_groups["policy"] and obs_groups["critic"]
    dummy_obs = {
        "base_lin_vel": torch.zeros(1, 3, device=device),
        "base_ang_vel": torch.zeros(1, 3, device=device),
        "projected_gravity": torch.zeros(1, 3, device=device),
        "velocity_commands": torch.zeros(1, 3, device=device),
        "joint_pos": torch.zeros(1, 29, device=device),
        "joint_vel": torch.zeros(1, 29, device=device),
        "actions": torch.zeros(1, 29, device=device),
        "height_scan": torch.zeros(1, map_scan_size, device=device),
    }

    # Create full model (both actor and critic)
    model_cfg = {
        "actor_hidden_dims": config.get("actor_hidden_dims", [512, 256, 128]),
        "critic_hidden_dims": config.get("critic_hidden_dims", [512, 256, 128]),
        "activation": config.get("activation", "elu"),
        "map_scan_dim": map_scan_dim,
        "mha_dim": mha_dim,
        "num_heads": num_heads,
        "cnn_downsample": cnn_downsample,
        "attach_global": attach_global,
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
    }

    policy = ActorCriticEncoder(
        obs=dummy_obs,
        obs_groups=obs_groups,
        num_actions=config["actions_dim"],
        **model_cfg
    ).to(device)

    # Load full state dict, but we'll only use actor for inference
    result = policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    print("Model loaded successfully")
    policy_runner = AMEPolicyWrapper(policy, device)

    # Warmup with flat obs (must include grid x,y in elevation — training uses full 3D coords)
    print("Warming up model...")
    proprio_warm = np.zeros(96, dtype=np.float32)
    proprio_warm[5] = -1.0
    elev_warm = make_flat_elevation_map(L, W)
    for _ in range(10):
        policy_runner.act(build_policy_obs(proprio_warm, elev_warm))
    print("Warmup complete")

    # Policy-order defaults: use training asset defaults (NOT yaml mujoco defaults).
    default_angles_policy = get_training_default_angles_policy()
    target_dof_pos = np.zeros(num_joints_in_model, dtype=np.float32)
    for policy_idx, mujoco_idx in enumerate(isaaclab_to_mujoco):
        if mujoco_idx >= 0 and mujoco_idx < num_joints_in_model:
            target_dof_pos[mujoco_idx] = (
                default_angles_policy[policy_idx]
            )

    qj = mj_data.qpos[7:]
    for policy_idx, mujoco_idx in enumerate(isaaclab_to_mujoco):
        if mujoco_idx >= 0 and mujoco_idx < len(qj):
            last_action[policy_idx] = (qj[mujoco_idx] - default_angles_policy[policy_idx]) / action_scale

    # Warn if yaml defaults differ from training (common source of collapse)
    yaml_policy_defaults = np.zeros(num_actions, dtype=np.float32)
    for policy_idx, mujoco_idx in enumerate(isaaclab_to_mujoco):
        if mujoco_idx >= 0:
            yaml_policy_defaults[policy_idx] = default_joint_angles[mujoco_idx]
    max_def_diff = np.max(np.abs(yaml_policy_defaults - default_angles_policy))
    if max_def_diff > 0.05:
        print(f"WARNING: g1_29dof.yaml default_joint_angles differ from training by up to {max_def_diff:.3f} rad")
        print("         Using training defaults from unitree.py for joint_pos_rel / actions.")

    print("Observation initialized (last_action synced to current posture)")
    print(f"Policy obs layout: proprio(96) + elevation({map_scan_size}) = {96 + map_scan_size}")
    print(f"Actor proprio_dim={policy.actor_proprio_dim}, map L={L} W={W}")

    # Sanity: policy at nominal proprio + flat elevation with correct (x,y,z) per cell.
    proprio_nom = np.zeros(96, dtype=np.float32)
    proprio_nom[5] = -1.0
    elev_nom = make_flat_elevation_map(L, W)
    act_nom = policy_runner.act(build_policy_obs(proprio_nom, elev_nom))
    print(
        f"Policy@nominal (gravity=[0,0,-1], joint_rel=0, elev with grid x/y/z): "
        f"range=({act_nom.min():.3f},{act_nom.max():.3f}), "
        f"hip_pitch={act_nom[0]:.3f}, knee_L={act_nom[9]:.3f}, knee_R={act_nom[10]:.3f}"
    )

    if print_elevation_z:
        print(
            f"Elevation debug ON: every {print_elevation_z_interval} policy step(s), "
            f"first {print_elevation_z_first_n} steps always"
        )
    running_main = True
    mj_per_step_duration = mj_model.opt.timestep * config["control_decimation"]
    with mujoco.viewer.launch_passive(mj_model, mj_data, show_left_ui=True, show_right_ui=True) as viewer:
        while viewer.is_running() and running_main and mj_data.time < config["simulation_duration"]:
            step_start = time.time()
            try:
                counter += 1
                policy_step = counter

                # 1) Policy inference once per control step.
                if policy_step > policy_delay_steps:
                    with cmd_lock:
                        current_cmd = cmd.copy()
                    if startup_step < startup_cmd_zero_steps:
                        current_cmd[:] = 0.0

                    qj = mj_data.qpos[7:]
                    dqj = mj_data.qvel[6:]
                    omega = mj_data.qvel[3:6]
                    gravity_orientation = get_projected_gravity(mj_model, mj_data)

                    qj_policy = np.zeros(num_actions, dtype=np.float32)
                    dqj_policy = np.zeros(num_actions, dtype=np.float32)
                    for policy_idx, mujoco_idx in enumerate(isaaclab_to_mujoco):
                        if mujoco_idx >= 0 and mujoco_idx < len(qj):
                            qj_policy[policy_idx] = qj[mujoco_idx]
                            dqj_policy[policy_idx] = dqj[mujoco_idx]

                    if terrain_scanner is not None:
                        elevation_map = terrain_scanner.elevation_map(mj_model, mj_data)
                    else:
                        elevation_map = np.zeros((3, W, L), dtype=np.float32)
                        elevation_map[2, :, :] = -0.79

                    proprio = build_proprio_obs(
                        omega,
                        gravity_orientation,
                        current_cmd[:3],
                        qj_policy,
                        dqj_policy,
                        default_angles_policy,
                        last_action,
                        num_actions,
                        ang_vel_scale,
                        dof_pos_scale,
                        dof_vel_scale,
                        cmd_scale,
                    )
                    obs_flat = build_policy_obs(proprio, elevation_map)

                    inference_start = time.perf_counter()
                    raw_action = policy_runner.act(obs_flat)
                    inference_time = (time.perf_counter() - inference_start) * 1000

                    raw_action = np.clip(raw_action, -100.0, 100.0)
                    if startup_blend_steps > 0:
                        alpha = min(float(startup_step + 1) / float(startup_blend_steps), 1.0)
                    else:
                        alpha = 1.0
                    action = (1.0 - alpha) * last_action + alpha * raw_action
                    last_action = raw_action.copy()

                    loco_action = action * action_scale + default_angles_policy
                    target_dof_pos = np.zeros(num_joints_in_model, dtype=np.float32)
                    for policy_idx, mujoco_idx in enumerate(isaaclab_to_mujoco):
                        if mujoco_idx >= 0 and mujoco_idx < num_joints_in_model:
                            target_dof_pos[mujoco_idx] = loco_action[policy_idx]

                    startup_step += 1

                    if print_elevation_z and (
                        policy_step <= print_elevation_z_first_n
                        or policy_step % print_elevation_z_interval == 0
                    ):
                        print(
                            format_elevation_debug(
                                elevation_map, policy_step, float(mj_data.time)
                            )
                        )

                    if args.verify_alignment and policy_step == policy_delay_steps + 1:
                        print("=== Alignment check (first policy step) ===")
                        print(f"  gravity: {gravity_orientation}")
                        if gravity_orientation[2] > -0.9:
                            print(
                                "  WARNING: robot already tilted before policy — "
                                "check PD gains / init pose, not gravity formula"
                            )
                        print(f"  cmd (raw/scaled): {current_cmd[:3]} / {current_cmd[:3] * cmd_scale}")
                        print(f"  base_ang_vel (scaled): {proprio[0:3]}")
                        print(f"  joint_pos_rel[0:5]: {proprio[9:14]}")
                        print(f"  joint_vel (scaled)[0:5]: {proprio[9+num_actions:9+num_actions+5]}")
                        print(f"  last_action[0:5]: {proprio[9+2*num_actions:9+2*num_actions+5]}")
                        print(f"  startup_blend alpha: {alpha:.3f} (effective action = blend of last/raw)")
                        ez = elevation_map[2]
                        print(f"  elev_z mean/min/max: {ez.mean():.3f} {ez.min():.3f} {ez.max():.3f}")
                        if ez.mean() > -0.5:
                            print(
                                "  WARNING: elev_z too high — robot likely fallen or "
                                "rays hit legs (expect mean ~ -0.75..-0.85 on flat ground)"
                            )
                        print(f"  raw_action range: ({raw_action.min():.3f}, {raw_action.max():.3f})")
                        imax = int(np.argmax(np.abs(raw_action)))
                        imin = int(np.argmin(raw_action))
                        print(
                            f"  raw_action extrema: max@{imax} {ISAACLAB_JOINT_ORDER[imax]}="
                            f"{raw_action[imax]:.3f}, min@{imin} {ISAACLAB_JOINT_ORDER[imin]}="
                            f"{raw_action[imin]:.3f}"
                        )
                        print("  leg joints (policy idx | name | q | q_target | raw | blended):")
                        for pname in (
                            "left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint",
                            "left_ankle_pitch_joint", "right_hip_pitch_joint", "right_knee_joint",
                            "right_ankle_pitch_joint",
                        ):
                            pi = ISAACLAB_JOINT_ORDER.index(pname)
                            print(
                                f"    [{pi:2d}] {pname:26s} q={qj_policy[pi]:6.3f} "
                                f"tgt={loco_action[pi]:6.3f} raw={raw_action[pi]:7.3f} "
                                f"blend={action[pi]:7.3f}"
                            )

                    if policy_step % 25 == 0:
                        print(f"Inference time: {inference_time:.3f}ms | Command: {current_cmd}")

                # 2) PD torque + batch physics steps.
                for _ in range(config["control_decimation"]):
                    tau = pd_control(
                        target_dof_pos[:num_joints_in_model],
                        mj_data.qpos[7:7 + num_joints_in_model],
                        kps[:num_joints_in_model],
                        np.zeros(num_joints_in_model),
                        mj_data.qvel[6:6 + num_joints_in_model],
                        kds[:num_joints_in_model],
                    )
                    tau = np.clip(tau, -torque_limit[:num_joints_in_model], torque_limit[:num_joints_in_model])
                    mj_data.ctrl[:num_joints_in_model] = tau
                    mujoco.mj_step(mj_model, mj_data)

            except KeyboardInterrupt:
                print("\nInterrupted by user, shutting down...")
                running_main = False
            except Exception as e:
                print(f"\nError during simulation step: {e}")
                running_main = False

            if terrain_scanner is not None:
                draw_hit_markers(viewer, terrain_scanner.ray_hits_w)
            else:
                viewer.user_scn.ngeom = 0

            viewer.sync()
            time_until_next_step = mj_per_step_duration - (time.time() - step_start)
            if time_until_next_step > 0:
                try:
                    time.sleep(time_until_next_step)
                except KeyboardInterrupt:
                    print("\nInterrupted during sleep, shutting down...")
                    running_main = False

    running[0] = False
    print("Simulation ended")
