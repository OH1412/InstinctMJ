from dataclasses import MISSING, dataclass

from instinct_mj.managers import CommandTermCfg

from .pose_velocity_command import PoseVelocityCommand


@dataclass(kw_only=True)
class PoseVelocityCommandCfg(CommandTermCfg):
    """Configuration for the position command generator."""

    class_type: type = PoseVelocityCommand

    entity_name: str = MISSING
    """Name of the entity in the environment for which commands are generated."""

    velocity_control_stiffness: float = 1.0
    """Scale factor to convert the position error to linear velocity command. Defaults to 1.0."""

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    only_positive_lin_vel_x: bool = False
    """Whether to only sample positive linear x velocity commands. Defaults to False."""

    @dataclass(kw_only=True)
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands. Only used in random_velocity_terrains."""

    random_velocity_terrain: list[str] = None
    """List of terrain types for which the velocity commands should be randomized."""

    velocity_ranges: dict = None
    """Dictionary containing velocity ranges for different terrains. If not None, the velocity ranges will be set based on the terrain type."""

    lin_vel_threshold: float = 0.15
    """Minimal threshold for the linear velocity command (in m/s)."""

    ang_vel_threshold: float = 0.15
    """Minimal threshold for the angular velocity command (in rad/s)."""

    lin_vel_metrics_std: float = 0.5
    """Standard deviation for the linear velocity metrics."""

    ang_vel_metrics_std: float = 0.5
    """Standard deviation for the angular velocity metrics."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    target_dis_threshold: float = 0.2
    """The distance threshold to the target position below which the command is set to zero. Defaults to 0.2."""

    patch_vis: bool = False
    """Whether to visualize all flat patches (can be expensive with many patches)."""

    corridor_target_layer_offsets: tuple[int, ...] = ()
    """Sample a fixed corridor goal relative to the spawn layer, including ground."""

    def build(self, env):
        return self.class_type(self, env)
