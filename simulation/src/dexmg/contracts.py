"""Native field contracts audited against all six official task HDF5 files.

A uses two PandaDex arms (robot0 is right); B uses one GR1 robot.
These are learning targets, not a promise that a 30-D B prediction can be
passed directly to the simulator's 24-D axis-angle controller.
"""

from dataclasses import dataclass
from typing import Any


GROUP_TASKS = {
    "A": ("two_arm_box_cleanup", "two_arm_drawer_cleanup", "two_arm_lift_tray"),
    "B": ("two_arm_coffee", "two_arm_pouring", "two_arm_can_sort_random"),
}
ACTION_FIELDS = {
    "A": (("right_rel_pos", 3), ("right_rel_rot_axis_angle", 3),
          ("right_gripper", 6), ("left_rel_pos", 3),
          ("left_rel_rot_axis_angle", 3), ("left_gripper", 6)),
    "B": (("right_abs_pos", 3), ("right_abs_rot_6d", 6),
          ("left_abs_pos", 3), ("left_abs_rot_6d", 6),
          ("right_gripper", 6), ("left_gripper", 6)),
}
STATE_FIELDS = {
    "A": (("robot0_eef_pos", 3), ("robot0_eef_quat", 4),
          ("robot0_gripper_qpos", 12), ("robot1_eef_pos", 3),
          ("robot1_eef_quat", 4), ("robot1_gripper_qpos", 12)),
    "B": (("robot0_right_eef_pos", 3), ("robot0_right_eef_quat", 4),
          ("robot0_right_gripper_qpos", 11), ("robot0_left_eef_pos", 3),
          ("robot0_left_eef_quat", 4), ("robot0_left_gripper_qpos", 11)),
}
ENV_NAMES = {
    "two_arm_box_cleanup": "TwoArmBoxCleanup",
    "two_arm_drawer_cleanup": "TwoArmDrawerCleanup",
    "two_arm_lift_tray": "TwoArmLiftTray",
    "two_arm_coffee": "TwoArmCoffee",
    "two_arm_pouring": "TwoArmPouring",
    # The official file slug differs from the actual robosuite environment.
    "two_arm_can_sort_random": "TwoArmCanSortBlue",
}
PROMPTS = {
    "two_arm_box_cleanup": "Place the lid onto the box to close it.",
    "two_arm_drawer_cleanup": "Open the drawer, place the mug inside, and close the drawer.",
    "two_arm_lift_tray": "Lift the tray with both hands while keeping both objects in the tray.",
    "two_arm_coffee": "Insert the coffee pod into the coffee machine and close the lid.",
    "two_arm_pouring": "Pour the ball from the cup into the bowl, then place the bowl upright on the pad.",
    "two_arm_can_sort_random": "Place the can into the bin matching its color.",
}


def validate_group(group: str) -> str:
    if group not in GROUP_TASKS:
        raise ValueError(f"Unknown group {group!r}; expected A or B")
    return group


def task_group(task: str) -> str:
    for group, tasks in GROUP_TASKS.items():
        if task in tasks:
            return group
    raise ValueError(f"Unknown DexMimicGen task {task!r}")


def action_keys(group: str) -> tuple[str, ...]:
    return tuple(key for key, _ in ACTION_FIELDS[validate_group(group)])


def state_keys(group: str) -> tuple[str, ...]:
    return tuple(key for key, _ in STATE_FIELDS[validate_group(group)])


def action_dim(group: str) -> int:
    return sum(dim for _, dim in ACTION_FIELDS[validate_group(group)])


def state_dim(group: str) -> int:
    return sum(dim for _, dim in STATE_FIELDS[validate_group(group)])


def camera_map(task: str) -> dict[str, str]:
    if task_group(task) == "A":
        return {"base_0_rgb": "agentview_image",
                "left_wrist_0_rgb": "robot1_eye_in_hand_image",
                "right_wrist_0_rgb": "robot0_eye_in_hand_image"}
    base = "frontview_image" if task == "two_arm_can_sort_random" else "agentview_image"
    return {"base_0_rgb": base,
            "left_wrist_0_rgb": "robot0_eye_in_left_hand_image",
            "right_wrist_0_rgb": "robot0_eye_in_right_hand_image"}


@dataclass(frozen=True)
class TaskContract:
    task: str
    group: str
    env_name: str
    action_fields: tuple[tuple[str, int], ...]
    state_fields: tuple[tuple[str, int], ...]
    prompt: str

    @property
    def action_dim(self) -> int:
        return sum(dim for _, dim in self.action_fields)

    @property
    def state_dim(self) -> int:
        return sum(dim for _, dim in self.state_fields)


def task_contract(task: str) -> TaskContract:
    group = task_group(task)
    return TaskContract(task, group, ENV_NAMES[task], ACTION_FIELDS[group],
                        STATE_FIELDS[group], PROMPTS[task])


def validate_env_args(task: str, env_args: dict[str, Any]) -> None:
    """Reject task/robot/controller/camera metadata outside the audited contract."""
    contract = task_contract(task)
    env_names = (("TwoArmCanSortBlue", "TwoArmCanSortRed", "TwoArmCanSortRandom")
                 if task == "two_arm_can_sort_random" else (contract.env_name,))
    if not isinstance(env_args, dict) or env_args.get("env_name") not in env_names:
        raise ValueError(f"{task}: env_name must be one of {env_names}")
    kwargs = env_args.get("env_kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError(f"{task}: env_kwargs metadata is required")
    expected_robots = (["PandaDexRH", "PandaDexLH"] if contract.group == "A" else
                       ["GR1ArmsOnly" if task == "two_arm_can_sort_random" else "GR1FixedLowerBody"])
    if kwargs.get("robots") != expected_robots:
        raise ValueError(f"{task}: robots must be {expected_robots}")
    cameras = kwargs.get("camera_names")
    required = {key.removesuffix("_image") for key in camera_map(task).values()}
    if not isinstance(cameras, (tuple, list)) or not required.issubset(cameras):
        raise ValueError(f"{task}: camera_names must include {sorted(required)}")
    controller = kwargs.get("controller_configs")
    if not isinstance(controller, dict):
        raise ValueError(f"{task}: controller_configs metadata is required")
    if contract.group == "A":
        part = controller.get("body_parts", {}).get("right", {})
        if (controller.get("type") != "BASIC" or part.get("type") != "OSC_POSE"
                or part.get("input_type") != "delta" or part.get("input_ref_frame") != "base"):
            raise ValueError(f"{task}: expected BASIC/OSC_POSE delta/base controller")
    else:
        config = controller.get("composite_controller_specific_configs", {})
        if (controller.get("type") != "WHOLE_BODY_MINK_IK"
                or config.get("ik_input_type") != "absolute"
                or config.get("ik_input_ref_frame") != "world"
                or config.get("ik_input_rotation_repr") != "axis_angle"):
            raise ValueError(f"{task}: expected WHOLE_BODY_MINK_IK absolute/world/axis_angle controller")
