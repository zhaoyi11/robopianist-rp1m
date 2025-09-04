# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base piano composer task."""

from typing import Sequence

import mujoco
import numpy as np
from dm_control import composer
from mujoco_utils import composer_utils, physics_utils

from robopianist.models.hands import HandSide, shadow_hand, allegro_hand, orca_hand
from robopianist.models.piano import piano

# Timestep of the physics simulation, in seconds.
_PHYSICS_TIMESTEP = 0.005

# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP = 0.05  # 20 Hz.

# Default position and orientation of the hands
# shadow hand
_LEFT_SHADOW_HAND_POSITION = (0.4, -0.15, 0.13)
_LEFT_SHADOW_HAND_QUATERNION = (-1, -1, 1, 1)
_RIGHT_SHADOW_HAND_POSITION = (0.4, 0.15, 0.13)
_RIGHT_SHADOW_HAND_QUATERNION = (-1, -1, 1, 1)

# allegro hand
_LEFT_ALLEGRO_HAND_POSITION = (0.1, -0.2, 0.15)
_LEFT_ALLEGRO_HAND_QUATERNION = (0, -0.7071, 0, 0.7071)
_RIGHT_ALLEGRO_HAND_POSITION = (0.1, 0.2, 0.15)
_RIGHT_ALLEGRO_HAND_QUATERNION = (0, -0.7071, 0, 0.7071)

# orca hand
_LEFT_ORCA_HAND_POSITION = (0.23, -0.2, 0.11)
_LEFT_ORCA_HAND_QUATERNION = (0.5, -0.5, -0.5, 0.5)
_RIGHT_ORCA_HAND_POSITION = (0.23, 0.2, 0.11)
_RIGHT_ORCA_HAND_QUATERNION = (0.5, -0.5, -0.5, 0.5)



# # Default position and orientation of the hands.
# # _LEFT_HAND_POSITION =  # allegro hand
# _LEFT_HAND_POSITION = (0.23, -0.2, 0.11) # orca hand

# # _LEFT_HAND_QUATERNION = (-1, -1, 1, 1) # shadow hand
# # _LEFT_HAND_QUATERNION = (0, -0.7071, 0, 0.7071) # allegro hand
# _LEFT_HAND_QUATERNION = (0.5, -0.5, -0.5, 0.5) # orca hand

# # _RIGHT_HAND_POSITION = (0.1, 0.2, 0.15) # allegro hand
# _RIGHT_HAND_POSITION = (0.23, 0.2, 0.11) # orca hand

# # _RIGHT_HAND_QUATERNION = (-1, -1, 1, 1) # shadow hand
# # _RIGHT_HAND_QUATERNION = (0, -0.7071, 0, 0.7071) # allegro hand
# _RIGHT_HAND_QUATERNION = (0.5, -0.5, -0.5, 0.5) # orca hand
_ATTACHMENT_YAW = 0  # Degrees.


class PianoOnlyTask(composer.Task):
    """Piano task with no hands."""

    def __init__(
        self,
        arena: composer_utils.Arena,
        change_color_on_activation: bool = False,
        add_piano_actuators: bool = False,
        key_scale: float = 1.0,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
    ) -> None:
        self._arena = arena
        self._piano = piano.Piano(
            change_color_on_activation=change_color_on_activation,
            add_actuators=add_piano_actuators,
            key_scale=key_scale,
        )
        arena.attach(self._piano)

        # Harden the piano keys.
        # The default solref parameters are (0.02, 1). In particular, the first
        # parameter specifies -stiffness, and so decreasing it makes the contacts
        # harder. The documentation recommends keeping the stiffness at least 2x larger
        # than the physics timestep, see:
        # https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=stiffness#solver-parameters
        self._piano.mjcf_model.default.geom.solref = (physics_timestep * 2, 1)

        self.set_timesteps(
            control_timestep=control_timestep, physics_timestep=physics_timestep
        )

    # Accessors.

    @property
    def root_entity(self):
        return self._arena

    @property
    def arena(self):
        return self._arena

    @property
    def piano(self) -> piano.Piano:
        return self._piano

    # Composer methods.

    def get_reward(self, physics) -> float:
        del physics  # Unused.
        return 0.0


class PianoTask(PianoOnlyTask):
    """Base class for piano tasks."""

    def __init__(
        self,
        arena: composer_utils.Arena,
        hand_name: str = "shadow",
        gravity_compensation: bool = False,
        change_color_on_activation: bool = False,
        primitive_fingertip_collisions: bool = False,
        reduced_action_space: bool = False,
        attachment_yaw: float = _ATTACHMENT_YAW,
        forearm_dofs: Sequence[str] = shadow_hand._DEFAULT_FOREARM_DOFS,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
    ) -> None:
        # enlarge the key scale for allegro hand
        if hand_name == "allegro":
            key_scale = 1.2
        else:
            key_scale = 1.0
        
        super().__init__(
            arena=arena,
            change_color_on_activation=change_color_on_activation,
            add_piano_actuators=False,
            key_scale=key_scale,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
        )
        if hand_name == "orca":
            self._right_hand = self._add_orca_hand(
                hand_side=HandSide.RIGHT,
                position=_RIGHT_ORCA_HAND_POSITION,
                quaternion=_RIGHT_ORCA_HAND_QUATERNION,
                gravity_compensation=gravity_compensation,
                primitive_fingertip_collisions=primitive_fingertip_collisions,
                reduced_action_space=reduced_action_space,
                attachment_yaw=attachment_yaw,
                forearm_dofs=forearm_dofs,
            )

            self._left_hand = self._add_orca_hand(
                hand_side=HandSide.LEFT,
                position=_LEFT_ORCA_HAND_POSITION,
                quaternion=_LEFT_ORCA_HAND_QUATERNION,
                gravity_compensation=gravity_compensation,
                primitive_fingertip_collisions=primitive_fingertip_collisions,
                reduced_action_space=reduced_action_space,
                attachment_yaw=attachment_yaw,
                forearm_dofs=forearm_dofs,
            )

        elif hand_name == "allegro":
            self._right_hand = self._add_allegro_hand(
                hand_side=HandSide.RIGHT,
                position=_RIGHT_ALLEGRO_HAND_POSITION,
                quaternion=_RIGHT_ALLEGRO_HAND_QUATERNION,
                gravity_compensation=gravity_compensation,
                primitive_fingertip_collisions=primitive_fingertip_collisions,
                reduced_action_space=reduced_action_space,
                attachment_yaw=attachment_yaw,
                forearm_dofs=forearm_dofs,
            )

            self._left_hand = self._add_allegro_hand(
                hand_side=HandSide.LEFT,
                position=_LEFT_ALLEGRO_HAND_POSITION,
                quaternion=_LEFT_ALLEGRO_HAND_QUATERNION,
                gravity_compensation=gravity_compensation,
                primitive_fingertip_collisions=primitive_fingertip_collisions,
                reduced_action_space=reduced_action_space,
                attachment_yaw=attachment_yaw,
                forearm_dofs=forearm_dofs,
            )

        elif hand_name == "shadow":
            self._right_hand = self._add_shadow_hand(
                hand_side=HandSide.RIGHT,
                position=_RIGHT_SHADOW_HAND_POSITION,
                quaternion=_RIGHT_SHADOW_HAND_QUATERNION,
                gravity_compensation=gravity_compensation,
                primitive_fingertip_collisions=primitive_fingertip_collisions,
                reduced_action_space=reduced_action_space,
                attachment_yaw=attachment_yaw,
                forearm_dofs=forearm_dofs,
            )

            self._left_hand = self._add_shadow_hand(
                hand_side=HandSide.LEFT,
                position=_LEFT_SHADOW_HAND_POSITION,
                quaternion=_LEFT_SHADOW_HAND_QUATERNION,
                gravity_compensation=gravity_compensation,
                primitive_fingertip_collisions=primitive_fingertip_collisions,
                reduced_action_space=reduced_action_space,
                attachment_yaw=attachment_yaw,
                forearm_dofs=forearm_dofs,
            )
        else:
            raise ValueError(f"Unknown hand {hand_name}. Available hands: shadow, allegro, orca")

    # Accessors.

    @property
    def left_hand(self) -> shadow_hand.ShadowHand:
        return self._left_hand

    @property
    def right_hand(self) -> shadow_hand.ShadowHand:
        return self._right_hand


    def _add_orca_hand(
        self,
        hand_side: HandSide,
        position,
        quaternion,
        gravity_compensation: bool,
        primitive_fingertip_collisions: bool,
        reduced_action_space: bool,
        attachment_yaw: float,
        forearm_dofs: Sequence[str],
    ) -> orca_hand.OrcaHand:

        joint_range = [-self._piano.size[1], self._piano.size[1]]

        # Offset the joint range by the hand's initial position.
        joint_range[0] -= position[1]
        joint_range[1] -= position[1]

        hand = orca_hand.OrcaHand(
            side=hand_side,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            restrict_wrist_yaw_range=False,
            reduced_action_space=reduced_action_space,
            forearm_dofs=forearm_dofs,
        )
        hand.root_body.pos = position

        # TODO: change the initial pose and the piano key size and action range !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        # Slightly rotate the forearms inwards (Z-axis) to mimic human posture.
        # rotate_axis = np.asarray([0, 0, 1], dtype=np.float64)
        # rotate_by = np.zeros(4, dtype=np.float64)
        # sign = -1 if hand_side == HandSide.LEFT else 1
        # angle = np.radians(sign * attachment_yaw)
        # mujoco.mju_axisAngle2Quat(rotate_by, rotate_axis, angle)
        # final_quaternion = np.zeros(4, dtype=np.float64)
        # # mujoco.mju_mulQuat(final_quaternion, rotate_by, quaternion)
        # hand.root_body.quat = final_quaternion
        hand.root_body.quat = quaternion

        if gravity_compensation:
            physics_utils.compensate_gravity(hand.mjcf_model)

        # Override forearm translation joint range.
        forearm_tx_joint = hand.mjcf_model.find("joint", "wrist_tx")
        if forearm_tx_joint is not None:
            forearm_tx_joint.range = joint_range
        forearm_tx_actuator = hand.mjcf_model.find("actuator", "wrist_tx")
        if forearm_tx_actuator is not None:
            forearm_tx_actuator.ctrlrange = joint_range

        self._arena.attach(hand)
        return hand

    # Helper methods.
    def _add_allegro_hand(
        self,
        hand_side: HandSide,
        position,
        quaternion,
        gravity_compensation: bool,
        primitive_fingertip_collisions: bool,
        reduced_action_space: bool,
        attachment_yaw: float,
        forearm_dofs: Sequence[str],
    ) -> allegro_hand.AllegroHand:
        joint_range = [-self._piano.size[1], self._piano.size[1]]

        # Offset the joint range by the hand's initial position.
        joint_range[0] -= position[1]
        joint_range[1] -= position[1]

        hand = allegro_hand.AllegroHand(
            side=hand_side,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            restrict_wrist_yaw_range=False,
            reduced_action_space=reduced_action_space,
            forearm_dofs=forearm_dofs,
        )
        hand.root_body.pos = position

        # TODO: change the initial pose and the piano key size and action range !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        # Slightly rotate the forearms inwards (Z-axis) to mimic human posture.
        # rotate_axis = np.asarray([0, 0, 1], dtype=np.float64)
        # rotate_by = np.zeros(4, dtype=np.float64)
        # sign = -1 if hand_side == HandSide.LEFT else 1
        # angle = np.radians(sign * attachment_yaw)
        # mujoco.mju_axisAngle2Quat(rotate_by, rotate_axis, angle)
        # final_quaternion = np.zeros(4, dtype=np.float64)
        # # mujoco.mju_mulQuat(final_quaternion, rotate_by, quaternion)
        # hand.root_body.quat = final_quaternion
        hand.root_body.quat = quaternion

        if gravity_compensation:
            physics_utils.compensate_gravity(hand.mjcf_model)

        # Override forearm translation joint range.
        forearm_tx_joint = hand.mjcf_model.find("joint", "wrist_tx")
        if forearm_tx_joint is not None:
            forearm_tx_joint.range = joint_range
        forearm_tx_actuator = hand.mjcf_model.find("actuator", "wrist_tx")
        if forearm_tx_actuator is not None:
            forearm_tx_actuator.ctrlrange = joint_range

        self._arena.attach(hand)
        return hand

    def _add_shadow_hand(
        self,
        hand_side: HandSide,
        position,
        quaternion,
        gravity_compensation: bool,
        primitive_fingertip_collisions: bool,
        reduced_action_space: bool,
        attachment_yaw: float,
        forearm_dofs: Sequence[str],
    ) -> shadow_hand.ShadowHand:
        joint_range = [-self._piano.size[1], self._piano.size[1]]

        # Offset the joint range by the hand's initial position.
        joint_range[0] -= position[1]
        joint_range[1] -= position[1]

        hand = shadow_hand.ShadowHand(
            side=hand_side,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            restrict_wrist_yaw_range=False,
            reduced_action_space=reduced_action_space,
            forearm_dofs=forearm_dofs,
        )
        hand.root_body.pos = position

        # Slightly rotate the forearms inwards (Z-axis) to mimic human posture.
        rotate_axis = np.asarray([0, 0, 1], dtype=np.float64)
        rotate_by = np.zeros(4, dtype=np.float64)
        sign = -1 if hand_side == HandSide.LEFT else 1
        angle = np.radians(sign * attachment_yaw)
        mujoco.mju_axisAngle2Quat(rotate_by, rotate_axis, angle)
        final_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(final_quaternion, rotate_by, quaternion)
        hand.root_body.quat = final_quaternion

        if gravity_compensation:
            physics_utils.compensate_gravity(hand.mjcf_model)

        # Override forearm translation joint range.
        forearm_tx_joint = hand.mjcf_model.find("joint", "forearm_tx")
        if forearm_tx_joint is not None:
            forearm_tx_joint.range = joint_range
        forearm_tx_actuator = hand.mjcf_model.find("actuator", "forearm_tx")
        if forearm_tx_actuator is not None:
            forearm_tx_actuator.ctrlrange = joint_range

        self._arena.attach(hand)
        return hand
