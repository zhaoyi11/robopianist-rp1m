"""Allegro Hand Composer class."""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_env import specs
from mujoco_utils import mjcf_utils, physics_utils, spec_utils, types

from robopianist.models.hands import base

################
## Constants ##
################


_HERE = Path(__file__).resolve().parent
_LEAP_HAND_DIR = _HERE / "third_party" / "leap_hand"


FINGERTIP_BODIES: Tuple[str, ...] = (
    "if_ds",
    "mf_ds",
    "rf_ds",
    "th_ds",
    )

FINGERTIP_COLORS: Tuple[Tuple[float, float, float], ...] = (
    # Important: the order of these colors should not be changed.
    (0.8, 0.2, 0.8),  # Purple.
    (0.8, 0.2, 0.2),  # Red.
    (0.2, 0.8, 0.8),  # Cyan.
    (0.2, 0.2, 0.8),  # Blue.
)

# Path to the shadow hand E3M5 XML file.
RIGHT_LEAP_HAND_XML = _LEAP_HAND_DIR / "right_hand.xml"
LEFT_LEAP_HAND_XML = _LEAP_HAND_DIR / "left_hand.xml"

@dataclass(frozen=True)
class Dof:
    """Wrist degree of freedom."""

    joint_type: str
    axis: Tuple[int, int, int]
    stiffness: float
    joint_range: Tuple[float, float]
    reflect: bool = False

_FINGERTIP_OFFSET = 0.03
_THUMBTIP_OFFSET = 0.045


class LeapHand(base.Hand):
    """An Leap Hand."""

    def _build(self, name: Optional[str] = None, 
               side: base.HandSide = base.HandSide.RIGHT,
               primitive_fingertip_collisions: bool = False,
               restrict_wrist_yaw_range: bool = False,
               reduced_action_space: bool = False,
               forearm_dofs: Sequence[str] = ("forearm_tx", "forearm_ty")
    )-> None:
        """Initializes a LeapHand.
        
        Args:
            name: Name of the hand. Used as a prefix in the MJCF name attributes.
            side: Which side (left or right) to model.
            primitive_fingertip_collisions: Whether to use capsule approximations for
                the fingertip colliders or the true meshes. Using primitive colliders
                speeds up the simulation.
            restric_wrist_yaw_range: Whether to restrict the range of the wrist yaw
                joint.
            forearm_dofs: Which dofs to add to the forearm.
        """
        if side == base.HandSide.RIGHT:
            self._prefix = "rh_"
            xml_file = RIGHT_LEAP_HAND_XML
        elif side == base.HandSide.LEFT:
            self._prefix = "lh_"
            xml_file = LEFT_LEAP_HAND_XML
        name = name or self._prefix + "leap_hand"

        self._hand_side = side
        self._mjcf_root = mjcf.from_path(str(xml_file))
        self._mjcf_root.model = name
        self._n_forearm_dofs = 0
        self._reduce_action_space = reduced_action_space
        self._forearm_dofs = forearm_dofs

        # self._add_dofs()

        self._parse_mjcf_elements()
        self._add_mjcf_elements()

        self._action_spec = None
    
    def _build_observables(self) -> "LeapHandObservables":
        return LeapHandObservables(self)
    
    def _parse_mjcf_elements(self) -> None:
        joints = mjcf_utils.safe_find_all(self._mjcf_root, "joint")
        actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")

        self._joints = tuple(joints)
        self._actuators = tuple(actuators)

    def _add_mjcf_elements(self) -> None:
        # Add sites to the tips of the fingers.
        fingertip_sites = []
        for tip_name in FINGERTIP_BODIES:
            tip_elem = mjcf_utils.safe_find(
                self._mjcf_root, "body", self._prefix + tip_name
            )
            offset = _THUMBTIP_OFFSET if tip_name == "thdistal" else _FINGERTIP_OFFSET
            tip_site = tip_elem.add(
                "site",
                name=tip_name + "_site",
                pos=(0.0, 0.0, offset),
                type="sphere",
                size=(0.004,),
                group=composer.SENSOR_SITES_GROUP,
            )
            fingertip_sites.append(tip_site)
        self._fingertip_sites = tuple(fingertip_sites)

        # Add joint torque sensors.
        joint_torque_sensors = []
        for joint_elem in self._joints:
            site_elem = joint_elem.parent.add(
                "site",
                name=joint_elem.name + "_site",
                size=(0.001, 0.001, 0.001),
                type="box",
                rgba=(0, 1, 0, 1),
                group=composer.SENSOR_SITES_GROUP,
            )
            torque_sensor_elem = joint_elem.root.sensor.add(
                "torque",
                site=site_elem,
                name=joint_elem.name + "_torque",
            )
            joint_torque_sensors.append(torque_sensor_elem)
        self._joint_torque_sensors = tuple(joint_torque_sensors)

        # Add velocity and force sensors to the actuators.
        actuator_velocity_sensors = []
        actuator_force_sensors = []
        for actuator_elem in self._actuators:
            velocity_sensor_elem = self._mjcf_root.sensor.add(
                "actuatorvel",
                actuator=actuator_elem,
                name=actuator_elem.name + "_velocity",
            )
            actuator_velocity_sensors.append(velocity_sensor_elem)

            force_sensor_elem = self._mjcf_root.sensor.add(
                "actuatorfrc",
                actuator=actuator_elem,
                name=actuator_elem.name + "_force",
            )
            actuator_force_sensors.append(force_sensor_elem)
        self._actuator_velocity_sensors = tuple(actuator_velocity_sensors)
        self._actuator_force_sensors = tuple(actuator_force_sensors)

        # Add touch sensors to the fingertips.
        fingertip_touch_sensors = []
        for tip_name in FINGERTIP_BODIES:
            tip_elem = mjcf_utils.safe_find(
                self._mjcf_root, "body", self._prefix + tip_name
            )
            offset = _THUMBTIP_OFFSET if tip_name == "thdistal" else _FINGERTIP_OFFSET
            touch_site = tip_elem.add(
                "site",
                name=tip_name + "_touch_site",
                pos=(0.0, 0.0, offset),
                type="sphere",
                size=(0.01,),
                group=composer.SENSOR_SITES_GROUP,
                rgba=(0, 1, 0, 0.6),
            )
            touch_sensor = self._mjcf_root.sensor.add(
                "touch",
                site=touch_site,
                name=tip_name + "_touch",
            )
            fingertip_touch_sensors.append(touch_sensor)
        self._fingertip_touch_sensors = tuple(fingertip_touch_sensors)
         
    # Accessors.

    @property
    def hand_side(self) -> base.HandSide:
        return self._hand_side

    @property
    def mjcf_model(self) -> types.MjcfRootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @property
    def n_forearm_dofs(self) -> int:
        return self._n_forearm_dofs

    @composer.cached_property
    def root_body(self) -> types.MjcfElement:
        return mjcf_utils.safe_find(self._mjcf_root, "body", self._prefix + "palm")

    @composer.cached_property
    def fingertip_bodies(self) -> Sequence[types.MjcfElement]:
        return tuple(
            mjcf_utils.safe_find(self._mjcf_root, "body", self._prefix + name)
            for name in FINGERTIP_BODIES
        )

    @property
    def joints(self) -> Sequence[types.MjcfElement]:
        return self._joints

    @property
    def actuators(self) -> Sequence[types.MjcfElement]:
        return self._actuators

    @property
    def joint_torque_sensors(self) -> Sequence[types.MjcfElement]:
        return self._joint_torque_sensors

    @property
    def fingertip_sites(self) -> Sequence[types.MjcfElement]:
        return self._fingertip_sites

    @property
    def actuator_velocity_sensors(self) -> Sequence[types.MjcfElement]:
        return self._actuator_velocity_sensors

    @property
    def actuator_force_sensors(self) -> Sequence[types.MjcfElement]:
        return self._actuator_force_sensors

    @property
    def fingertip_touch_sensors(self) -> Sequence[types.MjcfElement]:
        return self._fingertip_touch_sensors
    
    # Action specs.
    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        if self._action_spec is None:
            self._action_spec = spec_utils.create_action_spec(
                physics=physics, actuators=self.actuators, prefix=self.name
            )

        return self._action_spec

    def apply_action(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        del random_state  # Unused.
        physics.bind(self.actuators).ctrl = action


class LeapHandObservables(base.HandObservables):
    """LeapHand observables."""

    _entity: LeapHand

    @composer.observable
    def actuators_force(self):
        """Returns the actuator forces."""
        return observable.MJCFFeature("sensordata", self._entity.actuator_force_sensors)

    @composer.observable
    def actuators_velocity(self):
        """Returns the actuator velocities."""
        return observable.MJCFFeature(
            "sensordata", self._entity.actuator_velocity_sensors
        )

    @composer.observable
    def actuators_power(self):
        """Returns the actuator powers."""

        def _get_actuator_power(physics: mjcf.Physics) -> np.ndarray:
            force = physics.bind(self._entity.actuator_force_sensors).sensordata
            velocity = physics.bind(self._entity.actuator_velocity_sensors).sensordata
            return abs(force) * abs(velocity)

        return observable.Generic(raw_observation_callable=_get_actuator_power)

    @composer.observable
    def fingertip_positions(self):
        """Returns the fingertip positions in world coordinates."""

        def _get_fingertip_positions(physics: mjcf.Physics) -> np.ndarray:
            return physics.bind(self._entity.fingertip_sites).xpos.ravel()

        return observable.Generic(raw_observation_callable=_get_fingertip_positions)

    @composer.observable
    def fingertip_force(self):
        """Returns for each finger, the sum of forces felt at the fingertip."""

        def _get_fingertip_force(physics: mjcf.Physics) -> np.ndarray:
            return physics.bind(self._entity.fingertip_touch_sensors).sensordata

        return observable.Generic(raw_observation_callable=_get_fingertip_force)
