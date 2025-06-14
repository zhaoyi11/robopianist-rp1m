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
_ALLEGRO_HAND_DIR = _HERE / "third_party" / "wonik_allegro"

NQ = 16  # Number of joints.
NU = 16  # Number of actuators.


# TODO: check figertip body
FINGERTIP_BODIES: Tuple[str, ...] = (
    # Important: the order of these names should not be changed.
    "thdistal",
    "ffdistal",
    "mfdistal",
    "rfdistal",
    "lfdistal",
)

FINGERTIP_COLORS: Tuple[Tuple[float, float, float], ...] = (
    # Important: the order of these colors should not be changed.
    (0.8, 0.2, 0.8),  # Purple.
    (0.8, 0.2, 0.2),  # Red.
    (0.2, 0.8, 0.8),  # Cyan.
    (0.2, 0.2, 0.8),  # Blue.
)

# Path to the shadow hand E3M5 XML file.
RIGHT_ALLEGRO_HAND_XML = _ALLEGRO_HAND_DIR / "right_hand.xml"
LEFT_ALLEGRO_HAND_XML = _ALLEGRO_HAND_DIR / "left_hand.xml"


@dataclass(frozen=True)
class Dof:
    """Wrist degree of freedom."""

    joint_type: str
    axis: Tuple[int, int, int]
    stiffness: float
    joint_range: Tuple[float, float]
    reflect: bool = False


_WRIST_DOFS: Dict[str, Dof] = {
    "wrist_tx": Dof(
        joint_type="slide",
        pos=(0, 0, 0),
        axis=(-1, 0, 0),
        stiffness=300.0,
        # Note this is a dummy range, it will be set to the piano's length at task
        # initialization, see `robopianist/suite/tasks/base.py`.
        joint_range=(-1, 1),
    ),
    "wrist_ty": Dof(
        joint_type="slide",
        axis=(0, 0, 1),
        stiffness=300.0,
        joint_range=(0.0, 0.06),
    ),
    "wrist_pitch": Dof(
        joint_type="hinge",
        axis=(1, 0, 0),
        stiffness=50.0,
        joint_range=(-0.523599, 0.174533),
    ),
    "wrist_yaw": Dof(
        joint_type="hinge",
        axis=(0, 0, 1),
        stiffness=300.0,
        joint_range=(-0.698132, 0.488692),
        # reflect=True,
    ),
}

_FINGERTIP_OFFSET = 0.026


class AllegroHand(base.Hand):
    """An Allegro Hand."""

    def _build(self, name: Optional[str] = None, 
               side: base.HandSide = base.HandSide.RIGHT,
               primitive_fingertip_collisions: bool = False,
               restric_wrist_yaw_range: bool = False,
               reduced_action_space: bool = False,
               forearm_dofs: Sequence[str] = ("forearm_tx", "forearm_ty")
    )-> None:
        """Initializes a AllegroHand.
        
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
            xml_file = RIGHT_ALLEGRO_HAND_XML
        elif side == base.HandSide.LEFT:
            self._prefix = "lh_"
            xml_file = LEFT_ALLEGRO_HAND_XML
        name = name or self._prefix + "allegro_hand"

        self._hand_side = side
        self._mjcf_root = mjcf.from_path(str(xml_file))
        self._mjcf_root.model = name
        self._n_forearm_dofs = 0
        self._reduce_action_space = reduced_action_space
        self._forearm_dofs = forearm_dofs

        self._add_dofs()

        self._parse_mjcf_elements()
        self._add_mjcf_elements()

    
    def _build_observables(self) -> "AllegroHandObservables":
        return AllegroHandObservables(self)
    
    def _parse_mjcf_elements(self) -> None:
        joints = mjcf_utils.safe_find_all(self._mjcf_root, "joint")
        actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")

        self._joints = tuple(joints)
        self._actuators = tuple(actuators)

    def _add_mjcf_elements(self) -> None:
        # Add sites to the tips of the fingers.
        for name in FINGERTIP_BODIES:
            site = self._mjcf_root.worldbody.add(
                "site",
                
            )

    def _add_dofs(self) -> None:
        # Add the dofs.
        def _maybe_reflect_axis(
            axis: Sequence[float], reflect: bool
        ) -> Sequence[float]:
            if self._hand_side == base.HandSide.LEFT and reflect:
                return tuple([-a for a in axis])
            return axis
        
        for dof_name in self._forearm_dofs:
            if dof_name not in _WRIST_DOFS:
                raise ValueError(
                    f"Invalid forearm DOF: {dof_name}. Valid DOFs are: "
                    f"{_WRIST_DOFS}."
                )   
            
            dof = _WRIST_DOFS[dof_name]

            joint = self.root_body.add(
                "joint",
                type=dof.joint_type,
                name=dof_name,
                axis=_maybe_reflect_axis(dof.axis, dof.reflect),
                range=dof.joint_range,
            )

            joint.damping = physics_utils.get_critical_damping_from_stiffness(
                dof.stiffness, joint.full_identifier, self.mjcf_model
            )

            self._mjcf_root.actuator.add(
                "position",
                name=dof_name,
                joint=joint,
                ctrlrange=dof.joint_range,
                kp=dof.stiffness,
            )

            self._n_forearm_dofs += 1

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



class AllegroHandObservables(base.HandObservables):
    """AllegroHand observables."""

    _entity: AllegroHand

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
