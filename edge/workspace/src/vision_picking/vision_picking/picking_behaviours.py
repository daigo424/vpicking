#!/usr/bin/env python3
"""picking_controller_node.pyが組み立てるbehavior treeの葉ノード群。

各behaviourはpicking_robot_interface.RobotInterfaceの薄いラッパーで、実際の処理は
すべてRobotInterface側のメソッドに委譲する(ロジックの重複を避けるため)。
姿勢データ(coarse_pose/fine_pose)はpy_treesのblackboardを介してbehaviour間で共有する。
"""

import py_trees
from geometry_msgs.msg import Pose, Quaternion

from vision_picking.picking_robot_interface import DOWNWARD_ORIENTATION, WORLD_FRAME, RobotInterface

Status = py_trees.common.Status


def _position_xyz(pose: Pose) -> tuple[float, float, float]:
    return (pose.position.x, pose.position.y, pose.position.z)


class LookupTargetPose(py_trees.behaviour.Behaviour):
    """target_frameのworld系TFが取得できるまでRUNNINGを返し続け、取得できたらblackboardへ書き込む。"""

    def __init__(self, name: str, interface: RobotInterface, target_frame: str, blackboard_key: str) -> None:
        super().__init__(name=name)
        self._interface = interface
        self._target_frame = target_frame
        self._blackboard_key = blackboard_key
        self._waiting_logged = False
        self.blackboard = self.attach_blackboard_client(name=name)
        self.blackboard.register_key(key=blackboard_key, access=py_trees.common.Access.WRITE)

    def update(self) -> Status:
        pose = self._interface.lookup_target_pose(self._target_frame)
        if pose is None:
            if not self._waiting_logged:
                self._interface.node.get_logger().info(f"{self._target_frame}のTFを待機中...")
                self._waiting_logged = True
            return Status.RUNNING
        setattr(self.blackboard, self._blackboard_key, pose)
        return Status.SUCCESS


class RegisterCollisionObject(py_trees.behaviour.Behaviour):
    """positionはblackboard(pose)か固定値、orientationは省略時はposition側のposeの向きを使う。

    プレイス後の再登録(退避後)のように、位置は固定(プレイス目標)だが向きは把持時の
    ものを引き継ぎたい場合はorientation_keyを別途指定する。
    """

    def __init__(
        self,
        name: str,
        interface: RobotInterface,
        position_key: str | None = None,
        static_position: tuple[float, float, float] | None = None,
        orientation_key: str | None = None,
    ) -> None:
        super().__init__(name=name)
        assert position_key or static_position, "position_keyかstatic_positionのどちらかが必要"
        self._interface = interface
        self._position_key = position_key
        self._static_position = static_position
        self._orientation_key = orientation_key
        self.blackboard = self.attach_blackboard_client(name=name)
        for key in (position_key, orientation_key):
            if key:
                self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

    def update(self) -> Status:
        if self._position_key:
            position_pose = getattr(self.blackboard, self._position_key)
            x, y, z = _position_xyz(position_pose)
            orientation = position_pose.orientation
        else:
            x, y, z = self._static_position
            orientation = None
        if self._orientation_key:
            orientation = getattr(self.blackboard, self._orientation_key).orientation

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation = orientation
        self._interface.register_collision_object(WORLD_FRAME, "target_object", pose)
        return Status.SUCCESS


class RemoveCollisionObject(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, interface: RobotInterface) -> None:
        super().__init__(name=name)
        self._interface = interface

    def update(self) -> Status:
        self._interface.remove_collision_object("target_object")
        return Status.SUCCESS


class MoveTo(py_trees.behaviour.Behaviour):
    """positionはblackboard(pose)か固定値、orientationはblackboard(pose)か固定値から取る一発移動。"""

    def __init__(
        self,
        name: str,
        interface: RobotInterface,
        height_offset: float,
        log_label: str,
        position_key: str | None = None,
        static_position: tuple[float, float, float] | None = None,
        orientation_key: str | None = None,
        static_orientation: Quaternion = DOWNWARD_ORIENTATION,
        verify_and_correct: bool = False,
        yaw_free: bool = False,
    ) -> None:
        super().__init__(name=name)
        assert position_key or static_position, "position_keyかstatic_positionのどちらかが必要"
        self._interface = interface
        self._height_offset = height_offset
        self._log_label = log_label
        self._position_key = position_key
        self._static_position = static_position
        self._orientation_key = orientation_key
        self._static_orientation = static_orientation
        self._verify_and_correct = verify_and_correct
        self._yaw_free = yaw_free
        self.blackboard = self.attach_blackboard_client(name=name)
        for key in (position_key, orientation_key):
            if key:
                self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

    def update(self) -> Status:
        self._interface.node.get_logger().info(self._log_label)
        if self._position_key:
            x, y, z = _position_xyz(getattr(self.blackboard, self._position_key))
        else:
            x, y, z = self._static_position
        orientation = (
            getattr(self.blackboard, self._orientation_key).orientation
            if self._orientation_key
            else self._static_orientation
        )
        try:
            self._interface.move_to(
                (x, y, z + self._height_offset),
                orientation,
                verify_and_correct=self._verify_and_correct,
                yaw_free=self._yaw_free,
            )
        except RuntimeError as e:
            self._interface.node.get_logger().error(str(e))
            return Status.FAILURE
        return Status.SUCCESS


class SetGripper(py_trees.behaviour.Behaviour):
    def __init__(
        self, name: str, interface: RobotInterface, positions: list[float], log_label: str | None = None
    ) -> None:
        super().__init__(name=name)
        self._interface = interface
        self._positions = positions
        self._log_label = log_label

    def update(self) -> Status:
        if self._log_label:
            self._interface.node.get_logger().info(self._log_label)
        self._interface.set_gripper(self._positions)
        return Status.SUCCESS


class AttachTarget(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, interface: RobotInterface) -> None:
        super().__init__(name=name)
        self._interface = interface

    def update(self) -> Status:
        self._interface.attach_target()
        return Status.SUCCESS


class DetachTarget(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, interface: RobotInterface) -> None:
        super().__init__(name=name)
        self._interface = interface

    def update(self) -> Status:
        self._interface.detach_target()
        return Status.SUCCESS


class LogFingerPositions(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, interface: RobotInterface, label: str, position_key: str) -> None:
        super().__init__(name=name)
        self._interface = interface
        self._label = label
        self._position_key = position_key
        self.blackboard = self.attach_blackboard_client(name=name)
        self.blackboard.register_key(key=position_key, access=py_trees.common.Access.READ)

    def update(self) -> Status:
        target_xyz = _position_xyz(getattr(self.blackboard, self._position_key))
        self._interface.log_finger_positions(self._label, target_xyz)
        return Status.SUCCESS


class VerifyGrasped(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, interface: RobotInterface, position_key: str) -> None:
        super().__init__(name=name)
        self._interface = interface
        self._position_key = position_key
        self.blackboard = self.attach_blackboard_client(name=name)
        self.blackboard.register_key(key=position_key, access=py_trees.common.Access.READ)

    def update(self) -> Status:
        pick_z = getattr(self.blackboard, self._position_key).position.z
        try:
            self._interface.verify_grasped(pick_z)
        except RuntimeError as e:
            self._interface.node.get_logger().error(str(e))
            return Status.FAILURE
        return Status.SUCCESS


class VerifyPlaced(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, interface: RobotInterface) -> None:
        super().__init__(name=name)
        self._interface = interface

    def update(self) -> Status:
        self._interface.node.get_logger().info("ピック&プレイス完了")
        try:
            self._interface.verify_placed()
        except RuntimeError as e:
            self._interface.node.get_logger().error(str(e))
            return Status.FAILURE
        return Status.SUCCESS
