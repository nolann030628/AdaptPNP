import os 
import sys
import torch
import carb
import numpy as np
from typing import List, Optional
from termcolor import cprint

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix, rot_matrix_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver, Franka
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow, RmpFlowSmoothed
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config
from isaacsim.robot_motion.motion_generation.articulation_motion_policy import ArticulationMotionPolicy

sys.path.append(os.getcwd())
from Env_Config.Utils_Project.Set_Drive import set_drive
from Env_Config.Utils_Project.Transforms import quat_diff_rad, Rotation, get_pose_relat, get_pose_world, quat_mul
from Env_Config.Utils_Project.Code_Tools import float_truncate, dense_trajectory_points_generation

class Franka(Robot):
    def __init__(
        self, 
        world:World, 
        position:np.ndarray, 
        orientation:np.ndarray, 
        robot_name:str="Franka"
    )->None:
        # define world
        self.world = world
        # define Franka name
        self._name = robot_name
        # define Franka prim
        self._prim_path = "/World/"+self._name
        # get Franka usd file
        self.asset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Assets/Robots/franka/franka.usd")
        # define Franka positon
        self.position = position
        # define Franka orientation
        self.orientation = euler_angles_to_quat(orientation,degrees=True)
        
        # add Franka USD to stage
        add_reference_to_stage(self.asset_file, self._prim_path)
        # set Franka
        super().__init__(
            prim_path=self._prim_path,
            name=self._name,
            position=self.position,
            orientation=self.orientation,
            articulation_controller = None
        )
        # set Franka end effector
        self._end_effector_prim_path = self._prim_path + "/panda_rightfinger"
        gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"]
        gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
        gripper_closed_position = np.array([0.0, 0.0])
        deltas = np.array([0.05, 0.05]) / get_stage_units()
        self._gripper = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=gripper_dof_names,
            joint_opened_positions=gripper_open_position,
            joint_closed_positions=gripper_closed_position,
            action_deltas=deltas,
        )
        # add Franka into world (important!)
        self.world.scene.add(self)
        
        self.rmp_flow_config = load_supported_motion_policy_config("Franka", "RMPflow")
        self.rmp_flow = RmpFlow(**self.rmp_flow_config)
        self.rmp_flow.set_robot_base_pose(
            self.position, self.orientation
        )
        self.articulation_rmp = ArticulationMotionPolicy(self, self.rmp_flow, 1.0 / 60.0)
        self.articulation_controller = self.get_articulation_controller()
        
        # check whether point is reachable or not
        self.pre_pos_error = 0.0
        self.pre_ori_error = 0.0
        self.error_nochange_epoch = 0
        
        self.debug_tool = VisualCuboid(
            prim_path="/World/Debug_cube",
            name="debug_cube",
            scale=np.array([0.01, 0.01, 0.01]),
            color=np.array([1.0, 0.0, 0.0]),
            translation=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            visible=False,
        )
        
        self.debug_tool_2 = VisualCuboid(
            prim_path="/World/Debug_cube_1",
            name="debug_cube_1",
            scale=np.array([0.01, 0.01, 0.01]),
            color=np.array([0.0, 1.0, 0.0]),
            translation=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            visible=False,
        )
        
        
        
        return
    
    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        # self._end_effector = SingleRigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector = SingleRigidPrim(prim_path=self._prim_path+"/panda_hand", name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        self.disable_gravity()
        return
    
    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )
        return
    
    @property
    def end_effector(self) -> SingleRigidPrim:
        """[summary]

        Returns:
            SingleRigidPrim: [description]
        """
        return self._end_effector
    
    @property
    def gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._gripper
    
    def open_gripper(self) -> None:
        """[summary]"""
        self.gripper.open()
        for i in range(20):
            self.world.step()
        return
    
    def close_gripper(self) -> None:
        """[summary]"""
        self.gripper.close()
        for i in range(20):
            self.world.step()
        return
    
    def get_cur_ee_pos(self):
        """
        get current end_effector_position and end_effector orientation
        """
        # position, orientation = self.end_effector.get_world_pose()
        # ori_trans = np.array([0.0, 0.0, 0.0, 1.0])
        # orientation = quat_mul(ori_trans, orientation)
        position, orientation = self.rmp_flow.get_end_effector_as_prim().get_world_pose()
        return position, orientation
    
    def add_obstacle(self, obstacle):
        """
        add obstacle to franka motion
        make franka avoid potential collision smartly
        """
        self.rmp_flow.add_obstacle(obstacle, False)
        for i in range(10):
            self.world.step(render=True)
        return
    
    def warm_start(self):
        initial_pos, initial_ori = self._end_effector.get_world_pose()
        self.initial_pos = initial_pos + Rotation(initial_ori, np.array([0.0, 0.0, 0.1]))
        self.Rmpflow_Step_Action(self.initial_pos)
    
    def Rmpflow_Step_Action(self, position, orientation=None, quat_or_not=False):
        """
        Use RMPflow_controller to move the Franka
        """
        if orientation is None:
            orientation = None
        elif not quat_or_not:
            orientation = euler_angles_to_quat(orientation, degrees=True)
        else:
            orientation = orientation
        self.world.step(render=True)
        # set end effector target
        self.rmp_flow.set_end_effector_target(
            target_position=position, target_orientation=orientation
        )
        # update obstacle position and get target action
        self.rmp_flow.update_world()
        actions = self.articulation_rmp.get_next_articulation_action()
        # apply actions
        self._articulation_controller.apply_action(actions)
        
    def Rmpflow_Move(self, target_position, target_orientation=np.array([180.0, 0.0, 180.0]), quat_or_not=False):
        """
        Use RMPflow_controller to move the Franka
        """

        # 重置本地状态
        self.pre_pos_error = float("inf")
        self.pre_ori_error = float("inf")
        self.error_nochange_epoch = 0

        target_ee_position = target_position
        if target_orientation is None:
            target_ee_orientation = None
        elif not quat_or_not:
            target_ee_orientation = euler_angles_to_quat(target_orientation, degrees=True)
        else:
            target_ee_orientation = target_orientation
        
        while True:
            # get current end effector position
            gripper_pos, gripper_ori = self.get_cur_ee_pos()
            
            # debug
            # self.debug_tool.set_world_pose(gripper_pos, gripper_ori)
            # for i in range(10):
            #     self.world.step()
            
            # compute distance error
            pos_error = np.linalg.norm(target_ee_position - gripper_pos)
            ori_error = np.linalg.norm(target_ee_orientation - gripper_ori)
            
            pos_error_gap = abs(pos_error - self.pre_pos_error)
            self.pre_pos_error = pos_error

            ori_error_gap = abs(ori_error - self.pre_ori_error)
            self.pre_ori_error = ori_error
            
            if pos_error_gap < 1e-4 and ori_error_gap < 1e-4:
                self.error_nochange_epoch += 1
            
            if self.error_nochange_epoch > 30:
                cprint("Single Franka RMPflow controller failed", "red")
                return False
            if pos_error < 0.001 and ori_error < 0.001:
                cprint("Single Franka RMPflow controller success", "green")
                self.pre_pos_error = 0.0
                self.pre_ori_error = 0.0
                self.error_nochange_epoch = 0
                return True
            
            self.Rmpflow_Step_Action(target_ee_position, target_ee_orientation, quat_or_not=True)
                
    def Dense_Rmpflow_Move(
        self, 
        target_position, 
        target_orientation=np.array([180.0, 0.0, 180.0]),
        dense_sample_scale=0.02,
        quat_or_not=False
    ):
        if target_orientation is None:
            target_orientation = None
        elif not quat_or_not:
            target_orientation = euler_angles_to_quat(target_orientation, degrees=True)
        else:
            target_orientation = target_orientation
        
        gripper_pos, gripper_ori = self.get_cur_ee_pos()
        # ------ debug
        # cprint(gripper_pos, "green")
        # self.debug_tool_2.set_world_pose(gripper_pos, gripper_ori)
        # ------
        dense_sample_num = int(np.linalg.norm(gripper_pos - target_position) // dense_sample_scale)
        if dense_sample_num == 0:
            dense_sample_num = 1
        
        # cprint(dense_sample_num, "red")
        
        interp_pos = dense_trajectory_points_generation(
            start_pos=gripper_pos, 
            end_pos=target_position,
            num_points=dense_sample_num,
        )
        
        cprint("--Single Franka Dense_Rmpflow_Move Begin", "green")
        for i in range(len(interp_pos)):
            # ------ debug
            # debug_pos, debug_ori = self.get_cur_ee_pos()
            # cprint(debug_pos, "green")
            # self.debug_tool.set_world_pose(debug_pos, debug_ori)
            # for _ in range(100):
            #     self.world.step()
            # if i == 0:
            #     return
            # ------
            if i == len(interp_pos) - 1:
                # print(f"-------step {i}-------")
                # ----old
                for j in range(50):
                    self.Rmpflow_Step_Action(interp_pos[i], target_orientation, quat_or_not=True)
                    self.world.step()
                # ----new
                # self.Rmpflow_Move(interp_pos[i], target_orientation, quat_or_not=True)
            else:
                # print(f"-------step {i}-------")
                for j in range(5):
                    self.Rmpflow_Step_Action(interp_pos[i], target_orientation, quat_or_not=True)
                    self.world.step()
            
        cprint("--Single Franka Dense_Rmpflow_Move End", "green")
        return
        
        
        
    
    