import os
import sys
import numpy as np
import torch
from termcolor import cprint
import threading

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix, rot_matrix_to_quat
from isaacsim.core.prims import SingleXFormPrim

sys.path.append(os.getcwd())
from Env_Config.Robot.Franka import Franka
from Env_Config.Utils_Project.Transforms import quat_diff_rad, Rotation, get_pose_relat, get_pose_world
from Env_Config.Utils_Project.Code_Tools import float_truncate, dense_trajectory_points_generation

class Bimanual_Franka:
    def __init__(self, world:World, left_pos, left_ori, right_pos, right_ori):
        self.world = world
        self.left_franka = Franka(world, left_pos, left_ori, robot_name="Franka_Left")
        self.right_franka = Franka(world, right_pos, right_ori, robot_name="Franka_Right")
        
        self.left_pre_error = 0.0
        self.left_error_nochange_epoch = 0
        
        self.right_pre_error = 0.0
        self.right_error_nochange_epoch = 0
    
    def Gripper_Both_Open(self):
        self.left_franka.gripper.open()
        self.right_franka.gripper.open()
        for i in range(20):
            self.world.step(render=True)
        cprint("Gripper_Both_Open", "green")
            
    def Gripper_Both_Close(self):
        self.left_franka.gripper.close()
        self.right_franka.gripper.close()
        for i in range(20):
            self.world.step(render=True)
        cprint("Gripper_Both_Close", "green")
            
    def Gripper_Left_Open(self):
        self.left_franka.gripper.open()
        for i in range(20):
            self.world.step(render=True)
        cprint("Gripper_Left_Open", "green")
            
    def Gripper_Left_Close(self):
        self.left_franka.gripper.close()
        for i in range(20):
            self.world.step(render=True)
        cprint("Gripper_Left_Close", "green")
            
    def Gripper_Right_Open(self):
        self.right_franka.gripper.open()
        for i in range(20):
            self.world.step(render=True)
        cprint("Gripper_Right_Open", "green")
            
    def Gripper_Right_Close(self):
        self.right_franka.gripper.close()
        for i in range(20):
            self.world.step(render=True)
        cprint("Gripper_Right_Close", "green")
            

    def Rmpflow_Left_Move(
        self,
        target_position,
        target_orientation=np.array([180.0, 0.0, 0.0]),
        quat_or_not=False
    ):
        self.left_franka.Rmpflow_Move(target_position, target_orientation, quat_or_not)
        
    def Rmpflow_Right_Move(
        self,
        target_position,
        target_orientation=np.array([180.0, 0.0, 180.0]),
        quat_or_not=False
    ):
        self.right_franka.Rmpflow_Move(target_position, target_orientation, quat_or_not)
        
        
    def Rmpflow_Both_Move(
        self,
        left_target_position, 
        right_target_position,
        left_target_orientation=None,
        right_target_orientation=None,
        quat_or_not=False
    ):
        left_target_ee_position = left_target_position
        right_target_ee_position = right_target_position
        if left_target_orientation is not None:
            if not quat_or_not:
                left_target_ee_orientation = euler_angles_to_quat(left_target_orientation, degrees=True)
            else:
                left_target_ee_orientation = left_target_orientation
        else:
            left_target_ee_orientation = None
        if right_target_orientation is not None:
            if not quat_or_not:
                right_target_ee_orientation = euler_angles_to_quat(right_target_orientation, degrees=True)
            else:
                right_target_ee_orientation = right_target_orientation
        else:
            right_target_ee_orientation = None
            
        while True:
            # get current end effector position (left)
            left_pos, left_ori = self.left_franka.get_cur_ee_pos()
            # get current gripper position (left)
            left_gripper_pos = left_pos + Rotation(left_ori, np.array([0.0, 0.0, 0.1]))
            
            # get current end effector position (right)
            right_pos, right_ori = self.right_franka.get_cur_ee_pos()
            # get current gripper position (right)
            right_gripper_pos = right_pos + Rotation(right_ori, np.array([0.0, 0.0, 0.1]))
            
            # compute distance error (left)
            left_error = np.linalg.norm(left_target_position - left_gripper_pos)
            left_error_gap = abs(left_error - self.left_pre_error)
            self.left_pre_error = left_error
            if left_error_gap < 1e-4:
                self.left_error_nochange_epoch += 1
                
            # compute distance error (right)
            right_error = np.linalg.norm(right_target_position - right_gripper_pos)
            right_error_gap = abs(right_error - self.right_pre_error)
            self.right_pre_error = right_error
            if right_error_gap < 1e-4:
                self.right_error_nochange_epoch += 1
                
            if self.left_error_nochange_epoch > 100 and self.right_error_nochange_epoch > 100:
                cprint("Both Frankas RMPflow Controller failed", "red")
                return False
            elif self.left_error_nochange_epoch > 100 and right_error < 0.001:
                cprint("Only Right Franka RMPflow Controller succeeded", "yellow")
                return False
            elif self.right_error_nochange_epoch > 100 and left_error < 0.001:
                cprint("Only Left Franka RMPflow Controller succeeded", "yellow")
                return False
            elif left_error < 0.001 and right_error < 0.001:
                cprint("Both Frankas RMPflow Controller succeeded", "green")
                return True
            
            self.left_franka.Rmpflow_Step_Action(left_target_ee_position, left_target_ee_orientation, quat_or_not=True)
            self.right_franka.Rmpflow_Step_Action(right_target_ee_position, right_target_ee_orientation, quat_or_not=True)
            
    def Dense_Rmpflow_Left_Move(
        self, 
        target_position, 
        target_orientation=np.array([180.0, 0.0, 0.0]),
        dense_sample_scale=0.02,
        quat_or_not=False
    ):
        self.left_franka.Dense_Rmpflow_Move(target_position, target_orientation, dense_sample_scale, quat_or_not)
        
    def Dense_Rmpflow_Right_Move(
        self, 
        target_position, 
        target_orientation=np.array([180.0, 0.0, 180.0]),
        dense_sample_scale=0.02,
        quat_or_not=False
    ):
        self.right_franka.Dense_Rmpflow_Move(target_position, target_orientation, dense_sample_scale, quat_or_not)
        
    def Dense_Rmpflow_Both_Move(
        self, 
        left_target_position, 
        right_target_position, 
        left_target_orientation=np.array([180.0, 0.0, 0.0]),
        right_target_orientation=np.array([180.0, 0.0, 180.0]),
        dense_sample_scale=0.02,
        quat_or_not=False
    ):
        if left_target_orientation is not None:
            if not quat_or_not:
                left_target_ee_orientation = euler_angles_to_quat(left_target_orientation, degrees=True)
            else:
                left_target_ee_orientation = left_target_orientation
        if right_target_orientation is not None:
            if not quat_or_not:
                right_target_ee_orientation = euler_angles_to_quat(right_target_orientation, degrees=True)
            else:
                right_target_ee_orientation = right_target_orientation
        
        left_pos, left_ori = self.left_franka.get_cur_ee_pos()
        right_pos, right_ori = self.right_franka.get_cur_ee_pos()
        
        left_gripper_pos = left_pos + Rotation(left_ori, np.array([0.0, 0.0, 0.1]))
        right_gripper_pos = right_pos + Rotation(right_ori, np.array([0.0, 0.0, 0.1]))
        
        left_dense_sample_num = int(np.linalg.norm(left_target_position - left_gripper_pos) // dense_sample_scale)
        right_dense_sample_num = int(np.linalg.norm(right_target_position - right_gripper_pos) // dense_sample_scale)
        
        dense_sample_num = max(left_dense_sample_num, right_dense_sample_num)
        
        left_interp_pos = dense_trajectory_points_generation(
            start_pos=left_gripper_pos,
            end_pos=left_target_position,
            num_points=dense_sample_num,
        )
        right_interp_pos = dense_trajectory_points_generation(
            start_pos=right_gripper_pos,
            end_pos=right_target_position,
            num_points=dense_sample_num,
        )

        cprint("--Both Frankas Dense_Rmpflow_Move Begin", "green")
        for i in range(len(left_interp_pos)):
            print(f"-------step {i}-------")
            for j in range(5):
                self.left_franka.Rmpflow_Step_Action(left_interp_pos[i], left_target_ee_orientation, quat_or_not=True)
                self.right_franka.Rmpflow_Step_Action(right_interp_pos[i], right_target_ee_orientation, quat_or_not=True)
                self.world.step()
        cprint("--Both Frankas Dense_Rmpflow_Move End", "green")