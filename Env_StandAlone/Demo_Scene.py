from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# load external package
import os
import sys
import time
import numpy as np
import open3d as o3d
from termcolor import cprint
import threading
import cv2
import matplotlib.pyplot as plt
# load isaac-relevant package
import omni.replicator.core as rep
import isaacsim.core.utils.prims as prims_utils
from pxr import UsdGeom,UsdPhysics,PhysxSchema, Gf
from isaacsim.core.api import World
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import matplotlib.pyplot as plt


# load custom package
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Robot.Franka import Franka
from Env_Config.Robot.Bimanual_Franka import Bimanual_Franka
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Flatten_Judge import judge_fling
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from Env_Config.Utils_Project.Collision_Group import CollisionGroup
from Env_Config.Utils_Project.Set_Drive import set_drive
from Env_Config.Utils_Project.Transforms import quat_diff_rad, Rotation, get_pose_relat, get_pose_world
from Env_Config.Utils_Project.Code_Tools import float_truncate, dense_trajectory_points_generation
from Env_Config.Room.Load_Scene import Load_Scene


class Demo_Scene_Env(BaseEnv):
    def __init__(
        self, 
    ):
        # load BaseEnv
        super().__init__()
        
        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ------------------------------------ #

        # add ground
        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = os.path.join(self.root_path, "../Assets/Scene/floor/Collected_floor_material/Collected_WoodFloor001/WoodFloor001.usd"),
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )
        
        self.franka = Franka(
            self.world,
            position=np.array([0.5, 0.0, 0.746]),
            orientation=np.array([0.0, 0.0, 180.0]),
            robot_name="Franka",
        )
        
        
        self.workdesk = Load_Scene(
            position=np.array([0.0, 0, 0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            scale=np.array([0.55, 0.47, 0.63]),
            usd_path=os.path.join(self.root_path, "../Assets/Objects/desk/desk.usd"),
            prim_path="/World/WorkDesk",
            name="WorkDesk",
            mass=1e8
        )
        
        # set collision group
        self.collision_group = CollisionGroup(
            self.world,
            normal_object_path=["/World/WorkDesk", "/World/Franka"],
        )
        
        self.top_camera = Recording_Camera(camera_position=(0.0, 0.0, 5), camera_orientation=(0.0, 90.0, 90.0))
        
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        
        # initialize world
        self.reset()
        
        self.top_camera.initialize(
            depth_enable=True,
            segment_pc_enable=True,
            segment_prim_path_list=["/World/Object","/World/Franka_Left"]
        )
        
        
        self.thread_record = threading.Thread(target=self.top_camera.collect_rgb_graph_for_video)
        self.thread_record.daemon = True
        
        
        for i in range(100):
            self.step()    

        cprint("World Ready!", "green", "on_green")
        



def Demo_Scene():
        
    env = Demo_Scene_Env()



if __name__=="__main__":
    
    Demo_Scene()

    while simulation_app.is_running():
        simulation_app.update()
    
simulation_app.close()