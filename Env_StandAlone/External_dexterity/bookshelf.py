from isaacsim import SimulationApp
# simulation_app = SimulationApp({"headless": False})


import os
import sys
import numpy as np
from termcolor import cprint

# load isaac-relevant package

from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name

# load custom package
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Robot.Franka import Franka
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Collision_Group import CollisionGroup_TAMP
from Env_Config.Room.Load_Scene import Load_Scene
from Env_Config.Material.Transparent import Surface_Extend
# from vlm_plan.np import push_to_target


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

        visual_prim_path = find_unique_string_name(
            initial_name="/World/Looks/visual_material_1", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self.transparent_material = Surface_Extend(prim_path=visual_prim_path, opacity=0.8)
        
        
        self.franka = Franka(
            self.world,
            position=np.array([0.5, 0.0, 0.746]),
            orientation=np.array([0.0, 0.0, 180.0]),
            robot_name="Franka",
        )
        
        self.workdesk = Load_Scene(
            position=np.array([0.0, 0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            scale=np.array([0.55, 0.47, 0.63]),
            usd_path=os.path.join(self.root_path, "../Assets/Objects/desk_bookshelf/desk_bookshelf.usd"),
            prim_path="/World/WorkDesk",
            name="WorkDesk",
            mass=1e6
        )

        self.book1 = Load_Scene(
            position=np.array([-0.048, 0.146, 0.88]),
            orientation=np.array([90.0, 0.0, 180.0]),
            scale=np.array([0.0045, 0.0065, 0.005]),
            usd_path=os.path.join(self.root_path, "../Assets/Objects/red_book/red_book.usd"),
            prim_path="/World/Book1",
            name="Book1",
            # disable_rigid_body=True,
            mass=1e5
        )

        self.book2 = Load_Scene(
            position=np.array([-0.045, 0.08, 0.88]),
            orientation=np.array([-90.0, 0.0, 180.0]),
            scale=np.array([0.0045, 0.0065, 0.005]),
            usd_path=os.path.join(self.root_path, "../Assets/Objects/red_book/red_book.usd"),
            prim_path="/World/Book2",
            name="Book2",
            # disable_rigid_body=True,
            mass=1e5
        )
        

        self.object = Load_Scene(
            position=np.array([-0.045, 0.11, 0.88]),
            orientation=np.array([-90.0, 0.0, 180.0]),
            scale=np.array([0.0045, 0.0065, 0.007]),
            usd_path=os.path.join(self.root_path, "../Assets/Objects/blue_book/blue_book.usd"),
            prim_path="/World/Book3",
            name="Book3",
            # disable_rigid_body=True,
            mass=0.01
        )

        self.non_collision_object = Load_Scene(
            position=np.array([0.15, -0.15, 0.763]),
            orientation=np.array([180, 0.0, -90]),
            scale=np.array([0.0045, 0.0063, 0.007]),
            usd_path=os.path.join(self.root_path, "../Assets/Objects/blue_book/blue_book_trans.usd"),
            prim_path="/World/NonCollisionObject",
            name="NonCollisionObject",
            disable_rigid_body=True,
            mass=0.01,
            visual_material=self.transparent_material

        )

        # set collision group
        self.collision_group = CollisionGroup_TAMP(
            self.world,
            non_collision_path=["/World/NonCollisionObject"], 
            normal_collision_path=["/World/Book1", "/World/Book2", "/World/Book3"], 
            robot_path=["/World/Franka"],
            Workspace_path=["/World/WorkDesk"]
        )

        self.camera = Recording_Camera(
            prim_path="/World/camera",
            camera_position=(0.0, -5.3, 5.3), 
            camera_orientation=(50.0, 0.0, 0.0)
        )
        self.gripper_camera = Recording_Camera(
            prim_path="/World/gripper_camera",
            camera_position=(4.0, 0, 4.0), 
            camera_orientation=(50, 0, 90)
        )
        
        self.top_camera = Recording_Camera(
            prim_path="/World/top_camera",
            camera_position=(0.0, 0.0, 6.5), 
            camera_orientation=(0.0, 0.0, 0.0)
        )
        self.top_front_camera = Recording_Camera(
            prim_path="/World/top_front_camera",
            camera_position=(0.0, -2.4, 4.8), 
            camera_orientation=(0.96593, 0.25882, 0.0, 0.0), ori_type="quat"
        )

        self.front_camera = Recording_Camera(
            prim_path="/World/front_camera",
            camera_position=(4.0, 0, 4.0), 
            camera_orientation=(50, 0, 90)
        )
        
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        
        # initialize world
        self.reset()
        
        self.camera.initialize(
            depth_enable=True,
            segment_pc_enable=True,
            segment_prim_path_list=["/World/Book3"]
        )
        self.gripper_camera.initialize(
            depth_enable=True,
            segment_pc_enable=True,
            segment_prim_path_list=["/World/Book3"]
        )
        self.top_camera.initialize(
            depth_enable=True,
            segment_pc_enable=True,            
            segment_prim_path_list=["/World/Book3"] 
        )
        self.top_front_camera.initialize(
            depth_enable=True,
            segment_pc_enable=True,            
            segment_prim_path_list=["/World/Book3"] 
        )
        self.front_camera.initialize(
            depth_enable=True,
            segment_pc_enable=True,            
            segment_prim_path_list=["/World/Book3"] 
        )
       
        
        for i in range(100):
            self.step()    

        cprint("World Ready!", "green", "on_green")
        



def Demo_Scene():
        
    env = Demo_Scene_Env()



# if __name__=="__main__":
    
#     Demo_Scene()

#     while simulation_app.is_running():
#         simulation_app.update()
    
# simulation_app.close()