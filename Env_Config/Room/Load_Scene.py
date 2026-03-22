import numpy as np
import torch

from isaacsim.core.prims import XFormPrim, SingleXFormPrim, SingleRigidPrim, SingleGeometryPrim
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage

from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid, get_prim_type_name, get_prim_children, get_prim_path

class Load_Scene:
    def __init__(
        self,
        position=[0.0, 0.0, 0.0], 
        orientation=[0.0, 0.0, 0.0], 
        scale=[1.0, 1.0, 1.0], 
        usd_path:str=None, 
        prim_path:str="/World/Room",
        name="Room",
        mass=None,
        disable_rigid_body=False,
        visual_material = None
    ):
        self._position = position
        self._orientation = orientation
        self._scale = scale
        self._prim_path = find_unique_string_name(prim_path,is_unique_fn=lambda x: not is_prim_path_valid(x))
        self._usd_path = usd_path
        self._name=name
        self._mass=mass

        # add room to stage
        add_reference_to_stage(self._usd_path, self._prim_path)

        self._prim = SingleRigidPrim(
            prim_path=self._prim_path, 
            name=self._name, 
            scale=self._scale, 
            position=self._position, 
            orientation=euler_angles_to_quat(self._orientation, degrees=True),
            mass=self._mass
        )
        if visual_material is not None:
            self._prim.apply_visual_material(visual_material)
        if disable_rigid_body:
            self.disable_rigid_body_physics()

        self.prim = get_prim_at_path(self._prim_path)
        self.children_prim_list = get_prim_children(self.prim)
        self._target_prim = None
        self._target_prim_path = None
        for child_prim in self.children_prim_list:
            child_prim_path = get_prim_path(child_prim)
            if get_prim_type_name(child_prim_path) == "Xform":
                self._target_prim_path = child_prim_path
                break
        self._target_prim = SingleXFormPrim(self._target_prim_path)
        
        # print("target prim path: ", self._target_prim_path)
        # self._target_prim.get_world_pose()
        
    def disable_rigid_body_physics(self):
        
        self._prim.disable_rigid_body_physics()
        