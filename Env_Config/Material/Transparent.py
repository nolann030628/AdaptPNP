import numpy as np
from typing import Optional
from pxr import Gf, Sdf, UsdShade
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid

class Surface_Extend(PreviewSurface):
    def __init__(
        self,
        prim_path: str,
        name: str = "preview_surface",
        shader: Optional[UsdShade.Shader] = None,
        color: Optional[np.ndarray] = None,
        roughness: Optional[float] = None,
        metallic: Optional[float] = None,
        opacity: Optional[float] = None
    ):
        super().__init__(
            prim_path=prim_path,
            name=name,
            shader=shader,
            color=color,
            roughness=roughness,
            metallic=metallic,
        )
        if opacity is not None:
            self.shaders_list[0].CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        return
        
    