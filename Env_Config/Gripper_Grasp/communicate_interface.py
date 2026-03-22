import requests
import numpy as np
import json
import open3d

def get_response_from_flask(object_pc, scene_pc, scene_color):
    object_pc = np.asarray(object_pc).tolist()
    scene_pc = np.asarray(scene_pc).tolist()
    scene_color = np.asarray(scene_color).tolist()
    
    payload = {
        "object": object_pc,
        "scene": scene_pc,
        "color": scene_color
    }
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post("http://localhost:34123/grasp", headers=headers, data=json.dumps(payload))
    data = response.json()
    # print(data)
    return data
    
def vis_grasp(scene_pc, scene_color, gripper_mesh_data):
    scene_pcd = open3d.geometry.PointCloud()
    scene_pcd.points = open3d.utility.Vector3dVector(scene_pc)
    scene_pcd.colors = open3d.utility.Vector3dVector(scene_color)
    
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(np.array(gripper_mesh_data['vertices'], dtype=np.float32))
    mesh.triangles = open3d.utility.Vector3iVector(np.array(gripper_mesh_data['triangles'], dtype=np.int32))
    if gripper_mesh_data['colors']:
        mesh.vertex_colors = open3d.utility.Vector3dVector(np.array(gripper_mesh_data['colors'], dtype=np.float32))

    open3d.visualization.draw_geometries([scene_pcd, mesh])
