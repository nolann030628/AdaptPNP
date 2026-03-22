import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

checkpoint_path='./graspnet_flask/checkpoint.tar'
num_point=20000
num_view=300
collision_thresh=0.005
voxel_size=0.01


def set_seed(seed=42):                  
    np.random.seed(seed)                    
    torch.manual_seed(seed)                  
    torch.cuda.manual_seed(seed)              
    torch.cuda.manual_seed_all(seed)         

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"[INFO] Set all random seeds to {seed}")
    
def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def process_pointcloud(pc, color):
    pc = np.asarray(pc)
    color = np.asarray(color)
    if color.shape[0] != pc.shape[0]:
        color = np.zeros_like(pc)

    if pc.shape[0] >= num_point:
        indices = np.random.choice(pc.shape[0], num_point, replace=False)
    else:
        idx1 = np.arange(pc.shape[0])
        idx2 = np.random.choice(pc.shape[0], num_point - pc.shape[0], replace=True)
        indices = np.concatenate([idx1, idx2], axis=0)
    pc_sampled = pc[indices]
    color_sampled = color[indices]

    end_points = dict()
    pc_tensor = torch.from_numpy(pc_sampled[np.newaxis].astype(np.float32)).cuda()
    end_points['point_clouds'] = pc_tensor
    end_points['cloud_colors'] = color_sampled

    return end_points, pc, color

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.0, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:1]
    # print(gg.translations)
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def serialize_mesh(mesh: o3d.geometry.TriangleMesh):
    return {
        "type": "TriangleMesh",
        "vertices": np.asarray(mesh.vertices).tolist(),
        "triangles": np.asarray(mesh.triangles).tolist(),
        "colors": np.asarray(mesh.vertex_colors).tolist() if mesh.has_vertex_colors() else None,
    }

def find_points_in_bounding_sphere(object_points: np.ndarray, test_points: np.ndarray):
    center = np.mean(object_points, axis=0)
    distances = np.linalg.norm(object_points - center, axis=1)
    radius = np.max(distances)
    test_distances = np.linalg.norm(test_points - center, axis=1)
    indices = np.where(test_distances <= radius*1.05)[0]

    return center, radius, indices


def find_points_on_object_surface_min_k(
    object_points: np.ndarray, 
    test_points: np.ndarray, 
    epsilon: float = 0.02, 
    k: int = 5
):
    center = np.mean(object_points, axis=0)
    distances = np.linalg.norm(object_points - center, axis=1)
    radius = np.max(distances)
    test_distances = np.linalg.norm(test_points - center, axis=1)
    indices_within_radius = np.where(test_distances <= radius * 1.10)[0]
    
    candidates = [] 
    
    for idx in indices_within_radius:
        point = test_points[idx]

        point_distances = np.linalg.norm(object_points - point, axis=1)
        
        if np.min(point_distances) < epsilon:  
            distance_to_center = np.linalg.norm(point - center)
            candidates.append((idx, distance_to_center))

    candidates.sort(key=lambda x: x[1], reverse = True)

    selected_indices = [idx for idx, _ in candidates[:k]]

    if selected_indices:
        sel_points = test_points[selected_indices]
        center_x = center[0]
        if np.any(sel_points[:, 0] > center_x-0.02):
            selected_indices = [
                idx for idx in selected_indices if test_points[idx][0] > center_x-0.02
            ]
    
    return selected_indices

def object_grasp(object_point, scene_point, scene_color, net):
    end_points, pc, color = process_pointcloud(scene_point, scene_color)
    gg = get_grasps(net, end_points)
    if collision_thresh > 0:
        gg = collision_detection(gg, pc)
        
    gg_translations = np.asarray(gg.translations)

    indices = find_points_on_object_surface_min_k(object_point, gg_translations)
    
    if len(indices) == 0:
        return {
        "success": False,
        "score": None,
        "translation": None,
        "rotation": None,
        "gripper": None,
        }
    
    gg = gg[indices].sort_by_score()[:1]

    
    gripper = gg.to_open3d_geometry_list()[0]
    
    return {
        "success": True,
        "score": float(gg.scores),
        "translation": gg.translations.tolist(),
        "rotation": gg.rotation_matrices.tolist(),
        "gripper": serialize_mesh(gripper)
    }



from flask import Flask, request, jsonify
import numpy as np
import io

set_seed(42)
net = get_net()

app = Flask(__name__)

@app.route('/grasp', methods=['POST'])
def grasp():
    try:
        data = request.get_json()

        object_pc = np.array(data['object'], dtype=np.float32)  # (N1, 3)
        scene_pc = np.array(data['scene'], dtype=np.float32)    # (N2, 3)
        scene_color = np.array(data['color'], dtype=np.float32) # (N2, 3)

        assert object_pc.ndim == 2 and object_pc.shape[1] == 3
        assert scene_pc.ndim == 2 and scene_pc.shape[1] == 3
        assert scene_color.shape == scene_pc.shape

        result = object_grasp(object_pc, scene_pc, scene_color, net=net)
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=34123)