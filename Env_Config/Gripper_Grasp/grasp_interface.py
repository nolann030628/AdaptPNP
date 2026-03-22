import numpy as np
from isaacsim.core.utils.rotations import rot_matrix_to_quat
from Env_Config.Gripper_Grasp.communicate_interface import get_response_from_flask, vis_grasp


def grasp_checker(env, camera, pc_seg, pc_scene, color_scene, vis_grasp_flag=False):
    grasp_check = False
    pc_seg_camera = camera.convert_translation_world_to_camera(points_world=pc_seg)
    pc_scene_camera = camera.convert_translation_world_to_camera(points_world=pc_scene)
    response = get_response_from_flask(pc_seg_camera, pc_scene_camera, color_scene)
    # print(response)
    if not response['success']:
        print('No object pose found')
        return grasp_check, None, None
    else:
        print('Object pose found, score: ', response['score'])
        grasp_check = True
    
    gripper_data = response['gripper']
    if gripper_data is not None and vis_grasp_flag:
        vis_grasp(pc_scene_camera, color_scene, gripper_data)
        
    translation = np.asarray(response['translation'])
    rotation = np.asarray(response['rotation'])
    
    world_translation = camera.convert_translation_camera_to_world(translation)[0]
    
    world_rotation = camera.convert_rotation_camera_to_world(rotation)[0]
    robot_gripper_2_grasp_gripper = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    world_rotation = np.dot(world_rotation, robot_gripper_2_grasp_gripper)
    world_rotation_quat = rot_matrix_to_quat(world_rotation)
    
    print("world_translation\n", world_translation)
    print("world_rotation\n", world_rotation)
    if world_translation[2] > 0.9:
        return False, None, None
    obj_pos = env.object._prim.get_world_pose()[0]
    if obj_pos[1] > -0.18 and obj_pos[1] < 0.18:
        if world_translation[2] < obj_pos[2] - 0.005:
            return False, None, None
    if world_translation[0] < -0.3:
        return False, None, None
    
    return grasp_check, world_translation, world_rotation_quat

    


    