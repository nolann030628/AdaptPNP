import numpy as np
from Env_Config.Room.Object_Tools import set_prim_visible_group
from Env_Config.Gripper_Grasp.grasp_interface import grasp_checker
from isaacsim.core.utils.rotations import quat_to_euler_angles


def check_grasp(env, front=False):
    set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
    set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=False)
    for i in range(25):
        env.step()
    
    if front:
        pc_seg, color_seg = env.top_front_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            save_path="./pc_seg.ply",
            sample_flag=False,
            real_time_watch=False
        )

        pc_scene, color_scene = env.top_front_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            sample_flag=False,
            # workspace_x_limit=[-0.55, 0.55],
            workspace_z_limit=[0.50, None],
        )
    else:
        pc_seg, color_seg = env.gripper_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            save_path="./pc_seg.ply",
            sample_flag=False,
            real_time_watch=False
        )
        
        pc_scene, color_scene = env.gripper_camera.get_pointcloud_from_depth(
            show_original_pc_online=False,
            sample_flag=False,
            # workspace_x_limit=[-0.55, 0.55],
            workspace_z_limit=[0.50, None],
        )
    
    grasp_check, grasp_position, grasp_orientation = grasp_checker(env, env.top_front_camera, pc_seg, pc_scene, color_scene, vis_grasp_flag=False)

    return grasp_check, grasp_position, grasp_orientation


def execute_grasp(env, pos):
    front = False
    if(pos[0][1] < -0.18):
        front = True
    if(pos[0][2]>0.81):
        front = True

    rotation = quat_to_euler_angles(pos[1], degrees=True)
    if(rotation[1]<-10 and rotation[1]>-60):
        front = True

    grasp_check, grasp_position, grasp_orientation  = check_grasp(env, front)
    set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)
    set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=True)
    for i in range(25):
        env.step()
    print(grasp_check)
    if not grasp_check:
        return False, False

    env.franka.open_gripper()
    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([0.2, 0, 1.0]),
        target_orientation=np.array([180.0, 0.0, 0.0]),
        quat_or_not=False
    )

    x = env.object._prim.get_world_pose()[0][0]
    y = env.object._prim.get_world_pose()[0][1]
    z = env.object._prim.get_world_pose()[0][2]

    if(x>grasp_position[0] and grasp_position[2] < 0.86):
        pre_x = grasp_position[0] - 0.02
        grasp_position[0] = grasp_position[0] + 0.005
    elif(x <= grasp_position[0] and grasp_position[2] < 0.86):
        pre_x = grasp_position[0] + 0.02
        grasp_position[0] = grasp_position[0] - 0.005
    else:
        pre_x = x
    
    if(y>grasp_position[1]):
        pre_y = grasp_position[1] - 0.025
        grasp_position[1] = grasp_position[1] + 0.005
    else:
        pre_y = grasp_position[1] + 0.025
        grasp_position[1] = grasp_position[1] - 0.05
    
    if(z>grasp_position[2]):
        pre_z = grasp_position[2] - 0.05
    else:
        pre_z = grasp_position[2] + 0.05

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([0.2, 0, 1.0]),
        target_orientation=np.array([180.0, 0.0, 0.0]),
        quat_or_not=False
    )

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([pre_x, 0, 1.0]),
        target_orientation=grasp_orientation,
        quat_or_not=True
    )

    for _ in range(20): 
        env.step()

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([pre_x, pre_y, 1.0]),
        target_orientation=grasp_orientation,
        quat_or_not=True
    )

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([pre_x, pre_y, pre_z]),
        target_orientation=grasp_orientation,
        quat_or_not=True
    )

    for _ in range(20): 
        env.step()

    env.franka.Dense_Rmpflow_Move(
        target_position=grasp_position,
        target_orientation=grasp_orientation,
        quat_or_not=True
    )
    
    env.franka.Rmpflow_Move(
        target_position=grasp_position,
        target_orientation=grasp_orientation,
        quat_or_not=True
    )

    for _ in range(20): 
        env.step()
    
    env.franka.close_gripper()

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([grasp_position[0], grasp_position[1], 1.0]),
        target_orientation=grasp_orientation,
        quat_or_not=True
    )

    for _ in range(30): 
        env.step()

    if(env.object._prim.get_world_pose()[0][2] > z+0.08):
        return True, True
    else:
        return True, False