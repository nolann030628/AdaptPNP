import numpy as np
from scipy.spatial import ConvexHull
from isaacsim.core.utils.rotations import quat_to_euler_angles
from Env_Config.Room.Object_Tools import set_prim_visible_group


def check_pose(env, t_pos, pos_thresh=0.15, angle_thresh=20.0):
    obj_pos = env.object._prim.get_world_pose()
    
    t_xyz, t_quat = np.array(t_pos[0]), np.array(t_pos[1])
    o_xyz, o_quat = np.array(obj_pos[0]), np.array(obj_pos[1])

    pos_diff = np.linalg.norm(t_xyz - o_xyz)

    t_quat = t_quat / np.linalg.norm(t_quat)
    o_quat = o_quat / np.linalg.norm(o_quat)
    dot = abs(np.dot(t_quat, o_quat))
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = 2 * np.arccos(dot)
    angle_deg = np.degrees(angle_rad)

    print(f"pos_diff: {pos_diff:.4f} m, angle_deg: {angle_deg:.2f}°")
    
    is_close = (pos_diff < pos_thresh) and (angle_deg < angle_thresh)
    
    return is_close


def filter_contact_points_quadrant(pcd, center, cur_ori, target_ori, min_radius=0.0, eps=1e-9):
    def quat_to_R(q):
        q = np.asarray(q, float); q /= (np.linalg.norm(q)+eps)
        w,x,y,z = q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
        ])

    def signed_angle_2d(a, b):
        a = a / (np.linalg.norm(a)+eps)
        b = b / (np.linalg.norm(b)+eps)
        cr = a[0]*b[1] - a[1]*b[0]      
        dt = a[0]*b[0] + a[1]*b[1]      
        return np.arctan2(cr, dt)       

    Rx3 = quat_to_R(cur_ori)[:, 0]
    Tx3 = quat_to_R(target_ori)[:, 0]
    Rx = Rx3[:2]; Rx /= (np.linalg.norm(Rx)+eps)
    Tx = Tx3[:2]; Tx /= (np.linalg.norm(Tx)+eps)

    theta = signed_angle_2d(Rx, Tx)
    abs_theta = abs(theta)
    sign = 1.0 if theta >= 0 else -1.0

    Rperp = np.array([-Rx[1], Rx[0]])

    rel_xy = pcd[:, :2] - center[:2]
    rel_dot_Rx = rel_xy @ Rx         
    rel_dot_Rp = rel_xy @ Rperp       
    radii = np.linalg.norm(rel_xy, axis=1)

    half_sign = +1.0 if abs_theta < (np.pi/2) else -1.0
    mask_half = (rel_dot_Rx * half_sign) > 0.0

    mask_quarter = (rel_dot_Rp * sign) > 0.0

    mask_radius = radii > (min_radius + eps)

    mask = mask_half & mask_quarter & mask_radius
    return mask, np.degrees(theta) 


def push_to_target(env, t_pos, t_diff=0.05, o_diff = 15):
    env.franka.close_gripper()
    temp = 0
    try_num = 5
    loss_t = 1e9
    loss_o = 1e9
    I = 0.0            
    prev_err = 0.0     
    is_down = True

    if env.object._prim.get_world_pose()[0][2] > t_pos[0][2] + 0.03:
        is_down = rotate_down_target(env, t_pos)
    if not is_down:
        return False
    if quat_to_euler_angles(t_pos[1], degrees=True)[1]< -5 or quat_to_euler_angles(t_pos[1], degrees=True)[1]>5:
        o_diff = 30
        try_num = 3

    while (loss_t > t_diff or loss_o > o_diff) and temp < try_num:
        temp = temp + 1
        print("try: ", temp)
        print("loss_t1", loss_t)
        print("loss_o1", loss_o)
        
        if loss_t > t_diff:
            initial_xyz = env.object._prim.get_world_pose()[0]
            target_xyz  = np.array(t_pos[0], dtype=float)
            vector = target_xyz - initial_xyz
            vector[2] = 0
            set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
            set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=False)
            
            for i in range(25):
                env.step()
            pcd, _  = env.top_camera.get_point_cloud_data_from_segment(
                    sample_flag=True,
                    sampled_point_num=4096,
                    real_time_watch=False
            )
            set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)
            set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=True)

            hull = ConvexHull(pcd)
            hull_points = pcd[hull.vertices]
            d = np.linalg.norm(vector)
            u = vector / d
            u_neg = -u
            rel = hull_points - initial_xyz                   
            t = rel @ u_neg                         
            mask = t > 0.0
            rel_m = rel[mask]                            
            t_m = t[mask][:, None]                       
            perp_vec = rel_m - t_m * u_neg              
            perp_dist = np.linalg.norm(perp_vec, axis=1) 
            best_idx = np.argmin(perp_dist)
            contact_point = hull_points[mask][best_idx]

            offset = 0.03
            v_offset = offset * u_neg

            initial_position = contact_point + v_offset
            print("Initial xyz:", initial_xyz)
            print("Moving to initial position:", initial_position) 
            pre_initial_position = initial_position.copy()
            pre_initial_position[2] += 0.2

            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(pre_initial_position),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Rmpflow_Move(
                target_position=np.array(pre_initial_position),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(initial_position),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Rmpflow_Move(
                target_position=np.array(initial_position),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            if(o_diff<10):
                target = initial_position + vector - 0.5 * v_offset
            else:
                target = initial_position + vector- 0.8 * v_offset
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(target),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            
            middle_point = target.copy()
            middle_point[2] += 0.2
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(middle_point),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )

            for _ in range(15):
                env.step()
            if env.object._prim.get_world_pose()[0][2] < 0.748:
                return False
            
            loss_t = np.linalg.norm(env.object._prim.get_world_pose()[0]- target_xyz)

        object_center = env.object._prim.get_world_pose()[0]
        set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
        set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=False)
        for _ in range(15):
            env.step()
        if object_center[2] < 0.748:
            return False
        for i in range(25):
            env.step()
        pcd, _  = env.top_camera.get_point_cloud_data_from_segment(
                sample_flag=True,
                sampled_point_num=4096,
                real_time_watch=False
        )
        set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)
        set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=True)

        hull = ConvexHull(pcd)
        hull_points = pcd[hull.vertices] 
        mask, theta = filter_contact_points_quadrant(hull_points, object_center, t_pos[1],env.object._prim.get_world_pose()[1])
        loss_o = abs(theta)
        if loss_o > o_diff:
            candidate_points = hull_points[mask]
            if candidate_points.shape[0] > 0:
                dists = np.linalg.norm(candidate_points[:, :2] - object_center[:2], axis=1)
                best_idx = np.argmax(dists)
                contact_point = candidate_points[best_idx]

            r_vector = contact_point[:2] - object_center[:2]
            r_vector /= np.linalg.norm(r_vector) + 1e-9
            if theta < 0:
                perpe = np.array([-r_vector[1], r_vector[0]])  
            else:
                perpe = np.array([ r_vector[1],-r_vector[0]])  
            perpe = perpe / (np.linalg.norm(perpe) + 1e-9)

            if(loss_o > 10):
                Kp = 0.003
                Kd = 0.001
                de = loss_o - prev_err
                u = Kp * loss_o + Kd * de 
                u = np.clip(u, 0.0, 0.1) 
            else:
                Kp = 0.001
                Kd = 0.0005
                de = loss_o - prev_err
                u = Kp * loss_o + Kd * de 
                u = np.clip(u, 0.0, 0.05)
            prev_err = loss_o
            pre_offset = 0.05
            robot_xy_target = contact_point[:2] + u * perpe 
            robot_xy_start = contact_point[:2] - pre_offset * perpe
            initial_pos = [robot_xy_start[0], robot_xy_start[1], contact_point[2]]
            target_pos = [robot_xy_target[0], robot_xy_target[1], contact_point[2]]
            print("Initial position:", initial_pos)
            pre_initial_pos = initial_pos.copy()
            pre_initial_pos[2] += 0.3
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(pre_initial_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Rmpflow_Move(
                target_position=np.array(pre_initial_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(initial_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Rmpflow_Move(
                target_position=np.array(initial_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(target_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )
            env.franka.Rmpflow_Move(
                target_position=np.array(target_pos),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )

            middle = target_pos.copy()
            middle[2] += 0.5
            
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array(middle),
                target_orientation=np.array([180.0, 0.0, 0.0]),
            )

            for _ in range(15):
                env.step()

            if env.object._prim.get_world_pose()[0][2] < 0.748:
                return False

    if(loss_t > t_diff or loss_o > o_diff):
        print("Failed to push to target position.")
        return False
    
    else:
        print("Successfully pushed to target position!")
        return True


def move_to_target(env, target_position):
    ee_pos, ee_quat = env.franka.get_cur_ee_pos()  
    obj_pos, obj_quat = env.object._prim.get_world_pose() 
    obj_rotation = quat_to_euler_angles(obj_quat, degrees=True)

    target_pos = np.array(target_position[0], dtype=float)
    target_quat = np.array(target_position[1], dtype=float)
    target_rotation = quat_to_euler_angles(target_quat, degrees=True)

    vector = ee_pos - obj_pos
    new_gripper_pos = target_pos + vector

    new_gripper_pos[2] = 1.0 

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array(new_gripper_pos),
        target_orientation=np.array(ee_quat),
        quat_or_not=True
    )
    env.franka.Rmpflow_Move(
        target_position=np.array(new_gripper_pos),
        target_orientation=np.array(ee_quat),
        quat_or_not=True
    )
    rotation = quat_to_euler_angles(ee_quat, degrees=True)

    obj_pos, obj_quat = env.object._prim.get_world_pose()

    obj_rotation = quat_to_euler_angles(obj_quat, degrees=True)

    if(abs(obj_rotation[0]-target_rotation[0])>70):
        rotation[0] = rotation[0] - 30

        env.franka.Rmpflow_Move(
            target_position=np.array(new_gripper_pos),
            target_orientation=np.array(rotation),
            quat_or_not=False
        )

    new_gripper_pos[2] = target_pos[2] + 0.1

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array(new_gripper_pos),
        target_orientation=np.array(rotation),
        quat_or_not=False
    )

    ee_pos, ee_quat = env.franka.get_cur_ee_pos()

    print("Move to target position completed:", ee_pos)
    if env.object._prim.get_world_pose()[0][2] < 0.75:
        return False
    return True
    

def release(env, t_pos):
    ee_pos, ee_quat = env.franka.get_cur_ee_pos()
    obj_pos, obj_quat = env.object._prim.get_world_pose() 

    if obj_pos[2]< 0.8:
        return False

    vector = ee_pos - obj_pos 
    release_vector = vector / np.linalg.norm(vector)
    new_ee_pos = ee_pos + release_vector*0.2

    env.franka.open_gripper()

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array(new_ee_pos),
        target_orientation=ee_quat, 
        quat_or_not=True
    )

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([new_ee_pos[0], new_ee_pos[1], 1.0]),
        target_orientation=np.array([180.0, 0.0, 0.0]),
        quat_or_not=False
    )

    ik = check_pose(env, t_pos, pos_thresh=0.1, angle_thresh=180.0)
    return ik


def rotate_up_target(env,t_pos):
    env.franka.close_gripper()
    obj_pos, obj_quat = env.object._prim.get_world_pose()
    
    contact_point = obj_pos
    contact_point[0] = obj_pos[0] + 0.06
    contact_point[2] = 0.74

    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([0.2, 0.0, 0.765]),
        target_orientation=np.array([180.0, 0.0, 0.0]),
    )

    env.franka.Dense_Rmpflow_Move(
        target_position=contact_point,
        target_orientation=np.array([180.0, 0.0, 0.0]),
    )

    contact_point[0] = contact_point[0] - 0.01
    contact_point[2] = 0.77

    env.franka.Rmpflow_Move(
        target_position=contact_point,
        target_orientation=np.array([180.0, 15.0, 0.0]),
    )

    contact_point[0] = contact_point[0] + 0.02
    contact_point[2] = contact_point[2] - 0.05
    env.franka.Dense_Rmpflow_Move(
        target_position=contact_point,
        target_orientation=np.array([180.0, 37.0, 0.0]),
    )
    for i in range(20):
        env.step()
    
    env.franka.Dense_Rmpflow_Move(
        target_position=np.array([0.2, 0, 1.0]),
        target_orientation=np.array([180.0, 0.0, 0.0]),
        quat_or_not=False
    )

    ik = check_pose(env, t_pos, pos_thresh=1.0, angle_thresh=30.0)
    return ik
    

def rotate_down_target(env, t_pos):
    env.franka.close_gripper()
    def contactPoint(env):
        obj_pos = env.object._prim.get_world_pose()[0]
        set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
        set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=False) 
        for _ in range(25):
            env.step()

        pcd, _  = env.top_camera.get_point_cloud_data_from_segment(
            sample_flag=True,
            sampled_point_num=4096,
            real_time_watch=False
        )

        set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)
        set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=True)

        hull = ConvexHull(pcd)
        hull_points = pcd[hull.vertices] 
        
        idx = np.argmin(hull_points[:, 0])
        offset = -0.03
        if(obj_pos[0] < t_pos[0][0]):
            idx = np.argmin(hull_points[:, 0])
        else:
            idx = np.argmax(hull_points[:, 0])
        
        cp = np.array(hull_points[idx], dtype=float).copy() 
        cp[1] = float(obj_pos[1])
        cp[2] = float(np.max(hull_points[:, 2]))

        return cp

    obj_pos = env.object._prim.get_world_pose()[0]
    contact_point = np.array(contactPoint(env), dtype=float).reshape(3,).copy()
    if(obj_pos[0] < t_pos[0][0]):
        offset = -1
    else:
        offset = 1
    startpoint = contact_point.copy()
    startpoint[0] = startpoint[0] + offset * 0.03
    
    pre_point = startpoint.copy()
    pre_point[2] += -offset * 0.05

    end_point = startpoint.copy()
    end_point[0] += -offset * 0.3

    env.franka.Dense_Rmpflow_Move(
        target_position=pre_point,
        target_orientation=np.array([180.0, 0.0, 0.0]),
    )

    env.franka.Rmpflow_Move(
        target_position=startpoint,
        target_orientation=np.array([180.0, 0.0, 0.0]),
    )

    env.franka.Dense_Rmpflow_Move(
        target_position=end_point,
        target_orientation=np.array([180.0, 0.0, 0.0]),
    )

    z = env.object._prim.get_world_pose()[0][2]
    if z > t_pos[0][2] + 0.03:
        return False
    else:
        return True
