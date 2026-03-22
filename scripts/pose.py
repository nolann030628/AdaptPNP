import numpy as np
from isaacsim.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles
from scripts.gptutils import call_gpt_model
from PIL import Image, ImageDraw, ImageFont
import re
from Env_Config.Room.Object_Tools import set_prim_visible_group
from scripts.doubaoutils import get_2d_point

font_path = "./Assets/NVIDIASans_Md.ttf"

def add_number_to_image(image_path, number):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    font_size = 48
    font = ImageFont.truetype(font_path, font_size)

    text = str(number)
    text_position = (10, 10)
    text_color = (255, 0, 0)  # red color

    draw.text(text_position, text, font=font, fill=text_color)
    image.save(image_path)

    return image_path


def pixels_to_world(env, u, v, depth_image):
    debug = True
    h, w = depth_image.shape[:2]

    u_i = int(round(u))
    v_i = int(round(v))

    u_i = max(0, min(w - 1, u_i))
    v_i = max(0, min(h - 1, v_i))
    d = float(depth_image[v_i, u_i])
    if d <= 0:
        raise ValueError(f"Invalid depth {d} at pixel ({u_i},{v_i})")
    points_2d = np.array([[u_i, v_i]], dtype=float)  
    depth   = np.array([d],        dtype=float)    
    world_pt = env.top_camera.get_world_points_from_image_coords(points_2d, depth)
    if debug:
        print(f"[DBG] world_pt: {world_pt[0]}")

    return world_pt[0]  


def generate_pose(env, prompts, subtasks, idx, obs_image, subtask, action):
    init_pos = env.object._prim.get_world_pose()
    task = None
    if action == 'push' or action == 'rotate down' or action == 'rotate up':
        obj = subtask["parameters"]["object"]
        place = subtask["parameters"]["place"]
        if "table edge" in place:
            place = "bottom " + place
        task = f"Output the points (<point>x y</point>) for 'the {place}, note that the point should be as close to the {obj} as possible'"
    elif action == 'move to' :
        place = subtask["parameters"]["place"]
        task = f"Output the points (<point>x y</point>) for 'the {place}'"
    
    if task == None:
        print("format error")
        return False, None
    
    set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=False)
    set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
    for _ in range(20):
        env.step()
    env.top_camera.get_rgb_graph(save_or_not=True, save_path = "./images/topdown.png")
    set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=True)
    set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)

    point_2d=get_2d_point(obs_image, task)
    x, y = point_2d[0]
    point_3d=pixels_to_world(env, x, y, env.top_camera.get_depth_graph()) 
    
    if action == 'push':
        image_paths, valid_pos = pose_push_gen(point_3d[0], point_3d[1], env)
    elif action == 'rotate up':
        image_paths, valid_pos = pose_rotate_up_gen(point_3d[0], point_3d[1], env)
    elif action == 'rotate down':
        image_paths, valid_pos = pose_rotate_down_gen(point_3d[0], point_3d[1], env)
    

    current_subtask = subtask
    next_subtask = subtasks[idx + 1]
    pose_subtasks = [current_subtask, next_subtask]
    if image_paths is None or len(image_paths) == 0:
        return False, None
    num = choose_pose(pose_subtasks, image_paths, prompts['choose_pose'])
    if num < 0 or num >= len(valid_pos):
        num = len(valid_pos) 
        return False, None
    target_pos = valid_pos[num]

    env.object._prim.set_world_pose(init_pos[0], init_pos[1])
    for _ in range(20):
        env.step()
    
    return action, target_pos


def pose_push_gen(x,y,env):
    z_o = env.object._prim.get_world_pose()[0][2]
    x_o = env.object._prim.get_world_pose()[0][0]
    y_o = env.object._prim.get_world_pose()[0][1]
    z_height = z_o + 0.02
    rotation = env.object._prim.get_world_pose()[1]
    rotation_eu = quat_to_euler_angles(rotation, degrees=True)
    
    s = False
    num = 0
    image_paths = []
    valid_pos = []
    trial = 0

    new_position = [x,y,z_height]
    env.object._prim.set_world_pose(new_position, rotation)
    for _ in range(20):
        env.step()
    z_point = env.object._prim.get_world_pose()[0][2]
    if z_point > z_o+0.01 and z_point < z_o+0.03:
        offset = -0.03
        s = True
    else:
        offset = 0.03

    while(num<4) and trial < 15:
        if abs(x - x_o) > 0.08:
            random_x = x + np.random.uniform(-0.04, 0.04) + offset
        else:
            random_x = x
        
        if abs(y - y_o) > 0.08:
            random_y = y + np.random.uniform(-0.04, 0.04) + offset
        else:
            random_y = y
        
        random_position = np.array([random_x, random_y, z_height]) 
        random_angle = rotation_eu[2] + np.random.uniform(-20, 20)
        modified_rotation_eu = rotation_eu.copy()
        modified_rotation_eu[2] = random_angle 
        random_rotation = euler_angles_to_quat(modified_rotation_eu, degrees=True)
        new_position = random_position
        new_rotation = random_rotation
        
        env.object._prim.set_world_pose(new_position, new_rotation)
        trial += 1
        if trial == 14:
            print("Trial limit reached, exiting loop.")
            break
        for _ in range(100):
            env.step()

        if env.object._prim.get_world_pose()[0][2] < 0.7:
            continue
        elif s == False and env.object._prim.get_world_pose()[0][2] > (z_o+0.01):
            continue
        else:
            num = num + 1
            image_path = f"./images/image{num}.png"
            env.top_camera.get_rgb_graph(save_or_not=True, save_path=image_path)
            image_paths.append(image_path) 
            valid_pos.append(env.object._prim.get_world_pose())

    return image_paths, valid_pos


def pose_rotate_up_gen(x,y,env):
    z_o = env.object._prim.get_world_pose()[0][2]
    x_o = env.object._prim.get_world_pose()[0][0]
    z_height = z_o + abs(x - x_o)
    rotation = env.object._prim.get_world_pose()[1]
    rotation_eu = quat_to_euler_angles(rotation, degrees=True)

    num = 0
    image_paths = []
    valid_pos = []
    trial = 0
    while num<4 and trial < 15:
        random_x = x + np.random.uniform(-0.05, 0.05)
        random_y = y + np.random.uniform(-0.05, 0.05)
        random_z = z_height + np.random.uniform(-0.03, 0.03)
        random_position = np.array([random_x, random_y, random_z])
        random_x_angle = rotation_eu[0] + np.random.uniform(-280, -260)
        modified_rotation_eu = rotation_eu.copy()
        modified_rotation_eu[0] = random_x_angle 
        random_rotation = euler_angles_to_quat(modified_rotation_eu, degrees=True)

        env.object._prim.set_world_pose(random_position, random_rotation)
        trial += 1
        if trial == 14:
            print("Trial limit reached, exiting loop.")
            break
        for _ in range(100):
            env.step()

        if env.object._prim.get_world_pose()[0][2] < 0.7 or env.object._prim.get_world_pose()[0][0] < x:
            continue
        if env.object._prim.get_world_pose()[0][2] < z_o + 0.001:
            continue
        else:
            num = num + 1
            image_path = f"./images/image{num}.png"
            env.front_camera.get_rgb_graph(save_or_not=True, save_path=image_path)
            image_paths.append(image_path)
            valid_pos.append(env.object._prim.get_world_pose())

    return image_paths, valid_pos


def pose_rotate_down_gen(x,y,env):
    y = env.object._prim.get_world_pose()[0][1]
    z = env.object._prim.get_world_pose()[0][2]
    rotation = env.object._prim.get_world_pose()[1]
    rotation_eu = quat_to_euler_angles(rotation, degrees=True)
    
    num = 0
    image_paths = []
    valid_pos = []
    trial = 0
    while(num<4) and trial < 15:
        random_x = x + np.random.uniform(-0.03, 0.03)
        random_y = y + np.random.uniform(-0.03, 0.03)
        random_z = z
        random_position = np.array([random_x, random_y, random_z])
        random_x_angle = rotation_eu[2] + np.random.uniform(80, 100)
        modified_rotation_eu = rotation_eu.copy()
        modified_rotation_eu[0] = random_x_angle 
        random_rotation = euler_angles_to_quat(modified_rotation_eu, degrees=True)

        env.object._prim.set_world_pose(random_position, random_rotation)
        set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
        for i in range(15):
            env.step()
        trial += 1
        if trial == 14:
            print("Trial limit reached, exiting loop.")
            break
        for _ in range(150):
            env.step()
        if env.object._prim.get_world_pose()[0][2] < 0.7 or env.object._prim.get_world_pose()[0][2] > z-0.005:
            continue
        elif abs(quat_to_euler_angles(env.object._prim.get_world_pose()[1], degrees=True)[0]) < 150:
            continue
        else:
            num = num + 1
            image_path = f"./images/image{num}.png"
            env.front_camera.get_rgb_graph(save_or_not=True, save_path=image_path)
            image_paths.append(image_path)
            valid_pos.append(env.object._prim.get_world_pose())
        
        set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)

    return image_paths, valid_pos


def choose_pose(task, image_paths, prompt):
    for idx, image_path in enumerate(image_paths, start=1):
        add_number_to_image(image_path, idx)
    
    is_match = False
    p = 0
    
    while p < 5:
        p = p + 1
        plan = call_gpt_model(
            prompt1=prompt,
            task=task,
            images= image_paths,
        )

        print(plan)

        match = re.search(r'"image":\s*(\d+)', plan)

        if match:
            image_number = int(match.group(1))  
            print(f"Extracted image number: {image_number}")
            is_match = True
            break
        else:
            print("No image number found.")
            is_match = False
    
    if not is_match:
        print("No valid response after several trials, choose the first one")
        image_number = 1

    return image_number - 1  
