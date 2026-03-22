from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from scripts.doubaoutils import *
from scripts.gptutils import *
from scripts.plan import *
from scripts.pose import generate_pose
from scripts.grasp import execute_grasp
from scripts.low_level import *

# Set the scene here
from Env_StandAlone.External_dexterity.wall import Demo_Scene_Env

doubao_api_key = ""  # your Doubao Ark API Key
openai_api_key = ""  # your OpenAI API Key

def NP(env, task_instruction):
    set_doubao_api(doubao_api_key)
    set_openai_api(openai_api_key)
    env.top_camera.get_rgb_graph(save_or_not=True, save_path = "./images/topdown.png")
    obs_image = "./images/topdown.png"
    env.front_camera.get_rgb_graph(save_or_not=True, save_path="./images/front.png")
    front_image = "./images/front.png"
    prompt_directory = './scripts/prompts'
    prompts = load_prompts(prompt_directory)
    first_plan = plan(task_instruction, obs_image, prompts['firstplan'], prompts['action'])
    subtasks = split_actions(first_plan)
    plan_num = 0
    for idx, subtask in enumerate(subtasks):
        if plan_num > 3:
            print("Too many replans, aborting.")
            break
        temp = 0
        if subtask['action'] != 'move to' or subtask['action'] != 'release':
            obj_o = env.object._prim.get_world_pose()
        while temp < 3:
            if subtask['action'] == 'grasp':
                # pose, ik = execute_grasp(env, obj_o)
                pose = False
                ik = False
                print("[INFO] Grasp Pose:", pose)
                print("[INFO] IK Solution:", ik)
            else:
                pose = True
                if  idx == len(subtasks) - 1 or (idx == len(subtasks) - 2 and subtasks[idx+1]['action'] == 'release'):
                    act = subtask['action']
                    t_pos = env.non_collision_object._prim.get_world_pose()
                else:
                    act, t_pos = generate_pose(env, prompts, subtasks, idx, obs_image, subtask, subtask['action'])

                if not act:
                    ik = False
                elif "move to" in act:
                    ik = move_to_target(env, t_pos)
                    temp = 3
                elif "release" in act:
                    ik = release(env,t_pos)
                    temp = 3
                elif "push" in act:
                    ik = push_to_target(env, t_pos)
                elif "rotate up" in act:
                    ik = rotate_up_target(env, t_pos)
                elif "rotate down" in act:
                    ik = rotate_down_target(env, t_pos)
                
            if ik and pose:
                break
            else:
                if subtask['action'] != 'move to' or subtask['action'] != 'release':
                    env.object._prim.set_world_pose(obj_o[0], obj_o[1])
                    temp = temp + 1   
       
            env.franka.Dense_Rmpflow_Move(
                target_position=np.array([0.2, 0, 1.2]),
                target_orientation=np.array([180.0, 0.0, 0.0]),
                quat_or_not=False
            )
        
        if not pose or not ik:
            plan_num = plan_num + 1
            following_subtasks = subtasks[idx:]
            new_plan = replan(env, subtask, following_subtasks, obs_image, front_image, prompts, pose, ik)
            replan_subtasks = split_actions(new_plan)
            subtasks[idx+1:] = replan_subtasks
            env.object._prim.set_world_pose(obj_o[0], obj_o[1])
            
    
    if not check_pose(env, t_pos, 0.01, 10):
        suc = push_to_target(env, t_pos, 0.01, 10)
    
    if suc:
        print("successfully finish the task!")
    else:
        print("fail to finish the task!")

# Set the task_instruction here
def main():
    task_instruction = "Get the keyboard to the transparent target."
    env = Demo_Scene_Env()
    NP(env, task_instruction)


if __name__ == "__main__":
    main()
    while simulation_app.is_running():
        simulation_app.update()
    
simulation_app.close()
