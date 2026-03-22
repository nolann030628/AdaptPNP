import json
import os
from scripts.gptutils import *
from Env_Config.Room.Object_Tools import set_prim_visible_group

def load_prompts(directory):
    prompts = dict()
    
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        
        if os.path.isfile(path) and filename.endswith('.txt'):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            key = filename[:-4]  

            prompts[key] = content
    
    return prompts


def split_actions(plan):
    plan = plan.strip().replace('```json', '').replace('```', '').strip()
    actions = json.loads(plan)
    subtasks = []
    for action in actions:
        subtasks.append(action)
    return subtasks


def plan(task, obs_image, prompt_plan, prompt_action):
    plan = call_gpt_model(
        prompt1=prompt_plan,
        prompt2=prompt_action,
        task=task,
        images= [obs_image],
    )
    print(plan)
    return plan


def get_env(task, obs_image, prompt1):
    env = call_gpt_model(
        prompt1=prompt1,
        task=task,
        images= [obs_image],
    )
    return env


def replan(env, subtask, following_subtasks, topdown_img, front_img, prompts, grasp_pose=False, ik=False):
    set_prim_visible_group(prim_path_list=["/World/Franka"], visible=False)
    for i in range(15):
        env.step()

    if not grasp_pose:
        prompt = prompts['check_nopose']
        set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=False)
        for i in range(15):
            env.step()
    elif not ik:
        prompt = prompts['check_ik']

    env.front_camera.get_rgb_graph(save_or_not=True, save_path="./images/front.png")
    env.top_camera.get_rgb_graph(save_or_not=True, save_path="./images/topdown.png")
    set_prim_visible_group(prim_path_list=["/World/Franka"], visible=True)
    set_prim_visible_group(prim_path_list=["/World/NonCollisionObject"], visible=True)
    for i in range(15):
        env.step()

    get_err = False
    while not get_err :
        error_message = call_gpt_model(
            prompt1=prompt,
            task=subtask,
            images= [topdown_img, front_img],
        )
        print(error_message)
        error_message = error_message.strip().replace('```json', '').replace('```', '').strip()
        
        try:
            error = json.loads(error_message)
            error_type = error['error_type']
            get_err = True
        except json.JSONDecodeError as e:
            print(f"fail to check out the error")
            continue
    
    env.front_camera.get_rgb_graph(save_or_not=True, save_path="./images/front.png")
    
    if error_type == "object_blocked":
        extra_env = get_env(subtask, front_img, prompts['get_env_block'])
    else:
        extra_env = get_env(subtask, front_img, prompts['get_env_support'])
    
    replan = call_gpt_model(
        prompt1=prompts['grasp_replan_external'],
        prompt2=extra_env,
        prompt3=prompts['action'],
        task=following_subtasks,
        error_message=error_message,
    )
    
    print(replan)

    return replan

