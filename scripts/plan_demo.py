import json
import os
from gptutils import *

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

