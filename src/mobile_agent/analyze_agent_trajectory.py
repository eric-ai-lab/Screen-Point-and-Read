### This script is used to analyze the agent trajectory and determine which actions are more likely to fail, figuring out the most possible candidate actions for the agent to take.
import sys
import os
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
import requests
import base64
import pandas as pd
from typing import Tuple, List
from PIL import Image, ImageDraw, ImageFont
import logging
from copy import copy
import datetime
from time import sleep

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
AZURE_OPENAI_API_KEY = os.environ['AZURE_OPENAI_API_KEY']
AZURE_OPENAI_BASE = os.environ['AZURE_OPENAI_BASE']
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']
AZURE_OPENAI_API_VERSION = os.environ['AZURE_OPENAI_API_VERSION']

azure_openai_url = f"{AZURE_OPENAI_BASE}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
official_openai_url = "https://api.openai.com/v1/chat/completions"

system_font = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

time_delay_for_gpt4o = 3

prompt_dict = {
    "chain_of_lens_v1": {
        "system": """
You are a smart screen voice over tool that output concise natural language to user question based on the area pointed by the user (box 1).
Note: if user asks about the location, remember box 1 (box with label 1) is located inside the box 2 (box with label 2) and explain where box 1 in box 2 and where box 2 in the overall screen. 
Note: don't mention box 1 and box 2 directly in the output, instead, use "it", "this".""",
        "user": """
User question: what is this/what is it (box 1) showing and where it is located?
Your output: """  
    },
    "chain_of_lens_v2": {
        "system": None,
        "user": '''You are a smart screen reader that outputs concise natural language to answer questions from users based on the area (box 1) pointed out by the user shown as a red dot on the screen. The red dot is inside the box 1 in the first image, at (x,y) = ({{x}},{{y}}), where x and y are the normalized coordinates.
Note: box 1 is the box with label 1 and box 2 is the box with label 2, box 1 is located inside box 2
Note: the first image shows the box 1 from the view of box 2, and the second image shows the box 2 from the complete screen.
Note: if the user asks about the location, based on the layout, explain where box 1 is in box 2 and then explain where box 2 is in the overall screen. 
Note: don't mention box 1, box 2 or the red dot in the output.
            
User question: (1) what is this? (2) where it is located in the screen?
Your output should in format (1) … (2) …'''
    },
    "action_check_for_click_input_v1": {
        "system": """
You are verifying whether current action on Android phone could help us move along a given instruction to achieve a goal. In this setting, you only check input and click action: click on certain control, or input in text edit. We also describe the historical action and its corresponding working region. You need to judge whether current action and region are correct based on the instruction and history. Your output json should include required fields, like following:
1: If both of action and region are correct, you can output status field 'yes' with a short description in reason field.
2: If the action is wrong or the region is wrong, you can output status field 'no' with a short description in reason field. please also output the correct action and region in the correct_action and correct_region fields.
3: If actions are taken repeatedly on one region without making progress, you can output status field 'response' with a short description in reason field.""",
        "user": """
Historical action and region description: {{haction}}
Goal: {{goal}}
Instruction: {{instruction}}
Current action: {{action}}
Current region: {{region}}
Your question: Is the current action and region correct?
Your output: """  
    },
    "action_check_for_click_input_v2": {
        "system": None,
        "user": """
You are verifying whether current action on Android phone could help us move along a given instruction. In this setting, you only check input and click action: click on certain control, or input in text edit. We also describe the historical action and its corresponding working region. You need to judge whether current action and region are correct based on the instruction and history.  Your output json should include required fields, like following:
1: If both of action and region are correct, you can output status field 'yes' with a short description in reason field.
2: If the action is wrong or the region is wrong, you can output status field 'no' with a short description in reason field. please also output the correct action and region in the correct_action and correct_region fields.
3: If the same actions are taken repeatedly not related with the instruction, you can output status field 'response' with a short description in reason field.
4: (1) and (2) in current region describes the region where current action will be taken. They are used to help you understand the context of the action.
5: sometimes, the description in instructions are general and not specific, you need to understand the context of the action and region to make a decision. And click action can be used to refer to a general touch action, like tap and hold, long-press on text, etc.
Historical action and region description: {{haction}}
Instruction: {{instruction}}
Current action: {{action}}
Current region: {{region}}
Your question: Is the current action and region correct?
Your output: """  
    },
    "action_check_for_click_input_v3": {
        "system": None,
        "user": """
Given the following information in a mobile navigation task:
Historical action and region description: {{haction}}
Instruction: {{instruction}}
Current subgoal: {{action}}
Current region: {{region}}

The agent now is going to interact with the "Current region" described above. Should the agent proceed? 
Note: The agent should not proceed if the "Current region" is repeated too often in "Historical action and region description". 
Note: The agent may proceed if the "Current region" aligns with "Instruction" .
Note: The agent may proceed if the "Current region" matches the "Current subgoal". 

Please provide your answer in the following json format:
{
    "Analysis": "...",
    "Answer": "yes/no"
}"""  
    },
    "action_check_for_click_input_v4": {
        "system": None,
        "user": """
Given the following information in a mobile navigation task:
Historical action and region description: {{haction}}
Instruction: {{instruction}}
Current region: {{region}}

The agent now is going to interact with the "Current region" with the action: {{action}}. Should the agent proceed? 
Note: The agent should not proceed if the "Current region" is repeated too often in "Historical action and region description". 
Note: The agent may proceed if the "Current region" aligns with "Instruction" .

Please provide your answer in the following json format:
{
    "Analysis": "...",
    "Answer": "yes/no"
}"""
    },
    "action_check_for_response_v1": {
        "system": """
You are verifying whether current action on Android phone could help us move along a given instruction to achieve a goal. In this setting, you check response action: response means when you finish the plan or can't move forward. We also describe the historical action and its corresponding working region. You need to judge whether response action and given region are correct based on the instruction and history. Your output json should include required fields, like following:
1: If when you finish the plan or can't move forward, you can output status field 'yes' with a short description in reason field.
2: If you can find any alternative action on current region to make plan move forward, you can output status field 'no' with a short description in reason field. please also output the correct action and region in the correct_action and correct_region fields.""",
        "user": """
Historical action and region description: {{haction}}
Goal: {{goal}}
Instruction: {{instruction}}
Current region: {{region}}
Your question: Is response action and region correct?
Your output: """  
    },
    "action_check_for_response_v2": {
        "system": """
You are verifying whether current action on Android phone could help us move along a given instruction. In this setting, you check response action: response means when you finish the plan or can't move forward. We also describe the historical action and its corresponding working region. You need to judge whether response action and given region are correct based on the instruction and history. Your output json should include required fields, like following:
1: If when you finish the plan or can't move forward, you can output status field 'yes' with a short description in reason field.
2: If you can find any alternative action on current region to make plan move forward, you can output status field 'no' with a short description in reason field. please also output the correct action and region in the correct_action and correct_region fields.
3: (1) and (2) in current region describes the region where current action will be taken. They are used to help you understand the context of the action.
4: sometimes, the description in instructions are general and not specific, you need to understand the context of the action and region to make a decision.""",
        "user": """
Historical action and region description: {{haction}}
Instruction: {{instruction}}
Current region: {{region}}
Your question: Is response action and region correct?
Your output: """  
    }
}

overlap_len = []

# The reloader has already run - do what you want to do here, refer to https://stackoverflow.com/questions/9449101/how-to-stop-flask-from-initialising-twice-in-debug-mode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"analyze_agent_trajectory-{'{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())}.log"),
        ]
)

logger = logging.getLogger(__name__)

import traceback

def global_exception_handler(type, value, error_traceback):
    """
    refer to https://stackoverflow.com/questions/7075200/converting-exception-to-a-string-in-python-3
    """
    logger.exception(f"Uncaught exception {str(value)}")
    logger.error(str(type))
    logger.error(f"\n\t{''.join(traceback.format_tb(error_traceback))}")
    sys.exit()

sys.excepthook = global_exception_handler

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def action_response_judge_gpt4o(action, ssr, plan, goal, instruction, history_action, history_ssr, prompt='action_check_for_response_v2', use_official_openai_url=False):
    
    if use_official_openai_url:        
        url = official_openai_url
        headers = {
            "Content-Type": "application/json",
            "api-key": f"{OPENAI_API_KEY}"
        }
    else:
        url = azure_openai_url
        headers = {
            "Content-Type": "application/json",
            "api-key": f"{AZURE_OPENAI_API_KEY}"
        }

    haction = ""
    for index, (a, s) in enumerate(zip(history_action, history_ssr)):
        haction += f"{index+1}. action: {a}, region: {s}\n"
    user_request = prompt_dict[prompt]['user'].replace("{{haction}}", haction).replace("{{plan}}", plan).replace("{{goal}}", goal).replace("{{instruction}}", instruction).replace("{{region}}", ssr)
    
    if prompt_dict[prompt]['system'] is not None:
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    'role': 'system',
                    'content': prompt_dict[prompt]['system']
                },
                { 
                    "role": "user", 
                    "content": [  
                        { 
                            "type": "text", 
                            "text": user_request
                        }
                    ] 
                } 
            ],
            "max_tokens": 500
        }
    else:
        payload = {
            "model": "gpt-4o",
            "messages": [
                { 
                    "role": "user", 
                    "content": [  
                        { 
                            "type": "text", 
                            "text": user_request
                        }
                    ] 
                } 
            ],
            "max_tokens": 500
        }
    
    response = requests.post(url, headers=headers, json=payload)
    try:
        gpt_response = response.json()['choices'][0]['message']['content']
        sleep(time_delay_for_gpt4o)
        gpt_response = gpt_response.replace("```json", "").replace("```", "")
        from ast import literal_eval
        return literal_eval(gpt_response)
    except:
        raise Exception(response.json())

def action_click_input_judge_gpt4o(action, ssr, plan, goal, instruction, history_action, history_ssr, prompt='action_check_for_click_input_v4', use_official_openai_url=False):
    if use_official_openai_url:        
        url = official_openai_url
        headers = {
            "Content-Type": "application/json",
            "api-key": f"{OPENAI_API_KEY}"
        }
    else:
        url = azure_openai_url
        headers = {
            "Content-Type": "application/json",
            "api-key": f"{AZURE_OPENAI_API_KEY}"
        }

    haction = ""
    for index, (a, s) in enumerate(zip(history_action, history_ssr)):
        haction += f"{index+1}. action: {a}, region: {s}\n"
    user_request = prompt_dict[prompt]['user'].replace("{{haction}}", haction).replace("{{plan}}", plan).replace("{{goal}}", goal).replace("{{instruction}}", instruction).replace("{{action}}", action).replace("{{region}}", ssr)
    if prompt_dict[prompt]['system'] is not None:
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    'role': 'system',
                    'content': prompt_dict[prompt]['system']
                },
                { 
                    "role": "user", 
                    "content": [  
                        { 
                            "type": "text", 
                            "text": user_request
                        }
                    ] 
                } 
            ],
            "max_tokens": 500
        }
    else:
        payload = {
            "model": "gpt-4o",
            "messages": [
                { 
                    "role": "user", 
                    "content": [  
                        { 
                            "type": "text", 
                            "text": user_request
                        }
                    ] 
                } 
            ],
            "max_tokens": 500
        }
    response = requests.post(url, headers=headers, json=payload)
    try:
        gpt_response = response.json()['choices'][0]['message']['content']
        sleep(time_delay_for_gpt4o)
        if prompt in ["action_check_for_click_input_v3", "action_check_for_click_input_v4"]:
            import re
            
            def replace_quotes(match):
                analysis_value = match.group(2)
                answer_value = match.group(4)
                
                # Replace double quotes inside the values
                analysis_value_replaced = analysis_value.replace('"', '\\"')
                answer_value_replaced = answer_value.replace('"', '\\"')
                
                return f'{match.group(1)}{analysis_value_replaced}{match.group(3)}{answer_value_replaced}{match.group(5)}'
            
            gpt_response = gpt_response.replace("```json", "").replace("```", "")
            pattern = r'("Analysis":\s*")(.*?)(",\n\s*"Answer":\s*")(.*?)("\s*})'

            # Use re.sub with the replacement function
            result = re.sub(pattern, replace_quotes, gpt_response, flags=re.DOTALL)
            from ast import literal_eval
            result = literal_eval(gpt_response)
            return {
                "status": result['Answer'],
                "reason": result['Analysis'],
            }
        else:
            gpt_response = gpt_response.replace("```json", "").replace("```", "")
            from ast import literal_eval
            return literal_eval(gpt_response)
    except:
        logger.error(response.json())
        raise Exception(response.json())

def chain_of_lens_call_gpt4o(image_path_0, image_path_1, point=None, prompt='chain_of_lens_v2', use_official_openai_url=False):
    # Getting the base64 string
    base64_image_0 = encode_image(image_path_0)
    base64_image_1 = encode_image(image_path_1)

    if use_official_openai_url:        
        url = official_openai_url
        headers = {
            "Content-Type": "application/json",
            "api-key": f"{OPENAI_API_KEY}"
        }
    else:
        url = azure_openai_url
        headers = {
            "Content-Type": "application/json",
            "api-key": f"{AZURE_OPENAI_API_KEY}"
        }

    if point is None:
        prompt_text = prompt_dict[prompt]['user']
    else:
        prompt_text = prompt_dict[prompt]['user'].replace("{{x}}", f"{point[0]}").replace("{{y}}", f"{point[1]}")
        
    if prompt_dict[prompt]['system'] is None:
        payload = {
            "model": "gpt-4o",
            "messages": [
                { 
                    "role": "user", 
                    "content": [  
                        { 
                            "type": "text", 
                            "text": prompt_text
                        },
                        { 
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_0}"
                            }
                        },
                        { 
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_1}"
                            }
                        }
                    ] 
                } 
            ],
            "max_tokens": 500
        }
    else:
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    'role': 'system',
                    'content': prompt_text
                },
                { 
                    "role": "user", 
                    "content": [  
                        { 
                            "type": "text", 
                            "text": prompt_dict[prompt]['user']
                        },
                        { 
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_0}"
                            }
                        },
                        { 
                            "type": "image_url",
                            "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_1}"
                            }
                        }
                    ] 
                } 
            ],
            "max_tokens": 500
        }
    response = requests.post(url, headers=headers, json=payload)
    try:
        sleep(time_delay_for_gpt4o)
        return response.json()['choices'][0]['message']['content']
    except:
        logger.error(response.json())
        raise Exception(response.json())
    
def draw_transparent_dot(image_cv2, x, y, dot_radius=10):
    # Load the source image
    # Convert the image from BGR to RGB
    # image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    # Convert the image from NumPy array to PIL Image
    image_pil = Image.fromarray(image_cv2)

    # Convert the PIL Image to RGBA
    image = image_pil.convert("RGBA")

    # Create an overlay for the dot
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw the central solid dot
    draw.ellipse((x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius), fill=(0, 0, 255, 128))

    # Create a gradient effect around the dot
    gradient = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw_gradient = ImageDraw.Draw(gradient)
    
    for i in range(dot_radius*3, dot_radius, -1):
        alpha = int(128 * (dot_radius*4 - i) / (dot_radius * 2))
        draw_gradient.ellipse((x-i, y-i, x+i, y+i), fill=(0, 0, 255, alpha))
    
    # Composite the images together
    combined = Image.alpha_composite(image, gradient)
    combined = Image.alpha_composite(combined, overlay)

    # Convert to RGB to display
    combined = combined.convert("RGB")
    
    combined = np.array(combined)


    return combined
    
def pad_image_and_adjust_bbox(img, selected_bbox, pad_size_top, pad_size_bottom, pad_size_left, pad_size_right):
    # Pad the image
    img_padded = cv2.copyMakeBorder(img, pad_size_top, pad_size_bottom, pad_size_left, pad_size_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Adjust the bounding box
    selected_bbox_padded = list(selected_bbox)
    selected_bbox_padded[1] += pad_size_top  # Adjust the top coordinate
    selected_bbox_padded[0] += pad_size_left  # Adjust the left coordinate

    return img_padded, selected_bbox_padded

def crop_area(selected_bbox, outer_bbox, img, label = '1', point=None):
    """crop_area, label == 1 for small box, label == 12for big box

    Args:
        selected_bbox (_type_): selected bounding box
        img (_type_): selected image
        label (str, optional): label for the cropped region. Defaults to '1'.

    Returns:
        canvas: returned the canvas with the cropped area
    """
    if label == '1':
        img = draw_transparent_dot(img,point[0], point[1], min(10,max(1,min(selected_bbox[2], selected_bbox[3])//20)))
        max_boarder_left = min(200, selected_bbox[0], img.shape[1] - selected_bbox[0] - selected_bbox[2])
        max_boarder_top = min(200, selected_bbox[1], img.shape[0] - selected_bbox[1] - selected_bbox[3])
        if max_boarder_top <60:
            img, selected_bbox = pad_image_and_adjust_bbox(img, selected_bbox, 60, 60, 60, 60)
            max_boarder_left = min(200, selected_bbox[0], img.shape[1] - selected_bbox[0] - selected_bbox[2])
            max_boarder_top = min(200, selected_bbox[1], img.shape[0] - selected_bbox[1] - selected_bbox[3])
        copied_width = selected_bbox[2] + 2 * max_boarder_left
        copied_height = selected_bbox[3] + 2 * max_boarder_top
        # Calculate the coordinates of the top left corner of the copied area
        copied_x = selected_bbox[0] - max_boarder_left
        copied_y = selected_bbox[1] - max_boarder_top
        
    else:
    
        img, selected_bbox = pad_image_and_adjust_bbox(img, selected_bbox, 60, 60, 60, 60)
        copied_width = img.shape[1]
        copied_height = img.shape[0]
        copied_x = 0
        copied_y = 0
    # Calculate the coordinates of the top left corner of the bounding box
    bbox_x = selected_bbox[0]
    bbox_y = selected_bbox[1]

    # Calculate the width and height of the bounding box
    bbox_width = selected_bbox[2]
    bbox_height = selected_bbox[3]
    

    # Draw the bounding box on the canvas
    if label == '1':
        thic = 4
    else:
        thic = 5
    cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), (153, 0, 167), thic)

    # Add the label to the top left corner of the bounding box with a white background
    label_x = bbox_x
    # if label == '1' and max_boarder_top <30:
    #     label_y = bbox_y
    #     cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x + 25*thic//2, bbox_y+30*thic//2), (200, 0, 167), -1)
    #     cv2.putText(img, label, (label_x, label_y+30*thic//2), cv2.FONT_HERSHEY_SIMPLEX, 1*thic//2, (0, 0, 0), thic)
    # else:
    label_y = bbox_y-5*thic//2
    cv2.rectangle(img, (bbox_x, bbox_y-30*thic//2), (bbox_x + 25*thic//2, bbox_y), (200, 0, 167), -1)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1*thic//2, (0, 0, 0), thic)
    # Copy the area from the original image to the center of the canvas
    
    copied_y = min(copied_y, outer_bbox[1])
    copied_x = min(copied_x, outer_bbox[0])
    copied_y2 = max(outer_bbox[1]+outer_bbox[3], copied_y + copied_height)
    copied_x2 = max(outer_bbox[0]+outer_bbox[2], copied_x + copied_width)
    
    copied_area = img[copied_y:copied_y2, copied_x:copied_x2]

    return copied_area

def calculate_iou(bbox1, bbox2):
    x1, y1, x3, y3 = bbox1[:4]
    x2, y2, x4, y4 = bbox2[:4]
    
    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x3, x4)
    y_bottom = min(y3, y4)
    
    # If the rectangles do not intersect, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate the area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate the area of each bounding box
    bbox1_area = (x3 - x1) * (y3 - y1)
    bbox2_area = (x4 - x2) * (y4 - y2)
    
    # Calculate the union area
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate the IOU
    iou = intersection_area / union_area
    
    return iou

def conf():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_trajectory_file", type=str, default="../mobile_data/failed_agent_trajectory/trajectory.json")
    argparser.add_argument("--image_regions_file", type=str, default="image_to_bbox_and_scores.json")
    argparser.add_argument("--trajectory_file", type=str, default="plan_execution_annotated_local_status.json")
    argparser.add_argument("--statistic_file", type=str, default="ssr_statistic_agent.json")
    argparser.add_argument("--candidate_actions", nargs='+', type=str, default="click")
    argparser.add_argument("--with_history", type=bool, default=True)
    argparser.add_argument("--scoring_confident_threshold_of_small_region", type=float, default=0.05)
    argparser.add_argument("--scoring_confident_threshold_of_large_region", type=float, default=0.15)
    argparser.add_argument("--app_file", type=str, default="../mobile_data/failed_agent_trajectory.xlsx")
    argparser.add_argument("--include_extra_statistics", type=bool, default=False)
    argparser.add_argument("--small_box_override", type=bool, default=True)
    argparser.add_argument("--resume_previous_running_file", type=str, default=None)
    argparser.add_argument("--action_check_for_click_input_prompt", type=str, default='action_check_for_click_input_v4')
    argparser.add_argument("--use_official_openai_url", type=bool, default=False)
    return argparser.parse_args()

def identify_smallest_click_region(position: Tuple[int, int], bbox: List[Tuple[float, float, float, float]], threshold: float = .0) -> Tuple[float, float, float, float]:
    """identify the smallest region that contains the click location

    Args:
        position (Tuple[int, int]): click position
        bbox (List[Tuple[float, float, float, float]]): candidate bounding boxes

    Returns:
        Tuple[float, float, float, float]: returned the smallest region that contains the click location
    """
    x, y = position
    selected_bbox = None
    for bbox in sorted(bbox, key=lambda b: (b[2]-b[0])*(b[3]-b[1])):
        left, top, right, bottom = bbox
        if left - threshold <= x <= right + threshold and top - threshold <= y <= bottom + threshold:
            selected_bbox = bbox
            break
    return selected_bbox

def identify_smallest_input_region(bound: Tuple[int, int, int, int], bbox: List[Tuple[float, float, float, float]], threshold: float = 0.4) -> Tuple[float, float, float, float]:
    """identify the smallest region that contains the click location

    Args:
        position (Tuple[int, int]): click position
        bbox (List[Tuple[float, float, float, float]]): candidate bounding boxes

    Returns:
        Tuple[float, float, float, float]: returned the smallest region that contains the click location
    """
    selected_bbox = None
    for bbox in sorted(bbox, key=lambda b: (b[2]-b[0])*(b[3]-b[1])):
        iou = calculate_iou(bound, bbox)
        if iou >= threshold:
            selected_bbox = bbox
            break
    return selected_bbox

def central_distance(bbox1, bbox2):
    x1, y1, x3, y3 = bbox1[:4]
    x2, y2, x4, y4 = bbox2[:4]
    x1, y1, x3, y3 = (x1 + x3) / 2, (y1 + y3) / 2, (x2 + x4) / 2, (y2 + y4) / 2
    return ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5

def closer_smallest_bbox_heuristic(position: Tuple[int, int], 
                                   selected_large_bbox:Tuple[float, float, float, float], 
                                   bboxes: List[Tuple[float, float, float, float]], 
                                   threshold_iou_h:float = 0.85, 
                                   threshold_iou_l:float = 1e-3) -> Tuple[Tuple[float, float, float, float], float]:
    """identify the smallest region that contains the click location. This is for the case where the click location is not in any of the bounding boxes. Why 1e-3? 2560 * 1440 = 3686400, 1e-3 * 3686400 = 3686.4, which is the smallest region that can be identified by the human eye. 60 * 60 pixel on 2560 * 1440, estimate as 1/60 of the screen size.

    Args:
        position (Tuple[int, int]): click position
        bbox (List[Tuple[float, float, float, float]]): candidate bounding boxes

    Returns:
        Tuple[Tuple[float, float, float, float], float]: returned the smallest region that contains the click location, distance of heuristic
    """
    import math
    x, y = position
    selected_small_bbox_v1 = None; distance_v1 = float('inf')
    selected_small_bbox_v2 = None; distance_v2 = float('inf')
    for bbox in sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1])):
        iou = calculate_iou(selected_large_bbox, bbox)
        _, top, _, bottom = bbox
        if iou <= threshold_iou_h and iou >= threshold_iou_l and top <= y <= bottom:
            # see: for the case where the click location is well covered by the bounding box and the bounding box has a right size ratio. Specifically, the bounding box is not too large or too small
            d = min(y - top, bottom - y)
            if d < distance_v1:
                distance_v1 = d
                selected_small_bbox_v1 = bbox
        else:
            # see: if not found the bounding box that contains the click location, then find the bounding box that is closer to the click location
            d = central_distance(selected_large_bbox, bbox) 
            if d < distance_v2:
                distance_v2 = d
                selected_small_bbox_v2 = bbox
    if selected_small_bbox_v1 is not None:
        selected_small_bbox = selected_small_bbox_v1; distance = distance_v1
    else:
        selected_small_bbox = selected_small_bbox_v2; distance = distance_v2
    assert selected_small_bbox is not None
    return selected_small_bbox, distance

def inside_large_bbox(small_bbox: Tuple[float, float, float, float], large_bbox: Tuple[float, float, float, float], threshold: float = 5) -> bool:
    """check if the small bounding box is inside the large bounding box

    Args:
        small_bbox (Tuple[float, float, float, float]): small bounding box
        large_bbox (Tuple[float, float, float, float]): large bounding box

    Returns:
        bool: True if the small bounding box i  s inside the large bounding box
    """
    x1, y1, x3, y3 = small_bbox
    x2, y2, x4, y4 = large_bbox
    return (x1 >= x2 or x1 + threshold >= x2) and (y1 >= y2 or y1 + threshold >= y2) and (x3 <= x4 or x3 <= x4 + threshold) and (y3 <= y4 or y3 <= y4 + threshold)

def find_overlapped_small_large_bbox_pair(small_bboxes: List[Tuple[float, float, float, float]],
                                          large_bboxes: List[Tuple[float, float, float, float]]) -> List[Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]]:
    """find the overlapped small and large bounding box pair

    Args:
        small_bboxes (List[Tuple[float, float, float, float]]): small bounding boxes
        large_bboxes (List[Tuple[float, float, float, float]]): large bounding boxes

    Returns:
        Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]: get the pair of the small and paired large bounding boxes
    """
    bbox_pairs = []
    # see: to keep consistency, temp.ly remove the get_candidate_bbox
    # large_bboxes_candidates = get_candidate_bbox(large_bboxes)
    # small_bboxes_candidates = get_candidate_bbox(small_bboxes)
    large_bboxes_candidates, small_bboxes_candidates = large_bboxes, small_bboxes
        
    for large_bbox in large_bboxes_candidates:
        for small_bbox in small_bboxes_candidates:
            if inside_large_bbox(small_bbox, large_bbox):
                bbox_pairs.append((large_bbox, small_bbox))
                break
    return bbox_pairs

def get_candidate_bbox(bboxes):
    bboxes_candidates = []
    for index, cur_bbox in enumerate(bboxes):
        check_contain_sub_bbox = False
        for i in range(len(bboxes)):
            if i != index and inside_large_bbox(bboxes[i], cur_bbox):
                check_contain_sub_bbox = True
                break
        if not check_contain_sub_bbox:
            bboxes_candidates.append(cur_bbox)
    return bboxes_candidates

def draw_text_regions_pair(image_path, position, region_to_text, bound=5, fontsize=20):
    # Open the image
    img = Image.open(image_path)
    # Prepare the drawing context
    draw = ImageDraw.Draw(img)
    if position is not None:
        # You can adjust the radius and color as needed
        draw.ellipse([(position[0]-bound, position[1]-bound), (position[0]+bound, position[1]+bound)], fill='green')
    # Load a font
    if Path(system_font).exists():
        font = ImageFont.truetype(system_font, fontsize)
    else:# or specify a custom font
        font = ImageFont.load_default()
    cnt = 0
    
    for region, text in region_to_text:
        # Draw a red circle with a radius of 10 at the specified position
        # You can adjust the radius and color as needed
        draw.rectangle([(region[0], region[1]), (region[-2], region[-1])], outline='green', width=bound)
        if len(text) < 40:
            draw.text((region[0], region[1]), text, fill='green', font=font)
        else:
            index = int(text[:text.index(":")])
            if index >= 88 and index <= 137:
                draw.text((region[0], region[1] + cnt * 5), text[:10], fill='green', font=font)
                cnt += 1
            else:
                draw.text((region[0], region[1]), text, fill='green', font=font)
    # Save the modified image
    return img

def filter_valid_bboxes(bboxes, scoring_confident_threshold):
    l = [v for v in bboxes if v['score'] >= scoring_confident_threshold]
    l = sorted(l, key=lambda x: x['score'])
    return [v['region'] for v in l]

def calculate_outer_region(selected_small_bbox, selected_large_bbox, default_strategy=False):
    """as small and large bounding boxes may not overlap, calculate the outer region that contains both bounding boxes

    Args:
        selected_small_bbox (float, float, float, float): small bounding box
        selected_large_bbox (float, float, float, float): large bounding box

    Returns:
        _type_: new area including both small and large bounding boxes
    """
    if default_strategy:
        outer_area = selected_large_bbox
    else:
        outer_area = [min(selected_large_bbox[0], selected_small_bbox[0]), min(selected_large_bbox[1], selected_small_bbox[1]), \
                        max(selected_large_bbox[0]+selected_large_bbox[2], selected_small_bbox[0]+selected_small_bbox[2]), \
                            max(selected_large_bbox[1]+selected_large_bbox[3], selected_small_bbox[1]+selected_small_bbox[3])]
        outer_area[2] = outer_area[2] - outer_area[0]
        outer_area[3] = outer_area[3] - outer_area[1]
    return outer_area

def click_handler(step: dict, image_to_bbox_and_scores: dict, context) -> Tuple[str, str]:
    """click handler for the agent. The motivation is to use the click location and ssr to describe the region where actions occurred. To be specific, need to detect if
    1. the click location is correct given the plan, esp. step['attributing']['instruction_step']
    2. the click event is necessary given the history of the agent to avoid redundant actions

    Args:
        step (dict): self-described runtime ui context 
        image_to_bbox_and_scores (dict): available image to bounding box and scores with two keys "small_bbox", "large_bbox"
        context (_type_): ui_screen_image, receivedPlan, history_action, history_ssr
        
    Returns:
        Tuple[str, str]: action - type str, describing action on specific region (action + textural target); ssr_info - type str, describing the ssr information from gpt4v
    """
    action, ssr_info = None, None
    ui_screen_image, receivedPlan, goal, history_action, history_ssr, args = context
    action_check_for_click_input_prompt = args.action_check_for_click_input_prompt
    scoring_confident_threshold_of_small_region = args.scoring_confident_threshold_of_small_region
    scoring_confident_threshold_of_large_region = args.scoring_confident_threshold_of_large_region
    small_box_override = args.small_box_override
    img = cv2.imread(ui_screen_image)
    attributing_step = step['attributing']['instruction_step'] if 'attributing' in step and 'instruction_step' in step['attributing'] else ''
    # see: find the click location
    x, y = step['position']['x'], step['position']['y']
    small_bboxes = filter_valid_bboxes(image_to_bbox_and_scores['small_bbox'], scoring_confident_threshold_of_small_region)
    large_bboxes = filter_valid_bboxes(image_to_bbox_and_scores['large_bbox'], scoring_confident_threshold_of_large_region)
    selected_small_bbox, selected_smallest_with_heuristic, selected_large_bbox, heuristic_distance = identify_click_bboxes(img, x, y, small_bboxes, large_bboxes, small_box_override)
    export_region_for_click(step, ui_screen_image, x, y, selected_small_bbox, selected_smallest_with_heuristic, selected_large_bbox, heuristic_distance)
    # see: prepare gpt4v and let it generate the ssr information
    outer_area = calculate_outer_region(selected_small_bbox, selected_large_bbox)
    large_image = crop_and_save_small_bbox(ui_screen_image, img, selected_small_bbox, outer_area, (x, y))
    small_image = crop_and_save_large_bbox(ui_screen_image, img, selected_large_bbox, [0, 0, img.shape[1], img.shape[0]])
    refined_x, refined_y = get_relativeXY_in_large_box(x, y, selected_large_bbox)
    ssr_info = chain_of_lens_call_gpt4o(large_image, small_image, point=(refined_x, refined_y))
    action = step['description'] if "description" in step else step["failureReason"]
    evaluate_result = action_click_input_judge_gpt4o(action, ssr_info, receivedPlan, goal, attributing_step, history_action, history_ssr, prompt=action_check_for_click_input_prompt)
    return action, ssr_info, evaluate_result

def get_relativeXY_in_large_box(x, y, selected_large_bbox):
    x = '%.4f' % (x-selected_large_bbox[0]/ selected_large_bbox[2] )
    y = '%.4f' % (y-selected_large_bbox[1]/ selected_large_bbox[3])
    return x, y

def crop_and_save_large_bbox(ui_screen_image, img, selected_large_bbox, full_screen_region, index:int=None,):
    left, top, right, bottom = [int(v) for v in selected_large_bbox]
    selected_large_bbox_int = [left, top, right-left, bottom-top]
    large_canvas = crop_area(selected_large_bbox_int, full_screen_region, img.copy(), '2')
    # cv2.imshow("large_canvas", large_canvas)
    f = Path(ui_screen_image)
    if index is not None:
        target_fname = f"{f.parent.__str__()}/gpt4v/cache/{f.stem}_large_response{index}{f.suffix}"
    else:
        target_fname = f"{f.parent.__str__()}/gpt4v/cache/{f.stem}_large{f.suffix}"
    Path(target_fname).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(target_fname, large_canvas)
    return target_fname

def crop_and_save_small_bbox(ui_screen_image, img, selected_small_bbox, selected_large_bbox, click_point, index:int=None, boundary=100, size=500):
    left, top, right, bottom = [int(v) for v in selected_small_bbox]
    selected_small_bbox_int = [left, top, right-left, bottom-top]
    left, top, right, bottom = [int(v) for v in selected_large_bbox]
    selected_large_bbox_int = [left, top, right-left, bottom-top]
    small_canvas = crop_area(selected_small_bbox_int, selected_large_bbox_int, img.copy(), '1', point=click_point)
    canvas_width = max(small_canvas.shape[1], small_canvas.shape[0]) + boundary
    canvas_height = max(small_canvas.shape[1], small_canvas.shape[0]) + boundary
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    # Calculate the coordinates to place the small_canvas in the center of the canvas
    x_offset = (canvas_width - small_canvas.shape[1]) // 2
    y_offset = (canvas_height - small_canvas.shape[0]) // 2
    # Copy the small_canvas to the center of the canvas
    canvas[y_offset:y_offset+small_canvas.shape[0], x_offset:x_offset+small_canvas.shape[1]] = small_canvas
    # Resize the canvas to 500x500
    resized_canvas = cv2.resize(canvas, (size, size))
    # cv2.imshow("small_canvas", resized_canvas)
    f = Path(ui_screen_image)
    if index is not None:
        target_fname = f"{f.parent.__str__()}/gpt4v/cache/{f.stem}_small_response{index}{f.suffix}"
    else:
        target_fname = f"{f.parent.__str__()}/gpt4v/cache/{f.stem}_small{f.suffix}"
    Path(target_fname).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(target_fname, resized_canvas)
    return target_fname

def identify_click_bboxes(img, x, y, small_bboxes, large_bboxes, small_box_override):
    selected_small_bbox = identify_smallest_click_region((x, y), small_bboxes, threshold=50)
    selected_smallest_with_heuristic = False; heuristic_distance = -1
    selected_large_bbox = identify_smallest_click_region((x, y), large_bboxes)
    if selected_large_bbox is None:
        # see: using the heuristic to find the largest bounding box
        selected_large_bbox = [0, 0, img.shape[1], img.shape[0]]
    assert selected_large_bbox is not None
    if selected_small_bbox is None:
        if small_box_override:
            selected_small_bbox = selected_large_bbox
        else:
            selected_small_bbox, heuristic_distance = closer_smallest_bbox_heuristic((x, y), selected_large_bbox, small_bboxes)
        selected_smallest_with_heuristic = True
    assert selected_small_bbox is not None
    return selected_small_bbox,selected_smallest_with_heuristic,selected_large_bbox,heuristic_distance

def export_region_for_click(step, ui_screen_image, x, y, selected_small_bbox, selected_smallest_with_heuristic, selected_large_bbox, heuristic_distance):
    region1 = (selected_small_bbox, 'small_bbox_for_click')
    region2 = (selected_large_bbox, 'large_bbox_for_click')
    image_path = f"{Path(ui_screen_image).parent.__str__()}/annotated_images/vis/{step['image']}"
    img = draw_text_regions_pair(image_path, (x, y), [region2, region1])
    if selected_smallest_with_heuristic:
        export_image_path = f"{Path(ui_screen_image).parent.__str__()}/local_segment/heuristic/{step['image']}"
        Path(export_image_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(export_image_path)
        logger.info(f"Exported the heuristic image to {export_image_path} with heuristic_distance : {heuristic_distance}")
    else:
        export_image_path = f"{Path(ui_screen_image).parent.__str__()}/local_segment/threshold/{step['image']}"
        Path(export_image_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(export_image_path)
        logger.info(f"Exported the annotated image to {export_image_path}")
        
def export_region_for_input(step, ui_screen_image, bound, selected_small_bbox, selected_large_bbox):
    region1 = (selected_small_bbox, 'small_bbox_for_input')
    region2 = (selected_large_bbox, 'large_bbox_for_input')
    region = (bound, 'bound')
    image_path = f"{Path(ui_screen_image).parent.__str__()}/annotated_images/vis/{step['image']}"
    img = draw_text_regions_pair(image_path, None, [region2, region1, region])
    export_image_path = f"{Path(ui_screen_image).parent.__str__()}/local_segment/threshold/{step['image']}"
    Path(export_image_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(export_image_path)
    logger.info(f"Exported the annotated image to {export_image_path}")
    
def export_region_for_heuristic_action(step, ui_screen_image, selected_small_bbox, selected_large_bbox):
    region1 = (selected_small_bbox, 'small_bbox_for_heuristic_action')
    region2 = (selected_large_bbox, 'large_bbox_for_heuristic_action')
    image_path = f"{Path(ui_screen_image).parent.__str__()}/annotated_images/vis/{step['image']}"
    img = draw_text_regions_pair(image_path, None, [region2, region1])
    export_image_path = f"{Path(ui_screen_image).parent.__str__()}/local_segment/threshold/{step['image']}"
    Path(export_image_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(export_image_path)
    logger.info(f"Exported the annotated image to {export_image_path}")

def input_handler(step: dict, image_to_bbox_and_scores: dict, context, ui_screen_iou_threshold=0.9) -> Tuple[str, str]:
    action, ssr_info = None, None
    ui_screen_image, receivedPlan, goal, history_action, history_ssr, args = context
    scoring_confident_threshold_of_small_region = args.scoring_confident_threshold_of_small_region
    scoring_confident_threshold_of_large_region = args.scoring_confident_threshold_of_large_region
    img = cv2.imread(ui_screen_image)
    attributing_step = step['attributing']['instruction_step'] if 'attributing' in step and 'instruction_step' in step['attributing'] else ''
    bound = [step['bound']['left'], step['bound']['top'], step['bound']['right'], step['bound']['bottom']]
    simulated_input_click = (bound[0] + (bound[2] - bound[0]) // 2, bound[1] + (bound[3] - bound[1]) // 2) 
    full_screen_region = [0, 0, img.shape[1], img.shape[0]]  
    screen_iou = calculate_iou(bound, [0, 0, img.shape[1], img.shape[0]])    
    small_bboxes = filter_valid_bboxes(image_to_bbox_and_scores['small_bbox'], scoring_confident_threshold_of_small_region)
    large_bboxes = filter_valid_bboxes(image_to_bbox_and_scores['large_bbox'], scoring_confident_threshold_of_large_region)   
    if screen_iou > ui_screen_iou_threshold:
        # see: over the whole screen, then just by select the largest bounding box. just using small_bboxes and large_bboxes to find the pair small and large bounding boxes
        bbox_pairs = find_overlapped_small_large_bbox_pair(small_bboxes, large_bboxes)
        overlap_len.append(len(bbox_pairs))
        logger.info(f"Found {len(bbox_pairs)} overlapped small and large bounding box pairs.")
        if 'description' in step:
            action = step['description']
        elif "input" in step:
            action = step["input"]
        else:
            action = f"input something"
        for k, (small_bbox, large_bbox) in enumerate(bbox_pairs):
            outer_area = calculate_outer_region(small_bbox, large_bbox)
            small_image = crop_and_save_small_bbox(ui_screen_image, img, small_bbox, outer_area, simulated_input_click, index=k+1)
            large_image = crop_and_save_large_bbox(ui_screen_image, img, large_bbox, full_screen_region, index=k+1)
            refined_x, refined_y = get_relativeXY_in_large_box(simulated_input_click[0], simulated_input_click[1], large_bbox)
            ssr_info = chain_of_lens_call_gpt4o(large_image, small_image, point=(refined_x, refined_y), use_official_openai_url=args.use_official_openai_url)
            evaluate_result = action_response_judge_gpt4o(action, ssr_info, receivedPlan, goal, attributing_step, history_action, history_ssr, use_official_openai_url=args.use_official_openai_url)
            if 'status' in evaluate_result and evaluate_result['status'] == 'no':
                # see: find one alterative action for the invalid input behavior (here, we consider as a response type)
                return action, ssr_info, evaluate_result
        return action, "", {
            "status": "no",
            "reason": "no valid input region or valid alternative action found."
        }
    else:
        selected_small_bbox = identify_smallest_input_region(bound, small_bboxes, threshold=0.4)
        selected_large_bbox = identify_smallest_input_region(bound, large_bboxes, threshold=0.01)
        assert selected_small_bbox is not None
        assert selected_large_bbox is not None    # see: prepare gpt4v and let it generate the ssr information
        export_region_for_input(step, ui_screen_image, bound, selected_small_bbox, selected_large_bbox)  
        outer_area = calculate_outer_region(selected_small_bbox, selected_large_bbox)    
        small_image = crop_and_save_small_bbox(ui_screen_image, img, selected_small_bbox, outer_area, simulated_input_click)
        large_image = crop_and_save_large_bbox(ui_screen_image, img, selected_large_bbox, full_screen_region)
        refined_x, refined_y = get_relativeXY_in_large_box(simulated_input_click[0], simulated_input_click[1], selected_large_bbox)
        ssr_info = chain_of_lens_call_gpt4o(large_image, small_image, point=(refined_x, refined_y), use_official_openai_url=args.use_official_openai_url)
        if 'description' in step:
            action = step['description']
        elif "input" in step:
            action = step["input"]
        else:
            action = f"input something"
        evaluate_result = action_click_input_judge_gpt4o(action, ssr_info, receivedPlan, goal, attributing_step, history_action, history_ssr, use_official_openai_url=args.use_official_openai_url)
        return action, ssr_info, evaluate_result

def export_regions_for_response(step, ui_screen_image, bbox_pairs):
    l = []
    for i, (large_bbox, small_bbox) in enumerate(bbox_pairs):
        l.append((large_bbox, f"large_bbox:{i+1}"))
        l.append((small_bbox, f"small_bbox:{i+1}"))
    image_path = f"{Path(ui_screen_image).parent.__str__()}/annotated_images/vis/{step['image']}"
    img = draw_text_regions_pair(image_path, None, l)
    export_image_path = f"{Path(ui_screen_image).parent.__str__()}/local_segment/threshold/{step['image']}"
    Path(export_image_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(export_image_path)
    logger.info(f"Exported the annotated image to {export_image_path}")

def response_handler(step: dict, image_to_bbox_and_scores: dict, context):
    action, ssr_info = None, None
    ui_screen_image, receivedPlan, goal, history_action, history_ssr, args = context
    scoring_confident_threshold_of_small_region = args.scoring_confident_threshold_of_small_region
    scoring_confident_threshold_of_large_region = args.scoring_confident_threshold_of_large_region
    img = cv2.imread(ui_screen_image)
    attributing_step = step['attributing']['instruction_step'] if 'attributing' in step and 'instruction_step' in step['attributing'] else ''
    small_bboxes = filter_valid_bboxes(image_to_bbox_and_scores['small_bbox'], scoring_confident_threshold_of_small_region)
    large_bboxes = filter_valid_bboxes(image_to_bbox_and_scores['large_bbox'], scoring_confident_threshold_of_large_region)
    full_screen_region = [0, 0, img.shape[1], img.shape[0]]  
    # see: over the whole screen, then just by select the largest bounding box. just using small_bboxes and large_bboxes to find the pair small and large bounding boxes
    bbox_pairs = find_overlapped_small_large_bbox_pair(small_bboxes, large_bboxes)
    export_regions_for_response(step, ui_screen_image, bbox_pairs)
    overlap_len.append(len(bbox_pairs))    
    logger.info(f"Found {len(bbox_pairs)} overlapped small and large bounding box pairs.")
    for k, (small_bbox, large_bbox) in enumerate(bbox_pairs):
        simulated_input_click = (small_bbox[0] + (small_bbox[2] - small_bbox[0]) // 2, small_bbox[1] + (small_bbox[3] - small_bbox[1]) // 2) 
        outer_area = calculate_outer_region(small_bbox, large_bbox)
        small_image = crop_and_save_small_bbox(ui_screen_image, img, small_bbox, outer_area, simulated_input_click, index=k+1)
        large_image = crop_and_save_large_bbox(ui_screen_image, img, large_bbox, full_screen_region, index=k+1)
        refined_x, refined_y = get_relativeXY_in_large_box(simulated_input_click[0], simulated_input_click[1], simulated_input_click)
        ssr_info = chain_of_lens_call_gpt4o(large_image, small_image, point=(refined_x, refined_y), use_official_openai_url=args.use_official_openai_url)
        evaluate_result = action_response_judge_gpt4o(action, ssr_info, receivedPlan, goal, attributing_step, history_action, history_ssr, use_official_openai_url=args.use_official_openai_url)
        if 'status' in evaluate_result and evaluate_result['status'] == 'no':
            # see: find one action for the invalid input behavior
            return action, ssr_info, evaluate_result
    return action, "", {
        "status": "yes",
        "reason": "no alterative of response to make action move forward."
    }
    
action_to_handlers = {
    "click": click_handler,
    "input": input_handler,
    "response": response_handler,
}

def update_counter(statistics, trajectory_len_l, metric):
    if trajectory_len_l not in statistics:
        statistics[trajectory_len_l] = {}
    statistics[trajectory_len_l][metric] = statistics[trajectory_len_l].get(metric, 0) + 1

if __name__ == "__main__":
    args = conf()
    total_action = 0
    parent_folder = Path(args.input_trajectory_file).parent.__str__()
    candidate_actions = args.candidate_actions
    all_action_ready = True; not_implemented_actions = []
    for a in candidate_actions:
        if a not in action_to_handlers:
            logger.info(f"Action handler for {a} is not implemented.")
            not_implemented_actions.append(a)
            all_action_ready = False
    if not all_action_ready:
        raise NotImplementedError(f"Action handlers {not_implemented_actions} are not implemented.")
    if not Path(args.app_file).exists():
        raise ValueError(f"App file {args.app_file} can't be found.")
    
    app_details = pd.read_excel(args.app_file)
    
    if args.resume_previous_running_file is not None:
        # see: initialize the statistics from the previous running file
        with open(args.resume_previous_running_file, "r") as f:
            statistics = json.load(f)
    else:
        # see: initialize the statistics from scratch
        if args.include_extra_statistics:
            statistics = {
                "category": {
                    "count": 0,
                },
                "app": {
                    "count": 0,
                },
                "length-variation": {
                    "count": 0,
                },
                "action-variation": {
                    "count": 0,
                },
                "finished" : [
                ]
            }
        else:
            statistics = {
                "action-variation": {
                    "count": 0,
                },
                "finished" : [
                ]
            }
        for action in candidate_actions:
            statistics["action-variation"][action] = {
                "count": 0,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "fp_sample": [],
                "fn_sample": [],
            }
    output_statistic_file_with_datetime = f"{Path(args.statistic_file).stem}_{'{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())}{Path(args.statistic_file).suffix}"
    with open(args.input_trajectory_file, "r") as f:
        trajectories = json.load(f)
        for t in trajectories:
            t_folder = f"{parent_folder}/{t}"
            logger.info(f"Analyzing agent trajectory: {t_folder}")
            # see: clean up heuristic_images and 
            Path(f"{t_folder}/local_segment/heuristic").exists() and [p.unlink() for p in Path(f"{t_folder}/local_segment/heuristic").iterdir()] and Path(f"{t_folder}/local_segment/heuristic").rmdir()
            Path(f"{t_folder}/local_segment/threshold").exists() and [p.unlink() for p in Path(f"{t_folder}/local_segment/threshold").iterdir()] and Path(f"{t_folder}/local_segment/threshold").rmdir()
            trajectory_file = f"{t_folder}/{args.trajectory_file}"
            image_regions_file = f"{t_folder}/{args.image_regions_file}"
            image_to_bbox_and_scores = json.load(open(image_regions_file))
            trajectory = json.load(open(trajectory_file))
            receivedPlan = trajectory['plan']['receivedPlan']
            goal = trajectory['plan']['goal']
            history_action = []
            trajectory_with_evaluation = copy(trajectory)
            history_ssr = []
            relative_folder_root = t_folder.replace("../", "")
            target_row = app_details[app_details['execution_result_folder'] == relative_folder_root]
            category, app = target_row.category.tolist()[0], target_row.app_name.tolist()[0]
            if category is None or app is None:
                raise ValueError(f"Category or app name is not found in the app file {args.app_file} for {t_folder} folder for trajectory {trajectory_file}")
            for i, step in enumerate(trajectory['executions']):
                if 'action' in step:
                    # see: check if current folder has been processed already
                    step_image = f"{t_folder}/{step['image']}"
                    if step_image in statistics["finished"]:
                        continue
                    statistics["finished"].append(step_image)
                    if step['action'] in candidate_actions:
                        # see: only consider the actions that are in the candidate actions
                        action, ssr_info, evaluate_result = action_to_handlers[step['action']](step, image_to_bbox_and_scores[step['image']], (step_image, receivedPlan, goal, history_action, history_ssr, args))
                        history_action.append(action); history_ssr.append(ssr_info)
                        trajectory_with_evaluation['executions'][i]['ssr_info_from_gpt4v0'] = ssr_info
                        trajectory_with_evaluation['executions'][i]['action_evaluation_based_on_srr_info'] = evaluate_result
                        statistics["action-variation"]["count"] += 1
                        statistics["action-variation"][step['action']]["count"] += 1
                        if args.include_extra_statistics:
                            # see: count action_count
                            trajectory_len_l = f"trajectory-{i+1}"
                            statistics['length-variation']["count"] += 1; statistics['category']["count"] += 1; statistics['app']["count"] += 1
                            if trajectory_len_l not in statistics['length-variation']:
                                statistics['length-variation'][trajectory_len_l] = {}
                            statistics['length-variation'][trajectory_len_l]["count"] = statistics['length-variation'][trajectory_len_l].get("count", 0) + 1
                            if category not in statistics['category']:
                                statistics['category'][category] = {}
                            statistics['category'][category]["count"] = statistics['category'][category].get("count", 0) + 1
                            if app not in statistics['app']:
                                statistics['app'][app] = {}
                            statistics['app'][app]["count"] = statistics['app'][app].get("count", 0) + 1
                        if 'action_evaluation' in step:
                            # see: negative example
                            if 'status' in evaluate_result and (evaluate_result['status'] == 'no' or evaluate_result['status'] == 'response'):
                                # see: predicted as negative example
                                statistics["action-variation"][step['action']]["tn"] += 1
                                if args.include_extra_statistics:
                                    update_counter(statistics['length-variation'], trajectory_len_l, "tn")
                                    update_counter(statistics['category'], category, "tn")
                                    update_counter(statistics['app'], app, "tn")
                            else:
                                statistics["action-variation"][step['action']]["fn"] += 1
                                statistics["action-variation"][step['action']]["fn_sample"].append(
                                    {
                                        "trajectory": trajectory_file,
                                        "index": i,
                                        "step": step
                                    }
                                )
                                if args.include_extra_statistics:
                                    update_counter(statistics['length-variation'], trajectory_len_l, "fn")
                                    if "fn_sample" not in statistics['length-variation'][trajectory_len_l]:
                                        statistics['length-variation'][trajectory_len_l]["fn_sample"] = []
                                    statistics['length-variation'][trajectory_len_l]["fn_sample"].append(
                                        {
                                            "trajectory": trajectory_file,
                                            "index": i,
                                            "step": step
                                        }
                                    )
                                    update_counter(statistics['category'], category, "fn")
                                    update_counter(statistics['app'], app, "fn")
                        else:
                            # see: predicted positive example
                            if 'status' in evaluate_result and evaluate_result['status'] == 'yes':
                                statistics["action-variation"][step['action']]["tp"] += 1
                                if args.include_extra_statistics:
                                    update_counter(statistics['length-variation'], trajectory_len_l, "tp")
                                    update_counter(statistics['category'], category, "tp")
                                    update_counter(statistics['app'], app, "tp")
                            else:
                                statistics["action-variation"][step['action']]["fp"] += 1
                                statistics["action-variation"][step['action']]["fp_sample"].append(
                                    {
                                        "trajectory": trajectory_file,
                                        "index": i,
                                        "step": step
                                    }
                                )
                                if args.include_extra_statistics:
                                    update_counter(statistics['length-variation'], trajectory_len_l, "fp")
                                    if "fp_sample" not in statistics['length-variation'][trajectory_len_l]:
                                        statistics['length-variation'][trajectory_len_l]["fp_sample"] = []
                                    statistics['length-variation'][trajectory_len_l]["fp_sample"].append(
                                        {
                                            "trajectory": trajectory_file,
                                            "index": i,
                                            "step": step
                                        }
                                    )
                                    update_counter(statistics['category'], category, "fp")
                                    update_counter(statistics['app'], app, "fp")
                        with open(output_statistic_file_with_datetime, "w") as f:
                            json.dump(statistics, f, indent=1)
                    else:
                        history_action.append(step['description'])
                    total_action += 1
            with open(f"{t_folder}/trajectory_with_evaluation.json", "w") as f:
                json.dump(trajectory_with_evaluation, f, indent=1)
    logger.info(f"Total number of actions: {total_action}")
    logger.info(f"Average number of overlapped small and large bounding box pairs: {pd.Series(overlap_len).describe().to_dict()}")