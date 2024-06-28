import json
from src.generate_multilens_prompts import crop_area
from src.hierarchical_tree_construction import tree_construction
from src.select_target_path import select_target_path


data = json.load(open(''))
image_path = '/home/fanyue1997/ScreenReaderData/images/'
dino_output = json.load(open('/home/fanyue1997/ScreenReaderData//test_screendata/output_dino-4scale_r50_8xb2-90e_mobile_multi_bbox_mobile_pc_web_osworld/summary.json'))


prompt = '''You are a smart screen reader that outputs concise natural language to answer questions from users based on the area (box 1) pointed out by the user shown as a red dot on the screen. The red dot is inside the box 1 in the first image, at (x,y) = ({},{}), where x and y are the normalized coordinates.
Note: box 1 is the box with label 1 and box 2 is the box with label 2, box 1 is located inside box 2
Note: the first image shows what is inside box 2, and the second image shows the complete screen.
Note: if the user asks about the location, focus on the layout, explain where box 1 is in box 2 and then explain where box 2 is in the overall complete screen. 
Note: don't mention box 1, box 2 or the red dot in the output.
            
User question: (1) what is this? (2) where it is located in the screen?
Your output should in format (1) … (2) …'''

