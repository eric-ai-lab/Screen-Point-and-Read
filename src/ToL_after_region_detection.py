import json
from src.generate_multilens_prompts import crop_area
from src.hierarchical_tree_construction import tree_construction
from src.select_target_path import select_target_path
from src.call_gpt4o import call_gpt4o
import cv2, os


data = json.load(open(''))
image_path = ''
dino_output_path = ''


prompt = '''You are a smart screen reader that outputs concise natural language to answer questions from users based on the area (box 1) pointed out by the user shown as a red dot on the screen. The red dot is inside the box 1 in the first image, at (x,y) = ({},{}), where x and y are the normalized coordinates.
Note: box 1 is the box with label 1 and box 2 is the box with label 2, box 1 is located inside box 2
Note: the first image shows what is inside box 2, and the second image shows the complete screen.
Note: if the user asks about the location, focus on the layout, explain where box 1 is in box 2 and then explain where box 2 is in the overall complete screen. 
Note: don't mention box 1, box 2 or the red dot in the output.
            
User question: (1) what is this? (2) where it is located in the screen?
Your output should in format (1) … (2) …'''
for k in data.keys():
    item = data[k]
    img_filename = item["img_filename"]
    image = cv2.imread(os.path.join(image_path, img_filename))

    bboxes_small, bboxes_large = tree_construction(img_filename, dino_output_path['test_screendata/0.png'])
    bboxes_large.append([0,0,image.shape[1], image.shape[0]])

    results = []
    for p in ['bbox_point', 'neighbor_point']:
        # For Cycle Consistency Evaluation on the accuracy of the layout description generated, the agent needs to 
        # generate the description for both the target point and the reference point.
        x,y = item[p]
        selected_small_bbox, selected_large_bbox = select_target_path((x,y), image, bboxes_small, bboxes_large)
        outter_area = [min(selected_large_bbox[0], selected_small_bbox[0]), min(selected_large_bbox[1], selected_small_bbox[1]), \
                        max(selected_large_bbox[0]+selected_large_bbox[2], selected_small_bbox[0]+selected_small_bbox[2]), \
                            max(selected_large_bbox[1]+selected_large_bbox[3], selected_small_bbox[1]+selected_small_bbox[3])]
        outter_area[2] = outter_area[2] - outter_area[0]
        outter_area[3] = outter_area[3] - outter_area[1]
        
        cropped_image = crop_area(selected_small_bbox, outter_area, image.copy(),'1', [x,y])
        temp_image_path_1 = "temp_ssr_{}_1.jpeg".format(k)
        cv2.imwrite(temp_image_path_1, cropped_image)
        
        cropped_image = crop_area(selected_large_bbox, [0,0,image.shape[1], image.shape[0]], image.copy(), '2')
        temp_image_path_2 = "temp_ssr_{}_2.jpeg".format(k)
        cv2.imwrite(temp_image_path_2, cropped_image)
        
        result = call_gpt4o(prompt.format(x, y), temp_image_path_1,temp_image_path_2)
        os.remove(temp_image_path_1)
        os.remove(temp_image_path_2)
        results.append(result)
    item['eval_des'] = results[0]
    item['eval_reference_des'] = results[1]
