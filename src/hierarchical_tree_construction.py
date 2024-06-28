import cv2, os
import json
def tree_construction(img_filename, dino_path):
    """
    Constructs the Hierarchical Layout Trees based on the detected regions.

    Args:
        img_filename (str): The filename of the image.
        dino_path (str): The path to the DINO output file.

    Returns:
        tuple: A tuple containing two lists - bboxes_small and bboxes_large.
            - bboxes_small: A list of bounding boxes for small regions.
            - bboxes_large: A list of bounding boxes for large regions.
    """
    dino_output = json.load(open(dino_path))
    list_1 = dino_output[img_filename]
    bboxes_small = [[list_1['bboxes'][i][0], 
               list_1['bboxes'][i][1], 
               list_1['bboxes'][i][2]-list_1['bboxes'][i][0],
               list_1['bboxes'][i][3]-list_1['bboxes'][i][1]] for i in range(len(list_1['bboxes'])) \
                if list_1['scores'][i] > 0.05 and list_1['labels'][i] == 0]    
    
    bboxes_large = [[list_1['bboxes'][i][0], 
               list_1['bboxes'][i][1], 
               list_1['bboxes'][i][2]-list_1['bboxes'][i][0],
               list_1['bboxes'][i][3]-list_1['bboxes'][i][1]] for i in range(len(list_1['bboxes'])) \
                if list_1['scores'][i] > 0.15 and list_1['labels'][i] == 1]
    return bboxes_small, bboxes_large