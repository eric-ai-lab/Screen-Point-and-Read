import cv2, os
from tqdm import tqdm
def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IOU) of two bounding boxes in xywh format.
    
    Args:
        bbox1 (tuple): Bounding box coordinates in xywh format (x, y, w, h).
        bbox2 (tuple): Bounding box coordinates in xywh format (x, y, w, h).
    
    Returns:
        float: Intersection over Union (IOU) value.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)
    
    # Calculate the area of intersection
    intersection_area = w_intersection * h_intersection
    
    # Calculate the area of union
    union_area = w1 * h1 + w2 * h2 - intersection_area
    
    # Calculate the IOU
    iou = intersection_area / union_area
    
    return iou

def select_target_path(target_point, image, bboxes_small, bboxes_large):
    """
    Selects the target path in the Hierarchical Layout Tree based on the target point.
    The Hierarchical Layout Tree is represented by the detected regions.
    Args:
        target_point (tuple): The target point (x, y).
        image (numpy.ndarray): The image.
        bboxes_small (list): A list of bounding boxes for local regions in the Hierarchical Layout Tree.
        bboxes_large (list): A list of bounding boxes for global regions in the Hierarchical Layout Tree.
    Returns:
        tuple: A tuple containing two lists - selected_small_bbox and selected_large_bbox.
            - selected_small_bbox: The selected small bounding box.
            - selected_large_bbox: The selected large bounding box."""
    x,y = target_point

    for bbox in sorted(bboxes_small, key=lambda b: b[2]*b[3]):
        if bbox[0]-5 <= x <= bbox[0] + bbox[2]+5 and bbox[1]-5 <= y <= bbox[1] + bbox[3]+5:
            selected_small_bbox = bbox
            # cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 4)
            # print(f"Small bbox: {bbox}")
            break
        
    else:
        selected_small_bbox = []
    
    # If small bbox exists
    if selected_small_bbox != []:
        selected_large_bbox = []
        max_iou = 0
        # large bbox
        for bbox in bboxes_large:
            if calculate_iou(bbox, selected_small_bbox) > max_iou and  bbox[2]*bbox[3]> selected_small_bbox[2]*selected_small_bbox[3]:
                selected_large_bbox = bbox
                max_iou = calculate_iou(bbox, selected_small_bbox)
        
                
        if selected_large_bbox == []:
            selected_large_bbox = [0,0,image.shape[1], image.shape[0]]
        
    # if no small bbox
    else:

        # large bbox
        for bbox in sorted(bboxes_large, key=lambda b: b[2]*b[3]):
            if bbox[2]*bbox[3]< 0.9*image.shape[0]*image.shape[1] and bbox[0]-5 <= x <= bbox[0] + bbox[2]+5 and bbox[1]-5 <= y <= bbox[1] + bbox[3]+5:
                selected_large_bbox = bbox
                selected_small_bbox = selected_large_bbox
                # cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 4)
                # print(f"Small bbox: {bbox}")
                break
            
        else:
            selected_large_bbox = [0,0,image.shape[1], image.shape[0]]
            selected_small_bbox = [max(0, x-150), max(0, y-150), 300, 300]
            if selected_small_bbox[0] + selected_small_bbox[2] > image.shape[1]:
                selected_small_bbox[2] = image.shape[1] - selected_small_bbox[0]-1
                selected_small_bbox[1] = image.shape[0] - selected_small_bbox[3]-1
                

    selected_small_bbox = [int(i) for i in selected_small_bbox]
    selected_large_bbox = [int(i) for i in selected_large_bbox]
    return selected_small_bbox, selected_large_bbox