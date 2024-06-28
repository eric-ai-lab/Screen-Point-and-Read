import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

def draw_transparent_dot(image_cv2, x, y, dot_radius=10):
    """
    Draw a transparent dot on the image at the specified coordinates.
    """

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
def crop_area(selected_bbox, outer_bbox, img, label = '1', point = None):
    """
    Crop the specified area generating lenses from the screenshot based on the selected bounding box.

    Args:
        selected_bbox (tuple): The coordinates of the selected bounding box (x, y, width, height).
        outer_bbox (tuple): The coordinates of the outer bounding box (x, y, width, height).
        img (numpy.ndarray): The input screenshot.
        label (str, optional): The label indicating the type of lens. '1' means generating the lens 1 with local region drawn. '2' means generating the lens 2 with global region drawn. Defaults to '1'.
        point (tuple, optional): The coordinates of the point to be marked on the image. Defaults to None.

    Returns:
        numpy.ndarray: The cropped area from the original image.

    """
    if label == '1':
        img = draw_transparent_dot(img,point[0], point[1], min(5,max(1,min(selected_bbox[2], selected_bbox[3])//20)))
        max_boarder_left = min(200, selected_bbox[0], img.shape[1] - selected_bbox[0] - selected_bbox[2])
        max_boarder_top = min(200, selected_bbox[1], img.shape[0] - selected_bbox[1] - selected_bbox[3])
        if max_boarder_top <60:
            img, selected_bbox = pad_image_and_adjust_bbox(img, selected_bbox, 60, 0, 0, 0)
            max_boarder_left = min(200, selected_bbox[0], img.shape[1] - selected_bbox[0] - selected_bbox[2])
            max_boarder_top = min(200, selected_bbox[1], img.shape[0] - selected_bbox[1] - selected_bbox[3])
            # outer_bbox[2] += 120
            outer_bbox[3] += 60
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

