import os
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from plots import plot_ScoreCAM

# Load and resize the image
import cv2
import numpy as np

def highlight_yellow_region_save(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if image loading is successful
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the image to HSV
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Define the yellow range in HSV
    lower_yellow = np.array([25, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # Create a mask for the yellow range
    mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    
    # Create the result image
    result = np.zeros_like(image_rgb)
    result[mask > 0] = [255, 255, 255]  # White for the yellow region
    result[mask == 0] = [0, 0, 0]       # Black for the rest
    
    # Resize the result image to 320x320
    result_resized = cv2.resize(result, (320, 320))
    
    # Save the result image
    cv2.imwrite(output_path, cv2.cvtColor(result_resized, cv2.COLOR_RGB2BGR))
    
    return result_resized

# Path to save the processed image
output_path = "highlighted_yellow_image.png"

# Use the function to process and save the image
highlighted_image = highlight_yellow_region_save("path_to_your_image.png", output_path)

# Display the result if needed (for testing purposes)
if highlighted_image is not None:
    cv2.imshow("Highlighted Yellow Regions", cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#save_image(just_heatmap,"./Just_CAM_"+name+"/"+img_name.split('/')[-2]+"_"+img_name.split('/')[-1].split('.png')[0]+"_"+name+"_just_heatmap.png")    
