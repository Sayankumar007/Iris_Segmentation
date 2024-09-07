import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from cams import *
from global_vars import folder_paths



def reverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x[:, 0, :, :] = x[:,0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x


def visualize(img, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    # print(cam.shape)
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max())

    return result



#
def plot_Gradient_CAMs(img_name, model_dict, name):
  model = model_dict["arch"]
  target_layer = model.layer4[1]
  # wrapper for class activation mapping. Choose one of the following.
  # wrapped_model = CAM(model, target_layer)
  wrapped_model =GradCAM(model, target_layer)
  # wrapped_model = GradCAMpp(model, target_layer)
  # wrapped_model = SmoothGradCAMpp(model, target_layer, n_samples=25, stdev_spread=0.15)
  transform =transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
  ])
  img = Image.open(img_name)
  img = img.convert('RGB')
  tensor = transform(img)

  # reshape 4D tensor (N, C, H, W)
  tensor = tensor.unsqueeze(0)
  tensor = tensor.to('cuda')
  cam, _ = wrapped_model(tensor)
  # img = reverse_normalize(tensor)
  # heatmap = visualize(img, cam.cpu())
  # save_image(heatmap,"./CAMS_"+name+"/"+img_name.split('/')[-2]+"_"+img_name.split('/')[-1].split('.png')[0]+"_"+name+"_ScoreCAM.png")
  # save_image(just_heatmap,"./Just_CAM_"+name+"/"+img_name.split('/')[-2]+"_"+img_name.split('/')[-1].split('.png')[0]+"_"+name+"_just_heatmap.png")

  return cam.cpu()



#
def plot_ScoreCAM(img_name,model, name):
  wrapped_model = ScoreCAM(model)
  transform =transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
  ])
  img = Image.open(img_name)
  img = img.convert('RGB')
  tensor = transform(img)

  # reshape 4D tensor (N, C, H, W)
  tensor = tensor.unsqueeze(0)
  tensor = tensor.to('cuda')
  cam = wrapped_model(tensor)
  cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
  cam = 255 * cam.squeeze()
  # img = reverse_normalize(tensor)
  # heatmap = visualize(img, cam.cpu())
  # save_image(heatmap,"./CAMS_"+name+"/"+img_name.split('/')[-2]+"_"+img_name.split('/')[-1].split('.png')[0]+"_"+name+"_ScoreCAM.png")
  # save_image(just_heatmap,"./Just_CAM_"+name+"/"+img_name.split('/')[-2]+"_"+img_name.split('/')[-1].split('.png')[0]+"_"+name+"_just_heatmap.png")

  return cam.cpu()

def plot_CAMS(paths, model, name):
  # Iterate over each image in the input folder
  # for img_file in os.listdir(path):
  #     if img_file.endswith(".jpg") or img_file.endswith(".png"):
  #         img_path = os.path.join(path, img_file)
  for paths in folder_paths:
    for a_file in os.listdir(paths):
      if os.path.isfile("./raw_masks_gradcam/"+paths.split('/')[-1]+"/"+a_file):
        continue
      
      path = os.path.join(paths, a_file)
      cam = plot_Gradient_CAMs(path, model, name)
      
      # Find the minimum and maximum values in the image tensor
      min_val = cam.min()
      max_val = cam.max()

      # Normalize the tensor to the range 0-255
      normalized_tensor = 255 * (cam - min_val) / (max_val - min_val)

      # Scale the normalized tensor to the range 0-1
      scaled_tensor = normalized_tensor/ 255

      # Define the range for the binary mask (e.g., 0.3 to 0.7)
      lower_bound = 0.7
      upper_bound = 0.9

      # Create the binary mask
      binary_mask = ((scaled_tensor >= lower_bound) & (scaled_tensor <= upper_bound)).float()

      
      plt.imshow(binary_mask[0, 0, : , : ].numpy(), cmap='gray')
      plt.axis('off')

      # Save the figure
      # plt.savefig("./raw_masks/"+paths.split('/')[-1]+"/"+path.split('/')[-2]+"_"+path.split('/')[-1].split('.png')[0]+".png")
      plt.savefig("./raw_masks_gradcam/"+paths.split('/')[-1]+"/"+a_file)







