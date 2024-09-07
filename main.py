import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
from global_vars import *
from models import *
from plots import *
from train import build_model

os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


# If dataset not ready yet please follow the instructions of "load_data.py" file and run it first

# ********************* Loading Models ***********************
    
      
if os.path.exists("./checkpoint_resnet.pth"):
    resnet.load_state_dict(torch.load('./checkpoint_resnet.pth', map_location=torch.device(device)))      
      
if os.path.exists("./checkpoint_googlenet.pth"):
    googlenet.load_state_dict(torch.load('./checkpoint_googlenet.pth', map_location=torch.device(device)))

    




# ********** Defining model dictionaries for cams *************

resnet_dict = dict(type='resnet18', arch=resnet, layer_name='layer4_1',input_size=(224, 224))
googlenet_dict = dict(type='googlenet', arch=googlenet, layer_name='inception5b',input_size=(224, 224))
  
  
# training models one by one and generating CAMS with their metrics..

def func_resnet():
    # model = build_model(resnet, CHECKPOINT_PATH = './checkpoint_resnet.pth', name="resnet", EPOCHS=100, LEARNING_RATE=0.0001)
    plot_CAMS(paths, resnet_dict, "resnet" )
    # plot_Gradient_CAMs("./refactored_data/tests/Open/0001_2_1_2_21_002.png", resnet_dict, "resnet" )


def func_googlenet():
    # model = build_model(googlenet, CHECKPOINT_PATH = './checkpoint_googlenet.pth', name="googlenet", EPOCHS=100, LEARNING_RATE=0.0001)
    plot_CAMS(paths, googlenet_dict, "googlenet" )
    # plot_ScoreCAM("./refactored_data/tests/Open/0001_2_1_2_21_002.png", googlenet_dict, "googlenet" )




  
