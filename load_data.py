from zipfile import ZipFile
import os
import shutil
import re
import random
from PIL import Image
from glob import glob
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from global_vars import *
    
    

def prime_dataset():

    # Read Each Image With its Class Label
    images = []
    folders = CLASSES

    for folder in folders:
        t = folder
        x = os.listdir(f'{SOURCE_DIRECTORY}/{t}')
        for i in x:
            for j in re.split(r'[-;,\t\s]\s*', i):
                if j == '':
                    continue
                images.append({'Class': t, 'Image': j})

    # Partition Images into Training, Validation, and Testing
    for c in folders:
        os.makedirs(f'{TRAIN_DIRECTORY}{c}', exist_ok=True)
        os.makedirs(f'{VALID_DIRECTORY}{c}', exist_ok=True)
        os.makedirs(f'{TEST_DIRECTORY}{c}', exist_ok=True)
        os.makedirs(f'{RAW_DIRECTORY}{c}', exist_ok=True)
        

    counter = 0
    for c in folders:

        try:
            numOfFiles = len(next(os.walk(f'{SOURCE_DIRECTORY}{c}/'))[2])
            for files in random.sample(glob(f'{SOURCE_DIRECTORY}{c}/*'), int(numOfFiles * TRAIN_SPLIT)):
                shutil.move(files, f'{TRAIN_DIRECTORY}{c}')

            for files in random.sample(glob(f'{SOURCE_DIRECTORY}{c}/*'), int(numOfFiles * VALID_SPLIT)):
                shutil.move(files, f'{VALID_DIRECTORY}{c}')
                
            for files in random.sample(glob(f'{SOURCE_DIRECTORY}{c}/*'), int(numOfFiles * TEST_SPLIT)):
                shutil.move(files, f'{TEST_DIRECTORY}{c}')

            for files in glob(f'{SOURCE_DIRECTORY}{c}/*'):
                shutil.move(files, f'{RAW_DIRECTORY}{c}')
        except StopIteration:
            print(f"No files found in directory: {SOURCE_DIRECTORY}{c}")

        counter += 1

    # shutil.rmtree(SOURCE_DIRECTORY)
    
    


os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

prime_dataset()
