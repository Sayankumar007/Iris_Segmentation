CLASSES = ("Open", "Partial", "Closed", "Covered")

SEED = 42

TRAIN_SPLIT = 0.4
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
RAW_SPLIT = 0.4



IMAGE_SHAPE_2D = (224, 224)
IMAGE_SHAPE_3D = (3,224, 224)

SOURCE_DIRECTORY = './Data/'
REFACTORED_DIRECTORY = './refactored_data/'
TRAIN_DIRECTORY = './refactored_data/train/'
VALID_DIRECTORY = './refactored_data/valid/'
TEST_DIRECTORY = './refactored_data/tests/'
RAW_DIRECTORY = './refactored_data/raw/'


EPOCHS = 100
# LEARNING_RATE = 0.1
# LEARNING_RATE = 0.01
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.0001

paths=[ 
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Covered/0003_1_1_2_51_000.png",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Covered/0010_1_1_2_52_003.png",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Covered/0012_1_1_2_52_001.png",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Covered/0013_1_1_2_52_002.png",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Open/0003_2_1_2_20_006.png",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Open/0010_1_1_2_22_003.png",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Open/0011_2_1_2_24_002.png",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Open/0013_2_1_2_20_004.png",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Partial/0001_2_1_2_32_002.png",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Partial/0010_1_1_2_32_001.png",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Partial/0012_1_1_2_33_002.png",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/tests/Partial/0013_1_1_2_32_003.png"
]

folder_paths = [
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/raw/Covered",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/raw/Open",
  "/content/gdrive/MyDrive/Ritam_da/refactored_data/raw/Partial",

]