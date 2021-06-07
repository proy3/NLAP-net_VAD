"""
This file contains all the global variables that are used in other scripts.
"""
import os
import warnings
warnings.filterwarnings('ignore')

TENSORFLOW_DETERMINISM = True

# Root directory:
root_dir = os.getcwd()

# Default seed which will be used everywhere for reproducing results.
# It corresponds to the sum of four numbers which are the sums of the cubes of their digits.
DEFAULT_RAND_SEED = 153 + 370 + 371 + 407

DEFAULT_DATASET = 'shanghaitech'

DEFAULT_DATATYPE_CLS = 'MAData'
DEFAULT_MODEL_CLS = 'NextPredictionNet'
DEFAULT_AE_MODEL_CLS = 'ObjectCentricCAEs'

DEFAULT_MIN_DET_SCORE = 0.5
DEFAULT_MIN_TRAIN_DET_SCORE = 0.5
DEFAULT_MIN_TEST_DET_SCORE = 0.4

MAX_DET_SIZE_RATIO = 0.3
MIN_DET_PIXELS = 10

WHITE_COLOR = 255
ROI_SIZE = 64
FRAME_SIZE = 128
FRAME_DIFF_GAP = 3
DEFAULT_N_FRAMES_PER_VIDEO = 10
NORMALIZE_IMAGES = False
MAX_NORM_VAL = 1
MIN_NORM_VAL = 0
MAX_NORM = 1.0
MIN_NORM = 0.0
FRAMES_STEP = 1

USE_MOTION = False

HOLISTIC_METHOD = False

INFERENCE_MODE = False

RANDOM_SUBSET_TRAINING = False

NEW_TRAINING_SCHEME = False
FIRST_N_EPOCHS = 10
REDUCE_SUBSET_STEP = 5
REDUCE_SUBSET_FACTOR = 0.7
REDUCE_SUBSET_MIN = 40
REDUCE_EASY_ONLY = True

CHANGE_RANDOM_SUBSET = False
CHANGE_SUBSET_STEP = 20

DEFAULT_N_SAMPLES_PER_VIZ = 15
HORIZONTAL_MARGIN_SIZE = 10
VERTICAL_MARGIN_SIZE = 5

# Network architecture
USE_CROSS_DOMAIN_GENERATORS = True  # If false, use Auto-Encoders
USE_SHARED_ENCODERS = True
USE_SKIP_CONNECTIONS = True  # Between encoder and decoder of each generators
USE_SKIP_DIFF = False
USE_CONSISTENCY_LOSS = True  # Does not make sense when USE_CROSS_DOMAIN_GENERATORS is false
USE_ADVERSARIAL_LOSS = True  # Use GAN for training the generators

USE_INTENSITY_ONLY = True

SMOOTH_ADVERSARIAL_LABELS = True

# Network default parameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_N_EPOCHS = 300
DEFAULT_N_INNER_EPOCHS = 3
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_FINE_TUNE_LR = 1e-4
DEFAULT_G_LEARNING_RATE = 1e-3
DEFAULT_D_LEARNING_RATE = 1e-4
DEFAULT_FT_REDUCE_FACTOR = 0.1
DEFAULT_START_FINE_TUNE = 201

USE_SEPARABLE_CONV = False

DEFAULT_CONV_KERNEL_SIZE = 4
DEFAULT_CONV_STRIDES = 2
DEFAULT_CONV_LEAKY_RELU_ALPHA = 0.2
DEFAULT_OUT_ACTIVATION = 'sigmoid'
DEFAULT_CONV_DROPOUT_RATE = 0.1
DEFAULT_CONV_BATCH_NORM = False
DEFAULT_CONV_KERNEL_INIT = 'he_uniform'
DEFAULT_CONV_SKIP_DIFF = 'square'

DEFAULT_D_LOSS = 'mean_squared_error'
DEFAULT_D_LOSS_RATIO = 1.0
DEFAULT_G_LOSS_RATIO = 1.0
DEFAULT_G_LOSS_L1_RATIO = 0.0
DEFAULT_G_LOSS_L2_RATIO = 0.0
DEFAULT_G_LOSS_SS_RATIO = 1.0
DEFAULT_G_LOSS_GD_RATIO = 0.0

DEFAULT_D_OPT_METHOD = 'adam'
DEFAULT_G_OPT_METHOD = 'adam'
DEFAULT_OPT_GRAD_CLIP = 0.5

DEFAULT_SMOOTHING_FACTOR = 10
DEFAULT_NORMALIZE_SCORES = True
DEFAULT_BINARIZE_SCORES = False

DEFAULT_SCORE_USE_D = False
DEFAULT_SCORE_METHOD = 'ssim'
DEFAULT_SCORE_C_RATIO = 1.0
DEFAULT_SCORE_P_RATIO = 1.0
DEFAULT_SCORE_USE_GRID = False
DEFAULT_SCORE_GRID_SIZE = 2

USE_SVM_MODELS = True
DEFAULT_N_SVM_MODELS = 10

SCORE_AUTO_NAME = 'auto'
SCORE_METHODS = ['mae', 'mse', 'vse', 'ssim', 'psnr', 'nrmse']

IMAGE_EXTENSIONS = ['.ras', '.xwd', '.bmp', '.jpe', '.jpg', '.jpeg', '.xpm', '.ief', '.pbm', '.tif', '.gif',
                    '.ppm', '.xbm', '.tiff', '.rgb', '.pgm', '.png', '.pnm']
VIDEO_EXTENSIONS = ['.avi', '.mp4', '.mpeg']

DATASETS_PATH = os.environ['Datasets_ROOT']
PRETRAINED_PATH = os.environ['Pretrained_ROOT']
ESTIMATIONS_PATH = os.environ['Estimations_ROOT']
TEMP_PATH = os.environ['Temp_ROOT']

DET_METHOD_NAME = 'CenterNet'
FLOW_METHOD_NAME = 'FlowNet2'

DEFAULT_DET_TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
DEFAULT_DET_DATASET = 'coco'
DEFAULT_DET_ARCH = 'hourglass'  # 'hourglass'
DEFAULT_DET_MODEL = 'dla_2x'
DET_MODEL_NAMES = {'dla_34': 'dla_2x', 'hourglass': 'hg'}

DEFAULT_FLOW_ARCH = 'FlowNet2'

FRAMES_FILENAME = 'frames.npy'
DET_FILENAMES = {'ctdet_coco_hg': 'detections.npy',
                 'ctdet_coco_dla_2x': 'detections_v2.npy'}
DEFAULT_DET_MODEL_NAME = f'{DEFAULT_DET_TASK}_{DEFAULT_DET_DATASET}_{DET_MODEL_NAMES[DEFAULT_DET_ARCH]}'
FLOW_FILENAME = 'motion_flows.npy'

TRAIN_DIRNAME = 'training'
TEST_DIRNAME = 'testing'

VIDEOS_DIRNAME = 'videos'
FRAMES_DIRNAME = 'frames'
REGION_DIRNAME = 'region'
SCORES_DIRNAME = 'scores'

RESULTS_DIRNAME = 'results'
MODELS_DIRNAME = 'models'

SUMMARY_RESULTS_FILENAME = 'summary_results.txt'
SUMMARY_RUNNING_FILENAME = 'summary_running.csv'
COMMANDLINE_ARGS_FILENAME = 'commandline_args.txt'

ROC_FILENAME = 'roc_of_model.pdf'

ROI_DATAVIZ_NAME = 'st_roi_samples'

# load label to names mapping for visualization purposes
LABELS_TO_NAMES = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
                   8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter',
                   14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant',
                   22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie',
                   29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite',
                   35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket',
                   40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl',
                   47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog',
                   54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant', 60: 'bed',
                   61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard',
                   68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator',
                   74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier',
                   80: 'toothbrush'}
