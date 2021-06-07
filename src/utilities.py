"""
This file contains all the utilities used in other scripts.
"""
import os
import sys
import subprocess
import numpy as np
import cv2
from natsort import natsorted
import scipy.io as scio
from tqdm import tqdm
import datetime
import inspect
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from src import defines as ds


def return_data_dict(data_dict, data_size, num_frames=None, frame_index=None, bounding_box=None, label=None):
    """
    Returns a dict containing all given data.
    :param data_dict:
    :param data_size:
    :param num_frames:
    :param frame_index:
    :param bounding_box:
    :param label:
    :return:
    """
    dict_subset = dict()
    dict_subset['data'] = data_dict
    dict_subset['data_size'] = data_size
    if num_frames is not None:
        dict_subset['num_frames'] = num_frames
    if frame_index is not None:
        dict_subset['frame_index'] = frame_index
    if bounding_box is not None:
        dict_subset['bounding_box'] = bounding_box
    if label is not None:
        dict_subset['label'] = label
    return dict_subset


def return_score_dict(data, scores, bounding_box=None, label=None):
    """
    Returns a dict containing the essential video information and the corresponding scores.
    :param data:
    :param scores:
    :param bounding_box:
    :param label
    :return:
    """
    assert data['data_size'] == len(scores), 'Size of scores mismatch.'

    dict_scores = dict()
    dict_scores['data_size'] = data['data_size']
    dict_scores['num_frames'] = data['num_frames']
    dict_scores['frame_index'] = data['frame_index']
    dict_scores['scores'] = scores
    if bounding_box is not None:
        dict_scores['bounding_box'] = data['bounding_box']
    if label is not None:
        dict_scores['label'] = data['label']
    return dict_scores


def load_detections(src_dir, video_names, det_mdl_name):
    """
    Loads all the detections for all the given videos.
    :param src_dir:
    :param video_names:
    :param det_mdl_name:
    :return:
    """
    # Get all the detections
    return np.array([np.load(os.path.join(os.path.join(src_dir, v), ds.DET_FILENAMES[det_mdl_name]), allow_pickle=True)
                     for v in tqdm(video_names, file=sys.stdout, desc='Loading detections')])


def visualize_data(dict_subset, n_samples_per_viz=ds.DEFAULT_N_SAMPLES_PER_VIZ):
    """
    Produces visualization image with some randomly selected RoI samples.
    :param dict_subset
    :param n_samples_per_viz:
    :return: Viz image.
    """
    assert dict_subset['data_size'] >= n_samples_per_viz, f'The data size must be >= {n_samples_per_viz}.'

    # Show some randomly selected samples
    idx = np.array(list(range(dict_subset['data_size'])))
    np.random.shuffle(idx)
    idx = idx[:n_samples_per_viz]

    # Merge images for nice visualisation
    img_size = len(next(iter(dict_subset['data'].values()))[0])
    mg_h = np.ones((img_size, ds.HORIZONTAL_MARGIN_SIZE), np.float32)

    # stacked images horizontally
    stack_images = []
    for _, data in dict_subset['data'].items():
        stack_images.append(cv2.hconcat([cv2.hconcat([mg_h, d[:, :, 0], mg_h]) for d in data[idx]]))

    # Stack stacked images vertically
    mg_b = np.ones((ds.HORIZONTAL_MARGIN_SIZE, stack_images[0].shape[1]), np.float32)
    mg_v = np.ones((ds.VERTICAL_MARGIN_SIZE, stack_images[0].shape[1]), np.float32)

    stacked_images = cv2.vconcat([mg_b] + intersperse(stack_images, mg_v) + [mg_b])

    # Convert to uint8
    stacked_images *= ds.WHITE_COLOR
    return stacked_images.astype('uint8')


class GAData:
    """
    Structure of the spatio-temporal RoI sample.
    """
    def __init__(self,
                 dataset=ds.DEFAULT_DATASET,
                 rand_seed=ds.DEFAULT_RAND_SEED,
                 frame_diff_gap=ds.FRAME_DIFF_GAP,
                 min_det_score=ds.DEFAULT_MIN_DET_SCORE,
                 n_frames_per_video=ds.DEFAULT_N_FRAMES_PER_VIDEO,
                 normalize_images=ds.NORMALIZE_IMAGES,
                 frames_step=ds.FRAMES_STEP,
                 det_model_name=ds.DEFAULT_DET_MODEL_NAME,
                 estimations_path=ds.ESTIMATIONS_PATH):
        """
        Initializes the data extraction parameters.
        :param dataset:
        :param rand_seed:
        :param frame_diff_gap:
        :param min_det_score:
        :param n_frames_per_video:
        :param normalize_images:
        :param frames_step:
        :param det_model_name:
        :param estimations_path:
        """
        self.dataset = dataset
        self.rand_seed = rand_seed
        self.frame_diff_gap = frame_diff_gap
        self.min_det_score = min_det_score  # Unused
        self.n_frames_per_video = n_frames_per_video
        self.normalize_images = normalize_images
        self.frames_step = frames_step
        self.det_model_name = det_model_name  # Unused
        self.estimations_path = estimations_path

        self.det_model_name = None

        self.image_size = ds.FRAME_SIZE

        self.subset_dir_names = [ds.TRAIN_DIRNAME, ds.TEST_DIRNAME]
        self.subset_dirs = {s: os.path.join(self.estimations_path, self.dataset, s) for s in self.subset_dir_names}
        self.subset_videos = {s: natsorted([v for v in os.listdir(self.subset_dirs[s])
                                            if os.path.isdir(os.path.join(self.subset_dirs[s], v))])
                              for s in self.subset_dir_names}

        # Set the random seed
        np.random.seed(self.rand_seed)

    def get_random_train_subset(self):
        """
        Extracts the RoI data from randomly selected subset of frames for each video.
        :return:
        """
        # Get samples randomly distributed
        data_curr_a_int = []
        data_past_a_int = []
        data_past_a_gra = []

        for video in tqdm(self.subset_videos[ds.TRAIN_DIRNAME], file=sys.stdout,
                          total=len(self.subset_videos[ds.TRAIN_DIRNAME]),
                          desc='Extracting subset'):
            video_dir = os.path.join(self.subset_dirs[ds.TRAIN_DIRNAME], video)

            frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')

            n_frames = len(frames)

            idx = np.array(list(range(self.frame_diff_gap, n_frames)))
            np.random.shuffle(idx)
            idx = idx[:self.n_frames_per_video]

            for i in idx:
                curr_a_int = frames[i]
                past_a_int = frames[i - self.frame_diff_gap]

                # Resize RoIs
                curr_resized_a_int = cv2.resize(curr_a_int, (self.image_size, self.image_size))
                past_resized_a_int = cv2.resize(past_a_int, (self.image_size, self.image_size))

                # Compute gradient images
                past_resized_a_gra = compute_gradient_image(past_resized_a_int)

                if self.normalize_images:
                    # Normalize images
                    curr_resized_a_int = cv2.normalize(curr_resized_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_resized_a_int = cv2.normalize(past_resized_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_resized_a_gra = cv2.normalize(past_resized_a_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # Store results
                data_curr_a_int.append(curr_resized_a_int)
                data_past_a_int.append(past_resized_a_int)
                data_past_a_gra.append(past_resized_a_gra)

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_past_a_int = np.array(data_past_a_int)
        data_past_a_gra = np.array(data_past_a_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1))
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_past_a_gra) and data_size == len(data_past_a_int), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['past_appear_int'] = data_past_a_int
        dict_data['past_appear_gra'] = data_past_a_gra

        return return_data_dict(dict_data, data_size)

    def get_complete_train_set(self):
        """
        Extracts the RoI data from randomly selected subset of frames for each video.
        :return:
        """
        # Get samples randomly distributed
        data_curr_a_int = []
        data_past_a_int = []
        data_past_a_gra = []

        for video in tqdm(self.subset_videos[ds.TRAIN_DIRNAME], file=sys.stdout,
                          total=len(self.subset_videos[ds.TRAIN_DIRNAME]),
                          desc='Extracting subset'):
            video_dir = os.path.join(self.subset_dirs[ds.TRAIN_DIRNAME], video)

            frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')

            n_frames = len(frames)

            for i in range(self.frame_diff_gap, n_frames, self.frames_step):
                curr_a_int = frames[i]
                past_a_int = frames[i - self.frame_diff_gap]

                # Resize RoIs
                curr_resized_a_int = cv2.resize(curr_a_int, (self.image_size, self.image_size))
                past_resized_a_int = cv2.resize(past_a_int, (self.image_size, self.image_size))

                # Compute gradient images
                past_resized_a_gra = compute_gradient_image(past_resized_a_int)

                if self.normalize_images:
                    # Normalize images
                    curr_resized_a_int = cv2.normalize(curr_resized_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_resized_a_int = cv2.normalize(past_resized_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_resized_a_gra = cv2.normalize(past_resized_a_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # Store results
                data_curr_a_int.append(curr_resized_a_int)
                data_past_a_int.append(past_resized_a_int)
                data_past_a_gra.append(past_resized_a_gra)

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_past_a_int = np.array(data_past_a_int)
        data_past_a_gra = np.array(data_past_a_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1))
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_past_a_gra) and data_size == len(data_past_a_int), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['past_appear_int'] = data_past_a_int
        dict_data['past_appear_gra'] = data_past_a_gra

        return return_data_dict(dict_data, data_size)

    def get_video_data(self, subset_name, video_name):
        """
        Gets all the data for the given video.
        :param subset_name:
        :param video_name:
        :return:
        """
        assert subset_name in self.subset_dir_names, 'Invalid subset.'
        assert video_name in self.subset_videos[subset_name], 'Invalid video.'

        # Get samples randomly distributed
        data_curr_a_int = []
        data_past_a_int = []
        data_past_a_gra = []

        video_dir = os.path.join(self.subset_dirs[subset_name], video_name)

        frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')

        n_frames = len(frames)
        frames_idx = list(range(self.frame_diff_gap, n_frames))

        for i in frames_idx:
            curr_a_int = frames[i]
            past_a_int = frames[i - self.frame_diff_gap]

            # Resize RoIs
            curr_resized_a_int = cv2.resize(curr_a_int, (self.image_size, self.image_size))
            past_resized_a_int = cv2.resize(past_a_int, (self.image_size, self.image_size))

            # Compute gradient images
            past_resized_a_gra = compute_gradient_image(past_resized_a_int)

            if self.normalize_images:
                # Normalize images
                curr_resized_a_int = cv2.normalize(curr_resized_a_int, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                past_resized_a_int = cv2.normalize(past_resized_a_int, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                past_resized_a_gra = cv2.normalize(past_resized_a_gra, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            # Store results
            data_curr_a_int.append(curr_resized_a_int)
            data_past_a_int.append(past_resized_a_int)
            data_past_a_gra.append(past_resized_a_gra)

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_past_a_int = np.array(data_past_a_int)
        data_past_a_gra = np.array(data_past_a_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1))
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_past_a_gra) and data_size == len(data_past_a_int), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['past_appear_int'] = data_past_a_int
        dict_data['past_appear_gra'] = data_past_a_gra

        return return_data_dict(dict_data, data_size, frame_index=np.array(frames_idx), num_frames=n_frames)


class GARoIData:
    """
    Structure of the spatio-temporal RoI sample.
    """
    def __init__(self,
                 dataset=ds.DEFAULT_DATASET,
                 rand_seed=ds.DEFAULT_RAND_SEED,
                 frame_diff_gap=ds.FRAME_DIFF_GAP,
                 min_det_score=ds.DEFAULT_MIN_DET_SCORE,
                 n_frames_per_video=ds.DEFAULT_N_FRAMES_PER_VIDEO,
                 normalize_images=ds.NORMALIZE_IMAGES,
                 frames_step=ds.FRAMES_STEP,
                 det_model_name=ds.DEFAULT_DET_MODEL_NAME,
                 estimations_path=ds.ESTIMATIONS_PATH):
        """
        Initializes the data extraction parameters.
        :param dataset:
        :param rand_seed:
        :param frame_diff_gap:
        :param min_det_score:
        :param n_frames_per_video:
        :param normalize_images:
        :param frames_step:
        :param det_model_name:
        :param estimations_path:
        """
        self.dataset = dataset
        self.rand_seed = rand_seed
        self.frame_diff_gap = frame_diff_gap
        self.min_det_score = min_det_score
        self.n_frames_per_video = n_frames_per_video
        self.normalize_images = normalize_images
        self.frames_step = frames_step
        self.det_model_name = det_model_name
        self.estimations_path = estimations_path

        self.image_size = ds.ROI_SIZE

        self.subset_dir_names = [ds.TRAIN_DIRNAME, ds.TEST_DIRNAME]
        self.subset_dirs = {s: os.path.join(self.estimations_path, self.dataset, s) for s in self.subset_dir_names}
        self.subset_videos = {s: natsorted([v for v in os.listdir(self.subset_dirs[s])
                                            if os.path.isdir(os.path.join(self.subset_dirs[s], v))])
                              for s in self.subset_dir_names}
        self.subset_det = {s: None for s in self.subset_dir_names}

        # Set the random seed
        np.random.seed(self.rand_seed)

    def get_random_train_subset(self):
        """
        Extracts the RoI data from randomly selected subset of frames for each video.
        :return:
        """
        # Get detections if necessary
        if self.subset_det[ds.TRAIN_DIRNAME] is None:
            self.subset_det[ds.TRAIN_DIRNAME] = load_detections(self.subset_dirs[ds.TRAIN_DIRNAME],
                                                                self.subset_videos[ds.TRAIN_DIRNAME],
                                                                self.det_model_name)

        # Get samples randomly distributed
        data_curr_a_int = []
        data_past_a_int = []
        data_past_a_gra = []

        for video, det_data in tqdm(zip(self.subset_videos[ds.TRAIN_DIRNAME], self.subset_det[ds.TRAIN_DIRNAME]),
                                    file=sys.stdout, total=len(self.subset_videos[ds.TRAIN_DIRNAME]),
                                    desc='Extracting RoI subset'):
            video_dir = os.path.join(self.subset_dirs[ds.TRAIN_DIRNAME], video)

            frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')

            n_frames = len(frames)
            assert n_frames == len(det_data), 'Size of data mismatch.'

            idx = np.array(list(range(self.frame_diff_gap, n_frames)))
            np.random.shuffle(idx)
            idx = idx[:self.n_frames_per_video]

            for i in idx:
                curr_a_int = frames[i]
                past_a_int = frames[i - self.frame_diff_gap]

                # Detections will be a python dict: {category_id : [[x1, y1, x2, y2, score], ...], }
                curr_a_det = det_data[i]
                # Remove empty detections
                curr_a_det = {k: curr_a_det[k] for k in curr_a_det if curr_a_det[k].size}
                # transform ret into [[x1, y1, x2, y2, score], ...]
                curr_a_det = np.vstack(list(curr_a_det.values()))

                # Exclude detections that have lower score
                curr_a_det = curr_a_det[curr_a_det[:, 4] >= self.min_det_score, :]

                if not len(curr_a_det):
                    continue

                for box in curr_a_det:
                    (x1, y1, x2, y2) = box[:-1].astype(int)

                    # Bug fixed
                    if x1 < 0:
                        x1 = 0
                    if y1 < 0:
                        y1 = 0

                    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                        continue

                    # Resize RoIs
                    curr_roi_a_int = cv2.resize(curr_a_int[y1:y2, x1:x2], (self.image_size, self.image_size))
                    past_roi_a_int = cv2.resize(past_a_int[y1:y2, x1:x2], (self.image_size, self.image_size))

                    # Compute gradient images
                    past_roi_a_gra = compute_gradient_image(past_roi_a_int)

                    if self.normalize_images:
                        # Normalize images
                        curr_roi_a_int = cv2.normalize(curr_roi_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_a_int = cv2.normalize(past_roi_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_a_gra = cv2.normalize(past_roi_a_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # Store results
                    data_curr_a_int.append(curr_roi_a_int)
                    data_past_a_int.append(past_roi_a_int)
                    data_past_a_gra.append(past_roi_a_gra)

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_past_a_int = np.array(data_past_a_int)
        data_past_a_gra = np.array(data_past_a_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1))
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_past_a_gra) and data_size == len(data_past_a_int), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['past_appear_int'] = data_past_a_int
        dict_data['past_appear_gra'] = data_past_a_gra

        return return_data_dict(dict_data, data_size)

    def get_complete_train_set(self):
        """
        Extracts the RoI data from randomly selected subset of frames for each video.
        :return:
        """
        # Get detections if necessary
        if self.subset_det[ds.TRAIN_DIRNAME] is None:
            self.subset_det[ds.TRAIN_DIRNAME] = load_detections(self.subset_dirs[ds.TRAIN_DIRNAME],
                                                                self.subset_videos[ds.TRAIN_DIRNAME],
                                                                self.det_model_name)

        # Get samples randomly distributed
        data_curr_a_int = []
        data_past_a_int = []
        data_past_a_gra = []

        for video, det_data in tqdm(zip(self.subset_videos[ds.TRAIN_DIRNAME], self.subset_det[ds.TRAIN_DIRNAME]),
                                    file=sys.stdout, total=len(self.subset_videos[ds.TRAIN_DIRNAME]),
                                    desc='Extracting RoI subset'):
            video_dir = os.path.join(self.subset_dirs[ds.TRAIN_DIRNAME], video)

            frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')

            n_frames = len(frames)
            assert n_frames == len(det_data), 'Size of data mismatch.'

            for i in range(self.frame_diff_gap, n_frames, self.frames_step):
                curr_a_int = frames[i]
                past_a_int = frames[i - self.frame_diff_gap]

                # Detections will be a python dict: {category_id : [[x1, y1, x2, y2, score], ...], }
                curr_a_det = det_data[i]
                # Remove empty detections
                curr_a_det = {k: curr_a_det[k] for k in curr_a_det if curr_a_det[k].size}
                # transform ret into [[x1, y1, x2, y2, score], ...]
                curr_a_det = np.vstack(list(curr_a_det.values()))

                # Exclude detections that have lower score
                curr_a_det = curr_a_det[curr_a_det[:, 4] >= self.min_det_score, :]

                if not len(curr_a_det):
                    continue

                for box in curr_a_det:
                    (x1, y1, x2, y2) = box[:-1].astype(int)

                    # Bug fixed
                    if x1 < 0:
                        x1 = 0
                    if y1 < 0:
                        y1 = 0

                    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                        continue

                    # Resize RoIs
                    curr_roi_a_int = cv2.resize(curr_a_int[y1:y2, x1:x2], (self.image_size, self.image_size))
                    past_roi_a_int = cv2.resize(past_a_int[y1:y2, x1:x2], (self.image_size, self.image_size))

                    # Compute gradient images
                    past_roi_a_gra = compute_gradient_image(past_roi_a_int)

                    if self.normalize_images:
                        # Normalize images
                        curr_roi_a_int = cv2.normalize(curr_roi_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_a_int = cv2.normalize(past_roi_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_a_gra = cv2.normalize(past_roi_a_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # Store results
                    data_curr_a_int.append(curr_roi_a_int)
                    data_past_a_int.append(past_roi_a_int)
                    data_past_a_gra.append(past_roi_a_gra)

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_past_a_int = np.array(data_past_a_int)
        data_past_a_gra = np.array(data_past_a_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1))
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_past_a_gra) and data_size == len(data_past_a_int), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['past_appear_int'] = data_past_a_int
        dict_data['past_appear_gra'] = data_past_a_gra

        return return_data_dict(dict_data, data_size)

    def get_video_data(self, subset_name, video_name):
        """
        Gets all the data for the given video.
        :param subset_name:
        :param video_name:
        :return:
        """
        assert subset_name in self.subset_dir_names, 'Invalid subset.'
        assert video_name in self.subset_videos[subset_name], 'Invalid video.'

        # Get detections if necessary
        if self.subset_det[subset_name] is None:
            self.subset_det[subset_name] = load_detections(self.subset_dirs[subset_name],
                                                           self.subset_videos[subset_name],
                                                           self.det_model_name)
        det_data = self.subset_det[subset_name][self.subset_videos[subset_name].index(video_name)]

        # Get samples randomly distributed
        data_curr_a_int = []
        data_past_a_int = []
        data_past_a_gra = []
        frames_idx = []
        bounding_boxes = []

        video_dir = os.path.join(self.subset_dirs[subset_name], video_name)

        frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')

        n_frames = len(frames)
        assert n_frames == len(det_data), 'Size of data mismatch.'

        for i in range(self.frame_diff_gap, n_frames):
            curr_a_int = frames[i]
            past_a_int = frames[i - self.frame_diff_gap]

            # Detections will be a python dict: {category_id : [[x1, y1, x2, y2, score], ...], }
            curr_a_det = det_data[i]
            # Remove empty detections
            curr_a_det = {k: curr_a_det[k] for k in curr_a_det if curr_a_det[k].size}
            # transform ret into [[x1, y1, x2, y2, score], ...]
            curr_a_det = np.vstack(list(curr_a_det.values()))

            # Exclude detections that have lower score
            curr_a_det = curr_a_det[curr_a_det[:, 4] >= self.min_det_score, :]

            if not len(curr_a_det):
                continue

            for box in curr_a_det:
                (x1, y1, x2, y2) = box[:-1].astype(int)

                # Bug fixed
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0

                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue

                # Resize RoIs
                curr_roi_a_int = cv2.resize(curr_a_int[y1:y2, x1:x2], (self.image_size, self.image_size))
                past_roi_a_int = cv2.resize(past_a_int[y1:y2, x1:x2], (self.image_size, self.image_size))

                # Compute gradient images
                past_roi_a_gra = compute_gradient_image(past_roi_a_int)

                if self.normalize_images:
                    # Normalize images
                    curr_roi_a_int = cv2.normalize(curr_roi_a_int, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_roi_a_int = cv2.normalize(past_roi_a_int, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_roi_a_gra = cv2.normalize(past_roi_a_gra, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # Store results
                data_curr_a_int.append(curr_roi_a_int)
                data_past_a_int.append(past_roi_a_int)
                data_past_a_gra.append(past_roi_a_gra)
                frames_idx.append(i)
                bounding_boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_past_a_int = np.array(data_past_a_int)
        data_past_a_gra = np.array(data_past_a_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1))
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_past_a_gra) and data_size == len(data_past_a_int), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['past_appear_int'] = data_past_a_int
        dict_data['past_appear_gra'] = data_past_a_gra

        return return_data_dict(dict_data, data_size,
                                frame_index=np.array(frames_idx),
                                bounding_box=np.array(bounding_boxes), num_frames=n_frames)


class GAMRoIData:
    """
    Structure of the spatio-temporal RoI sample.
    """
    def __init__(self,
                 dataset=ds.DEFAULT_DATASET,
                 rand_seed=ds.DEFAULT_RAND_SEED,
                 frame_diff_gap=ds.FRAME_DIFF_GAP,
                 min_det_score=ds.DEFAULT_MIN_DET_SCORE,
                 n_frames_per_video=ds.DEFAULT_N_FRAMES_PER_VIDEO,
                 normalize_images=ds.NORMALIZE_IMAGES,
                 frames_step=ds.FRAMES_STEP,
                 det_model_name=ds.DEFAULT_DET_MODEL_NAME,
                 estimations_path=ds.ESTIMATIONS_PATH):
        """
        Initializes the data extraction parameters.
        :param dataset:
        :param rand_seed:
        :param frame_diff_gap:
        :param min_det_score:
        :param n_frames_per_video:
        :param normalize_images:
        :param frames_step:
        :param det_model_name:
        :param estimations_path:
        """
        self.dataset = dataset
        self.rand_seed = rand_seed
        self.frame_diff_gap = frame_diff_gap
        self.min_det_score = min_det_score
        self.n_frames_per_video = n_frames_per_video
        self.normalize_images = normalize_images
        self.frames_step = frames_step
        self.det_model_name = det_model_name
        self.estimations_path = estimations_path

        self.subset_dir_names = [ds.TRAIN_DIRNAME, ds.TEST_DIRNAME]
        self.subset_dirs = {s: os.path.join(self.estimations_path, self.dataset, s) for s in self.subset_dir_names}
        self.subset_videos = {s: natsorted([v for v in os.listdir(self.subset_dirs[s])
                                            if os.path.isdir(os.path.join(self.subset_dirs[s], v))])
                              for s in self.subset_dir_names}
        self.subset_det = {s: None for s in self.subset_dir_names}

        # Set the random seed
        np.random.seed(self.rand_seed)

    def get_random_train_subset(self):
        """
        Extracts the RoI data from randomly selected subset of frames for each video.
        :return:
        """
        # Get detections if necessary
        if self.subset_det[ds.TRAIN_DIRNAME] is None:
            self.subset_det[ds.TRAIN_DIRNAME] = load_detections(self.subset_dirs[ds.TRAIN_DIRNAME],
                                                                self.subset_videos[ds.TRAIN_DIRNAME],
                                                                self.det_model_name)

        # Get samples randomly distributed
        data_curr_a_int = []
        data_curr_a_gra = []
        data_curr_m_int = []
        data_curr_m_gra = []
        data_past_a_int = []
        data_past_a_gra = []
        data_past_m_int = []
        data_past_m_gra = []

        for video, det_data in tqdm(zip(self.subset_videos[ds.TRAIN_DIRNAME], self.subset_det[ds.TRAIN_DIRNAME]),
                                    file=sys.stdout, total=len(self.subset_videos[ds.TRAIN_DIRNAME]),
                                    desc='Extracting RoI subset'):
            video_dir = os.path.join(self.subset_dirs[ds.TRAIN_DIRNAME], video)

            frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')
            flows = np.load(os.path.join(video_dir, ds.FLOW_FILENAME), mmap_mode='r')

            n_frames = len(frames)
            assert n_frames == len(flows) and n_frames == len(det_data), 'Size of data mismatch.'

            idx = np.array(list(range(self.frame_diff_gap, n_frames)))
            np.random.shuffle(idx)
            idx = idx[:self.n_frames_per_video]

            for i in idx:
                curr_a_int = frames[i]
                past_a_int = frames[i - self.frame_diff_gap]
                curr_m_int = flows[i]
                past_m_int = flows[i - self.frame_diff_gap]

                # Detections will be a python dict: {category_id : [[x1, y1, x2, y2, score], ...], }
                curr_a_det = det_data[i]
                # Remove empty detections
                curr_a_det = {k: curr_a_det[k] for k in curr_a_det if curr_a_det[k].size}
                # transform ret into [[x1, y1, x2, y2, score], ...]
                curr_a_det = np.vstack(list(curr_a_det.values()))

                # Exclude detections that have lower score
                curr_a_det = curr_a_det[curr_a_det[:, 4] >= self.min_det_score, :]

                if not len(curr_a_det):
                    continue

                for box in curr_a_det:
                    (x1, y1, x2, y2) = box[:-1].astype(int)

                    # Bug fixed
                    if x1 < 0:
                        x1 = 0
                    if y1 < 0:
                        y1 = 0

                    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                        continue

                    # Resize RoIs
                    curr_roi_a_int = cv2.resize(curr_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                    past_roi_a_int = cv2.resize(past_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                    curr_roi_m_int = cv2.resize(curr_m_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                    past_roi_m_int = cv2.resize(past_m_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))

                    # Compute gradient images
                    curr_roi_a_gra = compute_gradient_image(curr_roi_a_int)
                    past_roi_a_gra = compute_gradient_image(past_roi_a_int)
                    curr_roi_m_gra = compute_gradient_image(curr_roi_m_int)
                    past_roi_m_gra = compute_gradient_image(past_roi_m_int)

                    if self.normalize_images:
                        # Normalize images
                        curr_roi_a_int = cv2.normalize(curr_roi_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_a_int = cv2.normalize(past_roi_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        curr_roi_a_gra = cv2.normalize(curr_roi_a_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_a_gra = cv2.normalize(past_roi_a_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        curr_roi_m_int = cv2.normalize(curr_roi_m_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_m_int = cv2.normalize(past_roi_m_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        curr_roi_m_gra = cv2.normalize(curr_roi_m_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_m_gra = cv2.normalize(past_roi_m_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # Store results
                    data_curr_a_int.append(curr_roi_a_int)
                    data_past_a_int.append(past_roi_a_int)
                    data_curr_m_int.append(curr_roi_m_int)
                    data_past_m_int.append(past_roi_m_int)
                    data_curr_a_gra.append(curr_roi_a_gra)
                    data_past_a_gra.append(past_roi_a_gra)
                    data_curr_m_gra.append(curr_roi_m_gra)
                    data_past_m_gra.append(past_roi_m_gra)

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_curr_a_gra = np.array(data_curr_a_gra)
        data_curr_m_int = np.array(data_curr_m_int)
        data_curr_m_gra = np.array(data_curr_m_gra)
        data_past_a_int = np.array(data_past_a_int)
        data_past_a_gra = np.array(data_past_a_gra)
        data_past_m_int = np.array(data_past_m_int)
        data_past_m_gra = np.array(data_past_m_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_curr_a_gra = data_curr_a_gra.reshape((*data_curr_a_gra.shape, 1))
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1))
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1))
            data_curr_m_int = data_curr_m_int.reshape((*data_curr_m_int.shape, 1))
            data_curr_m_gra = data_curr_m_gra.reshape((*data_curr_m_gra.shape, 1))
            data_past_m_int = data_past_m_int.reshape((*data_past_m_int.shape, 1))
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_curr_a_gra = data_curr_a_gra.reshape((*data_curr_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_curr_m_int = data_curr_m_int.reshape((*data_curr_m_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_curr_m_gra = data_curr_m_gra.reshape((*data_curr_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_m_int = data_past_m_int.reshape((*data_past_m_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_curr_a_gra) and \
               data_size == len(data_past_a_int) and \
               data_size == len(data_past_a_gra) and \
               data_size == len(data_curr_m_int) and \
               data_size == len(data_curr_m_gra) and \
               data_size == len(data_past_m_int) and \
               data_size == len(data_past_m_gra), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['curr_appear_gra'] = data_curr_a_gra
        dict_data['past_appear_int'] = data_past_a_int
        dict_data['past_appear_gra'] = data_past_a_gra
        dict_data['curr_motion_int'] = data_curr_m_int
        dict_data['curr_motion_gra'] = data_curr_m_gra
        dict_data['past_motion_int'] = data_past_m_int
        dict_data['past_motion_gra'] = data_past_m_gra

        return return_data_dict(dict_data, data_size)

    def get_complete_train_set(self):
        """
        Extracts the RoI data from randomly selected subset of frames for each video.
        :return:
        """
        # Get detections if necessary
        if self.subset_det[ds.TRAIN_DIRNAME] is None:
            self.subset_det[ds.TRAIN_DIRNAME] = load_detections(self.subset_dirs[ds.TRAIN_DIRNAME],
                                                                self.subset_videos[ds.TRAIN_DIRNAME],
                                                                self.det_model_name)

        # Get samples randomly distributed
        data_curr_a_int = []
        data_curr_a_gra = []
        data_curr_m_int = []
        data_curr_m_gra = []
        data_past_a_int = []
        data_past_a_gra = []
        data_past_m_int = []
        data_past_m_gra = []

        for video, det_data in tqdm(zip(self.subset_videos[ds.TRAIN_DIRNAME], self.subset_det[ds.TRAIN_DIRNAME]),
                                    file=sys.stdout, total=len(self.subset_videos[ds.TRAIN_DIRNAME]),
                                    desc='Extracting RoI subset'):
            video_dir = os.path.join(self.subset_dirs[ds.TRAIN_DIRNAME], video)

            frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')
            flows = np.load(os.path.join(video_dir, ds.FLOW_FILENAME), mmap_mode='r')

            n_frames = len(frames)
            assert n_frames == len(flows) and n_frames == len(det_data), 'Size of data mismatch.'

            for i in range(self.frame_diff_gap, n_frames, self.frames_step):
                curr_a_int = frames[i]
                past_a_int = frames[i - self.frame_diff_gap]
                curr_m_int = flows[i]
                past_m_int = flows[i - self.frame_diff_gap]

                # Detections will be a python dict: {category_id : [[x1, y1, x2, y2, score], ...], }
                curr_a_det = det_data[i]
                # Remove empty detections
                curr_a_det = {k: curr_a_det[k] for k in curr_a_det if curr_a_det[k].size}
                # transform ret into [[x1, y1, x2, y2, score], ...]
                curr_a_det = np.vstack(list(curr_a_det.values()))

                # Exclude detections that have lower score
                curr_a_det = curr_a_det[curr_a_det[:, 4] >= self.min_det_score, :]

                if not len(curr_a_det):
                    continue

                for box in curr_a_det:
                    (x1, y1, x2, y2) = box[:-1].astype(int)

                    # Bug fixed
                    if x1 < 0:
                        x1 = 0
                    if y1 < 0:
                        y1 = 0

                    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                        continue

                    # Resize RoIs
                    curr_roi_a_int = cv2.resize(curr_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                    past_roi_a_int = cv2.resize(past_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                    curr_roi_m_int = cv2.resize(curr_m_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                    past_roi_m_int = cv2.resize(past_m_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))

                    # Compute gradient images
                    curr_roi_a_gra = compute_gradient_image(curr_roi_a_int)
                    past_roi_a_gra = compute_gradient_image(past_roi_a_int)
                    curr_roi_m_gra = compute_gradient_image(curr_roi_m_int)
                    past_roi_m_gra = compute_gradient_image(past_roi_m_int)

                    if self.normalize_images:
                        # Normalize images
                        curr_roi_a_int = cv2.normalize(curr_roi_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_a_int = cv2.normalize(past_roi_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        curr_roi_a_gra = cv2.normalize(curr_roi_a_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_a_gra = cv2.normalize(past_roi_a_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        curr_roi_m_int = cv2.normalize(curr_roi_m_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_m_int = cv2.normalize(past_roi_m_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        curr_roi_m_gra = cv2.normalize(curr_roi_m_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_m_gra = cv2.normalize(past_roi_m_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # Store results
                    data_curr_a_int.append(curr_roi_a_int)
                    data_past_a_int.append(past_roi_a_int)
                    data_curr_m_int.append(curr_roi_m_int)
                    data_past_m_int.append(past_roi_m_int)
                    data_curr_a_gra.append(curr_roi_a_gra)
                    data_past_a_gra.append(past_roi_a_gra)
                    data_curr_m_gra.append(curr_roi_m_gra)
                    data_past_m_gra.append(past_roi_m_gra)

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_curr_a_gra = np.array(data_curr_a_gra)
        data_curr_m_int = np.array(data_curr_m_int)
        data_curr_m_gra = np.array(data_curr_m_gra)
        data_past_a_int = np.array(data_past_a_int)
        data_past_a_gra = np.array(data_past_a_gra)
        data_past_m_int = np.array(data_past_m_int)
        data_past_m_gra = np.array(data_past_m_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_curr_a_gra = data_curr_a_gra.reshape((*data_curr_a_gra.shape, 1))
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1))
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1))
            data_curr_m_int = data_curr_m_int.reshape((*data_curr_m_int.shape, 1))
            data_curr_m_gra = data_curr_m_gra.reshape((*data_curr_m_gra.shape, 1))
            data_past_m_int = data_past_m_int.reshape((*data_past_m_int.shape, 1))
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_curr_a_gra = data_curr_a_gra.reshape((*data_curr_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_curr_m_int = data_curr_m_int.reshape((*data_curr_m_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_curr_m_gra = data_curr_m_gra.reshape((*data_curr_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_m_int = data_past_m_int.reshape((*data_past_m_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_curr_a_gra) and \
               data_size == len(data_past_a_int) and \
               data_size == len(data_past_a_gra) and \
               data_size == len(data_curr_m_int) and \
               data_size == len(data_curr_m_gra) and \
               data_size == len(data_past_m_int) and \
               data_size == len(data_past_m_gra), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['curr_appear_gra'] = data_curr_a_gra
        dict_data['past_appear_int'] = data_past_a_int
        dict_data['past_appear_gra'] = data_past_a_gra
        dict_data['curr_motion_int'] = data_curr_m_int
        dict_data['curr_motion_gra'] = data_curr_m_gra
        dict_data['past_motion_int'] = data_past_m_int
        dict_data['past_motion_gra'] = data_past_m_gra

        return return_data_dict(dict_data, data_size)

    def get_video_data(self, subset_name, video_name):
        """
        Gets all the data for the given video.
        :param subset_name:
        :param video_name:
        :return:
        """
        assert subset_name in self.subset_dir_names, 'Invalid subset.'
        assert video_name in self.subset_videos[subset_name], 'Invalid video.'

        # Get detections if necessary
        if self.subset_det[subset_name] is None:
            self.subset_det[subset_name] = load_detections(self.subset_dirs[subset_name],
                                                           self.subset_videos[subset_name],
                                                           self.det_model_name)
        det_data = self.subset_det[subset_name][self.subset_videos[subset_name].index(video_name)]

        # Get samples randomly distributed
        data_curr_a_int = []
        data_curr_a_gra = []
        data_curr_m_int = []
        data_curr_m_gra = []
        data_past_a_int = []
        data_past_a_gra = []
        data_past_m_int = []
        data_past_m_gra = []
        frames_idx = []
        bounding_boxes = []

        video_dir = os.path.join(self.subset_dirs[subset_name], video_name)

        frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')
        flows = np.load(os.path.join(video_dir, ds.FLOW_FILENAME), mmap_mode='r')

        n_frames = len(frames)
        assert n_frames == len(det_data), 'Size of data mismatch.'

        for i in range(self.frame_diff_gap, n_frames):
            curr_a_int = frames[i]
            past_a_int = frames[i - self.frame_diff_gap]
            curr_m_int = flows[i]
            past_m_int = flows[i - self.frame_diff_gap]

            # Detections will be a python dict: {category_id : [[x1, y1, x2, y2, score], ...], }
            curr_a_det = det_data[i]
            # Remove empty detections
            curr_a_det = {k: curr_a_det[k] for k in curr_a_det if curr_a_det[k].size}
            # transform ret into [[x1, y1, x2, y2, score], ...]
            curr_a_det = np.vstack(list(curr_a_det.values()))

            # Exclude detections that have lower score
            curr_a_det = curr_a_det[curr_a_det[:, 4] >= self.min_det_score, :]

            if not len(curr_a_det):
                continue

            for box in curr_a_det:
                (x1, y1, x2, y2) = box[:-1].astype(int)

                # Bug fixed
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0

                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue

                # Resize RoIs
                curr_roi_a_int = cv2.resize(curr_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                past_roi_a_int = cv2.resize(past_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                curr_roi_m_int = cv2.resize(curr_m_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                past_roi_m_int = cv2.resize(past_m_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))

                # Compute gradient images
                curr_roi_a_gra = compute_gradient_image(curr_roi_a_int)
                past_roi_a_gra = compute_gradient_image(past_roi_a_int)
                curr_roi_m_gra = compute_gradient_image(curr_roi_m_int)
                past_roi_m_gra = compute_gradient_image(past_roi_m_int)

                if self.normalize_images:
                    # Normalize images
                    curr_roi_a_int = cv2.normalize(curr_roi_a_int, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_roi_a_int = cv2.normalize(past_roi_a_int, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    curr_roi_a_gra = cv2.normalize(curr_roi_a_gra, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_roi_a_gra = cv2.normalize(past_roi_a_gra, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    curr_roi_m_int = cv2.normalize(curr_roi_m_int, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_roi_m_int = cv2.normalize(past_roi_m_int, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    curr_roi_m_gra = cv2.normalize(curr_roi_m_gra, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_roi_m_gra = cv2.normalize(past_roi_m_gra, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # Store results
                data_curr_a_int.append(curr_roi_a_int)
                data_past_a_int.append(past_roi_a_int)
                data_curr_m_int.append(curr_roi_m_int)
                data_past_m_int.append(past_roi_m_int)
                data_curr_a_gra.append(curr_roi_a_gra)
                data_past_a_gra.append(past_roi_a_gra)
                data_curr_m_gra.append(curr_roi_m_gra)
                data_past_m_gra.append(past_roi_m_gra)
                frames_idx.append(i)
                bounding_boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_curr_a_gra = np.array(data_curr_a_gra)
        data_curr_m_int = np.array(data_curr_m_int)
        data_curr_m_gra = np.array(data_curr_m_gra)
        data_past_a_int = np.array(data_past_a_int)
        data_past_a_gra = np.array(data_past_a_gra)
        data_past_m_int = np.array(data_past_m_int)
        data_past_m_gra = np.array(data_past_m_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_curr_a_gra = data_curr_a_gra.reshape((*data_curr_a_gra.shape, 1))
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1))
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1))
            data_curr_m_int = data_curr_m_int.reshape((*data_curr_m_int.shape, 1))
            data_curr_m_gra = data_curr_m_gra.reshape((*data_curr_m_gra.shape, 1))
            data_past_m_int = data_past_m_int.reshape((*data_past_m_int.shape, 1))
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_curr_a_gra = data_curr_a_gra.reshape((*data_curr_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_int = data_past_a_int.reshape((*data_past_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_a_gra = data_past_a_gra.reshape((*data_past_a_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_curr_m_int = data_curr_m_int.reshape((*data_curr_m_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_curr_m_gra = data_curr_m_gra.reshape((*data_curr_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_m_int = data_past_m_int.reshape((*data_past_m_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_curr_a_gra) and \
               data_size == len(data_past_a_int) and \
               data_size == len(data_past_a_gra) and \
               data_size == len(data_curr_m_int) and \
               data_size == len(data_curr_m_gra) and \
               data_size == len(data_past_m_int) and \
               data_size == len(data_past_m_gra), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['curr_appear_gra'] = data_curr_a_gra
        dict_data['past_appear_int'] = data_past_a_int
        dict_data['past_appear_gra'] = data_past_a_gra
        dict_data['curr_motion_int'] = data_curr_m_int
        dict_data['curr_motion_gra'] = data_curr_m_gra
        dict_data['past_motion_int'] = data_past_m_int
        dict_data['past_motion_gra'] = data_past_m_gra

        return return_data_dict(dict_data, data_size,
                                frame_index=np.array(frames_idx),
                                bounding_box=np.array(bounding_boxes), num_frames=n_frames)


class GATRoIData:
    """
    Structure of the spatio-temporal RoI sample.
    """
    def __init__(self,
                 dataset=ds.DEFAULT_DATASET,
                 rand_seed=ds.DEFAULT_RAND_SEED,
                 frame_diff_gap=ds.FRAME_DIFF_GAP,
                 min_det_score=ds.DEFAULT_MIN_DET_SCORE,
                 n_frames_per_video=ds.DEFAULT_N_FRAMES_PER_VIDEO,
                 normalize_images=ds.NORMALIZE_IMAGES,
                 frames_step=ds.FRAMES_STEP,
                 det_model_name=ds.DEFAULT_DET_MODEL_NAME,
                 estimations_path=ds.ESTIMATIONS_PATH):
        """
        Initializes the data extraction parameters.
        :param dataset:
        :param rand_seed:
        :param frame_diff_gap:
        :param min_det_score:
        :param n_frames_per_video:
        :param normalize_images:
        :param frames_step:
        :param det_model_name:
        :param estimations_path:
        """
        self.dataset = dataset
        self.rand_seed = rand_seed
        self.frame_diff_gap = frame_diff_gap
        self.min_det_score = min_det_score
        self.n_frames_per_video = n_frames_per_video
        self.normalize_images = normalize_images
        self.frames_step = frames_step
        self.det_model_name = det_model_name
        self.estimations_path = estimations_path

        self.subset_dir_names = [ds.TRAIN_DIRNAME, ds.TEST_DIRNAME]
        self.subset_dirs = {s: os.path.join(self.estimations_path, self.dataset, s) for s in self.subset_dir_names}
        self.subset_videos = {s: natsorted([v for v in os.listdir(self.subset_dirs[s])
                                            if os.path.isdir(os.path.join(self.subset_dirs[s], v))])
                              for s in self.subset_dir_names}
        self.subset_det = {s: None for s in self.subset_dir_names}

        # Set the random seed
        np.random.seed(self.rand_seed)

    def get_random_train_subset(self):
        """
        Extracts the RoI data from randomly selected subset of frames for each video.
        :return:
        """
        # Get detections if necessary
        if self.subset_det[ds.TRAIN_DIRNAME] is None:
            self.subset_det[ds.TRAIN_DIRNAME] = load_detections(self.subset_dirs[ds.TRAIN_DIRNAME],
                                                                self.subset_videos[ds.TRAIN_DIRNAME],
                                                                self.det_model_name)

        # Get samples randomly distributed
        data_curr_a_int = []
        data_past_m_gra = []
        data_next_m_gra = []

        for video, det_data in tqdm(zip(self.subset_videos[ds.TRAIN_DIRNAME], self.subset_det[ds.TRAIN_DIRNAME]),
                                    file=sys.stdout, total=len(self.subset_videos[ds.TRAIN_DIRNAME]),
                                    desc='Extracting RoI subset'):
            video_dir = os.path.join(self.subset_dirs[ds.TRAIN_DIRNAME], video)

            frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')

            n_frames = len(frames)
            assert n_frames == len(det_data), 'Size of data mismatch.'

            idx = np.array(list(range(self.frame_diff_gap, n_frames - self.frame_diff_gap)))
            np.random.shuffle(idx)
            idx = idx[:self.n_frames_per_video]

            for i in idx:
                curr_a_int = frames[i]
                past_a_int = frames[i - self.frame_diff_gap]
                next_a_int = frames[i + self.frame_diff_gap]

                # Detections will be a python dict: {category_id : [[x1, y1, x2, y2, score], ...], }
                curr_a_det = det_data[i]
                # Remove empty detections
                curr_a_det = {k: curr_a_det[k] for k in curr_a_det if curr_a_det[k].size}
                # transform ret into [[x1, y1, x2, y2, score], ...]
                curr_a_det = np.vstack(list(curr_a_det.values()))

                # Exclude detections that have lower score
                curr_a_det = curr_a_det[curr_a_det[:, 4] >= self.min_det_score, :]

                if not len(curr_a_det):
                    continue

                for box in curr_a_det:
                    (x1, y1, x2, y2) = box[:-1].astype(int)

                    # Bug fixed
                    if x1 < 0:
                        x1 = 0
                    if y1 < 0:
                        y1 = 0

                    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                        continue

                    # Resize RoIs
                    curr_roi_a_int = cv2.resize(curr_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                    past_roi_a_int = cv2.resize(past_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                    next_roi_a_int = cv2.resize(next_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))

                    # Compute gradient images
                    curr_roi_a_gra = compute_gradient_image(curr_roi_a_int)
                    past_roi_a_gra = compute_gradient_image(past_roi_a_int)
                    next_roi_a_gra = compute_gradient_image(next_roi_a_int)

                    # Compute motion gradients
                    past_roi_m_gra = cv2.bitwise_xor(curr_roi_a_gra, past_roi_a_gra)
                    next_roi_m_gra = cv2.bitwise_xor(curr_roi_a_gra, next_roi_a_gra)

                    if self.normalize_images:
                        # Normalize images
                        curr_roi_a_int = cv2.normalize(curr_roi_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_m_gra = cv2.normalize(past_roi_m_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        next_roi_m_gra = cv2.normalize(next_roi_m_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # Store results
                    data_curr_a_int.append(curr_roi_a_int)
                    data_past_m_gra.append(past_roi_m_gra)
                    data_next_m_gra.append(next_roi_m_gra)

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_past_m_gra = np.array(data_past_m_gra)
        data_next_m_gra = np.array(data_next_m_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1))
            data_next_m_gra = data_next_m_gra.reshape((*data_next_m_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_next_m_gra = data_next_m_gra.reshape((*data_next_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_past_m_gra) and \
               data_size == len(data_next_m_gra), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['past_motion_gra'] = data_past_m_gra
        dict_data['next_motion_gra'] = data_next_m_gra

        return return_data_dict(dict_data, data_size)

    def get_complete_train_set(self):
        """
        Extracts the RoI data from randomly selected subset of frames for each video.
        :return:
        """
        # Get detections if necessary
        if self.subset_det[ds.TRAIN_DIRNAME] is None:
            self.subset_det[ds.TRAIN_DIRNAME] = load_detections(self.subset_dirs[ds.TRAIN_DIRNAME],
                                                                self.subset_videos[ds.TRAIN_DIRNAME],
                                                                self.det_model_name)

        # Get samples randomly distributed
        data_curr_a_int = []
        data_past_m_gra = []
        data_next_m_gra = []

        for video, det_data in tqdm(zip(self.subset_videos[ds.TRAIN_DIRNAME], self.subset_det[ds.TRAIN_DIRNAME]),
                                    file=sys.stdout, total=len(self.subset_videos[ds.TRAIN_DIRNAME]),
                                    desc='Extracting RoI subset'):
            video_dir = os.path.join(self.subset_dirs[ds.TRAIN_DIRNAME], video)

            frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')

            n_frames = len(frames)
            assert n_frames == len(det_data), 'Size of data mismatch.'

            for i in range(self.frame_diff_gap, n_frames - self.frame_diff_gap, self.frames_step):
                curr_a_int = frames[i]
                past_a_int = frames[i - self.frame_diff_gap]
                next_a_int = frames[i + self.frame_diff_gap]

                # Detections will be a python dict: {category_id : [[x1, y1, x2, y2, score], ...], }
                curr_a_det = det_data[i]
                # Remove empty detections
                curr_a_det = {k: curr_a_det[k] for k in curr_a_det if curr_a_det[k].size}
                # transform ret into [[x1, y1, x2, y2, score], ...]
                curr_a_det = np.vstack(list(curr_a_det.values()))

                # Exclude detections that have lower score
                curr_a_det = curr_a_det[curr_a_det[:, 4] >= self.min_det_score, :]

                if not len(curr_a_det):
                    continue

                for box in curr_a_det:
                    (x1, y1, x2, y2) = box[:-1].astype(int)

                    # Bug fixed
                    if x1 < 0:
                        x1 = 0
                    if y1 < 0:
                        y1 = 0

                    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                        continue

                    # Resize RoIs
                    curr_roi_a_int = cv2.resize(curr_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                    past_roi_a_int = cv2.resize(past_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                    next_roi_a_int = cv2.resize(next_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))

                    # Compute gradient images
                    curr_roi_a_gra = compute_gradient_image(curr_roi_a_int)
                    past_roi_a_gra = compute_gradient_image(past_roi_a_int)
                    next_roi_a_gra = compute_gradient_image(next_roi_a_int)

                    # Compute motion gradients
                    past_roi_m_gra = cv2.bitwise_xor(curr_roi_a_gra, past_roi_a_gra)
                    next_roi_m_gra = cv2.bitwise_xor(curr_roi_a_gra, next_roi_a_gra)

                    if self.normalize_images:
                        # Normalize images
                        curr_roi_a_int = cv2.normalize(curr_roi_a_int, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_roi_m_gra = cv2.normalize(past_roi_m_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        next_roi_m_gra = cv2.normalize(next_roi_m_gra, None,
                                                       alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # Store results
                    data_curr_a_int.append(curr_roi_a_int)
                    data_past_m_gra.append(past_roi_m_gra)
                    data_next_m_gra.append(next_roi_m_gra)

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_past_m_gra = np.array(data_past_m_gra)
        data_next_m_gra = np.array(data_next_m_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1))
            data_next_m_gra = data_next_m_gra.reshape((*data_next_m_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_next_m_gra = data_next_m_gra.reshape((*data_next_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_past_m_gra) and \
               data_size == len(data_next_m_gra), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['past_motion_gra'] = data_past_m_gra
        dict_data['next_motion_gra'] = data_next_m_gra

        return return_data_dict(dict_data, data_size)

    def get_video_data(self, subset_name, video_name):
        """
        Gets all the data for the given video.
        :param subset_name:
        :param video_name:
        :return:
        """
        assert subset_name in self.subset_dir_names, 'Invalid subset.'
        assert video_name in self.subset_videos[subset_name], 'Invalid video.'

        # Get detections if necessary
        if self.subset_det[subset_name] is None:
            self.subset_det[subset_name] = load_detections(self.subset_dirs[subset_name],
                                                           self.subset_videos[subset_name],
                                                           self.det_model_name)
        det_data = self.subset_det[subset_name][self.subset_videos[subset_name].index(video_name)]

        # Get samples randomly distributed
        data_curr_a_int = []
        data_past_m_gra = []
        data_next_m_gra = []
        frames_idx = []
        bounding_boxes = []

        video_dir = os.path.join(self.subset_dirs[subset_name], video_name)

        frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')

        n_frames = len(frames)
        assert n_frames == len(det_data), 'Size of data mismatch.'

        for i in range(self.frame_diff_gap, n_frames - self.frame_diff_gap):
            curr_a_int = frames[i]
            past_a_int = frames[i - self.frame_diff_gap]
            next_a_int = frames[i + self.frame_diff_gap]

            # Detections will be a python dict: {category_id : [[x1, y1, x2, y2, score], ...], }
            curr_a_det = det_data[i]
            # Remove empty detections
            curr_a_det = {k: curr_a_det[k] for k in curr_a_det if curr_a_det[k].size}
            # transform ret into [[x1, y1, x2, y2, score], ...]
            curr_a_det = np.vstack(list(curr_a_det.values()))

            # Exclude detections that have lower score
            curr_a_det = curr_a_det[curr_a_det[:, 4] >= self.min_det_score, :]

            if not len(curr_a_det):
                continue

            for box in curr_a_det:
                (x1, y1, x2, y2) = box[:-1].astype(int)

                # Bug fixed
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0

                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue

                # Resize RoIs
                curr_roi_a_int = cv2.resize(curr_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                past_roi_a_int = cv2.resize(past_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))
                next_roi_a_int = cv2.resize(next_a_int[y1:y2, x1:x2], (ds.ROI_SIZE, ds.ROI_SIZE))

                # Compute gradient images
                curr_roi_a_gra = compute_gradient_image(curr_roi_a_int)
                past_roi_a_gra = compute_gradient_image(past_roi_a_int)
                next_roi_a_gra = compute_gradient_image(next_roi_a_int)

                # Compute motion gradients
                past_roi_m_gra = cv2.bitwise_xor(curr_roi_a_gra, past_roi_a_gra)
                next_roi_m_gra = cv2.bitwise_xor(curr_roi_a_gra, next_roi_a_gra)

                if self.normalize_images:
                    # Normalize images
                    curr_roi_a_int = cv2.normalize(curr_roi_a_int, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_roi_m_gra = cv2.normalize(past_roi_m_gra, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    next_roi_m_gra = cv2.normalize(next_roi_m_gra, None,
                                                   alpha=ds.MIN_NORM_VAL, beta=ds.MAX_NORM_VAL,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # Store results
                data_curr_a_int.append(curr_roi_a_int)
                data_past_m_gra.append(past_roi_m_gra)
                data_next_m_gra.append(next_roi_m_gra)
                frames_idx.append(i)
                bounding_boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

        # Convert to arrays
        data_curr_a_int = np.array(data_curr_a_int)
        data_past_m_gra = np.array(data_past_m_gra)
        data_next_m_gra = np.array(data_next_m_gra)

        # Reshape and transform to float
        if self.normalize_images:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1))
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1))
            data_next_m_gra = data_next_m_gra.reshape((*data_next_m_gra.shape, 1))
        else:
            data_curr_a_int = data_curr_a_int.reshape((*data_curr_a_int.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_past_m_gra = data_past_m_gra.reshape((*data_past_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR
            data_next_m_gra = data_next_m_gra.reshape((*data_next_m_gra.shape, 1)).astype('float32') / ds.WHITE_COLOR

        # Get data size
        data_size = len(data_curr_a_int)
        assert data_size == len(data_past_m_gra) and \
               data_size == len(data_next_m_gra), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr_appear_int'] = data_curr_a_int
        dict_data['past_motion_gra'] = data_past_m_gra
        dict_data['next_motion_gra'] = data_next_m_gra

        return return_data_dict(dict_data, data_size,
                                frame_index=np.array(frames_idx),
                                bounding_box=np.array(bounding_boxes), num_frames=n_frames)


class MAData:
    """
    Structure of the spatio-temporal RoI sample.
    """
    def __init__(self,
                 dataset=ds.DEFAULT_DATASET,
                 frame_diff_gap=ds.FRAME_DIFF_GAP,
                 min_train_det_score=ds.DEFAULT_MIN_TRAIN_DET_SCORE,
                 min_test_det_score=ds.DEFAULT_MIN_TEST_DET_SCORE,
                 n_frames_per_video=ds.DEFAULT_N_FRAMES_PER_VIDEO,
                 normalize_images=ds.NORMALIZE_IMAGES,
                 frames_step=ds.FRAMES_STEP,
                 det_model_name=ds.DEFAULT_DET_MODEL_NAME,
                 use_motion=ds.USE_MOTION,
                 rand_seed=ds.DEFAULT_RAND_SEED,
                 estimations_path=ds.ESTIMATIONS_PATH):
        """
        Initializes the data extraction parameters.
        :param dataset:
        :param frame_diff_gap:
        :param min_train_det_score:
        :param min_test_det_score:
        :param n_frames_per_video:
        :param normalize_images:
        :param frames_step:
        :param det_model_name:
        :param use_motion:
        :param rand_seed:
        :param estimations_path:
        """
        self.dataset = dataset
        self.frame_diff_gap = frame_diff_gap
        self.n_frames_per_video = n_frames_per_video
        self.normalize_images = normalize_images
        self.frames_step = frames_step
        self.det_model_name = det_model_name
        self.use_motion = use_motion
        self.estimations_path = estimations_path

        self.subset_dir_names = [ds.TRAIN_DIRNAME, ds.TEST_DIRNAME]
        self.subset_dirs = {s: os.path.join(self.estimations_path, self.dataset, s) for s in self.subset_dir_names}
        self.subset_videos = {s: natsorted([v for v in os.listdir(self.subset_dirs[s])
                                            if os.path.isdir(os.path.join(self.subset_dirs[s], v))])
                              for s in self.subset_dir_names}
        self.subset_det = {s: None for s in self.subset_dir_names}
        self.min_det_score = {ds.TRAIN_DIRNAME: min_train_det_score,
                              ds.TEST_DIRNAME: min_test_det_score}

        self.frame_size = ds.FRAME_SIZE
        self.roi_size = ds.ROI_SIZE

        # Set the random seed
        np.random.seed(rand_seed)

    def get_subset(self,
                   subset_name,
                   video_name=None,
                   rand_select=False):
        """
        Extracts the data from frames.
        :return:
        """
        assert subset_name in self.subset_dir_names, 'Invalid subset.'
        assert video_name is None or video_name in self.subset_videos[subset_name], 'Invalid video.'

        # Get samples randomly distributed
        data_curr = []
        data_past = []
        data_next = []
        n_frames = None
        idx = None

        if video_name is None:
            videos = tqdm(self.subset_videos[subset_name], file=sys.stdout,
                          total=len(self.subset_videos[subset_name]),
                          desc='Extracting subset')
        else:
            videos = self.subset_videos[subset_name]

        for video in videos:
            if video_name is not None and video != video_name:
                continue

            video_dir = os.path.join(self.subset_dirs[subset_name], video)

            frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')

            n_frames = len(frames)

            if video_name is not None or rand_select:
                idx = np.array(list(range(self.frame_diff_gap, n_frames - self.frame_diff_gap)))
            else:
                idx = np.array(list(range(self.frame_diff_gap, n_frames - self.frame_diff_gap, self.frames_step)))

            if rand_select:
                np.random.shuffle(idx)
                idx = idx[:self.n_frames_per_video]

            for i in idx:
                # Resize RoIs
                curr_resized = cv2.resize(frames[i], (self.frame_size, self.frame_size))
                past_resized = cv2.resize(frames[i - self.frame_diff_gap], (self.frame_size, self.frame_size))
                next_resized = cv2.resize(frames[i + self.frame_diff_gap], (self.frame_size, self.frame_size))

                if self.use_motion:
                    curr_resized = compute_motion(curr_resized, past_resized)

                if self.normalize_images:
                    # Normalize images
                    curr_resized = cv2.normalize(curr_resized, None,
                                                 alpha=ds.MIN_NORM, beta=ds.MAX_NORM,
                                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    past_resized = cv2.normalize(past_resized, None,
                                                 alpha=ds.MIN_NORM, beta=ds.MAX_NORM,
                                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    next_resized = cv2.normalize(next_resized, None,
                                                 alpha=ds.MIN_NORM, beta=ds.MAX_NORM,
                                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # Store results
                data_curr.append(curr_resized)
                data_past.append(past_resized)
                data_next.append(next_resized)

        # Convert to arrays
        data_curr = np.array(data_curr)
        data_past = np.array(data_past)
        data_next = np.array(data_next)

        # Reshape and transform to float
        data_curr = data_curr.reshape((*data_curr.shape, 1))
        data_past = data_past.reshape((*data_past.shape, 1))
        data_next = data_next.reshape((*data_next.shape, 1))
        if not self.normalize_images:
            data_curr = (ds.MAX_NORM - ds.MIN_NORM) * data_curr.astype('float32') / ds.WHITE_COLOR + ds.MIN_NORM
            data_past = (ds.MAX_NORM - ds.MIN_NORM) * data_past.astype('float32') / ds.WHITE_COLOR + ds.MIN_NORM
            data_next = (ds.MAX_NORM - ds.MIN_NORM) * data_next.astype('float32') / ds.WHITE_COLOR + ds.MIN_NORM

        # Get data size
        data_size = len(data_curr)
        assert data_size == len(data_past) and data_size == len(data_next), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr'] = data_curr
        dict_data['past'] = data_past
        dict_data['next'] = data_next

        if video_name is None:
            return return_data_dict(dict_data, data_size)
        else:
            return return_data_dict(dict_data, data_size, frame_index=idx, num_frames=n_frames)

    def get_roi_subset(self,
                       subset_name,
                       video_name=None,
                       rand_select=False):
        """
        Extracts the RoI data from frames.
        :return:
        """
        assert subset_name in self.subset_dir_names, 'Invalid subset.'
        assert video_name is None or video_name in self.subset_videos[subset_name], 'Invalid video.'

        # Get detections if necessary
        if self.subset_det[subset_name] is None:
            self.subset_det[subset_name] = load_detections(self.subset_dirs[subset_name],
                                                           self.subset_videos[subset_name],
                                                           self.det_model_name)

        # Get samples randomly distributed
        data_curr = []
        data_past = []
        data_next = []
        n_frames = None
        frames_idx = []
        bounding_boxes = []

        if video_name is None:
            videos = tqdm(zip(self.subset_videos[subset_name], self.subset_det[subset_name]),
                          file=sys.stdout,
                          total=len(self.subset_videos[subset_name]),
                          desc='Extracting RoI subset')
        else:
            videos = zip(self.subset_videos[subset_name], self.subset_det[subset_name])

        for video, det_data in videos:
            if video_name is not None and video != video_name:
                continue

            video_dir = os.path.join(self.subset_dirs[subset_name], video)

            frames = np.load(os.path.join(video_dir, ds.FRAMES_FILENAME), mmap_mode='r')

            n_frames = len(frames)

            if video_name is not None or rand_select:
                idx = np.array(list(range(self.frame_diff_gap, n_frames - self.frame_diff_gap)))
            else:
                idx = np.array(list(range(self.frame_diff_gap, n_frames - self.frame_diff_gap, self.frames_step)))

            if rand_select:
                np.random.shuffle(idx)
                idx = idx[:self.n_frames_per_video]

            for i in idx:
                curr_img = frames[i]
                past_img = frames[i - self.frame_diff_gap]
                next_img = frames[i + self.frame_diff_gap]

                # Detections will be a python dict: {category_id : [[x1, y1, x2, y2, score], ...], }
                curr_a_det = det_data[i]
                # Remove empty detections
                curr_a_det = {k: curr_a_det[k] for k in curr_a_det if curr_a_det[k].size}
                # transform ret into [[x1, y1, x2, y2, score], ...]
                curr_a_det = np.vstack(list(curr_a_det.values()))

                # Exclude detections that have lower score
                curr_a_det = curr_a_det[curr_a_det[:, 4] >= self.min_det_score[subset_name], :]

                if not len(curr_a_det):
                    continue

                max_det_y_pixels = int(ds.MAX_DET_SIZE_RATIO * curr_img.shape[0])
                max_det_x_pixels = int(ds.MAX_DET_SIZE_RATIO * curr_img.shape[1])

                for box in curr_a_det:
                    (x1, y1, x2, y2) = box[:-1].astype(int)

                    # Bug fixed
                    if x1 < 0:
                        x1 = 0
                    if y1 < 0:
                        y1 = 0
                    if x2 > curr_img.shape[1]:
                        x2 = curr_img.shape[1]
                    if y2 > curr_img.shape[0]:
                        y2 = curr_img.shape[0]

                    if (x2 - x1) < ds.MIN_DET_PIXELS or (y2 - y1) < ds.MIN_DET_PIXELS:
                        continue

                    if self.dataset == 'ped1' or self.dataset == 'ped2':
                        # Fix whole frame detections on ped2
                        if (x2 - x1) > max_det_x_pixels and (y2 - y1) > max_det_y_pixels:
                            continue

                    # Resize RoIs
                    curr_resized = cv2.resize(curr_img[y1:y2, x1:x2], (self.roi_size, self.roi_size))
                    past_resized = cv2.resize(past_img[y1:y2, x1:x2], (self.roi_size, self.roi_size))
                    next_resized = cv2.resize(next_img[y1:y2, x1:x2], (self.roi_size, self.roi_size))

                    if self.use_motion:
                        curr_resized = compute_motion(curr_resized, past_resized)

                    if self.normalize_images:
                        # Normalize images
                        curr_resized = cv2.normalize(curr_resized, None,
                                                     alpha=ds.MIN_NORM, beta=ds.MAX_NORM,
                                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        past_resized = cv2.normalize(past_resized, None,
                                                     alpha=ds.MIN_NORM, beta=ds.MAX_NORM,
                                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        next_resized = cv2.normalize(next_resized, None,
                                                     alpha=ds.MIN_NORM, beta=ds.MAX_NORM,
                                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # Store results
                    data_curr.append(curr_resized)
                    data_past.append(past_resized)
                    data_next.append(next_resized)

                    frames_idx.append(i)
                    bounding_boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

        # Convert to arrays
        data_curr = np.array(data_curr)
        data_past = np.array(data_past)
        data_next = np.array(data_next)

        # Reshape and transform to float
        data_curr = data_curr.reshape((*data_curr.shape, 1))
        data_past = data_past.reshape((*data_past.shape, 1))
        data_next = data_next.reshape((*data_next.shape, 1))
        if not self.normalize_images:
            data_curr = (ds.MAX_NORM - ds.MIN_NORM) * data_curr.astype('float32') / ds.WHITE_COLOR + ds.MIN_NORM
            data_past = (ds.MAX_NORM - ds.MIN_NORM) * data_past.astype('float32') / ds.WHITE_COLOR + ds.MIN_NORM
            data_next = (ds.MAX_NORM - ds.MIN_NORM) * data_next.astype('float32') / ds.WHITE_COLOR + ds.MIN_NORM

        # Get data size
        data_size = len(data_curr)
        assert data_size == len(data_past) and data_size == len(data_next), 'Size of subset mismatch.'

        # Normalize data and return a dict containing data
        dict_data = dict()
        dict_data['curr'] = data_curr
        dict_data['past'] = data_past
        dict_data['next'] = data_next

        if video_name is None:
            return return_data_dict(dict_data, data_size)
        else:
            return return_data_dict(dict_data, data_size,
                                    frame_index=np.array(frames_idx),
                                    bounding_box=np.array(bounding_boxes), num_frames=n_frames)


class EST5EDT(datetime.tzinfo):

    def utcoffset(self, dt):
        return datetime.timedelta(hours=-5) + self.dst(dt)

    def dst(self, dt):
        d = datetime.datetime(dt.year, 3, 8)        # 2nd Sunday in March
        self.dston = d + datetime.timedelta(days=6-d.weekday())
        d = datetime.datetime(dt.year, 11, 1)       # 1st Sunday in Nov
        self.dstoff = d + datetime.timedelta(days=6-d.weekday())
        if self.dston <= dt.replace(tzinfo=None) < self.dstoff:
            return datetime.timedelta(hours=1)
        else:
            return datetime.timedelta(0)

    def tzname(self, dt):
        return 'EST5EDT'


def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout.decode())


def score_smoothing(score, sigma=30):
    # r = score.shape[0] //39
    # if r%2==0:
    #     r+=1
    r = 125
    if r > score.shape[0] // 2:
        r = score.shape[0] // 2 - 1
    if r % 2 == 0:
        r += 1
    gaussian_temp = np.ones(r*2-1)
    for i in range(r*2-1):
        gaussian_temp[i] = np.exp(-(i-r+1)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    new_score = score
    for i in range(r, score.shape[0]-r):
        new_score[i] = np.dot(score[i-r:i+r-1], gaussian_temp)
    return new_score


def get_results(scores_data,
                smoothing_factor=ds.DEFAULT_SMOOTHING_FACTOR,
                normalize=ds.DEFAULT_NORMALIZE_SCORES,
                binarize=ds.DEFAULT_BINARIZE_SCORES):
    """
    Gets the frame-level reconstruction error scores for all test sequences.
    :param scores_data:
    :param smoothing_factor:
    :param normalize:
    :param binarize:
    :return:
    """
    min_score = scores_data['scores'].min()
    max_score = scores_data['scores'].max()

    if normalize:
        scores_data['scores'] = np.interp(scores_data['scores'], (min_score, max_score), (0, 1))

    full_frames = list(range(1, scores_data['num_frames'] + 1))

    # get all fame number used in the test data
    test_frames = list(set(scores_data['frame_index']))
    test_frames.sort()

    max_scores = []
    for f in test_frames:
        frame_scores = scores_data['scores'][scores_data['frame_index'] == f]
        max_scores.append(np.max(frame_scores))

    if binarize:
        threshold = np.mean(max_scores)
        test_scores = [1 if x > threshold else 0 for x in max_scores]
    else:
        test_scores = max_scores

    # Put missing scores in frames to 0 (normal)
    test_frames = np.array(test_frames)
    match_index = [np.where(test_frames == f)[0] for f in range(scores_data['num_frames'])]
    full_scores = np.array([test_scores[i[0]] if i.size > 0 else min_score for i in match_index])

    # Smooth scores using Gaussian 1d filter
    if smoothing_factor:
        final_scores = gaussian_filter1d(full_scores, smoothing_factor)
    else:
        final_scores = full_scores

    return np.array(full_frames), final_scores


def get_gt_results(dataset=ds.DEFAULT_DATASET,
                   dataset_path=ds.DATASETS_PATH):
    """
    Get the frame-level ground-truth results.
    Inspired from:
    https://github.com/StevenLiuWen/ano_pred_cvpr2018/blob/master/Codes/evaluate.py
    :param dataset:
    :param dataset_path:
    :return:
    """
    tests_dir = os.path.join(dataset_path, dataset, ds.TEST_DIRNAME)

    if dataset == 'shanghaitech':
        gt_dir = os.path.join(tests_dir, 'test_frame_mask')

        video_path_list = [f for f in os.listdir(gt_dir)]
        video_path_list.sort()

        gt = []
        for video in video_path_list:
            gt.append(np.load(os.path.join(gt_dir, video)).tolist()[1:])
    else:
        gt_dir = os.path.join(dataset_path, dataset)
        gt_filename = '{}.mat'.format(dataset)

        abnormal_events = scio.loadmat(os.path.join(gt_dir, gt_filename), squeeze_me=True)['gt']

        if abnormal_events.ndim == 2:
            abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0], abnormal_events.shape[1])

        num_video = abnormal_events.shape[0]

        frames_dir = os.path.join(tests_dir, ds.FRAMES_DIRNAME)
        video_list = [v for v in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, v))]
        video_list.sort()

        assert num_video == len(video_list), 'ground truth does not match the number of testing videos. {} != {}' \
            .format(num_video, len(video_list))

        # get the total frames of sub video
        def get_video_length(sub_video_number):
            # video_name = video_name_template.format(sub_video_number)
            video_name = os.path.join(frames_dir, video_list[sub_video_number])
            assert os.path.isdir(video_name), '{} is not directory!'.format(video_name)
            return len(os.listdir(video_name))

        gt = []
        for i in range(num_video):
            length = get_video_length(i)

            sub_video_gt = np.zeros((length,), dtype=np.int8)
            sub_abnormal_events = abnormal_events[i]
            if sub_abnormal_events.ndim == 1:
                sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))

            _, num_abnormal = sub_abnormal_events.shape

            for j in range(num_abnormal):
                # (start - 1, end - 1)
                start = int(sub_abnormal_events[0, j]) - 1
                end = int(sub_abnormal_events[1, j])

                sub_video_gt[start: end] = 1

            gt.append(sub_video_gt.tolist()[1:])  # Skip the first frame

    return gt


def convert_videos_to_frames(videos_dir, frames_dir):
    """
    Converts all videos in the videos dir to frames and saves them to frames dir.
    :param videos_dir:
    :param frames_dir:
    :return:
    """
    videos = [v for v in os.listdir(videos_dir) if os.path.isfile(os.path.join(videos_dir, v)) and
              any(v[v.rfind('.'):] == ext for ext in ds.VIDEO_EXTENSIONS)]
    videos = natsorted(videos)
    for video in tqdm(videos, file=sys.stdout, desc='Extracting frames'):
        video_frames_dir = os.path.join(frames_dir, video[:video.rfind('.')])
        if not os.path.exists(video_frames_dir):
            os.makedirs(video_frames_dir)

        video_cap = cv2.VideoCapture(os.path.join(videos_dir, video))
        success, frame = video_cap.read()
        count = 0
        while success:
            # save frame as JPEG file
            cv2.imwrite(os.path.join(video_frames_dir, '{:04d}.jpg'.format(count)), frame)
            success, frame = video_cap.read()
            count += 1


def compute_gradient_image(gray_img):
    """
    Computes the spatial gradient image of the given grayscale image using Sobel operation.
    See: https://stackoverflow.com/questions/61943937/python-opencv-gradient-computing
    :param gray_img:
    :return:
    """
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
    grad_x = np.uint8(np.absolute(grad_x))
    grad_y = np.uint8(np.absolute(grad_y))
    grad_img = cv2.bitwise_or(grad_x, grad_y)
    return grad_img


def compute_tgi(gray_img_1, gray_img_2):
    """
    Computes the temporal gradient image between the two given image. Use absolute difference.
    :param gray_img_1:
    :param gray_img_2:
    :return:
    """
    # Compute gradient images
    img_1_gra = compute_gradient_image(gray_img_1)
    img_2_gra = compute_gradient_image(gray_img_2)

    # Compute motion gradient
    return cv2.bitwise_xor(img_1_gra, img_2_gra)


def compute_motion(gray_img_1, gray_img_2):
    """
    Estimates the motion image.
    Ref.: https://medium.com/@lope.ai/motion-detector-alarmusing-opencv-python-b3ff21c69e0c
    :param gray_img_1:
    :param gray_img_2:
    :return:
    """
    # Blur the images
    img_1_blurred = cv2.GaussianBlur(gray_img_1, (3, 3), 0)
    img_2_blurred = cv2.GaussianBlur(gray_img_2, (3, 3), 0)
    # Compute the absolute difference
    img_delta = cv2.absdiff(img_2_blurred, img_1_blurred)
    # Return the gradient of the difference
    return compute_gradient_image(img_delta)


def get_class_from_name(module, class_name):
    """
    Gets the class from the string name within the given module.
    :param module:
    :param class_name:
    :return: class
    """
    cls_names = [m[0] for m in inspect.getmembers(module, inspect.isclass)]
    assert class_name.lower() in [x.lower() for x in cls_names], 'Class name not found.'
    cls_name = cls_names[[x.lower() for x in cls_names].index(class_name.lower())]
    return getattr(module, cls_name)


def intersperse(lst, item):
    """
    Adds the item between each item in the list.
    :param lst:
    :param item:
    :return:
    """
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


def plot_video_anomaly_curve(frame_index, frame_score, true_score, filename):
    """
    Plots the video frame-level anomaly score.
    :param frame_index:
    :param frame_score:
    :param true_score:
    :param filename:
    :return:
    """
    assert len(true_score) == len(frame_score), 'Mismatch between the size of true and predicted scores.'
    # Plot a curve of scores and threshold_value
    fig = plt.figure(figsize=(10, 4))
    plt.plot(frame_index, frame_score, 'r', label='score', lw=3)
    plt.fill_between(frame_index, np.zeros(frame_index.shape), true_score, facecolor='c', label='ground-truth', lw=1)
    plt.xlabel('Frame')
    plt.ylabel('Score')
    plt.yticks(np.arange(0, 1.2, step=0.2))
    plt.legend(loc='best')
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def add_bool_parser(parser, arg_name, default_arg):
    """
    Adds a boolean parser.
    Ref.: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    :param parser:
    :param arg_name:
    :param default_arg:
    :return:
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + arg_name, dest=arg_name, action='store_true')
    group.add_argument('--no-' + arg_name, dest=arg_name, action='store_false')
    parser.set_defaults(**{arg_name: default_arg})
