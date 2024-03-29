"""
Script that extracts bounding boxes and flow estimations for a given dataset using CenterNet and FlowNet2.
"""
import numpy as np
import pandas as pd
import json
import time
import cv2
import sys
import os
import argparse
import datetime
from tqdm import tqdm
from natsort import natsorted
from src import defines as ds
from src import utilities as util

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default=ds.DEFAULT_DATASET, help='Name of the dataset.')
    parser.add_argument('--det_task', type=str, default=ds.DEFAULT_DET_TASK, help='CenterNet task.')
    parser.add_argument('--det_dataset', type=str, default=ds.DEFAULT_DET_DATASET, help='CenterNet dataset.')
    parser.add_argument('--det_arch', type=str, default=ds.DEFAULT_DET_ARCH, help='CenterNet architecture.')
    util.add_bool_parser(parser, 'save_frames', False)
    parser.add_argument('--pretrained_path', type=str, default=ds.PRETRAINED_PATH, help='Pretrained models path.')
    parser.add_argument('--centernet_path', type=str, default=os.environ['CenterNet_ROOT'], help='CenterNet path.')
    parser.add_argument('--datasets_path', type=str, default=ds.DATASETS_PATH, help='Datasets path.')
    parser.add_argument('--estimations_path', type=str, default=ds.ESTIMATIONS_PATH, help='Estimations path.')
    parser.add_argument('--temp_path', type=str, default=ds.TEMP_PATH, help='Temp folder path.')

    args = parser.parse_args()

    # Set the random seed
    np.random.seed(args.rand_seed)

    det_model_name = f'{args.det_task}_{args.det_dataset}_{ds.DET_MODEL_NAMES[args.det_arch]}'
    det_model_path = os.path.join(args.pretrained_path, ds.DET_METHOD_NAME, f'{det_model_name}.pth')
    assert os.path.isfile(det_model_path), 'Model path does not exist.'

    sys.path.insert(0, os.path.join(args.centernet_path, 'src'))
    sys.path.insert(0, os.path.join(args.centernet_path, 'src', 'lib'))
    sys.path.insert(0, os.path.join(args.centernet_path, 'src', 'lib', 'models', 'networks', 'DCNv2'))

    from detectors.detector_factory import detector_factory
    from opts import opts

    opt = opts().init('{} --load_model {} --arch {}'.format(args.det_task, det_model_path, args.det_arch).split(' '))
    detector = detector_factory[opt.task](opt)

    report_speeds = []

    for data_dirname in [ds.TRAIN_DIRNAME, ds.TEST_DIRNAME]:
        data_dir = os.path.join(args.datasets_path, args.dataset, data_dirname)

        # Extract frames from videos if necessary
        frames_dir = os.path.join(data_dir, ds.FRAMES_DIRNAME)
        if not os.path.exists(frames_dir):
            videos_dir = os.path.join(data_dir, ds.VIDEOS_DIRNAME)
            util.convert_videos_to_frames(videos_dir, frames_dir)

        video_names = [v for v in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, v))]
        video_names = natsorted(video_names)

        data_det_times = []
        data_total_times = []

        for video in tqdm(video_names, file=sys.stdout,
                          desc='Preprocessing {} - {}'.format(args.dataset, data_dirname)):
            video_dir = os.path.join(frames_dir, video)

            # Get the image size and extract spatial gradient images
            raw_img_h = None
            raw_img_w = None

            # Detect regions on images
            video_frames = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f)) and
                            any(f[f.rfind('.'):] == ext for ext in ds.IMAGE_EXTENSIONS)]
            video_frames = natsorted(video_frames)

            # Estimation times (in seconds)
            video_det_time = 0

            # Estimation dir
            video_est_dir = os.path.join(args.estimations_path, args.dataset, data_dirname, video)
            if not os.path.exists(video_est_dir):
                os.makedirs(video_est_dir)

            # Detection file
            frames_data = []
            video_det_data = []

            for frame_filename in tqdm(video_frames[1:], file=sys.stdout,
                                       desc='{} --> {} - {} - {}'.format(ds.DET_METHOD_NAME,
                                                                         args.dataset, data_dirname, video)):
                frame = cv2.imread(os.path.join(video_dir, frame_filename))
                # Get original size
                if raw_img_h is None:
                    raw_img_h, raw_img_w, _ = frame.shape

                # run detection process time
                start = time.time()
                ret = detector.run(frame)['results']
                det_time = time.time() - start

                if args.save_frames:
                    # Save frame
                    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames_data.append(gray_img)

                # Save detections
                video_det_data.append(ret)

                # Save running time
                video_det_time += det_time

            # Save arrays
            video_det_data = np.array(video_det_data)
            np.save(os.path.join(video_est_dir, ds.DET_FILENAMES[det_model_name]), video_det_data)
            if args.save_frames:
                frames_data = np.array(frames_data)
                np.save(os.path.join(video_est_dir, ds.FRAMES_FILENAME), frames_data)

            # Store video running summary (time)
            data_det_times.append(video_det_time / len(video_frames))

        # Store data running summary (time)
        dict_data_speeds = dict()
        dict_data_speeds['subset'] = data_dirname
        dict_data_speeds['n_videos'] = len(video_names)

        dict_data_speeds['min_det_time'] = np.min(data_det_times)
        dict_data_speeds['max_det_time'] = np.max(data_det_times)
        dict_data_speeds['avg_det_time'] = np.mean(data_det_times)
        dict_data_speeds['std_det_time'] = np.std(data_det_times)

        report_speeds.append(dict_data_speeds)

    # Report results
    dt = datetime.datetime.now(tz=util.EST5EDT())
    curr_script_name = os.path.splitext(os.path.basename(__file__))[0]
    results_dir = os.path.join(ds.root_dir, ds.RESULTS_DIRNAME, curr_script_name, dt.strftime('%Y-%m-%d_%Hh%Mm%Ss'))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save running summary
    pd.DataFrame(report_speeds).to_csv(os.path.join(results_dir, ds.SUMMARY_RUNNING_FILENAME), index=False)

    # Save commandline args
    with open(os.path.join(results_dir, ds.COMMANDLINE_ARGS_FILENAME), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
