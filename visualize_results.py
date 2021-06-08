"""
This script helps visualize the RoIs extracted from the estimated data.
"""
import json
import time
import cv2
import os
import argparse
import datetime
import copy
from src import defines as ds
from src import utilities as util


def show_window(win_name, img):
    cv2.startWindowThread()
    cv2.namedWindow(win_name)
    cv2.imshow(win_name, img)


def get_bool(prompt):
    while True:
        try:
            return {'y': True, 'n': False}[input(prompt).lower()]
        except KeyError:
            print('Invalid input please enter yes or no.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, default=ds.DEFAULT_DATASET, help='Name of the dataset.')
    parser.add_argument('-s', '--subset', type=str, default=ds.TRAIN_DIRNAME, help='Name of the subset.')
    parser.add_argument('-v', '--video', type=str, default=None, help='Name of the video.')
    parser.add_argument('-t', '--type', type=str, default=ds.DEFAULT_DATATYPE_CLS, help='Name of the datatype class.')
    parser.add_argument('--frame_diff_gap', type=int, default=ds.FRAME_DIFF_GAP, help='The temporal frame gap.')
    parser.add_argument('--min_train_det_score', type=float, default=ds.DEFAULT_MIN_TRAIN_DET_SCORE)
    parser.add_argument('--min_test_det_score', type=float, default=ds.DEFAULT_MIN_TEST_DET_SCORE)
    parser.add_argument('--det_model_name', type=str, default=ds.DEFAULT_DET_MODEL_NAME, help='Det model name.')
    parser.add_argument('--n_frames_per_video', type=int, default=ds.DEFAULT_N_FRAMES_PER_VIDEO, help='Frames/video.')
    util.add_bool_parser(parser, 'normalize_images', ds.NORMALIZE_IMAGES)
    parser.add_argument('--n_samples_per_viz', type=int, default=ds.DEFAULT_N_SAMPLES_PER_VIZ, help='Frames/viz.')
    parser.add_argument('--estimations_path', type=str, default=ds.ESTIMATIONS_PATH, help='Estimations path.')

    args = parser.parse_args()

    # Get datatype class
    data_cls = util.get_class_from_name(util, args.type)

    # Prepare arguments for data init
    data_args = copy.deepcopy(args.__dict__)
    data_args.pop('subset', None)
    data_args.pop('video', None)
    data_args.pop('type', None)
    data_args.pop('holistic', None)
    data_args.pop('n_samples_per_viz', None)

    # Initialize data class
    extract_data = data_cls(**data_args)

    # Results dir
    dt = datetime.datetime.now(tz=util.EST5EDT())
    curr_script_name = os.path.splitext(os.path.basename(__file__))[0]
    results_dir = os.path.join(ds.root_dir, ds.RESULTS_DIRNAME, curr_script_name, dt.strftime('%Y-%m-%d_%Hh%Mm%Ss'))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    while True:
        if args.holistic:
            # Extract data
            data = extract_data.get_subset(args.subset, args.video, rand_select=True)
        else:
            data = extract_data.get_roi_subset(args.subset, args.video, rand_select=True)

        print('Data size: {}'.format(data['data_size']))

        visualize_data = get_bool('Visualize data? ')
        while visualize_data:
            # Show some randomly selected samples
            viz_image = util.visualize_data(data, n_samples_per_viz=args.n_samples_per_viz)

            # Show the stacked image
            show_window('ST RoI data', viz_image)

            key = cv2.waitKey(0)
            if key == ord('s'):  # Save th image
                cv2.imwrite(os.path.join(results_dir, '{}_{}.png'.format(ds.ROI_DATAVIZ_NAME, time.time())), viz_image)
                print('Image saved.')
            elif key == ord('v'):  # Randomly visualize other samples
                continue
            else:
                break

        cv2.destroyAllWindows()

        if not get_bool('Do you want to redo? '):
            break

    # Save commandline args
    with open(os.path.join(results_dir, ds.COMMANDLINE_ARGS_FILENAME), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
