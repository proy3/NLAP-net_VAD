"""
This script first trains a anomaly detection model using train set and then tests it with the test set.
"""
import numpy as np
import pandas as pd
import json
import time
import os
import sys
import argparse
import datetime
import copy
from tqdm import tqdm
from src import defines as ds
from src import utilities as util
from src import networks as net
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default=ds.DEFAULT_MODEL_CLS, help='Name of the model class.')
    util.add_bool_parser(parser, 'inference_mode', ds.INFERENCE_MODE)
    parser.add_argument('--pretrained_model', type=str, default=None, help='Folder containing model.')
    util.add_bool_parser(parser, 'random_subset_training', ds.RANDOM_SUBSET_TRAINING)
    util.add_bool_parser(parser, 'new_training_scheme', ds.NEW_TRAINING_SCHEME)
    parser.add_argument('--first_n_epochs', type=int, default=ds.FIRST_N_EPOCHS, help='First n epochs.')
    util.add_bool_parser(parser, 'change_random_subset', ds.CHANGE_RANDOM_SUBSET)
    parser.add_argument('--change_subset_step', type=int, default=ds.CHANGE_SUBSET_STEP, help='Change subset step.')
    parser.add_argument('-d', '--dataset', type=str, default=ds.DEFAULT_DATASET, help='Name of the dataset.')
    parser.add_argument('--extract_datatype', type=str, default=ds.DEFAULT_DATATYPE_CLS, help='Extract data class.')
    parser.add_argument('--frame_diff_gap', type=int, default=ds.FRAME_DIFF_GAP, help='The temporal frame gap.')
    parser.add_argument('--min_train_det_score', type=float, default=ds.DEFAULT_MIN_TRAIN_DET_SCORE)
    parser.add_argument('--min_test_det_score', type=float, default=ds.DEFAULT_MIN_TEST_DET_SCORE)
    parser.add_argument('--n_frames_per_video', type=int, default=ds.DEFAULT_N_FRAMES_PER_VIDEO, help='Frames/video.')
    util.add_bool_parser(parser, 'normalize_images', ds.NORMALIZE_IMAGES)
    parser.add_argument('--frames_step', type=int, default=ds.FRAMES_STEP, help='The temporal frame step.')
    parser.add_argument('--det_model_name', type=str, default=ds.DEFAULT_DET_MODEL_NAME, help='Det model name.')
    util.add_bool_parser(parser, 'use_shared_encoders', ds.USE_SHARED_ENCODERS)
    util.add_bool_parser(parser, 'use_skip_connections', ds.USE_SKIP_CONNECTIONS)
    util.add_bool_parser(parser, 'use_adversarial_loss', ds.USE_ADVERSARIAL_LOSS)
    util.add_bool_parser(parser, 'smooth_adversarial_labels', ds.SMOOTH_ADVERSARIAL_LABELS)
    parser.add_argument('--batch_size', type=int, default=ds.DEFAULT_BATCH_SIZE, help='Batch size.')
    parser.add_argument('--n_epochs', type=int, default=ds.DEFAULT_N_EPOCHS, help='Number of epochs.')
    parser.add_argument('--n_inner_epochs', type=int, default=ds.DEFAULT_N_INNER_EPOCHS, help='Number inner epochs.')
    parser.add_argument('--d_learning_rate', type=float, default=ds.DEFAULT_D_LEARNING_RATE, help='D learning rate.')
    parser.add_argument('--g_learning_rate', type=float, default=ds.DEFAULT_G_LEARNING_RATE, help='G learning rate.')
    parser.add_argument('--ft_reduce_factor', type=float, default=ds.DEFAULT_FT_REDUCE_FACTOR, help='Fine tuning.')
    parser.add_argument('--start_fine_tune', type=int, default=ds.DEFAULT_START_FINE_TUNE, help='Start fine tuning.')
    util.add_bool_parser(parser, 'use_separable_conv', ds.USE_SEPARABLE_CONV)
    parser.add_argument('--conv_kernel_size', type=int, default=ds.DEFAULT_CONV_KERNEL_SIZE, help='Conv kernel size.')
    parser.add_argument('--conv_strides', type=int, default=ds.DEFAULT_CONV_STRIDES, help='Conv stride.')
    parser.add_argument('--conv_leaky_relu_alpha', type=float, default=ds.DEFAULT_CONV_LEAKY_RELU_ALPHA)
    parser.add_argument('--out_activation', type=str, default=ds.DEFAULT_OUT_ACTIVATION, help='Output activation.')
    parser.add_argument('--conv_dropout_rate', type=float, default=ds.DEFAULT_CONV_DROPOUT_RATE, help='Dropout rate.')
    util.add_bool_parser(parser, 'conv_batch_norm', ds.DEFAULT_CONV_BATCH_NORM)
    parser.add_argument('--conv_kernel_init', type=str, default=ds.DEFAULT_CONV_KERNEL_INIT, help='Kernel init.')
    parser.add_argument('--conv_skip_diff', type=str, default=ds.DEFAULT_CONV_SKIP_DIFF, help='Skip diff method.')
    parser.add_argument('--d_loss', type=str, default=ds.DEFAULT_D_LOSS, help='Discriminator loss.')
    parser.add_argument('--d_loss_ratio', type=float, default=ds.DEFAULT_D_LOSS_RATIO, help='D loss weight.')
    parser.add_argument('--g_loss_ratio', type=float, default=ds.DEFAULT_G_LOSS_RATIO, help='G loss weight.')
    parser.add_argument('--g_loss_l1_ratio', type=float, default=ds.DEFAULT_G_LOSS_L1_RATIO, help='Gen L1 loss.')
    parser.add_argument('--g_loss_l2_ratio', type=float, default=ds.DEFAULT_G_LOSS_L2_RATIO, help='Gen L2 loss.')
    parser.add_argument('--g_loss_ss_ratio', type=float, default=ds.DEFAULT_G_LOSS_SS_RATIO, help='Gen ss loss.')
    parser.add_argument('--g_loss_gd_ratio', type=float, default=ds.DEFAULT_G_LOSS_GD_RATIO, help='Gen gd loss.')
    parser.add_argument('--d_opt_method', type=str, default=ds.DEFAULT_D_OPT_METHOD, help='Discriminator optimizer.')
    parser.add_argument('--g_opt_method', type=str, default=ds.DEFAULT_G_OPT_METHOD, help='Generator optimizer.')
    parser.add_argument('--opt_grad_clip', type=float, default=ds.DEFAULT_OPT_GRAD_CLIP, help='Optimizer grad clip.')
    util.add_bool_parser(parser, 'normalize', ds.DEFAULT_NORMALIZE_SCORES)
    util.add_bool_parser(parser, 'binarize', ds.DEFAULT_BINARIZE_SCORES)
    parser.add_argument('--smoothing', type=int, default=ds.DEFAULT_SMOOTHING_FACTOR, help='Scores smoothing.')
    util.add_bool_parser(parser, 'score_use_d', ds.DEFAULT_SCORE_USE_D)
    parser.add_argument('--score_method', type=str, default=ds.DEFAULT_SCORE_METHOD, help='Anomaly scoring method.')
    util.add_bool_parser(parser, 'score_use_grid', ds.DEFAULT_SCORE_USE_GRID)
    parser.add_argument('--score_grid_size', type=int, default=ds.DEFAULT_SCORE_GRID_SIZE, help='Scores grid size.')

    args = parser.parse_args()

    # Result and model dir
    dt = datetime.datetime.now(tz=util.EST5EDT())
    curr_script_name = os.path.splitext(os.path.basename(__file__))[0]
    date_name = dt.strftime('%Y-%m-%d_%Hh%Mm%Ss')
    result_dir = os.path.join(ds.root_dir, ds.RESULTS_DIRNAME, curr_script_name, date_name)
    save_model_dir = os.path.join(ds.root_dir, ds.MODELS_DIRNAME, date_name)

    if args.pretrained_model is None:
        assert not args.inference_mode, 'Must specify the pretrained model path in inference mode.'
        load_model_dir = None
    else:
        load_model_dir = os.path.join(ds.root_dir, ds.MODELS_DIRNAME, args.pretrained_model)

    # Get model class
    model_cls = util.get_class_from_name(net, args.model)

    # Prepare arguments for model init
    model_args = copy.deepcopy(args.__dict__)
    model_args.pop('model', None)
    model_args.pop('inference_mode', None)
    model_args.pop('pretrained_model', None)
    model_args.pop('normalize', None)
    model_args.pop('binarize', None)
    model_args.pop('smoothing', None)
    model_args['load_model_dir_path'] = load_model_dir
    model_args['save_model_dir_path'] = save_model_dir
    model_args['result_dir_path'] = result_dir

    # Initialize model
    model = model_cls(**model_args)

    if model.holistic_method:
        args.det_model_name = None
        args.min_det_score = None

    if args.inference_mode:
        # Load the trained model
        model.load_models()
        train_time = 0
    else:
        # Build and train the model
        model.build_model()
        start = time.time()
        model.train()
        train_time = time.time() - start

    # Get the ground-truth anomaly labels
    gt_labels = util.get_gt_results(dataset=args.dataset)

    # Test the trained model for all test videos
    all_scores = []
    all_infer_time = []  # Inference time per frame

    for v, v_name in tqdm(enumerate(model.extract_data.subset_videos[ds.TEST_DIRNAME]), file=sys.stdout,
                          total=len(model.extract_data.subset_videos[ds.TEST_DIRNAME]),
                          desc='Applying trained models on test videos'):
        # Get the gt data
        gt_scores = np.array(gt_labels[v])

        # Apply the trained model
        start = time.time()
        ano_scores = model.test(ds.TEST_DIRNAME, v_name)
        all_infer_time.append((time.time() - start) / ano_scores['num_frames'])

        # Get smoothed results
        f_t, s_t = util.get_results(ano_scores,
                                    smoothing_factor=args.smoothing,
                                    normalize=args.normalize,
                                    binarize=args.binarize)
        all_scores.append(s_t.tolist())

        # Plot a curve of scores and threshold_value
        figure_name = 'anomaly_curve_in_video_{}.pdf'.format(os.path.splitext(v_name)[0])
        util.plot_video_anomaly_curve(f_t, s_t, gt_scores, os.path.join(result_dir, figure_name))

    # Clear all the loaded models
    model.close_session()

    # Average inference time per frame
    infer_time = np.mean(all_infer_time)

    print('Computing overall performance:')
    # Compute the AUC
    glob_res = net.get_global_result(gt_results=gt_labels,
                                     ts_results=all_scores,
                                     result_dir_path=result_dir)
    print('==============================================================')

    print('Reporting results and arguments:')
    # Report the result
    pd.DataFrame([{'Date': date_name,
                   'Dataset': args.dataset,
                   'Datatype': args.extract_datatype,
                   'Model': args.model,
                   'Scorer': model.ano_scorer.dist_name,
                   'AUC': glob_res['auc'],
                   'EER': glob_res['eer'],
                   'Thr': glob_res['thr'],
                   'train_time': train_time,
                   'infer_time': infer_time}]).to_csv(os.path.join(result_dir, ds.SUMMARY_RESULTS_FILENAME),
                                                      index=False)

    # Save commandline args
    with open(os.path.join(result_dir, ds.COMMANDLINE_ARGS_FILENAME), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print('==============================================================')
