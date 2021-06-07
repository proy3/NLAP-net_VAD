"""
This file contains all the unsupervised neural networks for semi-supervised video anomaly detection.
"""
import numpy as np
import joblib
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
from scipy.interpolate import interp1d
from keras.layers import Input, Conv2D, Concatenate, SpatialDropout2D, SeparableConv2D, MaxPooling2D, UpSampling2D
from keras.layers import LeakyReLU, Conv2DTranspose, BatchNormalization, Lambda
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.losses import mean_squared_error, mean_absolute_error
from keras_contrib.losses import DSSIMObjective
from keras.initializers import he_uniform, random_uniform, he_normal, random_normal
from keras.models import Model
from keras import optimizers
from keras.models import load_model
import tensorflow as tf
import keras.backend as kb
from tfdeterminism import patch
import matplotlib.pyplot as plt
import time
import random as rn
import os
import shutil
from tqdm import tqdm
import sys
import inspect
from src import defines as ds
from src import utilities as util


def reset_random_seeds(rand_seed=ds.DEFAULT_RAND_SEED):
    """
    Sets all the related seeds for reproducibility.
    :param rand_seed:
    :return:
    """
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    tf.set_random_seed(rand_seed)
    np.random.seed(rand_seed)
    rn.seed(rand_seed)


def configure_tf(use_tf_determinism=ds.TENSORFLOW_DETERMINISM):
    """
    Configure TensorFlow.
    :param use_tf_determinism:
    :return:
    """
    if use_tf_determinism:
        # The below is necessary in Python 3.2.3 onwards to
        # have reproducible behavior for certain hash-based operations.
        # See these references for further details:
        # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
        # https://github.com/fchollet/keras/issues/2280#issuecomment-306959926
        # For Stock TensorFlow Version < 2.1 (https://github.com/NVIDIA/framework-determinism)
        patch()

        # Force TensorFlow to use single thread.
        # Multiple threads are a potential source of
        # non-reproducible results.
        # For further details, see: https://stackoverflow.com/questions/42022950/
        # which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    else:
        config = tf.ConfigProto()

    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))


class LocalGARDiN:
    """
    Our object centered GARDiN baseline method.
    """

    def __init__(self,
                 use_tf_determinism=ds.TENSORFLOW_DETERMINISM,
                 random_subset_training=ds.RANDOM_SUBSET_TRAINING,
                 new_training_scheme=ds.NEW_TRAINING_SCHEME,
                 reduce_subset_step=ds.REDUCE_SUBSET_STEP,
                 reduce_subset_factor=ds.REDUCE_SUBSET_FACTOR,
                 dataset=ds.DEFAULT_DATASET,
                 frame_diff_gap=ds.FRAME_DIFF_GAP,
                 min_det_score=ds.DEFAULT_MIN_DET_SCORE,
                 n_frames_per_video=ds.DEFAULT_N_FRAMES_PER_VIDEO,
                 normalize_images=ds.NORMALIZE_IMAGES,
                 frames_step=ds.FRAMES_STEP,
                 estimations_path=ds.ESTIMATIONS_PATH,
                 extract_datatype=ds.DEFAULT_DATATYPE_CLS,
                 det_model_name=ds.DEFAULT_DET_MODEL_NAME,
                 use_cross_domain=ds.USE_CROSS_DOMAIN_GENERATORS,
                 use_skip_connections=ds.USE_SKIP_CONNECTIONS,
                 use_consistency_loss=ds.USE_CONSISTENCY_LOSS,
                 use_adversarial_loss=ds.USE_ADVERSARIAL_LOSS,
                 use_intensity_only=ds.USE_INTENSITY_ONLY,
                 smooth_adversarial_labels=ds.SMOOTH_ADVERSARIAL_LABELS,
                 batch_size=ds.DEFAULT_BATCH_SIZE,
                 n_epochs=ds.DEFAULT_N_EPOCHS,
                 learning_rate=ds.DEFAULT_LEARNING_RATE,
                 fine_tune_lr=ds.DEFAULT_FINE_TUNE_LR,
                 start_fine_tune=ds.DEFAULT_START_FINE_TUNE,
                 use_separable_conv=ds.USE_SEPARABLE_CONV,
                 conv_kernel_size=ds.DEFAULT_CONV_KERNEL_SIZE,
                 conv_strides=ds.DEFAULT_CONV_STRIDES,
                 conv_leaky_relu_alpha=ds.DEFAULT_CONV_LEAKY_RELU_ALPHA,
                 out_activation=ds.DEFAULT_OUT_ACTIVATION,
                 conv_dropout_rate=ds.DEFAULT_CONV_DROPOUT_RATE,
                 conv_batch_norm=ds.DEFAULT_CONV_BATCH_NORM,
                 conv_kernel_init=ds.DEFAULT_CONV_KERNEL_INIT,
                 d_loss=ds.DEFAULT_D_LOSS,
                 g_loss_l1_ratio=ds.DEFAULT_G_LOSS_L1_RATIO,
                 g_loss_l2_ratio=ds.DEFAULT_G_LOSS_L2_RATIO,
                 g_loss_ss_ratio=ds.DEFAULT_G_LOSS_SS_RATIO,
                 d_opt_method=ds.DEFAULT_D_OPT_METHOD,
                 g_opt_method=ds.DEFAULT_G_OPT_METHOD,
                 opt_grad_clip=ds.DEFAULT_OPT_GRAD_CLIP,
                 score_use_d=ds.DEFAULT_SCORE_USE_D,
                 score_method=ds.DEFAULT_SCORE_METHOD,
                 score_c_ratio=ds.DEFAULT_SCORE_C_RATIO,
                 score_p_ratio=ds.DEFAULT_SCORE_P_RATIO,
                 score_use_grid=ds.DEFAULT_SCORE_USE_GRID,
                 score_grid_size=ds.DEFAULT_SCORE_GRID_SIZE,
                 random_seed=ds.DEFAULT_RAND_SEED,
                 load_model_dir_path=None,
                 save_model_dir_path='',
                 result_dir_path=''):
        """
        Initializes hyper-parameters.
        :param use_tf_determinism:
        :param random_subset_training:
        :param new_training_scheme:
        :param reduce_subset_step:
        :param reduce_subset_factor:
        :param dataset:
        :param frame_diff_gap:
        :param min_det_score:
        :param n_frames_per_video:
        :param normalize_images:
        :param frames_step:
        :param estimations_path:
        :param extract_datatype:
        :param det_model_name:
        :param use_cross_domain:
        :param use_skip_connections:
        :param use_consistency_loss:
        :param use_adversarial_loss:
        :param use_intensity_only:
        :param smooth_adversarial_labels:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param fine_tune_lr:
        :param start_fine_tune:
        :param use_separable_conv:
        :param conv_kernel_size:
        :param conv_strides:
        :param conv_leaky_relu_alpha:
        :param out_activation:
        :param conv_dropout_rate:
        :param conv_batch_norm:
        :param conv_kernel_init:
        :param d_loss:
        :param g_loss_l1_ratio:
        :param g_loss_l2_ratio:
        :param g_loss_ss_ratio:
        :param d_opt_method:
        :param g_opt_method:
        :param opt_grad_clip:
        :param score_use_d:
        :param score_method:
        :param score_c_ratio:
        :param score_p_ratio:
        :param score_use_grid:
        :param score_grid_size:
        :param random_seed:
        :param load_model_dir_path:
        :param save_model_dir_path:
        :param result_dir_path:
        """
        # Configure TensorFlow
        configure_tf(use_tf_determinism)

        self.random_subset_training = random_subset_training

        self.new_training_scheme = new_training_scheme
        self.reduce_subset_step = reduce_subset_step
        self.reduce_subset_factor = reduce_subset_factor

        self.dataset = dataset
        self.frame_diff_gap = frame_diff_gap
        self.min_det_score = min_det_score
        self.n_frames_per_video = n_frames_per_video
        self.normalize_images = normalize_images
        self.frames_step = frames_step
        self.estimations_path = estimations_path
        self.extract_datatype = extract_datatype
        self.det_model_name = det_model_name
        self.rand_seed = random_seed

        # Get datatype class
        datatype_cls = util.get_class_from_name(util, self.extract_datatype)

        self.extract_data = datatype_cls(dataset=self.dataset,
                                         rand_seed=self.rand_seed,
                                         frame_diff_gap=self.frame_diff_gap,
                                         min_det_score=self.min_det_score,
                                         n_frames_per_video=self.n_frames_per_video,
                                         normalize_images=self.normalize_images,
                                         frames_step=self.frames_step,
                                         det_model_name=self.det_model_name,
                                         estimations_path=self.estimations_path)

        self.is_holistic = self.extract_data.det_model_name is None

        self.use_cross_domain = use_cross_domain
        self.use_skip_connections = use_skip_connections
        self.use_consistency_loss = use_consistency_loss
        self.use_adversarial_loss = use_adversarial_loss
        self.use_intensity_only = use_intensity_only

        self.score_use_d = score_use_d

        self.smooth_adversarial_labels = smooth_adversarial_labels

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.fine_tune_lr = fine_tune_lr
        self.start_fine_tune = start_fine_tune
        self.out_activation = out_activation
        self.d_loss = d_loss
        self.g_loss_l1_ratio = g_loss_l1_ratio
        self.g_loss_l2_ratio = g_loss_l2_ratio
        self.g_loss_ss_ratio = g_loss_ss_ratio
        self.d_opt_method = d_opt_method
        self.g_opt_method = g_opt_method
        self.opt_grad_clip = opt_grad_clip
        self.load_model_dir_path = load_model_dir_path
        self.save_model_dir_path = save_model_dir_path
        self.result_dir_path = result_dir_path

        self.conv_layout = Conv2DSetup(use_separable_conv=use_separable_conv,
                                       kernel_size=conv_kernel_size,
                                       strides=conv_strides,
                                       leaky_relu_alpha=conv_leaky_relu_alpha,
                                       dropout_rate=conv_dropout_rate,
                                       do_batch_normalize=conv_batch_norm,
                                       kernel_init=conv_kernel_init,
                                       rand_seed=random_seed)

        self.optimizer = Optimizer(self.learning_rate, self.opt_grad_clip)

        self.ano_scorer = QualityAssessmentScore(dist_name=score_method,
                                                 c_ratio=score_c_ratio,
                                                 p_ratio=score_p_ratio,
                                                 use_grid=score_use_grid,
                                                 grid_size=score_grid_size)

        self.input_shape = (self.extract_data.image_size, self.extract_data.image_size, 1)

        self.c_a_g = None
        self.p_a_g = None
        self.c_a_d = None
        self.p_a_d = None
        self.combined = None

        self.c_a_data_name = 'curr_appear_int'
        self.p_a_data_name = 'past_appear_int' if self.use_intensity_only else 'past_appear_gra'

        # Calculate output shape of D (PatchGAN)
        patch_size = int(self.extract_data.image_size / 2 ** 4)
        self.disc_patch = (patch_size, patch_size, 1)

        # Number of filters in the first layer of G and D
        self.conv_gf = 32
        self.conv_df = 32

        # Set the seed random numbers to obtain reproducible results
        reset_random_seeds(self.rand_seed)

        # SSIM loss function
        self.ssim_loss = DSSIMObjective()

        # It is useful to save encoders in order to reproduce the latent representation.
        self.c_a_g_name = 'curr_appear_g'
        self.p_a_g_name = 'past_appear_g'
        self.c_a_d_name = 'curr_appear_d'
        self.p_a_d_name = 'past_appear_d'

        # Save models to HDF5
        self.c_a_g_filename = self.c_a_g_name + '.h5'
        self.p_a_g_filename = self.p_a_g_name + '.h5'
        self.c_a_d_filename = self.c_a_d_name + '.h5'
        self.p_a_d_filename = self.p_a_d_name + '.h5'

        # Summary files
        self.c_a_g_summary_filename = self.c_a_g_name + '_summary.txt'
        self.p_a_g_summary_filename = self.p_a_g_name + '_summary.txt'
        self.c_a_d_summary_filename = self.c_a_d_name + '_summary.txt'
        self.p_a_d_summary_filename = self.p_a_d_name + '_summary.txt'

        if not os.path.exists(self.result_dir_path):
            os.makedirs(self.result_dir_path)

    def generator_loss(self, y_t, y_p):
        """
        Loss used in generators which combines l1, l2 and ssim losses.
        :param y_t:
        :param y_p:
        :return:
        """
        loss = self.g_loss_l1_ratio * mean_absolute_error(y_t, y_p)

        if self.g_loss_l2_ratio:
            loss += self.g_loss_l2_ratio * mean_squared_error(y_t, y_p)

        if self.g_loss_ss_ratio:
            loss += self.g_loss_ss_ratio * self.ssim_loss(y_t, y_p)

        return loss

    def build_model(self):
        """
        Builds and compiles all the models.
        :return:
        """
        if self.load_model_dir_path is None:
            if self.use_adversarial_loss:
                # Build and compile the discriminators
                self.c_a_d = self.build_discriminator()
                self.p_a_d = self.build_discriminator()

                self.c_a_d.name = self.c_a_d_name
                self.p_a_d.name = self.p_a_d_name

                self.c_a_d.compile(loss=self.d_loss,
                                   optimizer=self.optimizer.get_opt(self.d_opt_method),
                                   metrics=[])
                self.p_a_d.compile(loss=self.d_loss,
                                   optimizer=self.optimizer.get_opt(self.d_opt_method),
                                   metrics=[])

                # Print summary of the models
                print_model_summary(self.c_a_d, os.path.join(self.result_dir_path, self.c_a_d_summary_filename))
                print_model_summary(self.p_a_d, os.path.join(self.result_dir_path, self.p_a_d_summary_filename))

            # Build the generators
            self.c_a_g = self.build_generator()
            self.p_a_g = self.build_generator()

            self.c_a_g.name = self.c_a_g_name
            self.p_a_g.name = self.p_a_g_name
        else:
            # Load pretrained models
            self.load_models()

        # Input images from both domains
        inp_c_a = Input(shape=self.input_shape)
        inp_p_a = Input(shape=self.input_shape)

        # Inputs
        inp_model = [inp_c_a, inp_p_a]
        # Outputs
        out_model = []
        # Losses
        loss_model = []

        if self.use_cross_domain:
            # Translate images to the other domain
            gen_p_a = self.p_a_g(inp_c_a)
            gen_c_a = self.c_a_g(inp_p_a)
            # Store to outputs
            out_model.append(gen_p_a)
            out_model.append(gen_c_a)
            # Loss
            loss_model.append(self.generator_loss)
            loss_model.append(self.generator_loss)
        else:
            # Simply reconstruct
            gen_c_a = self.c_a_g(inp_c_a)
            gen_p_a = self.p_a_g(inp_p_a)
            # Store to outputs
            out_model.append(gen_c_a)
            out_model.append(gen_p_a)
            # Loss
            loss_model.append(self.generator_loss)
            loss_model.append(self.generator_loss)

        if self.use_cross_domain and self.use_consistency_loss:
            # Translate images back to original domain
            rec_c_a = self.c_a_g(gen_p_a)
            rec_p_a = self.p_a_g(gen_c_a)
            # Store to outputs
            out_model.append(rec_c_a)
            out_model.append(rec_p_a)
            # Loss
            loss_model.append(self.generator_loss)
            loss_model.append(self.generator_loss)

        if self.use_adversarial_loss:
            # For the combined model we will only train the generators
            self.c_a_d.trainable = False
            self.p_a_d.trainable = False
            # Discriminators determines validity of translated images
            val_c_a = self.c_a_d(gen_c_a)
            val_p_a = self.p_a_d(gen_p_a)
            # Store to outputs
            out_model.append(val_c_a)
            out_model.append(val_p_a)
            # Loss
            loss_model.append(self.d_loss)
            loss_model.append(self.d_loss)

        # Objectives
        # + Adversarial: Fool domain discriminators
        # + Translation: Minimize MAE between e.g. fake B and true B
        # + Cycle-consistency: Minimize MAE between reconstructed images and original
        self.combined = Model(inputs=inp_model, outputs=out_model)
        self.combined.compile(loss=loss_model, optimizer=self.optimizer.get_opt(self.g_opt_method), metrics=[])

        # Print summary of the models
        print_model_summary(self.c_a_g, os.path.join(self.result_dir_path, self.c_a_g_summary_filename))
        print_model_summary(self.p_a_g, os.path.join(self.result_dir_path, self.p_a_g_summary_filename))

    def close_session(self):
        """
        Clears all the sub models.
        :return:
        """
        kb.clear_session()
        self.combined = None
        self.c_a_d = None
        self.p_a_d = None
        self.c_a_g = None
        self.p_a_d = None

    def build_generator(self):
        """
        U-Net Generator
        :return:
        """
        # Image input
        d0 = Input(shape=self.input_shape)

        # Down-sampling
        d1 = self.conv_layout.conv_layer(d0, self.conv_gf, normalize=False)
        d2 = self.conv_layout.conv_layer(d1, self.conv_gf * 2)
        d3 = self.conv_layout.conv_layer(d2, self.conv_gf * 4)
        d4 = self.conv_layout.conv_layer(d3, self.conv_gf * 8)
        d5 = self.conv_layout.conv_layer(d4, self.conv_gf * 8)
        d6 = self.conv_layout.conv_layer(d5, self.conv_gf * 8)

        # Up-sampling
        if self.use_skip_connections:
            u1 = self.conv_layout.deconv_layer(d6, self.conv_gf * 8, d5)
            u2 = self.conv_layout.deconv_layer(u1, self.conv_gf * 8, d4)
            u3 = self.conv_layout.deconv_layer(u2, self.conv_gf * 4, d3)
            u4 = self.conv_layout.deconv_layer(u3, self.conv_gf * 2, d2)
            u5 = self.conv_layout.deconv_layer(u4, self.conv_gf, d1)
        else:
            u1 = self.conv_layout.deconv_layer(d6, self.conv_gf * 8)
            u2 = self.conv_layout.deconv_layer(u1, self.conv_gf * 8)
            u3 = self.conv_layout.deconv_layer(u2, self.conv_gf * 4)
            u4 = self.conv_layout.deconv_layer(u3, self.conv_gf * 2)
            u5 = self.conv_layout.deconv_layer(u4, self.conv_gf)

        output_img = Conv2DTranspose(1,
                                     kernel_size=self.conv_layout.kernel_size,
                                     strides=self.conv_layout.strides,
                                     padding='same',
                                     activation=self.out_activation)(u5)

        return Model(d0, output_img)

    def build_discriminator(self):
        """
        Builds discriminator.
        :return:
        """
        im = Input(shape=self.input_shape)

        d1 = self.conv_layout.conv_layer(im, self.conv_df, normalize=False)
        d2 = self.conv_layout.conv_layer(d1, self.conv_df * 2)
        d3 = self.conv_layout.conv_layer(d2, self.conv_df * 4)
        d4 = self.conv_layout.conv_layer(d3, self.conv_df * 8)

        validity = Conv2D(1,
                          kernel_size=self.conv_layout.kernel_size,
                          strides=1,
                          padding='same',
                          activation=self.out_activation)(d4)

        return Model(im, validity)

    def load_models(self):
        """
        Loads all the trained models.
        :return:
        """
        print('=====================================================================')
        print('Loading the trained models:')
        self.c_a_g = load_model(os.path.join(self.load_model_dir_path, self.c_a_g_filename),
                                custom_objects={'InstanceNormalization': InstanceNormalization})
        self.p_a_g = load_model(os.path.join(self.load_model_dir_path, self.p_a_g_filename),
                                custom_objects={'InstanceNormalization': InstanceNormalization})
        if os.path.isfile(os.path.join(self.load_model_dir_path, self.c_a_d_filename)):
            self.c_a_d = load_model(os.path.join(self.load_model_dir_path, self.c_a_d_filename),
                                    custom_objects={'InstanceNormalization': InstanceNormalization})
            self.p_a_d = load_model(os.path.join(self.load_model_dir_path, self.p_a_d_filename),
                                    custom_objects={'InstanceNormalization': InstanceNormalization})
        print('                                                                done!')

    def get_batch_idx(self, data_size):
        """
        Returns a randomly selected batch indexes.
        :param data_size:
        :return:
        """
        assert data_size > self.batch_size, 'Data size should be a lot bigger than the batch size.'
        idx = np.array(list(range(data_size)))
        # Create batch indexes
        np.random.shuffle(idx)
        good_n_samples = data_size - data_size % self.batch_size
        idx = idx[:good_n_samples]
        return np.array_split(idx, int(good_n_samples / self.batch_size))

    def expect_on_batch(self, data, batch_idx, real_label, output_predictions=False):
        """
        Returns the input and output requirements for training all the necessary models.
        :param data:
        :param batch_idx:
        :param real_label:
        :param output_predictions:
        :return:
        """
        # Output format
        expect_out = dict()
        # Inputs
        expect_out['inputs'] = {'c_a_data': data['data'][self.c_a_data_name][batch_idx],
                                'p_a_data': data['data'][self.p_a_data_name][batch_idx]}
        # Expected outputs
        expect_out['outputs'] = []
        # Produce generated data
        if self.use_cross_domain:
            if output_predictions:
                # Translate images to opposite domain
                expect_out['generated'] = {'p_a_data': self.p_a_g.predict(expect_out['inputs']['c_a_data']),
                                           'c_a_data': self.c_a_g.predict(expect_out['inputs']['p_a_data'])}
            # Expected outputs
            expect_out['outputs'].append(expect_out['inputs']['p_a_data'])
            expect_out['outputs'].append(expect_out['inputs']['c_a_data'])
            if self.use_consistency_loss:
                expect_out['outputs'].append(expect_out['inputs']['c_a_data'])
                expect_out['outputs'].append(expect_out['inputs']['p_a_data'])
        else:
            if output_predictions:
                # Reconstruct images
                expect_out['generated'] = {'c_a_data': self.c_a_g.predict(expect_out['inputs']['c_a_data']),
                                           'p_a_data': self.p_a_g.predict(expect_out['inputs']['p_a_data'])}
            # Expected outputs
            expect_out['outputs'].append(expect_out['inputs']['c_a_data'])
            expect_out['outputs'].append(expect_out['inputs']['p_a_data'])
        # In case of adversarial learning
        if self.use_adversarial_loss:
            expect_out['outputs'].append(real_label)
            expect_out['outputs'].append(real_label)
        return expect_out

    def train(self,
              save_model=True,
              plot_histories=True):
        """
        Fit the dae model.
        :param save_model:
        :param plot_histories:
        :return:
        """
        print('=====================================================================')
        print('Training the CNN models:')
        # Adversarial loss ground truths
        real = np.ones((self.batch_size,) + self.disc_patch)
        fake = np.zeros((self.batch_size,) + self.disc_patch)

        if self.smooth_adversarial_labels:
            # Use Label Smoothing for improving the training of the discriminator
            real_s = smooth_positive_labels(real)
        else:
            real_s = real

        all_d_train_loss = []
        all_g_train_loss = []
        all_c_a_d_lr = []
        all_p_a_d_lr = []
        all_combined_lr = []
        all_epochs = []
        all_train_sizes = []

        checkpoints_dir = os.path.join(ds.TEMP_PATH, f'checkpoints_{int(time.time())}')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        if self.random_subset_training:
            # Get train samples randomly selected
            train_data = self.extract_data.get_random_train_subset()
        else:
            # Get complete train samples
            train_data = self.extract_data.get_complete_train_set()

        for epoch in tqdm(range(1, self.n_epochs + 1), file=sys.stdout, desc='Training'):
            # Change the learning rate for other half of epochs
            if epoch == self.start_fine_tune:
                if self.use_adversarial_loss:
                    kb.set_value(self.c_a_d.optimizer.lr, self.fine_tune_lr)
                    kb.set_value(self.p_a_d.optimizer.lr, self.fine_tune_lr)
                kb.set_value(self.combined.optimizer.lr, self.fine_tune_lr)

            if self.new_training_scheme:
                if epoch > 1 and (epoch - 1) % self.reduce_subset_step == 0 \
                        and train_data['data_size'] > 2 * self.batch_size / self.reduce_subset_factor:
                    # Test on generators
                    test_losses = []
                    # Batch indexes
                    indexes = np.array(self.get_batch_idx(train_data['data_size']))
                    for bat_idx in tqdm(indexes, file=sys.stdout, desc=f'Testing on epoch {epoch:<3}'):
                        # Get expected data
                        exp_bat = self.expect_on_batch(train_data, bat_idx, real)
                        # Test the generators
                        g_loss = self.combined.test_on_batch(list(exp_bat['inputs'].values()), exp_bat['outputs'])
                        # Store the scores
                        test_losses.append(g_loss[0])
                    # Sort and take the data having the highest errors
                    train_data['data_size'] = int(train_data['data_size'] * self.reduce_subset_factor)
                    data_idx = indexes[np.argsort(-1 * np.array(test_losses))].flatten()
                    if len(data_idx) > train_data['data_size']:
                        data_idx = data_idx[:train_data['data_size']]
                    else:
                        train_data['data_size'] = len(data_idx)
                    train_data['data'][self.c_a_data_name] = train_data['data'][self.c_a_data_name][data_idx]
                    train_data['data'][self.p_a_data_name] = train_data['data'][self.p_a_data_name][data_idx]

            # Train models
            d_train_loss_list = []
            g_train_loss_list = []

            # Batch indexes
            indexes = np.array(self.get_batch_idx(train_data['data_size']))
            for bat_idx in tqdm(indexes, file=sys.stdout, desc=f'Training on epoch {epoch:<3}'):
                # Add more stochastic
                np.random.shuffle(bat_idx)
                # Get expected data
                exp_bat = self.expect_on_batch(train_data, bat_idx, real, output_predictions=self.use_adversarial_loss)

                if self.use_adversarial_loss:
                    # Train the discriminators (original images = real / translated = Fake)
                    d_c_loss_real = self.c_a_d.train_on_batch(exp_bat['inputs']['c_a_data'], real_s)
                    d_c_loss_fake = self.c_a_d.train_on_batch(exp_bat['generated']['c_a_data'], fake)
                    d_c_loss = 0.5 * np.add(d_c_loss_real, d_c_loss_fake)
                    # Past gradient discriminator loss
                    d_p_loss_real = self.p_a_d.train_on_batch(exp_bat['inputs']['p_a_data'], real_s)
                    d_p_loss_fake = self.p_a_d.train_on_batch(exp_bat['generated']['p_a_data'], fake)
                    d_p_loss = 0.5 * np.add(d_p_loss_real, d_p_loss_fake)
                    # Total discriminator loss
                    d_loss = 0.5 * np.add(d_c_loss, d_p_loss)
                    # Store the scores
                    d_train_loss_list.append(d_loss)

                # Train the generators
                g_loss = self.combined.train_on_batch(list(exp_bat['inputs'].values()), exp_bat['outputs'])
                # Store the scores
                g_train_loss_list.append(g_loss[0])

            # Take the average
            g_train_loss = float(np.mean(g_train_loss_list))
            # Get the learning rates - Just for test that ensures if it corresponds to the assigned lr
            combined_lr = float(kb.get_value(self.combined.optimizer.lr))
            # Store results
            all_g_train_loss.append(g_train_loss)
            all_combined_lr.append(combined_lr)
            all_epochs.append(epoch)
            all_train_sizes.append(train_data['data_size'])
            # Save models
            self.c_a_g.save(os.path.join(checkpoints_dir, f'{self.c_a_g_name}_epoch_{epoch}.h5'))
            self.p_a_g.save(os.path.join(checkpoints_dir, f'{self.p_a_g_name}_epoch_{epoch}.h5'))

            if self.use_adversarial_loss:
                # Take the average
                d_train_loss = float(np.mean(d_train_loss_list))
                # Get the learning rates - Just for test that ensures if it corresponds to the assigned lr
                c_a_d_lr = float(kb.get_value(self.c_a_d.optimizer.lr))
                p_a_d_lr = float(kb.get_value(self.p_a_d.optimizer.lr))
                # Store results
                all_d_train_loss.append(d_train_loss)
                all_c_a_d_lr.append(c_a_d_lr)
                all_p_a_d_lr.append(p_a_d_lr)
                # Save models
                self.c_a_d.save(os.path.join(checkpoints_dir, f'{self.c_a_d_name}_epoch_{epoch}.h5'))
                self.p_a_d.save(os.path.join(checkpoints_dir, f'{self.p_a_d_name}_epoch_{epoch}.h5'))
                # Plot the progress
                print(f'Epoch: {epoch:<3} -> train: [D loss: {d_train_loss:.6f}, G loss: {g_train_loss:.6f}], '
                      f'lr: [D: {c_a_d_lr:.3e}, G: {combined_lr:.3e}]')
            else:
                # Plot the progress
                print(f'Epoch: {epoch:<3} -> train: [G loss: {g_train_loss:.6f}], lr: [G: {combined_lr:.3e}]')
        print('                                                                done!')
        print('=====================================================================')

        if save_model:
            print('Saving models to disk:')
            if not os.path.exists(self.save_model_dir_path):
                os.makedirs(self.save_model_dir_path)
            # Save models
            self.c_a_g.save(os.path.join(self.save_model_dir_path, self.c_a_g_filename))
            self.p_a_g.save(os.path.join(self.save_model_dir_path, self.p_a_g_filename))
            if self.use_adversarial_loss:
                self.c_a_d.save(os.path.join(self.save_model_dir_path, self.c_a_d_filename))
                self.p_a_d.save(os.path.join(self.save_model_dir_path, self.p_a_d_filename))
            print('                                                                done!')
            print('=====================================================================')

        # Delete checkpoints dir
        shutil.rmtree(checkpoints_dir)

        if plot_histories:
            print('Plotting training histories:')
            # summarize loss
            fig = plt.figure()
            if self.use_adversarial_loss:
                plt.plot(all_epochs, all_d_train_loss, 'b--',
                         all_epochs, all_g_train_loss, 'g.-')
                plt.legend(['Discriminator', 'Generator'], loc='upper right')
            else:
                plt.plot(all_epochs, all_g_train_loss, 'g.-')
                plt.legend(['Generator'], loc='upper right')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            figure_name = 'model_loss_curves.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            # ======================================
            # summarize learning rates
            fig = plt.figure()
            if self.use_adversarial_loss:
                plt.plot(all_epochs, all_c_a_d_lr, 'b--',
                         all_epochs, all_p_a_d_lr, 'g.-',
                         all_epochs, all_combined_lr, 'k-')
                plt.legend(['c_a_int_d', 'p_a_gra_d', 'combined'], loc='upper right')
            else:
                plt.plot(all_epochs, all_combined_lr, 'k-')
                plt.legend(['combined'], loc='upper right')
            plt.ylabel('lr')
            plt.xlabel('epoch')
            figure_name = 'model_lr_curves.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            # ======================================
            # summarize train sizes
            fig = plt.figure()
            plt.plot(all_epochs, all_train_sizes, 'k-')
            plt.ylabel('size')
            plt.xlabel('epoch')
            figure_name = 'model_train_sizes.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            print('                                                                done!')
            print('=====================================================================')

    def predict_images(self, subset):
        """
        Predict using the trained models.
        :param subset:
        :return:
        """
        output = dict()
        if self.use_cross_domain:
            output[self.c_a_data_name] = self.c_a_g.predict(subset['data'][self.p_a_data_name])
            output[self.p_a_data_name] = self.p_a_g.predict(subset['data'][self.c_a_data_name])
        else:
            output[self.c_a_data_name] = self.c_a_g.predict(subset['data'][self.c_a_data_name])
            output[self.p_a_data_name] = self.p_a_g.predict(subset['data'][self.p_a_data_name])
        return output

    def test(self, subset_name, video_name):
        """
        Applies the trained models to produce reconstruction errors represented as anomaly scores.
        :param subset_name:
        :param video_name:
        :return:
        """
        # Get video samples
        data = self.extract_data.get_video_data(subset_name, video_name)

        # Predict using the trained models
        pred = self.predict_images(data)

        if self.score_use_d:
            assert self.c_a_d is not None and self.p_a_d is not None, 'The pretrained model is not GAN-based.'
            assert self.ano_scorer.dist_name != ds.SCORE_AUTO_NAME, 'D auto scorer is not currently supported.'
            # Take the prediction of discriminators
            pred_fake_c = self.c_a_d.predict(pred[self.c_a_data_name])
            pred_fake_p = self.p_a_d.predict(pred[self.p_a_data_name])
            real = np.ones(pred_fake_c.shape)
            # Compute the reconstruction errors
            ano_scores = np.array([self.ano_scorer.compute((c_r, c_g), (p_r, p_g))
                                   for c_r, c_g, p_r, p_g in zip(real[..., 0],
                                                                 pred_fake_c[..., 0],
                                                                 real[..., 0],
                                                                 pred_fake_p[..., 0])])
        else:
            if self.ano_scorer.dist_name == ds.SCORE_AUTO_NAME:
                # Extract random train subset
                tr_data = self.extract_data.get_random_train_subset()
                # Predict using the trained models
                tr_pred = self.predict_images(tr_data)
                # Get the variance of normalized scores for different quality measures
                tr_stats = dict()
                for m_name in tqdm(ds.SCORE_METHODS, desc='Finding the best scorer'):
                    # Change the scorer method name
                    self.ano_scorer.dist_name = m_name
                    # Compute the reconstruction errors
                    tr_ano_scores = np.array([self.ano_scorer.compute((c_r, c_g), (p_r, p_g))
                                              for c_r, c_g, p_r, p_g in zip(tr_data['data'][self.c_a_data_name][..., 0],
                                                                            tr_pred[self.c_a_data_name][..., 0],
                                                                            tr_data['data'][self.p_a_data_name][..., 0],
                                                                            tr_pred[self.p_a_data_name][..., 0])])
                    # Normalize the scores and compute the variance
                    tr_stats[m_name] = np.var((tr_ano_scores - tr_ano_scores.min()) / tr_ano_scores.ptp())
                # Get the scorer method that gives the highest variance
                self.ano_scorer.dist_name = max(tr_stats, key=tr_stats.get)
            # Compute the reconstruction errors
            ano_scores = np.array([self.ano_scorer.compute((c_r, c_g), (p_r, p_g))
                                   for c_r, c_g, p_r, p_g in zip(data['data'][self.c_a_data_name][..., 0],
                                                                 pred[self.c_a_data_name][..., 0],
                                                                 data['data'][self.p_a_data_name][..., 0],
                                                                 pred[self.p_a_data_name][..., 0])])

        # Remove the train det if necessary in order to save RAM
        if self.extract_data.subset_det[ds.TRAIN_DIRNAME] is not None:
            self.extract_data.subset_det[ds.TRAIN_DIRNAME] = None

        return util.return_score_dict(data, ano_scores)


class NextPredictionNet:
    """
    Our Next Prediction Network that can either predict the next appearance or the next motion image.
    """
    def __init__(self,
                 use_tf_determinism=ds.TENSORFLOW_DETERMINISM,
                 random_subset_training=ds.RANDOM_SUBSET_TRAINING,
                 change_random_subset=ds.CHANGE_RANDOM_SUBSET,
                 change_subset_step=ds.CHANGE_SUBSET_STEP,
                 new_training_scheme=ds.NEW_TRAINING_SCHEME,
                 first_n_epochs=ds.FIRST_N_EPOCHS,
                 reduce_subset_step=ds.REDUCE_SUBSET_STEP,
                 reduce_subset_factor=ds.REDUCE_SUBSET_FACTOR,
                 reduce_subset_min=ds.REDUCE_SUBSET_MIN,
                 reduce_easy_only=ds.REDUCE_EASY_ONLY,
                 dataset=ds.DEFAULT_DATASET,
                 frame_diff_gap=ds.FRAME_DIFF_GAP,
                 min_train_det_score=ds.DEFAULT_MIN_TRAIN_DET_SCORE,
                 min_test_det_score=ds.DEFAULT_MIN_TEST_DET_SCORE,
                 n_frames_per_video=ds.DEFAULT_N_FRAMES_PER_VIDEO,
                 normalize_images=ds.NORMALIZE_IMAGES,
                 frames_step=ds.FRAMES_STEP,
                 estimations_path=ds.ESTIMATIONS_PATH,
                 extract_datatype=ds.DEFAULT_DATATYPE_CLS,
                 det_model_name=ds.DEFAULT_DET_MODEL_NAME,
                 use_motion=ds.USE_MOTION,
                 holistic_method=ds.HOLISTIC_METHOD,
                 use_shared_encoders=ds.USE_SHARED_ENCODERS,
                 use_skip_connections=ds.USE_SKIP_CONNECTIONS,
                 use_skip_diff=ds.USE_SKIP_DIFF,
                 use_adversarial_loss=ds.USE_ADVERSARIAL_LOSS,
                 smooth_adversarial_labels=ds.SMOOTH_ADVERSARIAL_LABELS,
                 batch_size=ds.DEFAULT_BATCH_SIZE,
                 n_epochs=ds.DEFAULT_N_EPOCHS,
                 n_inner_epochs=ds.DEFAULT_N_INNER_EPOCHS,
                 d_learning_rate=ds.DEFAULT_D_LEARNING_RATE,
                 g_learning_rate=ds.DEFAULT_G_LEARNING_RATE,
                 ft_reduce_factor=ds.DEFAULT_FT_REDUCE_FACTOR,
                 start_fine_tune=ds.DEFAULT_START_FINE_TUNE,
                 use_separable_conv=ds.USE_SEPARABLE_CONV,
                 conv_kernel_size=ds.DEFAULT_CONV_KERNEL_SIZE,
                 conv_strides=ds.DEFAULT_CONV_STRIDES,
                 conv_leaky_relu_alpha=ds.DEFAULT_CONV_LEAKY_RELU_ALPHA,
                 out_activation=ds.DEFAULT_OUT_ACTIVATION,
                 conv_dropout_rate=ds.DEFAULT_CONV_DROPOUT_RATE,
                 conv_batch_norm=ds.DEFAULT_CONV_BATCH_NORM,
                 conv_kernel_init=ds.DEFAULT_CONV_KERNEL_INIT,
                 conv_skip_diff=ds.DEFAULT_CONV_SKIP_DIFF,
                 d_loss=ds.DEFAULT_D_LOSS,
                 d_loss_ratio=ds.DEFAULT_D_LOSS_RATIO,
                 g_loss_ratio=ds.DEFAULT_G_LOSS_RATIO,
                 g_loss_l1_ratio=ds.DEFAULT_G_LOSS_L1_RATIO,
                 g_loss_l2_ratio=ds.DEFAULT_G_LOSS_L2_RATIO,
                 g_loss_ss_ratio=ds.DEFAULT_G_LOSS_SS_RATIO,
                 g_loss_gd_ratio=ds.DEFAULT_G_LOSS_GD_RATIO,
                 d_opt_method=ds.DEFAULT_D_OPT_METHOD,
                 g_opt_method=ds.DEFAULT_G_OPT_METHOD,
                 opt_grad_clip=ds.DEFAULT_OPT_GRAD_CLIP,
                 score_use_d=ds.DEFAULT_SCORE_USE_D,
                 score_method=ds.DEFAULT_SCORE_METHOD,
                 score_use_grid=ds.DEFAULT_SCORE_USE_GRID,
                 score_grid_size=ds.DEFAULT_SCORE_GRID_SIZE,
                 random_seed=ds.DEFAULT_RAND_SEED,
                 load_model_dir_path=None,
                 save_model_dir_path='',
                 result_dir_path=''):
        """
        Initializes hyper-parameters.
        :param use_tf_determinism:
        :param random_subset_training:
        :param change_random_subset:
        :param change_subset_step:
        :param new_training_scheme:
        :param first_n_epochs:
        :param reduce_subset_step:
        :param reduce_subset_factor:
        :param reduce_subset_min:
        :param reduce_easy_only:
        :param dataset:
        :param frame_diff_gap:
        :param min_train_det_score:
        :param min_test_det_score:
        :param n_frames_per_video:
        :param normalize_images:
        :param frames_step:
        :param estimations_path:
        :param extract_datatype:
        :param det_model_name:
        :param use_motion:
        :param holistic_method:
        :param use_shared_encoders:
        :param use_skip_connections:
        :param use_skip_diff:
        :param use_adversarial_loss:
        :param smooth_adversarial_labels:
        :param batch_size:
        :param n_epochs:
        :param n_inner_epochs:
        :param d_learning_rate:
        :param g_learning_rate:
        :param ft_reduce_factor:
        :param start_fine_tune:
        :param use_separable_conv:
        :param conv_kernel_size:
        :param conv_strides:
        :param conv_leaky_relu_alpha:
        :param out_activation:
        :param conv_dropout_rate:
        :param conv_batch_norm:
        :param conv_kernel_init:
        :param conv_skip_diff:
        :param d_loss:
        :param d_loss_ratio:
        :param g_loss_ratio:
        :param g_loss_l1_ratio:
        :param g_loss_l2_ratio:
        :param g_loss_ss_ratio:
        :param g_loss_gd_ratio:
        :param d_opt_method:
        :param g_opt_method:
        :param opt_grad_clip:
        :param score_use_d:
        :param score_method:
        :param score_use_grid:
        :param score_grid_size:
        :param random_seed:
        :param load_model_dir_path:
        :param save_model_dir_path:
        :param result_dir_path:
        """
        # Configure TensorFlow
        configure_tf(use_tf_determinism)

        self.random_subset_training = random_subset_training
        self.change_random_subset = change_random_subset
        self.change_subset_step = change_subset_step

        self.new_training_scheme = new_training_scheme
        self.first_n_epochs = first_n_epochs
        self.reduce_subset_step = reduce_subset_step
        self.reduce_subset_factor = reduce_subset_factor
        self.reduce_subset_min = reduce_subset_min
        self.reduce_easy_only = reduce_easy_only

        self.dataset = dataset
        self.frame_diff_gap = frame_diff_gap
        self.min_train_det_score = min_train_det_score
        self.min_test_det_score = min_test_det_score
        self.n_frames_per_video = n_frames_per_video
        self.normalize_images = normalize_images
        self.frames_step = frames_step
        self.estimations_path = estimations_path
        self.extract_datatype = extract_datatype
        self.det_model_name = det_model_name
        self.use_motion = use_motion
        self.holistic_method = holistic_method
        self.rand_seed = random_seed

        # Get datatype class
        datatype_cls = util.get_class_from_name(util, self.extract_datatype)

        self.extract_data = datatype_cls(dataset=self.dataset,
                                         frame_diff_gap=self.frame_diff_gap,
                                         min_train_det_score=self.min_train_det_score,
                                         min_test_det_score=self.min_test_det_score,
                                         n_frames_per_video=self.n_frames_per_video,
                                         normalize_images=self.normalize_images,
                                         frames_step=self.frames_step,
                                         det_model_name=self.det_model_name,
                                         use_motion=self.use_motion,
                                         rand_seed=self.rand_seed,
                                         estimations_path=self.estimations_path)

        self.use_shared_encoders = use_shared_encoders
        self.use_skip_connections = use_skip_connections
        self.use_skip_diff = use_skip_diff
        self.use_adversarial_loss = use_adversarial_loss

        self.score_use_d = score_use_d

        self.smooth_adversarial_labels = smooth_adversarial_labels

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_inner_epochs = n_inner_epochs
        self.d_learning_rate = d_learning_rate
        self.g_learning_rate = g_learning_rate
        self.ft_reduce_factor = ft_reduce_factor
        self.start_fine_tune = start_fine_tune
        self.out_activation = out_activation
        self.d_loss = d_loss
        self.d_loss_ratio = d_loss_ratio
        self.g_loss_ratio = g_loss_ratio
        self.g_loss_l1_ratio = g_loss_l1_ratio
        self.g_loss_l2_ratio = g_loss_l2_ratio
        self.g_loss_ss_ratio = g_loss_ss_ratio
        self.g_loss_gd_ratio = g_loss_gd_ratio
        self.d_opt_method = d_opt_method
        self.g_opt_method = g_opt_method
        self.opt_grad_clip = opt_grad_clip
        self.load_model_dir_path = load_model_dir_path
        self.save_model_dir_path = save_model_dir_path
        self.result_dir_path = result_dir_path

        self.conv_layout = Conv2DSetup(use_separable_conv=use_separable_conv,
                                       kernel_size=conv_kernel_size,
                                       strides=conv_strides,
                                       leaky_relu_alpha=conv_leaky_relu_alpha,
                                       dropout_rate=conv_dropout_rate,
                                       do_batch_normalize=conv_batch_norm,
                                       kernel_init=conv_kernel_init,
                                       skip_diff=conv_skip_diff,
                                       rand_seed=random_seed)

        self.g_optimizer = Optimizer(self.g_learning_rate, self.opt_grad_clip)
        self.d_optimizer = Optimizer(self.d_learning_rate, self.opt_grad_clip)

        self.ano_scorer = PredictionQualityAssessment(dist_name=score_method,
                                                      use_grid=score_use_grid,
                                                      grid_size=score_grid_size)

        if self.holistic_method:
            self.image_size = self.extract_data.frame_size
        else:
            self.image_size = self.extract_data.roi_size

        self.input_shape = (self.image_size, self.image_size, 1)

        self.gen_model = None
        self.dis_model = None
        self.combined = None

        self.curr_data_name = 'curr'
        self.past_data_name = 'past'
        self.next_data_name = 'next'

        # Calculate output shape of D (PatchGAN)
        patch_size = int(self.image_size / 2 ** 4)
        self.disc_patch = (patch_size, patch_size, 1)

        # Number of filters in the first layer of G and D
        self.conv_gf = int(self.image_size / 2)
        self.conv_df = int(self.image_size / 2)

        # Set the seed random numbers to obtain reproducible results
        reset_random_seeds(self.rand_seed)

        # SSIM loss function
        self.ssim_loss = DSSIMObjective()

        # It is useful to save encoders in order to reproduce the latent representation.
        self.gen_model_name = 'gen_model'
        self.dis_model_name = 'dis_model'

        # Save models to HDF5
        self.gen_model_filename = self.gen_model_name + '.h5'
        self.dis_model_filename = self.dis_model_name + '.h5'

        # Summary files
        self.gen_model_summary_filename = self.gen_model_name + '_summary.txt'
        self.dis_model_summary_filename = self.dis_model_name + '_summary.txt'

        if not os.path.exists(self.result_dir_path):
            os.makedirs(self.result_dir_path)

    def generator_loss(self, y_t, y_p):
        """
        Loss used in generators which combines l1, l2 and ssim losses.
        :param y_t:
        :param y_p:
        :return:
        """
        loss = self.g_loss_l1_ratio * mean_absolute_error(y_t, y_p)

        if self.g_loss_l2_ratio:
            loss += self.g_loss_l2_ratio * mean_squared_error(y_t, y_p)

        if self.g_loss_ss_ratio:
            loss += self.g_loss_ss_ratio * self.ssim_loss(y_t, y_p)

        if self.g_loss_gd_ratio:
            y_t_dx, y_t_dy = tf.image.image_gradients(y_t)
            y_p_dx, y_p_dy = tf.image.image_gradients(y_p)
            loss += self.g_loss_gd_ratio * (mean_squared_error(y_t_dx, y_p_dx) + mean_squared_error(y_t_dy, y_p_dy))

        return loss

    def build_model(self):
        """
        Builds and compiles all the models.
        :return:
        """
        if self.load_model_dir_path is None:
            if self.use_adversarial_loss:
                # Build and compile the discriminator
                self.dis_model = self.build_discriminator()

                self.dis_model.name = self.dis_model_name

                self.dis_model.compile(loss=self.d_loss,
                                       optimizer=self.d_optimizer.get_opt(self.d_opt_method),
                                       metrics=[])

                # Print summary of the models
                print_model_summary(self.dis_model, os.path.join(self.result_dir_path, self.dis_model_summary_filename))

            # Build the generators
            self.gen_model = self.build_generator()

            self.gen_model.name = self.gen_model_name

            self.gen_model.compile(loss=self.generator_loss,
                                   optimizer=self.g_optimizer.get_opt(self.g_opt_method),
                                   metrics=[])
        else:
            # Load pretrained models
            self.load_models()

        # Input images from both domains
        inp_past = Input(shape=self.input_shape)
        inp_curr = Input(shape=self.input_shape)

        # Inputs
        inp_model = [inp_past, inp_curr]
        # Outputs
        out_model = []
        # Losses
        loss_model = []
        # Loss weights
        loss_ratios = []

        # Predict the next image
        gen_next = self.gen_model(inp_model)
        # Store to outputs
        out_model.append(gen_next)
        # Loss
        loss_model.append(self.generator_loss)
        loss_ratios.append(self.g_loss_ratio)

        if self.use_adversarial_loss:
            # For the combined model we will only train the generator
            self.dis_model.trainable = False
            # Discriminator determines validity of predicted next image
            val_next = self.dis_model(gen_next)
            # Store to outputs
            out_model.append(val_next)
            # Loss
            loss_model.append(self.d_loss)
            loss_ratios.append(self.d_loss_ratio)

        # Objectives
        # + Adversarial: Fool domain discriminators
        # + Translation: Minimize MAE between e.g. fake B and true B
        # + Cycle-consistency: Minimize MAE between reconstructed images and original
        self.combined = Model(inputs=inp_model, outputs=out_model)
        self.combined.compile(loss=loss_model,
                              loss_weights=loss_ratios,
                              optimizer=self.g_optimizer.get_opt(self.g_opt_method),
                              metrics=[])

        # Print summary of the models
        print_model_summary(self.gen_model, os.path.join(self.result_dir_path, self.gen_model_summary_filename))

    def close_session(self):
        """
        Clears all the sub models.
        :return:
        """
        kb.clear_session()
        self.combined = None
        self.dis_model = None
        self.gen_model = None

    def build_generator(self):
        """
        U-Net Generator
        :return:
        """
        # Image inputs
        d0_1 = Input(shape=self.input_shape)
        d0_2 = Input(shape=self.input_shape)

        if self.use_shared_encoders:
            conv_d1 = self.conv_layout.get_conv_2d(self.conv_gf)
            conv_d2 = self.conv_layout.get_conv_2d(self.conv_gf * 2)
            conv_d3 = self.conv_layout.get_conv_2d(self.conv_gf * 4)
            conv_d4 = self.conv_layout.get_conv_2d(self.conv_gf * 8)
            conv_d5 = self.conv_layout.get_conv_2d(self.conv_gf * 8)
            conv_d6 = self.conv_layout.get_conv_2d(self.conv_gf * 8)

            # Encoder 1: Down-sampling
            d1_1 = self.conv_layout.conv_layer(conv_d1(d0_1), normalize=False)
            d2_1 = self.conv_layout.conv_layer(conv_d2(d1_1))
            d3_1 = self.conv_layout.conv_layer(conv_d3(d2_1))
            d4_1 = self.conv_layout.conv_layer(conv_d4(d3_1))
            d5_1 = self.conv_layout.conv_layer(conv_d5(d4_1))
            d6_1 = self.conv_layout.conv_layer(conv_d6(d5_1))

            # Encoder 2: Down-sampling
            d1_2 = self.conv_layout.conv_layer(conv_d1(d0_2), normalize=False)
            d2_2 = self.conv_layout.conv_layer(conv_d2(d1_2))
            d3_2 = self.conv_layout.conv_layer(conv_d3(d2_2))
            d4_2 = self.conv_layout.conv_layer(conv_d4(d3_2))
            d5_2 = self.conv_layout.conv_layer(conv_d5(d4_2))
            d6_2 = self.conv_layout.conv_layer(conv_d6(d5_2))
        else:
            # Encoder 1: Down-sampling
            d1_1 = self.conv_layout.conv_layer(d0_1, self.conv_gf, normalize=False)
            d2_1 = self.conv_layout.conv_layer(d1_1, self.conv_gf * 2)
            d3_1 = self.conv_layout.conv_layer(d2_1, self.conv_gf * 4)
            d4_1 = self.conv_layout.conv_layer(d3_1, self.conv_gf * 8)
            d5_1 = self.conv_layout.conv_layer(d4_1, self.conv_gf * 8)
            d6_1 = self.conv_layout.conv_layer(d5_1, self.conv_gf * 8)

            # Encoder 2: Down-sampling
            d1_2 = self.conv_layout.conv_layer(d0_2, self.conv_gf, normalize=False)
            d2_2 = self.conv_layout.conv_layer(d1_2, self.conv_gf * 2)
            d3_2 = self.conv_layout.conv_layer(d2_2, self.conv_gf * 4)
            d4_2 = self.conv_layout.conv_layer(d3_2, self.conv_gf * 8)
            d5_2 = self.conv_layout.conv_layer(d4_2, self.conv_gf * 8)
            d6_2 = self.conv_layout.conv_layer(d5_2, self.conv_gf * 8)

        # CFV Fusion
        if self.use_skip_diff:
            d6 = self.conv_layout.diff_layer([d6_1, d6_2])
        else:
            d6 = Concatenate()([d6_1, d6_2])

        # Up-sampling
        if self.use_skip_connections:
            if self.use_skip_diff:
                d5 = [self.conv_layout.diff_layer([d5_1, d5_2])]
                d4 = [self.conv_layout.diff_layer([d4_1, d4_2])]
                d3 = [self.conv_layout.diff_layer([d3_1, d3_2])]
                d2 = [self.conv_layout.diff_layer([d2_1, d2_2])]
                d1 = [self.conv_layout.diff_layer([d1_1, d1_2])]
            else:
                d5 = [d5_1, d5_2]
                d4 = [d4_1, d4_2]
                d3 = [d3_1, d3_2]
                d2 = [d2_1, d2_2]
                d1 = [d1_1, d1_2]
            u1 = self.conv_layout.deconv_layer(d6, self.conv_gf * 8, d5)
            u2 = self.conv_layout.deconv_layer(u1, self.conv_gf * 8, d4)
            u3 = self.conv_layout.deconv_layer(u2, self.conv_gf * 4, d3)
            u4 = self.conv_layout.deconv_layer(u3, self.conv_gf * 2, d2)
            u5 = self.conv_layout.deconv_layer(u4, self.conv_gf, d1)
        else:
            u1 = self.conv_layout.deconv_layer(d6, self.conv_gf * 8)
            u2 = self.conv_layout.deconv_layer(u1, self.conv_gf * 8)
            u3 = self.conv_layout.deconv_layer(u2, self.conv_gf * 4)
            u4 = self.conv_layout.deconv_layer(u3, self.conv_gf * 2)
            u5 = self.conv_layout.deconv_layer(u4, self.conv_gf)

        output_img = Conv2DTranspose(1,
                                     kernel_size=self.conv_layout.kernel_size,
                                     strides=self.conv_layout.strides,
                                     padding='same',
                                     activation=self.out_activation)(u5)

        return Model([d0_1, d0_2], output_img)

    def build_discriminator(self):
        """
        Builds discriminator.
        :return:
        """
        im = Input(shape=self.input_shape)

        d1 = self.conv_layout.conv_layer(im, self.conv_df, normalize=False, add_dropout=True)
        d2 = self.conv_layout.conv_layer(d1, self.conv_df * 2, add_dropout=True)
        d3 = self.conv_layout.conv_layer(d2, self.conv_df * 4, add_dropout=True)
        d4 = self.conv_layout.conv_layer(d3, self.conv_df * 8, add_dropout=True)

        validity = Conv2D(1,
                          kernel_size=self.conv_layout.kernel_size,
                          strides=1,
                          padding='same',
                          activation=self.out_activation)(d4)

        return Model(im, validity)

    def load_models(self):
        """
        Loads all the trained models.
        :return:
        """
        print('=====================================================================')
        print('Loading the trained model:')
        self.gen_model = load_model(os.path.join(self.load_model_dir_path, self.gen_model_filename),
                                    custom_objects={'InstanceNormalization': InstanceNormalization,
                                                    'generator_loss': self.generator_loss})
        if os.path.isfile(os.path.join(self.load_model_dir_path, self.dis_model_filename)):
            self.dis_model = load_model(os.path.join(self.load_model_dir_path, self.dis_model_filename),
                                        custom_objects={'InstanceNormalization': InstanceNormalization})
        print('                                                                done!')

    def get_data(self,
                 subset_name,
                 video_name=None,
                 rand_select=False):
        """
        Gets the appropriate subset.
        :param subset_name:
        :param video_name:
        :param rand_select:
        :return:
        """
        if self.holistic_method:
            # Get train samples randomly selected
            return self.extract_data.get_subset(subset_name, video_name, rand_select)
        else:
            # Get complete train samples
            return self.extract_data.get_roi_subset(subset_name, video_name, rand_select)

    def get_batch_idx(self, data_size):
        """
        Returns a randomly selected batch indexes.
        :param data_size:
        :return:
        """
        assert data_size > self.batch_size, 'Data size should be a lot bigger than the batch size.'
        idx = np.array(list(range(data_size)))
        # Create batch indexes
        np.random.shuffle(idx)
        good_n_samples = data_size - data_size % self.batch_size
        idx = idx[:good_n_samples]
        return np.array_split(idx, int(good_n_samples / self.batch_size))

    def get_inner_epoch_idx(self, batch_idx_size):
        """
        Returns a randomly selected inner epoch indexes.
        :param batch_idx_size:
        :return:
        """
        if batch_idx_size > self.n_inner_epochs:
            n_inner_epochs = self.n_inner_epochs
        else:
            n_inner_epochs = batch_idx_size
        idx = np.array(list(range(batch_idx_size)))
        # Create inner epoch indexes
        np.random.shuffle(idx)
        good_n_idx = batch_idx_size - batch_idx_size % n_inner_epochs
        inner_idx = np.array_split(idx[:good_n_idx], int(good_n_idx / n_inner_epochs))
        if len(idx[good_n_idx:]):
            inner_idx += [idx[good_n_idx:]]
        return inner_idx

    def expect_on_batch(self, data, batch_idx, real_label=None, output_predictions=False):
        """
        Returns the input and output requirements for training all the necessary models.
        :param data:
        :param batch_idx:
        :param real_label:
        :param output_predictions:
        :return:
        """
        # Output format
        expect_out = dict()
        # Inputs
        expect_out['inp'] = {self.past_data_name: data['data'][self.past_data_name][batch_idx],
                             self.curr_data_name: data['data'][self.curr_data_name][batch_idx]}
        # Expected outputs
        expect_out['out'] = {}
        # Produce generated data
        if output_predictions:
            # Predict the next image
            expect_out['gen'] = {self.next_data_name: self.gen_model.predict(list(expect_out['inp'].values()))}
        # Expected outputs
        expect_out['out'][self.next_data_name] = data['data'][self.next_data_name][batch_idx]
        # In case of adversarial learning
        if self.use_adversarial_loss and real_label is not None:
            expect_out['out'][self.dis_model_name] = real_label
        return expect_out

    def train(self,
              save_model=True,
              plot_histories=True):
        """
        Fit the dae model.
        :param save_model:
        :param plot_histories:
        :return:
        """
        print('=====================================================================')
        print('Training the CNN models:')
        # Adversarial loss ground truths
        real = np.ones((self.batch_size,) + self.disc_patch)
        fake = np.zeros((self.batch_size,) + self.disc_patch)

        if self.smooth_adversarial_labels:
            # Use Label Smoothing for improving the training of the discriminator
            real_s = smooth_positive_labels(real)
        else:
            real_s = real

        all_d_train_loss = []
        all_g_train_loss = []
        all_dis_model_lr = []
        all_combined_lr = []
        all_epochs = []
        all_train_sizes = []

        checkpoints_dir = os.path.join(ds.TEMP_PATH, f'checkpoints_{int(time.time())}')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # Fit the generator for some epochs
        if self.new_training_scheme and not self.random_subset_training:
            if self.first_n_epochs:
                # Get the random subset train set
                train_data = self.get_data(ds.TRAIN_DIRNAME, rand_select=True)
                # Train G only
                self.gen_model.fit([train_data['data'][self.past_data_name], train_data['data'][self.curr_data_name]],
                                   train_data['data'][self.next_data_name],
                                   batch_size=self.batch_size,
                                   epochs=self.first_n_epochs)
            # Get the complete train set
            train_data = self.get_data(ds.TRAIN_DIRNAME)
        else:
            # Get the train set
            train_data = self.get_data(ds.TRAIN_DIRNAME, rand_select=self.random_subset_training)

        for epoch in tqdm(range(1, self.n_epochs + 1), file=sys.stdout, desc='Training', ascii=True):
            # Change the learning rate for other half of epochs
            if epoch == self.start_fine_tune:
                if self.use_adversarial_loss:
                    kb.set_value(self.dis_model.optimizer.lr, self.d_learning_rate * self.ft_reduce_factor)
                kb.set_value(self.combined.optimizer.lr, self.g_learning_rate * self.ft_reduce_factor)
            elif self.load_model_dir_path and epoch < self.start_fine_tune:
                continue

            if self.change_random_subset:
                if epoch > 1 and (epoch - 1) % self.change_subset_step == 0:
                    train_data = self.get_data(ds.TRAIN_DIRNAME, rand_select=True)

            if self.new_training_scheme:
                if epoch > 1 and (epoch - 1) % self.reduce_subset_step == 0 \
                        and train_data['data_size'] > self.reduce_subset_min * self.batch_size:
                    # Test on generators
                    test_losses = []
                    # Store the previous data size
                    data_size = train_data['data_size']
                    # Batch indexes
                    indexes = np.array(self.get_batch_idx(train_data['data_size']))
                    for bat_idx in tqdm(indexes, file=sys.stdout, desc=f'Testing on epoch {epoch:<3}', leave=False):
                        # Get expected data
                        exp_bat = self.expect_on_batch(train_data, bat_idx, real)
                        # Test the generators
                        g_loss = self.combined.test_on_batch(list(exp_bat['inp'].values()),
                                                             list(exp_bat['out'].values()))
                        # Store the scores
                        if isinstance(g_loss, (list, tuple, set, np.ndarray)):
                            g_loss = g_loss[0]
                        test_losses.append(g_loss)
                    # Sort and take the data having the highest errors
                    train_data['data_size'] = int(data_size * self.reduce_subset_factor)
                    data_idx = indexes[np.argsort(-1 * np.array(test_losses))].flatten()
                    # Reduce data size
                    if len(data_idx) > train_data['data_size']:
                        if self.reduce_easy_only:
                            data_idx = data_idx[:train_data['data_size']]
                        else:
                            d_start = int((data_size - train_data['data_size']) / 2)
                            data_idx = data_idx[d_start:train_data['data_size'] + d_start]
                    else:
                        train_data['data_size'] = len(data_idx)
                    train_data['data'][self.past_data_name] = train_data['data'][self.past_data_name][data_idx]
                    train_data['data'][self.curr_data_name] = train_data['data'][self.curr_data_name][data_idx]
                    train_data['data'][self.next_data_name] = train_data['data'][self.next_data_name][data_idx]

            # Train models
            d_train_loss_list = []
            g_train_loss_list = []

            # Batch indexes
            indexes = np.array(self.get_batch_idx(train_data['data_size']))
            # Inner epoch indexes
            inner_indexes = self.get_inner_epoch_idx(len(indexes))

            for inner_idx in tqdm(inner_indexes, file=sys.stdout, desc=f'Training on epoch {epoch:<3}', leave=False):
                # Get inner batch indexes
                if self.use_adversarial_loss:
                    for bat_idx in indexes[inner_idx]:
                        # Add more stochastic
                        np.random.shuffle(bat_idx)
                        # Get expected data
                        exp_bat = self.expect_on_batch(train_data, bat_idx, real, output_predictions=True)
                        # Train the discriminator (original images = real / translated = Fake)
                        d_loss_real = self.dis_model.train_on_batch(exp_bat['out'][self.next_data_name], real_s)
                        d_loss_fake = self.dis_model.train_on_batch(exp_bat['gen'][self.next_data_name], fake)
                        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                        # Store the scores
                        if isinstance(d_loss, (list, tuple, set, np.ndarray)):
                            d_loss = d_loss[0]
                        d_train_loss_list.append(d_loss)

                for bat_idx in indexes[inner_idx]:
                    # Add more stochastic
                    np.random.shuffle(bat_idx)
                    # Get expected data
                    exp_bat = self.expect_on_batch(train_data, bat_idx, real)
                    # Train the generators
                    g_loss = self.combined.train_on_batch(list(exp_bat['inp'].values()),
                                                          list(exp_bat['out'].values()))
                    # Store the scores
                    if isinstance(g_loss, (list, tuple, set, np.ndarray)):
                        g_loss = g_loss[0]
                    g_train_loss_list.append(g_loss)

            # Take the average
            g_train_loss = float(np.mean(g_train_loss_list))
            # Get the learning rates - Just for test that ensures if it corresponds to the assigned lr
            combined_lr = float(kb.get_value(self.combined.optimizer.lr))
            # Store results
            all_g_train_loss.append(g_train_loss)
            all_combined_lr.append(combined_lr)
            all_epochs.append(epoch)
            all_train_sizes.append(train_data['data_size'])
            # Save models
            self.gen_model.save(os.path.join(checkpoints_dir, f'{self.gen_model_name}_epoch_{epoch}.h5'))

            if self.use_adversarial_loss:
                # Take the average
                d_train_loss = float(np.mean(d_train_loss_list))
                # Get the learning rates - Just for test that ensures if it corresponds to the assigned lr
                dis_model_lr = float(kb.get_value(self.dis_model.optimizer.lr))
                # Store results
                all_d_train_loss.append(d_train_loss)
                all_dis_model_lr.append(dis_model_lr)
                # Save models
                self.dis_model.save(os.path.join(checkpoints_dir, f'{self.dis_model_name}_epoch_{epoch}.h5'))
                # Plot the progress
                print(f'\nEpoch: {epoch:<3} -> train: [D loss: {d_train_loss:.6f}, G loss: {g_train_loss:.6f}], '
                      f'lr: [D: {dis_model_lr:.3e}, G: {combined_lr:.3e}]')
            else:
                # Plot the progress
                print(f'\nEpoch: {epoch:<3} -> train: [G loss: {g_train_loss:.6f}], lr: [G: {combined_lr:.3e}]')
        print('                                                                done!')
        print('=====================================================================')

        if save_model:
            print('Saving models to disk:')
            if not os.path.exists(self.save_model_dir_path):
                os.makedirs(self.save_model_dir_path)
            # Save models
            self.gen_model.save(os.path.join(self.save_model_dir_path, self.gen_model_filename))
            if self.use_adversarial_loss:
                self.dis_model.save(os.path.join(self.save_model_dir_path, self.dis_model_filename))
            print('                                                                done!')
            print('=====================================================================')

        # Delete checkpoints dir
        shutil.rmtree(checkpoints_dir)

        if plot_histories:
            print('Plotting training histories:')
            # summarize loss
            fig = plt.figure()
            if self.use_adversarial_loss:
                plt.plot(all_epochs, all_d_train_loss, 'b--',
                         all_epochs, all_g_train_loss, 'g.-')
                plt.legend(['Discriminator', 'Generator'], loc='upper right')
            else:
                plt.plot(all_epochs, all_g_train_loss, 'g.-')
                plt.legend(['Generator'], loc='upper right')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            figure_name = 'model_loss_curves.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            # ======================================
            # summarize learning rates
            fig = plt.figure()
            if self.use_adversarial_loss:
                plt.plot(all_epochs, all_dis_model_lr, 'b--',
                         all_epochs, all_combined_lr, 'g.-')
                plt.legend(['dis_model', 'combined'], loc='upper right')
            else:
                plt.plot(all_epochs, all_combined_lr, 'k-')
                plt.legend(['combined'], loc='upper right')
            plt.ylabel('lr')
            plt.xlabel('epoch')
            figure_name = 'model_lr_curves.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            # ======================================
            # summarize train sizes
            fig = plt.figure()
            plt.plot(all_epochs, all_train_sizes, 'k-')
            plt.ylabel('size')
            plt.xlabel('epoch')
            figure_name = 'model_train_sizes.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            print('                                                                done!')
            print('=====================================================================')

    def predict_images(self, subset):
        """
        Predict using the trained models.
        :param subset:
        :return:
        """
        output = dict()
        output[self.next_data_name] = self.gen_model.predict([subset['data'][self.past_data_name],
                                                              subset['data'][self.curr_data_name]])
        return output

    def test(self, subset_name, video_name):
        """
        Applies the trained models to produce reconstruction errors represented as anomaly scores.
        :param subset_name:
        :param video_name:
        :return:
        """
        # Get video samples
        data = self.get_data(subset_name, video_name)

        # Predict using the trained models
        pred = self.predict_images(data)

        if self.score_use_d:
            assert self.dis_model is not None, 'The pretrained model is not GAN-based.'
            assert self.ano_scorer.dist_name != ds.SCORE_AUTO_NAME, 'D auto scorer is not currently supported.'
            # Take the prediction of discriminators
            pred_real = self.dis_model.predict(data['data'][self.next_data_name])
            pred_fake = self.dis_model.predict(pred[self.next_data_name])
            # Compute the reconstruction errors
            ano_scores = np.array([np.sum(np.square(r - f))
                                   for r, f in zip(pred_real[..., 0], pred_fake[..., 0])])
        else:
            if self.ano_scorer.dist_name == ds.SCORE_AUTO_NAME:
                # Extract random train subset
                tr_data = self.get_data(ds.TRAIN_DIRNAME, rand_select=True)
                # Predict using the trained models
                tr_pred = self.predict_images(tr_data)
                # Get the variance of normalized scores for different quality measures
                tr_stats = dict()
                for m_name in tqdm(ds.SCORE_METHODS, desc='Finding the best scorer'):
                    # Change the scorer method name
                    self.ano_scorer.dist_name = m_name
                    # Compute the reconstruction errors
                    tr_ano_scores = np.array([self.ano_scorer.compute(r, g)
                                              for r, g in zip(tr_data['data'][self.next_data_name][..., 0],
                                                              tr_pred[self.next_data_name][..., 0])])
                    # Normalize the scores and compute the variance
                    tr_stats[m_name] = np.var((tr_ano_scores - tr_ano_scores.min()) / tr_ano_scores.ptp())
                # Get the scorer method that gives the highest variance
                self.ano_scorer.dist_name = max(tr_stats, key=tr_stats.get)
            # Compute the reconstruction errors
            ano_scores = np.array([self.ano_scorer.compute(r, g)
                                   for r, g in zip(data['data'][self.next_data_name][..., 0],
                                                   pred[self.next_data_name][..., 0])])

        # Remove the train det if necessary in order to save RAM
        if self.extract_data.subset_det[ds.TRAIN_DIRNAME] is not None:
            self.extract_data.subset_det[ds.TRAIN_DIRNAME] = None

        return util.return_score_dict(data, ano_scores)


class ObjectCentricCAEs:
    """
    Our implementation of the object-centric method proposed in Ionescu et al. 2019.
    """
    def __init__(self,
                 use_tf_determinism=ds.TENSORFLOW_DETERMINISM,
                 random_subset_training=ds.RANDOM_SUBSET_TRAINING,
                 dataset=ds.DEFAULT_DATASET,
                 frame_diff_gap=ds.FRAME_DIFF_GAP,
                 min_train_det_score=ds.DEFAULT_MIN_TRAIN_DET_SCORE,
                 min_test_det_score=ds.DEFAULT_MIN_TEST_DET_SCORE,
                 n_frames_per_video=ds.DEFAULT_N_FRAMES_PER_VIDEO,
                 normalize_images=ds.NORMALIZE_IMAGES,
                 frames_step=ds.FRAMES_STEP,
                 estimations_path=ds.ESTIMATIONS_PATH,
                 extract_datatype=ds.DEFAULT_DATATYPE_CLS,
                 det_model_name=ds.DEFAULT_DET_MODEL_NAME,
                 use_motion=ds.USE_MOTION,
                 holistic_method=ds.HOLISTIC_METHOD,
                 batch_size=ds.DEFAULT_BATCH_SIZE,
                 n_epochs=ds.DEFAULT_N_EPOCHS,
                 learning_rate=ds.DEFAULT_LEARNING_RATE,
                 ft_reduce_factor=ds.DEFAULT_FT_REDUCE_FACTOR,
                 start_fine_tune=ds.DEFAULT_START_FINE_TUNE,
                 g_loss_l1_ratio=ds.DEFAULT_G_LOSS_L1_RATIO,
                 g_loss_l2_ratio=ds.DEFAULT_G_LOSS_L2_RATIO,
                 g_loss_ss_ratio=ds.DEFAULT_G_LOSS_SS_RATIO,
                 g_loss_gd_ratio=ds.DEFAULT_G_LOSS_GD_RATIO,
                 g_opt_method=ds.DEFAULT_G_OPT_METHOD,
                 opt_grad_clip=ds.DEFAULT_OPT_GRAD_CLIP,
                 use_svm_models=ds.USE_SVM_MODELS,
                 n_svm_models=ds.DEFAULT_N_SVM_MODELS,
                 random_seed=ds.DEFAULT_RAND_SEED,
                 load_model_dir_path=None,
                 save_model_dir_path='',
                 result_dir_path=''):
        """
        Initializes hyper-parameters.
        :param use_tf_determinism:
        :param random_subset_training:
        :param dataset:
        :param frame_diff_gap:
        :param min_train_det_score:
        :param min_test_det_score:
        :param n_frames_per_video:
        :param normalize_images:
        :param frames_step:
        :param estimations_path:
        :param extract_datatype:
        :param det_model_name:
        :param use_motion:
        :param holistic_method:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param ft_reduce_factor:
        :param start_fine_tune:
        :param g_loss_l1_ratio:
        :param g_loss_l2_ratio:
        :param g_loss_ss_ratio:
        :param g_loss_gd_ratio:
        :param g_opt_method:
        :param opt_grad_clip:
        :param use_svm_models:
        :param n_svm_models:
        :param random_seed:
        :param load_model_dir_path:
        :param save_model_dir_path:
        :param result_dir_path:
        """
        # Configure TensorFlow
        configure_tf(use_tf_determinism)

        self.random_subset_training = random_subset_training

        self.dataset = dataset
        self.frame_diff_gap = frame_diff_gap
        self.min_train_det_score = min_train_det_score
        self.min_test_det_score = min_test_det_score
        self.n_frames_per_video = n_frames_per_video
        self.normalize_images = normalize_images
        self.frames_step = frames_step
        self.estimations_path = estimations_path
        self.extract_datatype = extract_datatype
        self.det_model_name = det_model_name
        self.use_motion = use_motion
        self.holistic_method = holistic_method
        self.rand_seed = random_seed

        self.use_svm_models = use_svm_models
        self.n_svm_models = n_svm_models

        # Get datatype class
        datatype_cls = util.get_class_from_name(util, self.extract_datatype)

        self.extract_data = datatype_cls(dataset=self.dataset,
                                         frame_diff_gap=self.frame_diff_gap,
                                         min_train_det_score=self.min_train_det_score,
                                         min_test_det_score=self.min_test_det_score,
                                         n_frames_per_video=self.n_frames_per_video,
                                         normalize_images=self.normalize_images,
                                         frames_step=self.frames_step,
                                         det_model_name=self.det_model_name,
                                         use_motion=self.use_motion,
                                         rand_seed=self.rand_seed,
                                         estimations_path=self.estimations_path)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.ft_reduce_factor = ft_reduce_factor
        self.start_fine_tune = start_fine_tune
        self.g_loss_l1_ratio = g_loss_l1_ratio
        self.g_loss_l2_ratio = g_loss_l2_ratio
        self.g_loss_ss_ratio = g_loss_ss_ratio
        self.g_loss_gd_ratio = g_loss_gd_ratio
        self.g_opt_method = g_opt_method
        self.opt_grad_clip = opt_grad_clip
        self.load_model_dir_path = load_model_dir_path
        self.save_model_dir_path = save_model_dir_path
        self.result_dir_path = result_dir_path

        self.optimizer = Optimizer(self.learning_rate, self.opt_grad_clip)

        if self.holistic_method:
            self.image_size = self.extract_data.frame_size
        else:
            self.image_size = self.extract_data.roi_size

        self.input_shape = (self.image_size, self.image_size, 1)
        self.latent_shape = None

        self.enc_model = None
        self.dec_model = None
        self.combined = None
        self.svm_models = None
        self.emb_scaler = None

        self.curr_data_name = 'curr'
        self.past_data_name = 'past'
        self.next_data_name = 'next'

        # Number of filters in the first layer of G and D
        self.conv_gf = int(self.image_size / 2)

        # Set the seed random numbers to obtain reproducible results
        reset_random_seeds(self.rand_seed)

        # SSIM loss function
        self.ssim_loss = DSSIMObjective()

        # It is useful to save encoders in order to reproduce the latent representation.
        self.enc_model_name = 'enc_model'
        self.dec_model_name = 'dec_model'
        self.emb_scaler_name = 'emb_scaler'

        # Save models to HDF5
        self.enc_model_filename = self.enc_model_name + '.h5'
        self.dec_model_filename = self.dec_model_name + '.h5'

        self.svm_models_filenames = ['svm_model_{}.sav'.format(i) for i in range(self.n_svm_models)]
        self.emb_scaler_filename = self.emb_scaler_name + '.pkl'

        # Summary files
        self.enc_model_summary_filename = self.enc_model_name + '_summary.txt'
        self.dec_model_summary_filename = self.dec_model_name + '_summary.txt'

        if not os.path.exists(self.result_dir_path):
            os.makedirs(self.result_dir_path)

    def generator_loss(self, y_t, y_p):
        """
        Loss used in generators which combines l1, l2 and ssim losses.
        :param y_t:
        :param y_p:
        :return:
        """
        loss = self.g_loss_l1_ratio * mean_absolute_error(y_t, y_p)

        if self.g_loss_l2_ratio:
            loss += self.g_loss_l2_ratio * mean_squared_error(y_t, y_p)

        if self.g_loss_ss_ratio:
            loss += self.g_loss_ss_ratio * self.ssim_loss(y_t, y_p)

        if self.g_loss_gd_ratio:
            y_t_dx, y_t_dy = tf.image.image_gradients(y_t)
            y_p_dx, y_p_dy = tf.image.image_gradients(y_p)
            loss += self.g_loss_gd_ratio * (mean_squared_error(y_t_dx, y_p_dx) + mean_squared_error(y_t_dy, y_p_dy))

        return loss

    def build_model(self):
        """
        Builds and compiles all the models.
        :return:
        """
        if self.load_model_dir_path is None:
            # Build the encoders
            past_enc = self.build_encoder()
            curr_enc = self.build_encoder()
            next_enc = self.build_encoder()

            # Get the latent shape
            self.latent_shape = curr_enc.layers[-1].output_shape[1:]

            # Build the decoders
            past_dec = self.build_decoder()
            curr_dec = self.build_decoder()
            next_dec = self.build_decoder()

            # Build encoders model
            inp_past = Input(shape=self.input_shape)
            inp_curr = Input(shape=self.input_shape)
            inp_next = Input(shape=self.input_shape)

            self.enc_model = Model(inputs=[inp_past, inp_curr, inp_next],
                                   outputs=[past_enc(inp_past), curr_enc(inp_curr), next_enc(inp_next)])

            # Build the decoders model
            emb_past = Input(shape=self.latent_shape)
            emb_curr = Input(shape=self.latent_shape)
            emb_next = Input(shape=self.latent_shape)

            self.dec_model = Model(inputs=[emb_past, emb_curr, emb_next],
                                   outputs=[past_dec(emb_past), curr_dec(emb_curr), next_dec(emb_next)])
        else:
            # Load pretrained models
            self.load_models(load_svm_models=False)

        # Build the generator model
        inp_past = Input(shape=self.input_shape)
        inp_curr = Input(shape=self.input_shape)
        inp_next = Input(shape=self.input_shape)

        self.combined = Model(inputs=[inp_past, inp_curr, inp_next],
                              outputs=self.dec_model(self.enc_model([inp_past, inp_curr, inp_next])))
        self.combined.compile(loss=self.generator_loss,
                              optimizer=self.optimizer.get_opt(self.g_opt_method),
                              metrics=[])

        # Print summary of the models
        print_model_summary(self.enc_model, os.path.join(self.result_dir_path, self.enc_model_summary_filename))
        print_model_summary(self.dec_model, os.path.join(self.result_dir_path, self.dec_model_summary_filename))

    def close_session(self):
        """
        Clears all the sub models.
        :return:
        """
        kb.clear_session()
        self.enc_model = None
        self.dec_model = None
        self.combined = None
        self.svm_models = None

    def build_encoder(self):
        """
        Builds encoder.
        :return:
        """
        # Image inputs
        _input = Input(shape=self.input_shape)

        _x = Conv2D(32, (3, 3), activation='relu', padding='same')(_input)
        _x = MaxPooling2D((2, 2), padding='same')(_x)
        _x = Conv2D(32, (3, 3), activation='relu', padding='same')(_x)
        _x = MaxPooling2D((2, 2), padding='same')(_x)
        _x = Conv2D(16, (3, 3), activation='relu', padding='same')(_x)
        _encoded = MaxPooling2D((2, 2), padding='same')(_x)

        return Model(_input, _encoded)

    def build_decoder(self):
        """
        Builds decoder.
        :return:
        """
        _latent_input = Input(shape=self.latent_shape)

        _x = Conv2D(16, (3, 3), activation='relu', padding='same')(_latent_input)
        _x = UpSampling2D((2, 2))(_x)
        _x = Conv2D(32, (3, 3), activation='relu', padding='same')(_x)
        _x = UpSampling2D((2, 2))(_x)
        _x = Conv2D(32, (3, 3), activation='relu', padding='same')(_x)
        _x = UpSampling2D((2, 2))(_x)
        _decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(_x)

        return Model(_latent_input, _decoded)

    def load_models(self, load_svm_models=True):
        """
        Loads all the trained models.
        :param load_svm_models:
        :return:
        """
        print('=====================================================================')
        print('Loading the trained model:')
        self.enc_model = load_model(os.path.join(self.load_model_dir_path, self.enc_model_filename),
                                    custom_objects={'InstanceNormalization': InstanceNormalization,
                                                    'generator_loss': self.generator_loss})
        self.dec_model = load_model(os.path.join(self.load_model_dir_path, self.dec_model_filename),
                                    custom_objects={'InstanceNormalization': InstanceNormalization,
                                                    'generator_loss': self.generator_loss})
        if self.use_svm_models and load_svm_models:
            # Check if there's any svm model file, otherwise train svm models
            svm_models_filenames = [f for f in os.listdir(self.load_model_dir_path)
                                    if os.path.isfile(os.path.join(self.load_model_dir_path, f)) and
                                    os.path.splitext(f)[1] == '.sav']

            if len(svm_models_filenames):
                self.svm_models_filenames = svm_models_filenames
                self.n_svm_models = len(svm_models_filenames)
                self.svm_models = [joblib.load(os.path.join(self.load_model_dir_path, f)) for f in svm_models_filenames]
                self.emb_scaler = joblib.load(os.path.join(self.load_model_dir_path, self.emb_scaler_filename))
            else:
                self.train_svm_models()
        print('                                                                done!')

    def get_data(self,
                 subset_name,
                 video_name=None,
                 rand_select=False):
        """
        Gets the appropriate subset.
        :param subset_name:
        :param video_name:
        :param rand_select:
        :return:
        """
        if self.holistic_method:
            # Get train samples randomly selected
            return self.extract_data.get_subset(subset_name, video_name, rand_select)
        else:
            # Get complete train samples
            return self.extract_data.get_roi_subset(subset_name, video_name, rand_select)

    def get_batch_idx(self, data_size):
        """
        Returns a randomly selected batch indexes.
        :param data_size:
        :return:
        """
        assert data_size > self.batch_size, 'Data size should be a lot bigger than the batch size.'
        idx = np.array(list(range(data_size)))
        # Create batch indexes
        np.random.shuffle(idx)
        good_n_samples = data_size - data_size % self.batch_size
        idx = idx[:good_n_samples]
        return np.array_split(idx, int(good_n_samples / self.batch_size))

    def expect_on_batch(self, data, batch_idx):
        """
        Returns the input and output requirements for training all the necessary models.
        :param data:
        :param batch_idx:
        :return:
        """
        # Output format
        expect_out = dict()
        # Inputs
        expect_out['inp'] = {self.past_data_name: data['data'][self.past_data_name][batch_idx],
                             self.curr_data_name: data['data'][self.curr_data_name][batch_idx],
                             self.next_data_name: data['data'][self.next_data_name][batch_idx]}
        # Expected outputs
        expect_out['out'] = expect_out['inp']
        return expect_out

    def train(self,
              save_model=True,
              plot_histories=True):
        """
        Fit the dae model.
        :param save_model:
        :param plot_histories:
        :return:
        """
        print('=====================================================================')
        print('Training the CNN models:')
        all_g_train_loss = []
        all_combined_lr = []
        all_epochs = []
        all_train_sizes = []

        checkpoints_dir = os.path.join(ds.TEMP_PATH, f'checkpoints_{int(time.time())}')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # Get the train set
        train_data = self.get_data(ds.TRAIN_DIRNAME, rand_select=self.random_subset_training)

        for epoch in tqdm(range(1, self.n_epochs + 1), file=sys.stdout, desc='Training', ascii=True):
            # Change the learning rate for other half of epochs
            if epoch == self.start_fine_tune:
                kb.set_value(self.combined.optimizer.lr, self.learning_rate * self.ft_reduce_factor)
            elif self.load_model_dir_path and epoch < self.start_fine_tune:
                continue

            # Train models
            g_train_loss_list = []

            # Batch indexes
            indexes = np.array(self.get_batch_idx(train_data['data_size']))

            for bat_idx in tqdm(indexes, file=sys.stdout, desc=f'Training on epoch {epoch:<3}', leave=False):
                # Add more stochastic
                np.random.shuffle(bat_idx)
                # Get expected data
                exp_bat = self.expect_on_batch(train_data, bat_idx)
                # Train the generators
                g_loss = self.combined.train_on_batch(list(exp_bat['inp'].values()),
                                                      list(exp_bat['out'].values()))
                # Store the scores
                if isinstance(g_loss, (list, tuple, set, np.ndarray)):
                    g_loss = g_loss[0]
                g_train_loss_list.append(g_loss)

            # Take the average
            g_train_loss = float(np.mean(g_train_loss_list))
            # Get the learning rates - Just for test that ensures if it corresponds to the assigned lr
            combined_lr = float(kb.get_value(self.combined.optimizer.lr))
            # Store results
            all_g_train_loss.append(g_train_loss)
            all_combined_lr.append(combined_lr)
            all_epochs.append(epoch)
            all_train_sizes.append(train_data['data_size'])
            # Save models
            self.enc_model.save(os.path.join(checkpoints_dir, f'{self.enc_model_name}_epoch_{epoch}.h5'))
            self.dec_model.save(os.path.join(checkpoints_dir, f'{self.dec_model_name}_epoch_{epoch}.h5'))

            # Plot the progress
            print(f'\nEpoch: {epoch:<3} -> train: [G loss: {g_train_loss:.6f}], lr: [G: {combined_lr:.3e}]')
        print('                                                                done!')
        print('=====================================================================')

        if save_model:
            print('Saving models to disk:')
            if not os.path.exists(self.save_model_dir_path):
                os.makedirs(self.save_model_dir_path)
            # Save models
            self.enc_model.save(os.path.join(self.save_model_dir_path, self.enc_model_filename))
            self.dec_model.save(os.path.join(self.save_model_dir_path, self.dec_model_filename))
            print('                                                                done!')
            print('=====================================================================')

        # Delete checkpoints dir
        shutil.rmtree(checkpoints_dir)

        if self.use_svm_models:
            # Train SVM models
            self.train_svm_models(train_data)

        if plot_histories:
            print('Plotting training histories:')
            # summarize loss
            fig = plt.figure()
            plt.plot(all_epochs, all_g_train_loss, 'g.-')
            plt.legend(['Generator'], loc='upper right')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            figure_name = 'model_loss_curves.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            # ======================================
            # summarize learning rates
            fig = plt.figure()
            plt.plot(all_epochs, all_combined_lr, 'k-')
            plt.legend(['combined'], loc='upper right')
            plt.ylabel('lr')
            plt.xlabel('epoch')
            figure_name = 'model_lr_curves.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            # ======================================
            # summarize train sizes
            fig = plt.figure()
            plt.plot(all_epochs, all_train_sizes, 'k-')
            plt.ylabel('size')
            plt.xlabel('epoch')
            figure_name = 'model_train_sizes.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            print('                                                                done!')
            print('=====================================================================')

    def get_cfv_data(self,
                     data,
                     fit_scaler=False):
        """
        Gets the compressed feature vector of the latent space of the given data.
        :param data:
        :param fit_scaler
        :return:
        """
        # Get the encoded data
        enc_data = self.predict_embeddings(data)
        # Flatten and concatenate latent data
        emb_data = np.hstack((enc_data[self.past_data_name].reshape(len(enc_data[self.past_data_name]), -1),
                              enc_data[self.curr_data_name].reshape(len(enc_data[self.curr_data_name]), -1),
                              enc_data[self.next_data_name].reshape(len(enc_data[self.next_data_name]), -1)))
        # Normalize data
        if fit_scaler:
            self.emb_scaler = preprocessing.StandardScaler().fit(emb_data)
        return self.emb_scaler.transform(emb_data)

    def train_svm_models(self,
                         train_data=None,
                         save_model=True,
                         viz_clusters=True):
        """
        Trains all the SVM models which will be used to form one-versus-rest classification scheme.
        :param train_data:
        :param save_model:
        :param viz_clusters:
        :return:
        """
        print('=====================================================================')
        print('Training the SVM models:')
        # Get train data if necessary
        if train_data is None:
            if self.random_subset_training:
                # Get train samples randomly selected
                train_data = self.extract_data.get_random_train_subset()
            else:
                # Get complete train samples
                train_data = self.extract_data.get_complete_train_set()
        # Get the cfv data
        train_cfv_data = self.get_cfv_data(train_data, fit_scaler=True)
        # Cluster the latent data
        cfv_clusters = KMeans(n_clusters=self.n_svm_models).fit(train_cfv_data)
        # Visualize the training latent clusters if necessary
        if viz_clusters:
            reduced_cfv_clusters = TSNE(n_components=2).fit_transform(train_cfv_data)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Principal Component 1', fontsize=15)
            ax.set_ylabel('Principal Component 2', fontsize=15)
            clusters_targets = [str(n).zfill(2) for n in range(1, self.n_svm_models + 1)]
            clusters_colors = ['m', 'k', 'g', 'c', 'y', 'r', 'b', 'orange', 'olive', 'indigo']
            for i in range(self.n_svm_models):
                _reduced = reduced_cfv_clusters[cfv_clusters.labels_ == i]
                ax.scatter(_reduced[:, 0], _reduced[:, 1], c=clusters_colors[i], s=50)
            ax.legend(clusters_targets)
            ax.grid()
            figure_name = 'model_latent_clusters.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
        # Fit the SVM models
        self.svm_models = []
        for i in tqdm(range(self.n_svm_models), file=sys.stdout, desc='Training SVMs'):
            svm_clf = SVC(gamma='auto')
            svm_clf.fit(train_cfv_data, np.where(cfv_clusters.labels_ == i, 0, 1))
            self.svm_models.append(svm_clf)
        print('                                                                done!')
        print('=====================================================================')
        if save_model:
            print('Saving models to disk:')
            joblib.dump(self.emb_scaler, os.path.join(self.save_model_dir_path, self.emb_scaler_filename))
            for i, model_filename in enumerate(self.svm_models_filenames):
                joblib.dump(self.svm_models[i], os.path.join(self.save_model_dir_path, model_filename))
            print('                                                                done!')
            print('=====================================================================')

    def predict_embeddings(self, subset):
        """
        Predict using the trained models.
        :param subset:
        :return:
        """
        output = dict()
        out = self.enc_model.predict([subset['data'][self.past_data_name],
                                      subset['data'][self.curr_data_name],
                                      subset['data'][self.next_data_name]])
        output[self.past_data_name] = out[0]
        output[self.curr_data_name] = out[1]
        output[self.next_data_name] = out[2]
        return output

    def predict_images(self, subset):
        """
        Predict using the trained models.
        :param subset:
        :return:
        """
        output = dict()
        out = self.combined.predict([subset['data'][self.past_data_name],
                                     subset['data'][self.curr_data_name],
                                     subset['data'][self.next_data_name]])
        output[self.past_data_name] = out[0]
        output[self.curr_data_name] = out[1]
        output[self.next_data_name] = out[2]
        return output

    def test(self, subset_name, video_name):
        """
        Applies the trained models to produce reconstruction errors represented as anomaly scores.
        :param subset_name:
        :param video_name:
        :return:
        """
        # Get video samples
        data = self.get_data(subset_name, video_name)

        if self.use_svm_models:
            # Get cfv data
            cfv_data = self.get_cfv_data(data)
            # Apply trained SVM
            svm_scores = np.array([m.predict(cfv_data).flatten() for m in self.svm_models])
            # Compute one-versus-rest SVM anomaly scores
            ano_scores = np.array([np.min(svm_scores[:, i]) for i in range(svm_scores.shape[1])])
        else:
            ano_scores = None

        # Remove the train det if necessary in order to save RAM
        if self.extract_data.subset_det[ds.TRAIN_DIRNAME] is not None:
            self.extract_data.subset_det[ds.TRAIN_DIRNAME] = None

        return util.return_score_dict(data, ano_scores)


class ContrastiveModelV1:
    """
    First simple version of our proposed contrastive method.
    """
    def __init__(self,
                 use_tf_determinism=ds.TENSORFLOW_DETERMINISM,
                 random_subset_training=ds.RANDOM_SUBSET_TRAINING,
                 dataset=ds.DEFAULT_DATASET,
                 frame_diff_gap=ds.FRAME_DIFF_GAP,
                 min_train_det_score=ds.DEFAULT_MIN_TRAIN_DET_SCORE,
                 min_test_det_score=ds.DEFAULT_MIN_TEST_DET_SCORE,
                 n_frames_per_video=ds.DEFAULT_N_FRAMES_PER_VIDEO,
                 normalize_images=ds.NORMALIZE_IMAGES,
                 frames_step=ds.FRAMES_STEP,
                 estimations_path=ds.ESTIMATIONS_PATH,
                 extract_datatype=ds.DEFAULT_DATATYPE_CLS,
                 det_model_name=ds.DEFAULT_DET_MODEL_NAME,
                 use_motion=ds.USE_MOTION,
                 holistic_method=ds.HOLISTIC_METHOD,
                 batch_size=ds.DEFAULT_BATCH_SIZE,
                 n_epochs=ds.DEFAULT_N_EPOCHS,
                 learning_rate=ds.DEFAULT_LEARNING_RATE,
                 ft_reduce_factor=ds.DEFAULT_FT_REDUCE_FACTOR,
                 start_fine_tune=ds.DEFAULT_START_FINE_TUNE,
                 use_separable_conv=ds.USE_SEPARABLE_CONV,
                 conv_kernel_size=ds.DEFAULT_CONV_KERNEL_SIZE,
                 conv_strides=ds.DEFAULT_CONV_STRIDES,
                 conv_leaky_relu_alpha=ds.DEFAULT_CONV_LEAKY_RELU_ALPHA,
                 out_activation=ds.DEFAULT_OUT_ACTIVATION,
                 conv_dropout_rate=ds.DEFAULT_CONV_DROPOUT_RATE,
                 conv_batch_norm=ds.DEFAULT_CONV_BATCH_NORM,
                 conv_kernel_init=ds.DEFAULT_CONV_KERNEL_INIT,
                 g_loss_l1_ratio=ds.DEFAULT_G_LOSS_L1_RATIO,
                 g_loss_l2_ratio=ds.DEFAULT_G_LOSS_L2_RATIO,
                 g_loss_ss_ratio=ds.DEFAULT_G_LOSS_SS_RATIO,
                 g_loss_gd_ratio=ds.DEFAULT_G_LOSS_GD_RATIO,
                 g_opt_method=ds.DEFAULT_G_OPT_METHOD,
                 opt_grad_clip=ds.DEFAULT_OPT_GRAD_CLIP,
                 use_svm_models=ds.USE_SVM_MODELS,
                 n_svm_models=ds.DEFAULT_N_SVM_MODELS,
                 random_seed=ds.DEFAULT_RAND_SEED,
                 load_model_dir_path=None,
                 save_model_dir_path='',
                 result_dir_path=''):
        """
        Initializes hyper-parameters.
        :param use_tf_determinism:
        :param random_subset_training:
        :param dataset:
        :param frame_diff_gap:
        :param min_train_det_score:
        :param min_test_det_score:
        :param n_frames_per_video:
        :param normalize_images:
        :param frames_step:
        :param estimations_path:
        :param extract_datatype:
        :param det_model_name:
        :param use_motion:
        :param holistic_method:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param ft_reduce_factor:
        :param start_fine_tune:
        :param use_separable_conv:
        :param conv_kernel_size:
        :param conv_strides:
        :param conv_leaky_relu_alpha:
        :param out_activation:
        :param conv_dropout_rate:
        :param conv_batch_norm:
        :param conv_kernel_init:
        :param g_loss_l1_ratio:
        :param g_loss_l2_ratio:
        :param g_loss_ss_ratio:
        :param g_loss_gd_ratio:
        :param g_opt_method:
        :param opt_grad_clip:
        :param use_svm_models:
        :param n_svm_models:
        :param random_seed:
        :param load_model_dir_path:
        :param save_model_dir_path:
        :param result_dir_path:
        """
        # Configure TensorFlow
        configure_tf(use_tf_determinism)

        self.random_subset_training = random_subset_training

        self.dataset = dataset
        self.frame_diff_gap = frame_diff_gap
        self.min_train_det_score = min_train_det_score
        self.min_test_det_score = min_test_det_score
        self.n_frames_per_video = n_frames_per_video
        self.normalize_images = normalize_images
        self.frames_step = frames_step
        self.estimations_path = estimations_path
        self.extract_datatype = extract_datatype
        self.det_model_name = det_model_name
        self.use_motion = use_motion
        self.holistic_method = holistic_method
        self.rand_seed = random_seed

        self.use_svm_models = use_svm_models
        self.n_svm_models = n_svm_models

        # Get datatype class
        datatype_cls = util.get_class_from_name(util, self.extract_datatype)

        self.extract_data = datatype_cls(dataset=self.dataset,
                                         frame_diff_gap=self.frame_diff_gap,
                                         min_train_det_score=self.min_train_det_score,
                                         min_test_det_score=self.min_test_det_score,
                                         n_frames_per_video=self.n_frames_per_video,
                                         normalize_images=self.normalize_images,
                                         frames_step=self.frames_step,
                                         det_model_name=self.det_model_name,
                                         use_motion=self.use_motion,
                                         rand_seed=self.rand_seed,
                                         estimations_path=self.estimations_path)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.ft_reduce_factor = ft_reduce_factor
        self.start_fine_tune = start_fine_tune
        self.out_activation = out_activation
        self.g_loss_l1_ratio = g_loss_l1_ratio
        self.g_loss_l2_ratio = g_loss_l2_ratio
        self.g_loss_ss_ratio = g_loss_ss_ratio
        self.g_loss_gd_ratio = g_loss_gd_ratio
        self.g_opt_method = g_opt_method
        self.opt_grad_clip = opt_grad_clip
        self.load_model_dir_path = load_model_dir_path
        self.save_model_dir_path = save_model_dir_path
        self.result_dir_path = result_dir_path

        self.conv_layout = Conv2DSetup(use_separable_conv=use_separable_conv,
                                       kernel_size=conv_kernel_size,
                                       strides=conv_strides,
                                       leaky_relu_alpha=conv_leaky_relu_alpha,
                                       dropout_rate=conv_dropout_rate,
                                       do_batch_normalize=conv_batch_norm,
                                       kernel_init=conv_kernel_init,
                                       rand_seed=random_seed)

        self.optimizer = Optimizer(self.learning_rate, self.opt_grad_clip)

        if self.holistic_method:
            self.image_size = self.extract_data.frame_size
        else:
            self.image_size = self.extract_data.roi_size

        self.input_shape = (self.image_size, self.image_size, 1)
        self.latent_shape = None

        self.enc_model = None
        self.dec_model = None
        self.combined = None
        self.svm_models = None

        self.curr_data_name = 'curr'
        self.past_data_name = 'past'
        self.next_data_name = 'next'

        # Number of filters in the first layer of G and D
        self.conv_gf = int(self.image_size / 2)

        # Set the seed random numbers to obtain reproducible results
        reset_random_seeds(self.rand_seed)

        # SSIM loss function
        self.ssim_loss = DSSIMObjective()

        # It is useful to save encoders in order to reproduce the latent representation.
        self.enc_model_name = 'enc_model'
        self.dec_model_name = 'dec_model'

        # Save models to HDF5
        self.enc_model_filename = self.enc_model_name + '.h5'
        self.dec_model_filename = self.dec_model_name + '.h5'

        self.svm_models_filenames = ['svm_model_{}.sav'.format(i) for i in range(self.n_svm_models)]

        # Summary files
        self.enc_model_summary_filename = self.enc_model_name + '_summary.txt'
        self.dec_model_summary_filename = self.dec_model_name + '_summary.txt'

        if not os.path.exists(self.result_dir_path):
            os.makedirs(self.result_dir_path)

    def generator_loss(self, y_t, y_p):
        """
        Loss used in generators which combines l1, l2 and ssim losses.
        :param y_t:
        :param y_p:
        :return:
        """
        loss = self.g_loss_l1_ratio * mean_absolute_error(y_t, y_p)

        if self.g_loss_l2_ratio:
            loss += self.g_loss_l2_ratio * mean_squared_error(y_t, y_p)

        if self.g_loss_ss_ratio:
            loss += self.g_loss_ss_ratio * self.ssim_loss(y_t, y_p)

        if self.g_loss_gd_ratio:
            y_t_dx, y_t_dy = tf.image.image_gradients(y_t)
            y_p_dx, y_p_dy = tf.image.image_gradients(y_p)
            loss += self.g_loss_gd_ratio * (mean_squared_error(y_t_dx, y_p_dx) + mean_squared_error(y_t_dy, y_p_dy))

        return loss

    def build_model(self):
        """
        Builds and compiles all the models.
        :return:
        """
        if self.load_model_dir_path is None:
            # Build the encoders
            past_enc = self.build_encoder()
            curr_enc = self.build_encoder()
            next_enc = self.build_encoder()

            # Get the latent shape
            self.latent_shape = curr_enc.layers[-1].output_shape[1:]

            # Build the decoders
            past_dec = self.build_decoder()
            curr_dec = self.build_decoder()
            next_dec = self.build_decoder()

            # Build encoders model
            inp_past = Input(shape=self.input_shape)
            inp_curr = Input(shape=self.input_shape)
            inp_next = Input(shape=self.input_shape)

            self.enc_model = Model(inputs=[inp_past, inp_curr, inp_next],
                                   outputs=[past_enc(inp_past), curr_enc(inp_curr), next_enc(inp_next)])

            # Build the decoders model
            emb_past = Input(shape=self.latent_shape)
            emb_curr = Input(shape=self.latent_shape)
            emb_next = Input(shape=self.latent_shape)

            self.dec_model = Model(inputs=[emb_past, emb_curr, emb_next],
                                   outputs=[past_dec(emb_past), curr_dec(emb_curr), next_dec(emb_next)])
        else:
            # Load pretrained models
            self.load_models(load_svm_models=False)

        # Build the generator model
        inp_past = Input(shape=self.input_shape)
        inp_curr = Input(shape=self.input_shape)
        inp_next = Input(shape=self.input_shape)

        self.combined = Model(inputs=[inp_past, inp_curr, inp_next],
                              outputs=self.dec_model(self.enc_model([inp_past, inp_curr, inp_next])))
        self.combined.compile(loss=self.generator_loss,
                              optimizer=self.optimizer.get_opt(self.g_opt_method),
                              metrics=[])

        # Print summary of the models
        print_model_summary(self.enc_model, os.path.join(self.result_dir_path, self.enc_model_summary_filename))
        print_model_summary(self.dec_model, os.path.join(self.result_dir_path, self.dec_model_summary_filename))

    def close_session(self):
        """
        Clears all the sub models.
        :return:
        """
        kb.clear_session()
        self.enc_model = None
        self.dec_model = None
        self.combined = None
        self.svm_models = None

    def build_encoder(self):
        """
        Builds encoder.
        :return:
        """
        # Image inputs
        d0 = Input(shape=self.input_shape)

        d1 = self.conv_layout.conv_layer(d0, self.conv_gf, normalize=False)
        d2 = self.conv_layout.conv_layer(d1, self.conv_gf * 2)
        d3 = self.conv_layout.conv_layer(d2, self.conv_gf * 4)
        d4 = self.conv_layout.conv_layer(d3, self.conv_gf * 8)
        d5 = self.conv_layout.conv_layer(d4, self.conv_gf * 8)
        d6 = self.conv_layout.conv_layer(d5, self.conv_gf * 8)

        return Model(d0, d6)

    def build_decoder(self):
        """
        Builds decoder.
        :return:
        """
        u0 = Input(shape=self.latent_shape)

        u1 = self.conv_layout.deconv_layer(u0, self.conv_gf * 8)
        u2 = self.conv_layout.deconv_layer(u1, self.conv_gf * 8)
        u3 = self.conv_layout.deconv_layer(u2, self.conv_gf * 4)
        u4 = self.conv_layout.deconv_layer(u3, self.conv_gf * 2)
        u5 = self.conv_layout.deconv_layer(u4, self.conv_gf)

        output_img = Conv2DTranspose(1,
                                     kernel_size=self.conv_layout.kernel_size,
                                     strides=self.conv_layout.strides,
                                     padding='same',
                                     activation=self.out_activation)(u5)

        return Model(u0, output_img)

    def load_models(self, load_svm_models=True):
        """
        Loads all the trained models.
        :param load_svm_models:
        :return:
        """
        print('=====================================================================')
        print('Loading the trained model:')
        self.enc_model = load_model(os.path.join(self.load_model_dir_path, self.enc_model_filename),
                                    custom_objects={'InstanceNormalization': InstanceNormalization,
                                                    'generator_loss': self.generator_loss})
        self.dec_model = load_model(os.path.join(self.load_model_dir_path, self.dec_model_filename),
                                    custom_objects={'InstanceNormalization': InstanceNormalization,
                                                    'generator_loss': self.generator_loss})
        if self.use_svm_models and load_svm_models:
            # Check if there's any svm model file, otherwise train svm models
            svm_models_filenames = [f for f in os.listdir(self.load_model_dir_path)
                                    if os.path.isfile(os.path.join(self.load_model_dir_path, f)) and
                                    os.path.splitext(f)[1] == '.sav']

            if len(svm_models_filenames):
                self.svm_models_filenames = svm_models_filenames
                self.n_svm_models = len(svm_models_filenames)
                self.svm_models = [joblib.load(os.path.join(self.load_model_dir_path, f)) for f in svm_models_filenames]
            else:
                self.train_svm_models()
        print('                                                                done!')

    def get_data(self,
                 subset_name,
                 video_name=None,
                 rand_select=False):
        """
        Gets the appropriate subset.
        :param subset_name:
        :param video_name:
        :param rand_select:
        :return:
        """
        if self.holistic_method:
            # Get train samples randomly selected
            return self.extract_data.get_subset(subset_name, video_name, rand_select)
        else:
            # Get complete train samples
            return self.extract_data.get_roi_subset(subset_name, video_name, rand_select)

    def get_batch_idx(self, data_size):
        """
        Returns a randomly selected batch indexes.
        :param data_size:
        :return:
        """
        assert data_size > self.batch_size, 'Data size should be a lot bigger than the batch size.'
        idx = np.array(list(range(data_size)))
        # Create batch indexes
        np.random.shuffle(idx)
        good_n_samples = data_size - data_size % self.batch_size
        idx = idx[:good_n_samples]
        return np.array_split(idx, int(good_n_samples / self.batch_size))

    def expect_on_batch(self, data, batch_idx):
        """
        Returns the input and output requirements for training all the necessary models.
        :param data:
        :param batch_idx:
        :return:
        """
        # Output format
        expect_out = dict()
        # Inputs
        expect_out['inp'] = {self.past_data_name: data['data'][self.past_data_name][batch_idx],
                             self.curr_data_name: data['data'][self.curr_data_name][batch_idx],
                             self.next_data_name: data['data'][self.next_data_name][batch_idx]}
        # Expected outputs
        expect_out['out'] = expect_out['inp']
        return expect_out

    def train(self,
              save_model=True,
              plot_histories=True):
        """
        Fit the dae model.
        :param save_model:
        :param plot_histories:
        :return:
        """
        print('=====================================================================')
        print('Training the CNN models:')
        all_g_train_loss = []
        all_combined_lr = []
        all_epochs = []
        all_train_sizes = []

        checkpoints_dir = os.path.join(ds.TEMP_PATH, f'checkpoints_{int(time.time())}')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # Get the train set
        train_data = self.get_data(ds.TRAIN_DIRNAME, rand_select=self.random_subset_training)

        for epoch in tqdm(range(1, self.n_epochs + 1), file=sys.stdout, desc='Training', ascii=True):
            # Change the learning rate for other half of epochs
            if epoch == self.start_fine_tune:
                kb.set_value(self.combined.optimizer.lr, self.learning_rate * self.ft_reduce_factor)
            elif self.load_model_dir_path and epoch < self.start_fine_tune:
                continue

            # Train models
            g_train_loss_list = []

            # Batch indexes
            indexes = np.array(self.get_batch_idx(train_data['data_size']))

            for bat_idx in tqdm(indexes, file=sys.stdout, desc=f'Training on epoch {epoch:<3}', leave=False):
                # Add more stochastic
                np.random.shuffle(bat_idx)
                # Get expected data
                exp_bat = self.expect_on_batch(train_data, bat_idx)
                # Train the generators
                g_loss = self.combined.train_on_batch(list(exp_bat['inp'].values()),
                                                      list(exp_bat['out'].values()))
                # Store the scores
                if isinstance(g_loss, (list, tuple, set, np.ndarray)):
                    g_loss = g_loss[0]
                g_train_loss_list.append(g_loss)

            # Take the average
            g_train_loss = float(np.mean(g_train_loss_list))
            # Get the learning rates - Just for test that ensures if it corresponds to the assigned lr
            combined_lr = float(kb.get_value(self.combined.optimizer.lr))
            # Store results
            all_g_train_loss.append(g_train_loss)
            all_combined_lr.append(combined_lr)
            all_epochs.append(epoch)
            all_train_sizes.append(train_data['data_size'])
            # Save models
            self.enc_model.save(os.path.join(checkpoints_dir, f'{self.enc_model_name}_epoch_{epoch}.h5'))
            self.dec_model.save(os.path.join(checkpoints_dir, f'{self.dec_model_name}_epoch_{epoch}.h5'))

            # Plot the progress
            print(f'\nEpoch: {epoch:<3} -> train: [G loss: {g_train_loss:.6f}], lr: [G: {combined_lr:.3e}]')
        print('                                                                done!')
        print('=====================================================================')

        if save_model:
            print('Saving models to disk:')
            if not os.path.exists(self.save_model_dir_path):
                os.makedirs(self.save_model_dir_path)
            # Save models
            self.enc_model.save(os.path.join(self.save_model_dir_path, self.enc_model_filename))
            self.dec_model.save(os.path.join(self.save_model_dir_path, self.dec_model_filename))
            print('                                                                done!')
            print('=====================================================================')

        # Delete checkpoints dir
        shutil.rmtree(checkpoints_dir)

        if self.use_svm_models:
            # Train SVM models
            self.train_svm_models(train_data)

        if plot_histories:
            print('Plotting training histories:')
            # summarize loss
            fig = plt.figure()
            plt.plot(all_epochs, all_g_train_loss, 'g.-')
            plt.legend(['Generator'], loc='upper right')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            figure_name = 'model_loss_curves.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            # ======================================
            # summarize learning rates
            fig = plt.figure()
            plt.plot(all_epochs, all_combined_lr, 'k-')
            plt.legend(['combined'], loc='upper right')
            plt.ylabel('lr')
            plt.xlabel('epoch')
            figure_name = 'model_lr_curves.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            # ======================================
            # summarize train sizes
            fig = plt.figure()
            plt.plot(all_epochs, all_train_sizes, 'k-')
            plt.ylabel('size')
            plt.xlabel('epoch')
            figure_name = 'model_train_sizes.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
            print('                                                                done!')
            print('=====================================================================')

    def get_cfv_data(self,
                     data):
        """
        Gets the compressed feature vector of the latent space of the given data.
        :param data:
        :return:
        """
        # Get the encoded data
        enc_data = self.predict_embeddings(data)
        # Flatten and concatenate latent data
        return np.hstack((enc_data[self.past_data_name].reshape(len(enc_data[self.past_data_name]), -1),
                          enc_data[self.curr_data_name].reshape(len(enc_data[self.curr_data_name]), -1),
                          enc_data[self.next_data_name].reshape(len(enc_data[self.next_data_name]), -1)))

    def train_svm_models(self,
                         train_data=None,
                         save_model=True,
                         viz_clusters=True):
        """
        Trains all the SVM models which will be used to form one-versus-rest classification scheme.
        :param train_data:
        :param save_model:
        :param viz_clusters:
        :return:
        """
        print('=====================================================================')
        print('Training the SVM models:')
        # Get train data if necessary
        if train_data is None:
            if self.random_subset_training:
                # Get train samples randomly selected
                train_data = self.extract_data.get_random_train_subset()
            else:
                # Get complete train samples
                train_data = self.extract_data.get_complete_train_set()
        # Get the cfv data
        train_cfv_data = self.get_cfv_data(train_data)
        # Cluster the latent data
        cfv_clusters = KMeans(n_clusters=self.n_svm_models).fit(train_cfv_data)
        # Visualize the training latent clusters if necessary
        if viz_clusters:
            reduced_cfv_clusters = TSNE(n_components=2).fit_transform(train_cfv_data)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Principal Component 1', fontsize=15)
            ax.set_ylabel('Principal Component 2', fontsize=15)
            clusters_targets = [str(n).zfill(2) for n in range(1, self.n_svm_models + 1)]
            clusters_colors = ['m', 'k', 'g', 'c', 'y', 'r', 'b', 'orange', 'olive', 'indigo']
            for i in range(self.n_svm_models):
                _reduced = reduced_cfv_clusters[cfv_clusters.labels_ == i]
                ax.scatter(_reduced[:, 0], _reduced[:, 1], c=clusters_colors[i], s=50)
            ax.legend(clusters_targets)
            ax.grid()
            figure_name = 'model_latent_clusters.pdf'
            fig.savefig(os.path.join(self.result_dir_path, figure_name), bbox_inches='tight')
            plt.close(fig)
        # Fit the SVM models
        self.svm_models = []
        for i in tqdm(range(self.n_svm_models), file=sys.stdout, desc='Training SVMs'):
            svm_clf = SVC(gamma='auto')
            svm_clf.fit(train_cfv_data, np.where(cfv_clusters.labels_ == i, 1, -1))
            self.svm_models.append(svm_clf)
        print('                                                                done!')
        print('=====================================================================')
        if save_model:
            print('Saving models to disk:')
            for i, model_filename in enumerate(self.svm_models_filenames):
                joblib.dump(self.svm_models[i], os.path.join(self.save_model_dir_path, model_filename))
            print('                                                                done!')
            print('=====================================================================')

    def predict_embeddings(self, subset):
        """
        Predict using the trained models.
        :param subset:
        :return:
        """
        output = dict()
        out = self.enc_model.predict([subset['data'][self.past_data_name],
                                      subset['data'][self.curr_data_name],
                                      subset['data'][self.next_data_name]])
        output[self.past_data_name] = out[0]
        output[self.curr_data_name] = out[1]
        output[self.next_data_name] = out[2]
        return output

    def predict_images(self, subset):
        """
        Predict using the trained models.
        :param subset:
        :return:
        """
        output = dict()
        out = self.combined.predict([subset['data'][self.past_data_name],
                                     subset['data'][self.curr_data_name],
                                     subset['data'][self.next_data_name]])
        output[self.past_data_name] = out[0]
        output[self.curr_data_name] = out[1]
        output[self.next_data_name] = out[2]
        return output

    def test(self, subset_name, video_name):
        """
        Applies the trained models to produce reconstruction errors represented as anomaly scores.
        :param subset_name:
        :param video_name:
        :return:
        """
        # Get video samples
        data = self.get_data(subset_name, video_name)

        if self.use_svm_models:
            # Get cfv data
            cfv_data = self.get_cfv_data(data)
            # Apply trained SVM
            svm_scores = np.array([m.predict(cfv_data).flatten() for m in self.svm_models])
            # Compute one-versus-rest SVM anomaly scores
            ano_scores = np.array([-np.max(svm_scores[:, i]) for i in range(svm_scores.shape[1])])
        else:
            ano_scores = None

        # Remove the train det if necessary in order to save RAM
        if self.extract_data.subset_det[ds.TRAIN_DIRNAME] is not None:
            self.extract_data.subset_det[ds.TRAIN_DIRNAME] = None

        return util.return_score_dict(data, ano_scores)


class QualityAssessmentScore:
    """
    Quality assessment scoring class which will be used for computing the anomaly score.
    Each function is use to measure the similarity between given images.
    """
    def __init__(self, dist_name, c_ratio, p_ratio, use_grid, grid_size):
        """
        Initialization.
        :param dist_name:
        :param c_ratio:
        :param p_ratio:
        :param use_grid:
        :param grid_size:
        """
        self.dist_name = dist_name
        self.c_ratio = c_ratio
        self.p_ratio = p_ratio
        self.use_grid = use_grid
        self.grid_size = grid_size

    def compute(self, c_pair, p_pair):
        """
        Gets the member method by its str name.
        :param c_pair:
        :param p_pair:
        :return:
        """
        cs = ps = 0
        # Get the method from 'self'. Default to a lambda.
        method_name = f'compute_{self.dist_name}'
        method = getattr(self, method_name, lambda: 'Invalid scoring method.')

        if self.use_grid:
            # Computes grid-based score
            assert len(c_pair[0]) == len(p_pair[0]), 'Current and past images do not have the same size.'
            n = int(len(c_pair[0]) / self.grid_size)
            x = list(range(0, len(c_pair[0]), n))
            # Get max errors
            if self.c_ratio > 0:
                cs = np.max([method(c_pair[0][i:i + n, j:j + n], c_pair[1][i:i + n, j:j + n]) for i in x for j in x])
            if self.p_ratio > 0:
                ps = np.max([method(p_pair[0][i:i + n, j:j + n], p_pair[1][i:i + n, j:j + n]) for i in x for j in x])
        else:
            # Computes score on whole images
            if self.c_ratio > 0:
                cs = method(c_pair[0], c_pair[1])
            if self.p_ratio > 0:
                ps = method(p_pair[0], p_pair[1])

        # Cumulated error
        return self.c_ratio * cs + self.p_ratio * ps

    def compute_mae(self, real_img, fake_img):
        """
        Computes the MAE score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return (np.abs(real_img - fake_img)).mean(axis=None)

    def compute_mse(self, real_img, fake_img):
        """
        Computes the MSE score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return (np.square(real_img - fake_img)).mean(axis=None)

    def compute_vse(self, real_img, fake_img):
        """
        Computes the MSE score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return (np.square(real_img - fake_img)).var(axis=None)

    def compute_ssim(self, real_img, fake_img):
        """
        Computes the SSIM score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return 1 - ssim(real_img, fake_img, data_range=1)

    def compute_psnr(self, real_img, fake_img):
        """
        Computes the PSNR score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return 1 / psnr(real_img, fake_img, data_range=1)

    def compute_nrmse(self, real_img, fake_img):
        """
        Computes the NRMSE score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return nrmse(real_img, fake_img)


class PredictionQualityAssessment:
    """
    Quality assessment scoring class which will be used for computing the anomaly score.
    Each function is use to measure the similarity between given images.
    """
    def __init__(self, dist_name, use_grid, grid_size):
        """
        Initialization.
        :param dist_name:
        :param use_grid:
        :param grid_size:
        """
        self.dist_name = dist_name
        self.use_grid = use_grid
        self.grid_size = grid_size

    def compute(self, real_img, pred_img):
        """
        Gets the member method by its str name.
        :param real_img:
        :param pred_img:
        :return:
        """
        # Get the method from 'self'. Default to a lambda.
        method_name = f'compute_{self.dist_name}'
        method = getattr(self, method_name, lambda: 'Invalid scoring method.')

        if self.use_grid:
            # Computes grid-based score
            n = int(len(real_img) / self.grid_size)
            x = list(range(0, len(real_img), n))
            # Get max errors
            score = np.max([method(real_img[i:i + n, j:j + n], pred_img[i:i + n, j:j + n]) for i in x for j in x])
        else:
            # Computes score on whole images
            score = method(real_img, pred_img)

        return score

    def compute_mae(self, real_img, fake_img):
        """
        Computes the MAE score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return (np.abs(real_img - fake_img)).mean(axis=None)

    def compute_mse(self, real_img, fake_img):
        """
        Computes the MSE score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return (np.square(real_img - fake_img)).mean(axis=None)

    def compute_vse(self, real_img, fake_img):
        """
        Computes the MSE score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return (np.square(real_img - fake_img)).var(axis=None)

    def compute_ssim(self, real_img, fake_img):
        """
        Computes the SSIM score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return 1 - ssim(real_img, fake_img, data_range=ds.MAX_NORM - ds.MIN_NORM)

    def compute_psnr(self, real_img, fake_img):
        """
        Computes the PSNR score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return 1 / psnr(real_img, fake_img, data_range=ds.MAX_NORM - ds.MIN_NORM)

    def compute_nrmse(self, real_img, fake_img):
        """
        Computes the NRMSE score given the real and the generated images.
        :param real_img:
        :param fake_img:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'compute_{self.dist_name}', 'Not the right score function.'
        return nrmse(real_img, fake_img)


class Optimizer:
    def __init__(self, lr, clip):
        self.lr = lr
        self.clip = clip

    def get_opt(self, opt):
        """Dispatch method"""
        method_name = 'opt_' + str(opt)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: 'Invalid optimizer.')
        # Call the method as we return it
        return method()

    def opt_adam(self):
        return optimizers.Adam(lr=self.lr, clipvalue=self.clip)

    def opt_sgd(self):
        return optimizers.SGD(lr=self.lr, clipvalue=self.clip)

    def opt_rmsprop(self):
        return optimizers.RMSprop(lr=self.lr, clipvalue=self.clip)


class Conv2DSetup:
    def __init__(self,
                 use_separable_conv=ds.USE_SEPARABLE_CONV,
                 kernel_size=ds.DEFAULT_CONV_KERNEL_SIZE,
                 strides=ds.DEFAULT_CONV_STRIDES,
                 leaky_relu_alpha=ds.DEFAULT_CONV_LEAKY_RELU_ALPHA,
                 dropout_rate=ds.DEFAULT_CONV_DROPOUT_RATE,
                 do_batch_normalize=ds.DEFAULT_CONV_BATCH_NORM,
                 kernel_init=ds.DEFAULT_CONV_KERNEL_INIT,
                 skip_diff=ds.DEFAULT_CONV_SKIP_DIFF,
                 rand_seed=ds.DEFAULT_RAND_SEED):
        """
        Conv layer initialization.
        :param use_separable_conv:
        :param kernel_size:
        :param strides:
        :param leaky_relu_alpha:
        :param dropout_rate:
        :param do_batch_normalize:
        :param kernel_init:
        :param skip_diff:
        :param rand_seed:
        """
        self.use_separable_conv = use_separable_conv
        self.kernel_size = kernel_size
        self.strides = strides
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout_rate = dropout_rate
        self.do_batch_normalize = do_batch_normalize
        self.kernel_init = kernel_init
        self.skip_diff = skip_diff
        self.rand_seed = rand_seed

    def get_kernel_init(self):
        """Dispatch method"""
        method_name = f'init_{self.kernel_init}'
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: 'Invalid kernel init.')
        # Call the method as we return it
        return method()

    def init_he_uniform(self):
        return he_uniform(seed=self.rand_seed)

    def init_he_normal(self):
        return he_normal(seed=self.rand_seed)

    def init_random_uniform(self):
        return random_uniform(seed=self.rand_seed)

    def init_random_normal(self):
        return random_normal(seed=self.rand_seed)

    def diff_layer(self, pair_of_tensors):
        """
        Calls the Lambda layer with the difference.
        :param pair_of_tensors:
        :return:
        """
        diff_name = f'diff_{self.skip_diff}'
        # Get the method from 'self'. Default to a lambda.
        diff_method = getattr(self, diff_name, lambda: 'Invalid diff.')
        # Call the method as we return it
        return Lambda(diff_method)(pair_of_tensors)

    def diff_square(self, pair_of_tensors):
        """
        Tensor: for constructing the squared difference layer.
        :param pair_of_tensors:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'diff_{self.skip_diff}', 'Not the right diff function.'
        x, y = pair_of_tensors
        return kb.square(x - y)

    def diff_abs(self, pair_of_tensors):
        """
        Tensor: for constructing the squared difference layer.
        :param pair_of_tensors:
        :return:
        """
        assert inspect.currentframe().f_code.co_name == f'diff_{self.skip_diff}', 'Not the right diff function.'
        x, y = pair_of_tensors
        return kb.abs(x - y)

    def get_conv_2d(self,
                    filters,
                    kernel_size=None,
                    strides=None):
        """
        Gets the appropriate conv 2d layer method.
        :param filters:
        :param kernel_size:
        :param strides:
        :return:
        """
        conv_args = {'filters': filters,
                     'kernel_size': self.kernel_size if kernel_size is None else kernel_size,
                     'strides': self.strides if strides is None else strides,
                     'padding': 'same',
                     'kernel_initializer': self.get_kernel_init()}
        if self.use_separable_conv:
            return SeparableConv2D(**conv_args)
        else:
            return Conv2D(**conv_args)

    def conv_layer(self,
                   layer_input,
                   filters=None,
                   kernel_size=None,
                   strides=None,
                   add_dropout=False,
                   normalize=True):
        """
        Builds a Unet style convolutional layer with activation and batch normalization.
        :param layer_input:
        :param filters:
        :param kernel_size:
        :param strides:
        :param add_dropout:
        :param normalize:
        :return:
        """
        if filters is None:
            d = layer_input
        else:
            d = self.get_conv_2d(filters, kernel_size, strides)(layer_input)
        d = LeakyReLU(alpha=self.leaky_relu_alpha)(d)
        if add_dropout and self.dropout_rate:
            d = SpatialDropout2D(self.dropout_rate, seed=self.rand_seed)(d)
        if normalize:
            if self.do_batch_normalize:
                d = BatchNormalization(momentum=0.8)(d)
            else:
                d = InstanceNormalization()(d)
        return d

    def deconv_layer(self,
                     layer_input,
                     filters,
                     skip_input=None,
                     add_dropout=False,
                     normalize=True):
        """
        Builds a Unet style de-convolutional layer with activation, batch normalization and skip connection.
        :param layer_input:
        :param filters:
        :param skip_input:
        :param add_dropout:
        :param normalize:
        :return:
        """
        u = Conv2DTranspose(filters, kernel_size=self.kernel_size, strides=self.strides, padding='same',
                            kernel_initializer=self.get_kernel_init())(layer_input)
        u = LeakyReLU(alpha=self.leaky_relu_alpha)(u)
        if add_dropout and self.dropout_rate:
            u = SpatialDropout2D(self.dropout_rate, seed=self.rand_seed)(u)
        if normalize:
            if self.do_batch_normalize:
                u = BatchNormalization(momentum=0.8)(u)
            else:
                u = InstanceNormalization()(u)
        if skip_input is not None:
            u = Concatenate()([u] + skip_input)
        return u


def squared_differences(pair_of_tensors):
    """
    Tensor: for constructing the squared difference layer.
    :param pair_of_tensors:
    :return:
    """
    x, y = pair_of_tensors
    return kb.square(x - y)


def absolute_differences(pair_of_tensors):
    """
    Tensor: for constructing the squared difference layer.
    :param pair_of_tensors:
    :return:
    """
    x, y = pair_of_tensors
    return kb.abs(x - y)


def smooth_positive_labels(y):
    """
    Smooth positive label class=1 to [0.7, 1.2].
    :param y:
    :return:
    """
    return y - 0.3 + (np.random.random(y.shape) * 0.5)


def smooth_negative_labels(y):
    """
    Smooth positive label class=0 to [0.0, 0.3].
    :param y:
    :return:
    """
    return y + (np.random.random(y.shape) * 0.3)


def binary_focal_loss(gamma=2.0, alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = kb.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = kb.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = kb.clip(pt_0, epsilon, 1. - epsilon)

        return - kb.sum(alpha * kb.pow(1. - pt_1, gamma) * kb.log(pt_1)) \
               - kb.sum((1 - alpha) * kb.pow(pt_0, gamma) * kb.log(1. - pt_0))

    return binary_focal_loss_fixed


def pixel_wise_mse(target, output):
    """
    Computes the pixel-wise mean square error loss function.
    :param target:
    :param output:
    :return:
    """
    eps = kb.epsilon()
    output = kb.clip(output, eps, 1. - eps)
    return kb.sum(kb.sum(kb.pow(target - output, 2), axis=-1), axis=-1) / (ds.ROI_SIZE ** 2)


def pixel_wise_mae(target, output):
    """
    Computes the pixel-wise mean square error loss function.
    :param target:
    :param output:
    :return:
    """
    eps = kb.epsilon()
    output = kb.clip(output, eps, 1. - eps)
    return kb.sum(kb.sum(kb.abs(target - output), axis=-1), axis=-1) / (ds.ROI_SIZE ** 2)


def psnr_loss(target, output):
    """
    Computes the PSNR loss function.
    :param target:
    :param output:
    :return:
    """
    eps = kb.epsilon()
    output = kb.clip(output, eps, 1. - eps)
    max_pixel = 1.0
    return (10.0 * kb.log((max_pixel ** 2) / (kb.mean(kb.square(target - output), axis=-1)))) / 2.303


def print_model_summary(model, summary_path):
    """
    Writes and prints the summary of the given model.
    :param model:
    :param summary_path:
    :return:
    """
    with open(summary_path, 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    with open(summary_path, 'r') as fin:
        print(fin.read(), end='')


def get_global_result(gt_results,
                      ts_results,
                      result_dir_path='',
                      plot_figure=True):
    """
    Computes the ROC and AUC using all the sequences.
    :param gt_results: contains only the gt labels, NOT the frame indexes
    :param ts_results: contains only the scores, NOT the frame indexes
    :param result_dir_path:
    :param plot_figure:
    :return:
    """
    # Check if the sizes are correct
    assert len(gt_results) == len(ts_results), 'Something is wrong with the length of result array.'
    flat_ts_results = []
    flat_gt_results = []
    for gt_seq, ts_seq in zip(gt_results, ts_results):
        assert len(gt_seq) == len(ts_seq), 'Something is wrong with the length of result sequence array.'
        flat_ts_results += ts_seq
        flat_gt_results += gt_seq
    # Flatten both results.
    flat_ts_results = np.array(flat_ts_results)
    flat_gt_results = np.array(flat_gt_results)
    assert len(flat_gt_results) == len(flat_ts_results), 'Something is wrong with the dimension of result array.'
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(flat_gt_results, flat_ts_results)
    # Compute AUC and EER
    glob_results = dict()
    glob_results['auc'] = auc(fpr, tpr)
    glob_results['eer'] = fpr[np.nanargmin(np.absolute((fpr + tpr - 1)))]
    glob_results['thr'] = interp1d(fpr, thresholds)(glob_results['eer'])
    if plot_figure:
        # Save figure
        fig = plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % glob_results['auc'])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        fig.savefig(os.path.join(result_dir_path, ds.ROC_FILENAME), bbox_inches='tight')
    return glob_results
