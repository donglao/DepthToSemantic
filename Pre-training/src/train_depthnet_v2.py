import os, argparse
from depthnet_v2 import train


parser = argparse.ArgumentParser()

# Training input filepaths
parser.add_argument('--train_images_left_path',
    type=str, required=True, help='Path to list of training left camera image paths')
parser.add_argument('--train_images_right_path',
    type=str, required=True, help='Path to list of training right camera image paths')
parser.add_argument('--train_intrinsics_left_path',
    type=str, required=True, help='Path to list of training left camera intrinsics paths')
parser.add_argument('--train_intrinsics_right_path',
    type=str, required=True, help='Path to list of training right camera intrinsics paths')
parser.add_argument('--train_focal_length_baseline_left_path',
    type=str, required=True, help='Path to list of training left camera focal length baseline paths')
parser.add_argument('--train_focal_length_baseline_right_path',
    type=str, required=True, help='Path to list of training right camera focal length baseline paths')

# Validation input file paths
parser.add_argument('--val_image_path',
    type=str, default=None, help='Path to list of validation image paths')
parser.add_argument('--val_focal_length_baseline_path',
    type=str, default=None, help='Path to list of validation focal length baseline paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default=None, help='Path to list of validation ground truth depth paths')

# Input settings
parser.add_argument('--n_batch',
    type=int, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, help='Width of each sample')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 255], help='Range of image intensities after normalization')

# Network settings
parser.add_argument('--encoder_type',
    nargs='+', type=str, default=['resnet18'], help='Encoder type: resnet18')
parser.add_argument('--n_filters_encoder',
    nargs='+', type=int, default=[32, 64, 96, 128, 256], help='Space delimited list of filters to use in each block of image encoder')
parser.add_argument('--decoder_type',
    nargs='+', type=str, default=['multiscale'], help='Decoder type: multiscale')
parser.add_argument('--n_resolution_decoder_output',
    type=int, default=1, help='Number of resolutions to output')
parser.add_argument('--n_filters_decoder',
    nargs='+', type=int, default=[256, 128, 96, 64, 32], help='Space delimited list of filters to use in each block of depth decoder')
parser.add_argument('--max_predict_inverse_depth',
    type=float, default=0.30, help='Maximum value of predicted inverse depth')

# Weight settings
parser.add_argument('--weight_initializer',
    type=str, default='xavier_normal', help='Weight initialization type: kaiming_uniform, kaiming_normal, xavier_uniform, xavier_normal')
parser.add_argument('--activation_func',
    type=str, default='leaky_relu', help='Activation function after each layer: relu, leaky_relu, elu, sigmoid')
parser.add_argument('--use_batch_norm',
    action='store_true', help='If set, then use batch norm')

# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, help='Space delimited list to change learning rate')

# Augmentation settings
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=[1.00, 0.50, 0.25], help='Probabilities to use data augmentation. Note: there is small chance that no augmentation take place even when we enter augmentation pipeline.')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=[50, 55, 60], help='If not -1, then space delimited list to change augmentation probability')
parser.add_argument('--augmentation_random_swap_left_right',
    action='store_true', help='If set, peform random swapping between left and right images')
parser.add_argument('--augmentation_random_brightness',
    nargs='+', type=float, default=[-1, -1], help='If does not contain -1, apply random brightness')
parser.add_argument('--augmentation_random_contrast',
    nargs='+', type=float, default=[-1, -1], help='If does not contain -1, apply random contrast')
parser.add_argument('--augmentation_random_saturation',
    nargs='+', type=float, default=[-1, -1], help='If does not contain -1, apply random saturation')

# Loss settings
parser.add_argument('--supervision_type',
    nargs='+', type=str, help='monocular, stereo')
parser.add_argument('--w_color',
    type=float, help='Weight of image reconstruction loss')
parser.add_argument('--w_structure',
    type=float, help='Weight of ssim loss')
parser.add_argument('--w_smoothness',
    type=float, help='Weight of smoothness loss')
parser.add_argument('--w_weight_decay_depth',
    type=float, default=0.00, help='Weight of weight decay regularization for depth')
parser.add_argument('--w_weight_decay_pose',
    type=float, default=0.00, help='Weight of weight decay regularization for depth')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, default=os.path.join('trained_depthnet', 'model'), help='Path to save checkpoints')
parser.add_argument('--n_checkpoint',
    type=int, default=5000, help='Number of iterations for each checkpoint')
parser.add_argument('--n_summary',
    type=int, default=5000, help='Number of iterations before logging summary')
parser.add_argument('--n_summary_display',
    type=int, default=4, help='Number of samples to include in visual display summary')
parser.add_argument('--validation_start_step',
    type=int, default=5000, help='Number of steps before starting validation')
parser.add_argument('--depth_model_restore_path',
    type=str, default=None, help='Path to restore depth model from checkpoint')
parser.add_argument('--pose_model_restore_path',
    type=str, default=None, help='Path to restore pose model from checkpoint')

# Hardware settings
parser.add_argument('--device',
    type=str, help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    train(train_images_left_path=args.train_images_left_path,
          train_images_right_path=args.train_images_right_path,
          train_intrinsics_left_path=args.train_intrinsics_left_path,
          train_intrinsics_right_path=args.train_intrinsics_right_path,
          train_focal_length_baseline_left_path=args.train_focal_length_baseline_left_path,
          train_focal_length_baseline_right_path=args.train_focal_length_baseline_right_path,
          val_image_path=args.val_image_path,
          val_focal_length_baseline_path=args.val_focal_length_baseline_path,
          val_ground_truth_path=args.val_ground_truth_path,
          # Input settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          normalized_image_range=args.normalized_image_range,
          # Networks settings
          encoder_type=args.encoder_type,
          n_filters_encoder=args.n_filters_encoder,
          decoder_type=args.decoder_type,
          n_resolution_decoder_output=args.n_resolution_decoder_output,
          n_filters_decoder=args.n_filters_decoder,
          max_predict_inverse_depth=args.max_predict_inverse_depth,
          # Weight settings
          weight_initializer=args.weight_initializer,
          activation_func=args.activation_func,
          use_batch_norm=args.use_batch_norm,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          # Augmentation settings
          augmentation_probabilities=args.augmentation_probabilities,
          augmentation_schedule=args.augmentation_schedule,
          augmentation_random_swap_left_right=args.augmentation_random_swap_left_right,
          augmentation_random_brightness=args.augmentation_random_brightness,
          augmentation_random_contrast=args.augmentation_random_contrast,
          augmentation_random_saturation=args.augmentation_random_saturation,
          # Loss function settings
          supervision_type=args.supervision_type,
          w_color=args.w_color,
          w_structure=args.w_structure,
          w_smoothness=args.w_smoothness,
          w_weight_decay_depth=args.w_weight_decay_depth,
          w_weight_decay_pose=args.w_weight_decay_pose,
          # Evaluation settings
          min_evaluate_depth=args.min_evaluate_depth,
          max_evaluate_depth=args.max_evaluate_depth,
          # Checkpoint settings
          checkpoint_path=args.checkpoint_path,
          n_checkpoint=args.n_checkpoint,
          n_summary=args.n_summary,
          n_summary_display=args.n_summary_display,
          validation_start_step=args.validation_start_step,
          depth_model_restore_path=args.depth_model_restore_path,
          pose_model_restore_path=args.pose_model_restore_path,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)
