import os, argparse
from monodepth2 import train


parser = argparse.ArgumentParser()

# Training input filepaths
parser.add_argument('--train_images_path',
    type=str, required=True, help='Path to list of training camera image paths')
parser.add_argument('--train_intrinsics_path',
    type=str, required=True, help='Path to list of training camera intrinsics paths')
parser.add_argument('--train_ground_truth_path',
    type=str, default=None, help='Path to list of training ground truth paths')

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
parser.add_argument('--network_modules',
    nargs='+', type=str, default=['depth', 'pose'], help='Space delimited list of network modules to build')
parser.add_argument('--scale_factor_depth',
    type=float, default=5.4, help='Scale factor of depth')
parser.add_argument('--min_predict_depth',
    type=float, default=0.10, help='Minimum value of predicted depth')
parser.add_argument('--max_predict_depth',
    type=float, default=100.0, help='Maximum value of predicted depth')

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
parser.add_argument('--augmentation_random_crop_type',
            nargs='+', type=str, default=['none'], help='Random crop type for data augmentation: horizontal, vertical, anchored, bottom')
parser.add_argument('--augmentation_random_crop_shape',
            nargs='+', type=int, default=[-1, -1], help='Random crop to (height, width) shape')
parser.add_argument('--augmentation_random_brightness',
    nargs='+', type=float, default=[-1, -1], help='If does not contain -1, apply random brightness')
parser.add_argument('--augmentation_random_contrast',
    nargs='+', type=float, default=[-1, -1], help='If does not contain -1, apply random contrast')
parser.add_argument('--augmentation_random_saturation',
    nargs='+', type=float, default=[-1, -1], help='If does not contain -1, apply random saturation')
parser.add_argument('--augmentation_random_crop_to_shape',
    nargs='+', type=int, default=[-1, -1], help='Shape after cropping')
parser.add_argument('--augmentation_random_flip_type',
    nargs='+', type=str, default=['none'], help='Random flip type for data augmentation: horizontal, vertical')

# Loss settings
parser.add_argument('--w_photometric',
    type=float, default=1.00, help='Weight of image reconstruction loss')
parser.add_argument('--w_smoothness',
    type=float, default=0.10, help='Weight of smoothness loss')

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

    train(train_images_path=args.train_images_path,
          train_intrinsics_path=args.train_intrinsics_path,
          train_ground_truth_path=args.train_ground_truth_path,
          # Input settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          normalized_image_range=args.normalized_image_range,
          # Networks settings
          encoder_type=args.encoder_type,
          network_modules=args.network_modules,
          scale_factor_depth=args.scale_factor_depth,
          min_predict_depth=args.min_predict_depth,
          max_predict_depth=args.max_predict_depth,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          # Augmentation settings
          augmentation_probabilities=args.augmentation_probabilities,
          augmentation_schedule=args.augmentation_schedule,
          augmentation_random_crop_shape=args.augmentation_random_crop_shape,
          augmentation_random_crop_type=args.augmentation_random_crop_type,
          augmentation_random_brightness=args.augmentation_random_brightness,
          augmentation_random_contrast=args.augmentation_random_contrast,
          augmentation_random_saturation=args.augmentation_random_saturation,
          augmentation_random_crop_to_shape=args.augmentation_random_crop_to_shape,
          augmentation_random_flip_type=args.augmentation_random_flip_type,
          # Loss function settings
          w_photometric=args.w_photometric,
          w_smoothness=args.w_smoothness,
          # Checkpoint settings
          checkpoint_path=args.checkpoint_path,
          n_checkpoint=args.n_checkpoint,
          n_summary=args.n_summary,
          n_summary_display=args.n_summary_display,
          validation_start_step=args.validation_start_step,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)
