import os, argparse
from reconstruction import train


parser = argparse.ArgumentParser()

# Training input filepaths
parser.add_argument('--train_images_left_path',
    type=str, required=True, help='Path to list of training left camera image paths')
parser.add_argument('--train_images_right_path',
    type=str, required=True, help='Path to list of training right camera image paths')

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
parser.add_argument('--remove_percent_range',
    nargs='+', type=float, default=[0.02, 0.10], help='Min and max percent to remove')
parser.add_argument('--remove_patch_size',
    nargs='+', type=int, default=[3, 3], help='Patch size to remove')

parser.add_argument('--w_reconstruction',
    type=float, help='Weight of image reconstruction loss')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, default=os.path.join('trained_flow', 'model'), help='Path to save checkpoints')
parser.add_argument('--n_checkpoint',
    type=int, default=5000, help='Number of iterations for each checkpoint')
parser.add_argument('--n_summary',
    type=int, default=5000, help='Number of iterations before logging summary')
parser.add_argument('--n_summary_display',
    type=int, default=4, help='Number of samples to include in visual display summary')

# Hardware settings
parser.add_argument('--device',
    type=str, help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    train(train_images_left_path=args.train_images_left_path,
          train_images_right_path=args.train_images_right_path,
          # Input settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          normalized_image_range=args.normalized_image_range,
          # Network settings
          encoder_type=args.encoder_type,
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
          remove_percent_range=args.remove_percent_range,
          remove_patch_size=args.remove_patch_size,
          w_reconstruction=args.w_reconstruction,
          # Checkpoint settings
          checkpoint_path=args.checkpoint_path,
          n_checkpoint=args.n_checkpoint,
          n_summary=args.n_summary,
          n_summary_display=args.n_summary_display,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)
