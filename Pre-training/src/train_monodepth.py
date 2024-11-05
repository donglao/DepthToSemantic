import argparse
from monodepth_main import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_image0_path',
    type=str, required=True, help='Path to list of training image 0 paths')
parser.add_argument('--train_image1_path',
    type=str, required=True, help='Path to list of training image 1 paths')
parser.add_argument('--train_camera_path',
    type=str, required=True, help='Path to list of training camera parameter paths')
parser.add_argument('--val_image0_path',
    type=str, default='', help='Path to list of validation image paths')
parser.add_argument('--val_camera_path',
    type=str, default='', help='Path to list of validation camera parameter paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default='', help='Path to list of validation ground truth depth paths')
parser.add_argument('--load_triplet',
    action='store_true', help='If set, then use load triplet')
parser.add_argument('--use_resize',
    action='store_true', help='If set, then use resize instead of crop')
# Batch parameters
parser.add_argument('--n_batch',
    type=int, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, help='Width of each sample')
parser.add_argument('--encoder_type',
    type=str, help='Encoder type.')
parser.add_argument('--decoder_type',
    type=str, help='Decoder type.')
parser.add_argument('--activation_func',
    type=str, help='Activation func')
parser.add_argument('--n_pyramid',
    type=int, help='Number of levels in image pyramid')
# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, help='Space delimited list to change learning rate')
parser.add_argument('--use_augment',
    action='store_true', help='If set, then use data augmentation')
parser.add_argument('--w_color',
    type=float, help='Weight of image reconstruction loss')
parser.add_argument('--w_ssim',
    type=float, help='Weight of ssim loss')
parser.add_argument('--w_smoothness',
    type=float, help='Weight of smoothness loss')
parser.add_argument('--w_left_right',
    type=float, help='Weight of left-right consistency loss')
# Depth range settings
parser.add_argument('--scale_factor',
    type=float, help='Scale factor for disparity')
parser.add_argument('--min_evaluate_depth',
    type=float, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, help='Maximum value of depth to evaluate')
# Checkpoint settings
parser.add_argument('--n_summary',
    type=int, help='Number of iterations for logging summary')
parser.add_argument('--n_checkpoint',
    type=int, help='Number of iterations for each checkpoint')
parser.add_argument('--checkpoint_path',
    type=str, help='Path to save checkpoints')
# Hardware settings
parser.add_argument('--device',
    type=str, help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

  train(train_image0_path=args.train_image0_path,
        train_image1_path=args.train_image1_path,
        train_camera_path=args.train_camera_path,
        val_image0_path=args.val_image0_path,
        val_camera_path=args.val_camera_path,
        val_ground_truth_path=args.val_ground_truth_path,
        load_triplet=args.load_triplet,
        use_resize=args.use_resize,
        # Batch settings
        n_batch=args.n_batch,
        n_height=args.n_height,
        n_width=args.n_width,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        activation_func=args.activation_func,
        n_pyramid=args.n_pyramid,
        # Training settings
        learning_rates=args.learning_rates,
        learning_schedule=args.learning_schedule,
        use_augment=args.use_augment,
        w_color=args.w_color,
        w_ssim=args.w_ssim,
        w_smoothness=args.w_smoothness,
        w_left_right=args.w_left_right,
        # Depth range settings
        scale_factor=args.scale_factor,
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        # Checkpoint settings
        n_summary=args.n_summary,
        n_checkpoint=args.n_checkpoint,
        checkpoint_path=args.checkpoint_path,
        # Hardware settings
        device=args.device,
        n_thread=args.n_thread)
