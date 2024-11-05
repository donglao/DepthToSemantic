import argparse
from monodepth2 import run


parser = argparse.ArgumentParser()


# Input file paths
parser.add_argument('--image_path',
    type=str, required=True, help='Path to list of image paths')
parser.add_argument('--intrinsics_path',
    type=str, default=None, help='Path to list of intrinsics paths')
parser.add_argument('--ground_truth_path',
    type=str, default=None, help='Path to list of ground truth depth paths')

# Input settings
parser.add_argument('--n_height',
    type=int, help='Height to resize each sample')
parser.add_argument('--n_width',
    type=int, help='Width to resize each sample')
parser.add_argument('--load_image_triplets',
    action='store_true', help='If set, load image triplets return three images')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 255], help='Range of image intensities after normalization')

# Evaluation settings
parser.add_argument('--max_evaluate_sample',
    type=int, default=-1, help='Maximum number of samples to evaluate')
parser.add_argument('--median_scale_depth',
    action='store_true', help='If set, then scale depth based on ratio of median output and ground truth')
parser.add_argument('--min_evaluate_depth',
    type=float, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--monodepth2_encoder_restore_path',
    type=str, default=None, help='Path to restore monodepth2 encoder from checkpoint')
parser.add_argument('--monodepth2_decoder_restore_path',
    type=str, default=None, help='Path to restore monodepth2 depth decoder from checkpoint')
parser.add_argument('--posenet_model_restore_path',
    type=str, default=None, help='Path to restore pose network from checkpoint')

# Output settings
parser.add_argument('--output_dirpath',
    type=str, required=True, help='Path directory to save outputs')
parser.add_argument('--save_outputs',
    action='store_true', help='If set, then save outputs to file')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set, then save outputs using input filenames')

# Tasks settings
parser.add_argument('--tasks',
    nargs='+', type=str, default=['depth'], help='List of tasks to infer: depth, rigid_flow')

# Hardware settings
parser.add_argument('--device',
    type=str, help='Device to use: gpu, cpu')


args = parser.parse_args()

if __name__ == '__main__':

    run(image_path=args.image_path,
        intrinsics_path=args.intrinsics_path,
        ground_truth_path=args.ground_truth_path,
        # Input settings
        n_height=args.n_height,
        n_width=args.n_width,
        load_image_triplets=args.load_image_triplets,
        normalized_image_range=args.normalized_image_range,
        # Evaluation settings
        max_evaluate_sample=args.max_evaluate_sample,
        median_scale_depth=args.median_scale_depth,
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        # Checkpoint settings
        monodepth2_encoder_restore_path=args.monodepth2_encoder_restore_path,
        monodepth2_decoder_restore_path=args.monodepth2_decoder_restore_path,
        posenet_model_restore_path=args.posenet_model_restore_path,
        # Output settings
        output_dirpath=args.output_dirpath,
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        # Task settings
        tasks=args.tasks,
        # Hardware settings
        device=args.device)
