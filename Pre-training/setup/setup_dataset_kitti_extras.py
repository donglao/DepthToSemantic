import os, sys, argparse
import torch
import numpy as np
sys.path.insert(0, 'src')
import datasets, data_utils, loss_utils
from posenet_model import PoseNetModel
from monodepth2_model import Monodepth2Model
from transforms import Transforms


'''
Input paths
'''
KITTI_RAW_DATA_DIRPATH = os.path.join('data', 'kitti_raw_data')

TRAIN_REF_DIRPATH = os.path.join('training', 'kitti')
VAL_REF_DIRPATH = os.path.join('validation', 'kitti')
TEST_REF_DIRPATH = os.path.join('testing', 'kitti')


# Training data file path
TRAIN_IMAGES_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_images_left.txt')
TRAIN_IMAGES_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_images_right.txt')
TRAIN_INTRINSICS_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_intrinsics_left.txt')
TRAIN_INTRINSICS_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_intrinsics_right.txt')

# Nonstatic data meaning camera was non static
TRAIN_NONSTATIC_IMAGES_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_images_left.txt')
TRAIN_NONSTATIC_IMAGES_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_images_right.txt')
TRAIN_NONSTATIC_INTRINSICS_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_intrinsics_left.txt')
TRAIN_NONSTATIC_INTRINSICS_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_intrinsics_right.txt')

'''
Output paths
'''
KITTI_RAW_DATA_DERIVED_DIRPATH = os.path.join(
    'data', 'kitti_raw_data_unsupervised_segmentation')

TRAIN_DEPTH_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_depth_left.txt')
TRAIN_DEPTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_depth_right.txt')

# Nonstatic data meaning camera was non static
TRAIN_NONSTATIC_DEPTH_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_depth_left.txt')
TRAIN_NONSTATIC_DEPTH_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_depth_right.txt')
TRAIN_NONSTATIC_RIGID_FLOWS_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_rigid_flows_left.txt')
TRAIN_NONSTATIC_RIGID_FLOWS_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_rigid_flows_right.txt')


def run(task,
        monodepth2_model,
        posenet_model,
        dataloader,
        load_image_triplets,
        transforms,
        output_paths=None,
        verbose=False):
    '''
    Runs monodepth2 and posenet to output depth or rigid flow
    if output paths are provided, then will save outputs to a predetermined list of paths

    Arg(s):
        task : str
            task to run, depth or rigid_flow
        monodepth2_model : Object
            Monodepth2 class instance
        posenet_model : Object
            PoseNet class instance
        dataloader : torch.utils.data.DataLoader
            dataloader that outputs an image and a range map
        load_image_triplets : bool
            if set then split image into triplet
        transforms : Object
            Transform class instance
        output_paths : list[str]
            list of paths to store output depth
        verbose : bool
            if set then print to console
    Returns:
        list[numpy[float32]] : list of numpy arrays if output paths is None else no return value
    '''

    device = monodepth2_model.device

    outputs = []

    n_sample = len(dataloader)

    if output_paths is not None:
        assert len(output_paths) == n_sample

    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):

            image_info = inputs[-1]
            shape = [1, 1, int(image_info['height']), int(image_info['width'])]

            inputs = inputs[:-1]

            inputs = [
                in_.to(device) for in_ in inputs
            ]

            # Fetch data
            if load_image_triplets:
                image0, image1, image2, intrinsics = inputs

                # Normalize images
                [image0, image1, image2] = transforms.transform(
                    images_arr=[image0, image1, image2],
                    random_transform_probability=0.0)
            else:
                image0, intrinsics = inputs

                # Normalize images
                [image0] = transforms.transform(
                    images_arr=[image0],
                    random_transform_probability=0.0)

            # Forward through the network
            depth = monodepth2_model.forward_depth(image0)
            depth = torch.nn.functional.interpolate(
                input=depth,
                size=shape[-2:],
                mode='bilinear',
                align_corners=True)

            if task == 'depth':
                pass
            elif task == 'rigid_flow':
                pose0to1 = posenet_model.forward(image0, image1)
                pose0to2 = posenet_model.forward(image0, image2)

                points0 = loss_utils.backproject_to_camera(depth, intrinsics, shape)

                # Get rigid flow based on projected coordinates
                n_batch, _, n_height, n_width = shape
                xy0 = loss_utils.meshgrid(
                    n_batch,
                    n_height,
                    n_width,
                    homogeneous=False,
                    device=device)

                # Get rigid flow from 0 to 1
                xy0to1 = loss_utils.project_to_pixel(points0, pose0to1, intrinsics, shape)
                rigid_flow0to1 = xy0to1 - xy0

                # Get rigid flow from 0 to 2
                xy0to2 = loss_utils.project_to_pixel(points0, pose0to2, intrinsics, shape)
                rigid_flow0to2 = xy0to2 - xy0

                rigid_flows = torch.cat([rigid_flow0to1, rigid_flow0to2], dim=-1)
            else:
                raise ValueError('Unsupported task: {}'.format(task))

            # Convert to numpy (if not converted already)
            if task == 'depth':
                output = np.squeeze(depth.detach().cpu().numpy())
            elif task == 'rigid_flow':
                rigid_flows = np.squeeze(rigid_flows.detach().cpu().numpy())
                output = np.transpose(rigid_flows, (1, 2, 0))
            if verbose:
                print('Processed {}/{} samples'.format(idx + 1, n_sample), end='\r')

            # Return output depths as a list if we do not store them
            if output_paths is None:
                output.append(output)
            else:
                if task == 'depth':
                    data_utils.save_depth(output, output_paths[idx])
                elif task == 'rigid_flow':
                    data_utils.save_flow(output, output_paths[idx])

    if output_paths is None:
        return outputs

def setup_dataset_kitti_training(n_height,
                                 n_width,
                                 normalized_image_range,
                                 monodepth2_encoder_restore_path,
                                 monodepth2_decoder_restore_path,
                                 posenet_model_restore_path,
                                 paths_only):
    '''
    Creates depth and forward and backward rigid flows

    Arg(s):
        n_height : int
            height to resize each image
        n_width : int
            width to resize each image
        normalized_image_range : list[float]
            range of image intensity after normalization
        monodepth2_encoder_restore_path : str
            path to monodepth2 encoder checkpoint
        monodepth2_decoder_restore_path : str
            path to monodepth2 decoder checkpoint
        posenet_model_restore_path : str
            path to posenet checkpoint
        paths_only : bool
            if set, then only produces paths
    '''

    # Read input paths
    train_images_left_paths = data_utils.read_paths(TRAIN_IMAGES_LEFT_FILEPATH)
    train_images_right_paths = data_utils.read_paths(TRAIN_IMAGES_RIGHT_FILEPATH)

    train_intrinsics_left_paths = data_utils.read_paths(TRAIN_INTRINSICS_LEFT_FILEPATH)
    train_intrinsics_right_paths = data_utils.read_paths(TRAIN_INTRINSICS_RIGHT_FILEPATH)

    train_nonstatic_images_left_paths = data_utils.read_paths(TRAIN_NONSTATIC_IMAGES_LEFT_FILEPATH)
    train_nonstatic_images_right_paths = data_utils.read_paths(TRAIN_NONSTATIC_IMAGES_RIGHT_FILEPATH)

    train_nonstatic_intrinsics_left_paths = data_utils.read_paths(TRAIN_NONSTATIC_INTRINSICS_LEFT_FILEPATH)
    train_nonstatic_intrinsics_right_paths = data_utils.read_paths(TRAIN_NONSTATIC_INTRINSICS_RIGHT_FILEPATH)

    # Preallocate output paths
    train_depth_left_paths = [
        train_images_left_path \
            .replace('image', 'predicted_depth')
        for train_images_left_path in train_images_left_paths
    ]

    train_depth_right_paths = [
        train_images_right_path \
            .replace('image', 'predicted_depth')
        for train_images_right_path in train_images_right_paths
    ]

    train_nonstatic_depth_left_paths = [
        train_nonstatic_images_left_path \
            .replace('image', 'predicted_depth')
        for train_nonstatic_images_left_path in train_nonstatic_images_left_paths
    ]

    train_nonstatic_depth_right_paths = [
        train_nonstatic_images_right_path \
            .replace('image', 'predicted_depth')
        for train_nonstatic_images_right_path in train_nonstatic_images_right_paths
    ]

    train_nonstatic_rigid_flows_left_paths = [
        train_nonstatic_images_left_path \
            .replace('image', 'predicted_rigid_flows') \
            .replace('.png', '.npy')
        for train_nonstatic_images_left_path in train_nonstatic_images_left_paths
    ]

    train_nonstatic_rigid_flows_right_paths = [
        train_nonstatic_images_right_path \
            .replace('image', 'predicted_rigid_flows') \
            .replace('.png', '.npy')
        for train_nonstatic_images_right_path in train_nonstatic_images_right_paths
    ]

    # Combine left and right camera paths to a single list
    train_images_paths = train_images_left_paths + train_images_right_paths
    train_intrinsics_paths = train_intrinsics_left_paths + train_intrinsics_right_paths
    train_nonstatic_images_paths = train_nonstatic_images_left_paths + train_nonstatic_images_right_paths
    train_nonstatic_intrinsics_paths = train_nonstatic_intrinsics_left_paths + train_nonstatic_intrinsics_right_paths

    train_depth_paths = train_depth_left_paths + train_depth_right_paths
    train_nonstatic_rigid_flows_paths = train_nonstatic_rigid_flows_left_paths + train_nonstatic_rigid_flows_right_paths

    # Create output directories
    inputs = [
        [
            'rigid_flow',
            train_nonstatic_images_paths,
            train_nonstatic_intrinsics_paths,
            train_nonstatic_rigid_flows_paths
        ], [
            'depth',
            train_images_paths,
            train_intrinsics_paths,
            train_depth_paths
        ]
    ]

    # Set up monodepth2 and posenet
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    monodepth2_model = Monodepth2Model(
        device=device)
    monodepth2_model.eval()

    assert monodepth2_encoder_restore_path is not None and monodepth2_encoder_restore_path != ''
    assert monodepth2_decoder_restore_path is not None and monodepth2_decoder_restore_path != ''

    monodepth2_model.restore_model(
        encoder_restore_path=monodepth2_encoder_restore_path,
        decoder_depth_restore_path=monodepth2_decoder_restore_path)

    posenet_model = PoseNetModel(
        encoder_type='resnet18',
        rotation_parameterization='axis',
        weight_initializer='kaiming_uniform',
        activation_func='relu',
        device=device)
    posenet_model.eval()

    assert posenet_model_restore_path is not None and posenet_model_restore_path != ''

    posenet_model.restore_model(posenet_model_restore_path)

    # Set up transforms
    transforms = Transforms(
        normalized_image_range=normalized_image_range)

    for task, images_paths, intrinsics_paths, output_paths in inputs:

        # Create output directories
        output_dirpaths = set([os.path.dirname(path) for path in output_paths])

        for dirpath in output_dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

        n_sample = len(images_paths)

        # Instantiate dataloader
        dataloader = torch.utils.data.DataLoader(
            datasets.Monodepth2InferenceDataset(
                image_paths=images_paths,
                intrinsics_paths=intrinsics_paths,
                resize_shape=(n_height, n_width),
                load_image_triplets=True,
                return_image_info=True),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        if not paths_only:
            print('Generating {} for {} samples'.format(task, n_sample))

            # Write to pseudo ground truth to disk
            run(task=task,
                monodepth2_model=monodepth2_model,
                posenet_model=posenet_model,
                dataloader=dataloader,
                load_image_triplets=True,
                transforms=transforms,
                output_paths=output_paths,
                verbose=True)

    # Write to file
    print('Storing {} training left depth maps file paths into: {}'.format(
        len(train_depth_left_paths),
        TRAIN_DEPTH_LEFT_FILEPATH))
    data_utils.write_paths(
        TRAIN_DEPTH_LEFT_FILEPATH,
        train_depth_left_paths)

    print('Storing {} training right depth maps file paths into: {}'.format(
        len(train_depth_right_paths),
        TRAIN_DEPTH_RIGHT_FILEPATH))
    data_utils.write_paths(
        TRAIN_DEPTH_RIGHT_FILEPATH,
        train_depth_right_paths)

    print('Storing {} nonstatic training left depth maps file paths into: {}'.format(
        len(train_nonstatic_depth_left_paths),
        TRAIN_NONSTATIC_DEPTH_LEFT_FILEPATH))
    data_utils.write_paths(
        TRAIN_NONSTATIC_DEPTH_LEFT_FILEPATH,
        train_nonstatic_depth_left_paths)

    print('Storing {} nonstatic training right depth maps file paths into: {}'.format(
        len(train_nonstatic_depth_right_paths),
        TRAIN_NONSTATIC_DEPTH_RIGHT_FILEPATH))
    data_utils.write_paths(
        TRAIN_NONSTATIC_DEPTH_RIGHT_FILEPATH,
        train_nonstatic_depth_right_paths)

    print('Storing {} nonstatic training left rigid flows file paths into: {}'.format(
        len(train_nonstatic_rigid_flows_left_paths),
        TRAIN_NONSTATIC_RIGID_FLOWS_LEFT_FILEPATH))
    data_utils.write_paths(
        TRAIN_NONSTATIC_RIGID_FLOWS_LEFT_FILEPATH,
        train_nonstatic_rigid_flows_left_paths)

    print('Storing {} nonstatic training right rigid flows file paths into: {}'.format(
        len(train_nonstatic_rigid_flows_right_paths),
        TRAIN_NONSTATIC_RIGID_FLOWS_RIGHT_FILEPATH))
    data_utils.write_paths(
        TRAIN_NONSTATIC_RIGID_FLOWS_RIGHT_FILEPATH,
        train_nonstatic_rigid_flows_right_paths)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_height',
        type=int, help='Height to resize each sample')
    parser.add_argument('--n_width',
        type=int, help='Width to resize each sample')
    parser.add_argument('--normalized_image_range',
        nargs='+', type=float, default=[0, 255], help='Range of image intensities after normalization')
    parser.add_argument('--monodepth2_encoder_restore_path',
        type=str, default=None, help='Path to restore monodepth2 encoder from checkpoint')
    parser.add_argument('--monodepth2_decoder_restore_path',
        type=str, default=None, help='Path to restore monodepth2 depth decoder from checkpoint')
    parser.add_argument('--posenet_model_restore_path',
        type=str, default=None, help='Path to restore pose network from checkpoint')
    parser.add_argument('--paths_only',
        action='store_true', help='If set, then generate paths only')

    args = parser.parse_args()

    # Create directories for output files
    for dirpath in [TRAIN_REF_DIRPATH, VAL_REF_DIRPATH, TEST_REF_DIRPATH]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    # Set up dataset
    setup_dataset_kitti_training(
        args.n_height,
        args.n_width,
        args.normalized_image_range,
        args.monodepth2_encoder_restore_path,
        args.monodepth2_decoder_restore_path,
        args.posenet_model_restore_path,
        args.paths_only)
