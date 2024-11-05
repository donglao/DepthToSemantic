import os, sys, glob, argparse, cv2, json
import numpy as np
import multiprocessing as mp
from PIL import Image
sys.path.insert(0, 'src')
import data_utils


'''
Paths for Cityscapes dataset
'''
CITYSCAPES_DATA_DIRPATH = os.path.join('data', 'cityscapes')

'''
Output paths
'''
CITYSCAPES_DATA_DERIVED_DIRPATH = os.path.join(
    'data',
    'cityscapes_derived')

TRAIN_REF_DIRPATH = os.path.join('training', 'cityscapes')
VAL_REF_DIRPATH = os.path.join('validation', 'cityscapes')
TEST_REF_DIRPATH = os.path.join('testing', 'cityscapes')

# Training data file path
TRAIN_IMAGES_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'cityscapes_train_images_left.txt')
TRAIN_IMAGES_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'cityscapes_train_images_right.txt')
TRAIN_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'cityscapes_train_intrinsics.txt')
TRAIN_FOCAL_LENGTH_BASELINE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'cityscapes_train_focal_length_baseline.txt')
TRAIN_DISPARITY_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'cityscapes_train_disparity.txt')
TRAIN_DEPTH_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'cityscapes_train_depth.txt')
TRAIN_IMAGES_ALL_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'cityscapes_train_images_all.txt')
TRAIN_INTRINSICS_ALL_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'cityscapes_train_intrinsics_all.txt')


def convert_intrinsics_and_focal_length_baseline(path, paths_only=False):
    '''
    Load intrinsics and baseline in JSON and saves in numpy

    Arg(s):
        path : str
            path to calibration JSON file
        paths_only : bool
            boolean flag if set then create paths only
    Returns:
        str : path to calibration numpy file
        str : path to focal length and baseline numpy file
    '''

    # Load intrinsics from JSON
    with open(path) as json_file:
        json_dict = json.load(json_file)
        intrinsics_dict = json_dict['intrinsic']
        extrinsics_dict = json_dict['extrinsic']

    output_path_intrinsics = path \
        .replace(CITYSCAPES_DATA_DIRPATH, CITYSCAPES_DATA_DERIVED_DIRPATH) \
        .replace('json', 'npy')

    filename_json = os.path.basename(path)
    filename_focal_length_baseline = \
        filename_json.replace('camera', 'focal_length_baseline')

    output_path_focal_length_baseline = path \
        .replace(CITYSCAPES_DATA_DIRPATH, CITYSCAPES_DATA_DERIVED_DIRPATH) \
        .replace(filename_json, filename_focal_length_baseline) \
        .replace('json', 'npy')

    output_dirpath = os.path.dirname(output_path_intrinsics)

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # Create intrinsics matrix
    intrinsics = np.eye(3)

    intrinsics[0, 0] = intrinsics_dict['fx']
    intrinsics[1, 1] = intrinsics_dict['fy']
    intrinsics[0, 2] = intrinsics_dict['u0'] * 0.60
    intrinsics[1, 2] = intrinsics_dict['v0'] * 0.60

    # Create focal length and baseline
    focal_length_baseline = np.array([intrinsics_dict['fx'], extrinsics_dict['baseline']])

    # Store as numpy
    if not paths_only:
        np.save(output_path_intrinsics, intrinsics)
        np.save(output_path_focal_length_baseline, focal_length_baseline)

    return (output_path_intrinsics, output_path_focal_length_baseline)

def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple[str]
            path to image path at current time step,
            path to image path at previous time step,
            path to image path at next time step,
            boolean flag if set then create paths only
    Returns:
        str : output concatenated image path
    '''

    image_curr_path, \
        image_prev_path, \
        image_next_path, \
        paths_only = inputs

    if not paths_only:
        # Read images and concatenate together
        image_curr = cv2.imread(image_curr_path)
        image_prev = cv2.imread(image_prev_path)
        image_next = cv2.imread(image_next_path)

        n_height, n_width = image_curr.shape[0:2]

        crop_height_start = int(n_height * 0.10)
        crop_height_end = int(n_height * 0.70)
        crop_width_start = int(n_width * 0.10)
        crop_width_end = int(n_width * 0.70)

        image_curr = image_curr[crop_height_start:crop_height_end, crop_width_start:crop_width_end, :]
        image_prev = image_prev[crop_height_start:crop_height_end, crop_width_start:crop_width_end, :]
        image_next = image_next[crop_height_start:crop_height_end, crop_width_start:crop_width_end, :]

        images = np.concatenate([image_curr, image_prev, image_next], axis=1)

    images_path = image_curr_path \
        .replace(CITYSCAPES_DATA_DIRPATH, CITYSCAPES_DATA_DERIVED_DIRPATH)

    # Create output directories
    output_dirpath = os.path.dirname(images_path)

    if not os.path.exists(output_dirpath):
        try:
            os.makedirs(output_dirpath)
        except FileExistsError:
            pass

    if not paths_only:
        # Write to disk
        cv2.imwrite(images_path, images)

    return images_path

def process_disparity(inputs, paths_only=False):
    '''
    Processes a single disparity

    Arg(s):
        inputs : tuple[str]
            path to disparity at current time step,
            path to focal length and baseline,
            boolean flag if set then create paths only
    Returns:
        str : output concatenated image path
    '''

    disparity_path, focal_length_baseline_path, paths_only = inputs

    if not paths_only:
        # Read disparity
        disparity = Image.open(disparity_path).convert('I')

        # Read focal length and baseline
        focal_length_baseline = np.load(focal_length_baseline_path)
        focal_length_baseline = np.prod(focal_length_baseline)

        # Convert unsigned int 16/32 to disparity values
        disparity = np.asarray(disparity, np.uint16) - 1
        disparity = np.asarray(disparity / 256.0, np.float32)

        depth = np.where(
            disparity > 0,
            focal_length_baseline / disparity,
            np.zeros_like(disparity))

        depth = np.where(
            depth < 2,
            np.zeros_like(depth),
            depth)

        n_height, n_width = depth.shape
        crop_height_start = int(n_height * 0.10)
        crop_height_end = int(n_height * 0.70)
        crop_width_start = int(n_width * 0.10)
        crop_width_end = int(n_width * 0.70)

        depth = depth[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
        disparity = disparity[crop_height_start:crop_height_end, crop_width_start:crop_width_end]

    output_disparity_path = disparity_path \
        .replace(CITYSCAPES_DATA_DIRPATH, CITYSCAPES_DATA_DERIVED_DIRPATH)

    output_depth_path = disparity_path \
        .replace(CITYSCAPES_DATA_DIRPATH, CITYSCAPES_DATA_DERIVED_DIRPATH) \
        .replace('disparity', 'depth')

    # Create output directories
    output_disparity_dirpath = os.path.dirname(output_disparity_path)
    output_depth_dirpath = os.path.dirname(output_depth_path)

    for output_dirpath in [output_disparity_dirpath, output_depth_dirpath]:

        if not os.path.exists(output_dirpath):
            try:
                os.makedirs(output_dirpath)
            except FileExistsError:
                pass

    if not paths_only:
        # Write to disk
        disparity = Image.fromarray(disparity.astype(np.uint32), mode='I')
        disparity.save(output_disparity_path)

        data_utils.save_depth(depth, output_depth_path)

    return (output_disparity_path, output_depth_path)

def setup_dataset_cityscapes_training(paths_only=False, n_thread=8):
    '''
    Fetch image, intrinsics, focal length and baseline paths for training

    Arg(s):
        paths_only : bool
            if set, then only produces paths
        n_thread : int
            number of threads to use in multiprocessing
    '''

    # Allocate list to hold output paths
    train_images_left_paths = []
    train_images_right_paths = []
    train_intrinsics_paths = []
    train_focal_length_baseline_paths = []
    train_disparity_paths = []
    train_depth_paths = []

    '''
    Fetch directories for training
    '''
    # Example: data/cityscape/leftImg8bit/train/aachen/aachen_000...png
    train_image_left_dirpaths = glob.glob(
        os.path.join(CITYSCAPES_DATA_DIRPATH, 'leftImg8bit', 'train', '*/'))

    train_image_left_extra_dirpaths = glob.glob(
        os.path.join(CITYSCAPES_DATA_DIRPATH, 'leftImg8bit', 'train_extra', '*/'))

    train_image_right_dirpaths = glob.glob(
        os.path.join(CITYSCAPES_DATA_DIRPATH, 'rightImg8bit', 'train', '*/'))

    train_image_right_extra_dirpaths = glob.glob(
        os.path.join(CITYSCAPES_DATA_DIRPATH, 'rightImg8bit', 'train_extra', '*/'))

    train_intrinsics_dirpaths = glob.glob(
        os.path.join(CITYSCAPES_DATA_DIRPATH, 'camera', 'train', '*/'))

    train_intrinsics_extra_dirpaths = glob.glob(
        os.path.join(CITYSCAPES_DATA_DIRPATH, 'camera', 'train_extra', '*/'))

    train_disparity_dirpaths = glob.glob(
        os.path.join(CITYSCAPES_DATA_DIRPATH, 'disparity', 'train', '*/'))

    train_disparity_extra_dirpaths = glob.glob(
        os.path.join(CITYSCAPES_DATA_DIRPATH, 'disparity', 'train_extra', '*/'))

    # Combine train and train extra dirpaths
    train_image_left_dirpaths = \
        sorted(train_image_left_dirpaths + train_image_left_extra_dirpaths)

    train_image_right_dirpaths = \
        sorted(train_image_right_dirpaths + train_image_right_extra_dirpaths)

    train_intrinsics_dirpaths = \
        sorted(train_intrinsics_dirpaths + train_intrinsics_extra_dirpaths)

    train_disparity_dirpaths = \
        sorted(train_disparity_dirpaths + train_disparity_extra_dirpaths)

    n_dirpath = len(train_image_left_dirpaths)

    '''
    Check directories match
    '''
    assert n_dirpath == len(train_image_right_dirpaths)
    assert n_dirpath == len(train_intrinsics_dirpaths)
    assert n_dirpath == len(train_disparity_dirpaths)

    for index in range(n_dirpath):
        image_left_dirpath_basename = os.path.basename(train_image_left_dirpaths[index][:-1])
        image_right_dirpath_basename = os.path.basename(train_image_right_dirpaths[index][:-1])
        intrinsics_dirpath_basename = os.path.basename(train_intrinsics_dirpaths[index][:-1])
        disparity_dirpath_basename = os.path.basename(train_disparity_dirpaths[index][:-1])

        assert image_left_dirpath_basename == image_right_dirpath_basename
        assert image_left_dirpath_basename == intrinsics_dirpath_basename
        assert image_left_dirpath_basename == disparity_dirpath_basename

    '''
    Process each directory
    '''
    for dirpath_index in range(n_dirpath):

        dirpath_basename = \
            os.path.basename(train_image_left_dirpaths[dirpath_index][:-1])

        # Get directory
        image_left_dirpath = train_image_left_dirpaths[dirpath_index]
        image_right_dirpath = train_image_right_dirpaths[dirpath_index]
        intrinsics_dirpath = train_intrinsics_dirpaths[dirpath_index]
        disparity_dirpath = train_disparity_dirpaths[dirpath_index]

        # Fetch data paths
        image_left_paths = \
            sorted(glob.glob(os.path.join(image_left_dirpath, '*.png')))
        image_right_paths = \
            sorted(glob.glob(os.path.join(image_right_dirpath, '*.png')))
        intrinsics_paths = \
            sorted(glob.glob(os.path.join(intrinsics_dirpath, '*.json')))
        disparity_paths = \
            sorted(glob.glob(os.path.join(disparity_dirpath, '*.png')))

        if dirpath_basename == 'troisdorf':
            # Note: troisdorf is missing troisdorf_000000_000073_disparity.png
            image_left_paths.remove(os.path.join(image_left_dirpath, 'troisdorf_000000_000073_leftImg8bit.png'))
            image_right_paths.remove(os.path.join(image_right_dirpath, 'troisdorf_000000_000073_rightImg8bit.png'))
            intrinsics_paths.remove(os.path.join(intrinsics_dirpath, 'troisdorf_000000_000073_camera.json'))

        n_sample = len(image_left_paths)

        assert n_sample == len(image_right_paths), 'Directory={}'.format(dirpath_basename)
        assert n_sample == len(intrinsics_paths), 'Directory={}'.format(dirpath_basename)
        assert n_sample == len(disparity_paths), 'Directory={}'.format(dirpath_basename)

        for index_sample in range(n_sample):
            image_left_ref = os.path.join(*(os.path.basename(image_left_paths[index_sample]).split('_')[0:3]))
            image_right_ref = os.path.join(*(os.path.basename(image_right_paths[index_sample]).split('_')[0:3]))
            intrinsics_ref = os.path.join(*(os.path.basename(intrinsics_paths[index_sample]).split('_')[0:3]))
            disparity_ref = os.path.join(*(os.path.basename(disparity_paths[index_sample]).split('_')[0:3]))

            assert image_left_ref == image_right_ref
            assert image_left_ref == intrinsics_ref
            assert image_left_ref == disparity_ref

        # Process frames
        image_left_pool_inputs = []
        image_right_pool_inputs = []
        disparity_pool_inputs = []

        for index_sample in range(1, n_sample-1):

            # Process intrinsics
            intrinsics_path, \
                focal_length_baseline_path = convert_intrinsics_and_focal_length_baseline(
                    intrinsics_paths[index_sample],
                    paths_only=paths_only)

            train_intrinsics_paths.append(intrinsics_path)
            train_focal_length_baseline_paths.append(focal_length_baseline_path)

            # Fetch data tuple to be processed
            image_left_curr_path = image_left_paths[index_sample]
            image_left_prev_path = image_left_paths[index_sample-1]
            image_left_next_path = image_left_paths[index_sample+1]

            image_right_curr_path = image_right_paths[index_sample]
            image_right_prev_path = image_right_paths[index_sample-1]
            image_right_next_path = image_right_paths[index_sample+1]

            disparity_path = disparity_paths[index_sample]

            # Add image paths to pool
            image_left_pool_inputs.append((
                image_left_curr_path,
                image_left_prev_path,
                image_left_next_path,
                paths_only))

            image_right_pool_inputs.append((
                image_right_curr_path,
                image_right_prev_path,
                image_right_next_path,
                paths_only))

            # Add disparity paths to pool
            disparity_pool_inputs.append((
                disparity_path,
                focal_length_baseline_path,
                paths_only))

        with mp.Pool(n_thread) as pool:
            image_left_pool_results = pool.map(process_frame, image_left_pool_inputs)

            image_right_pool_results = pool.map(process_frame, image_right_pool_inputs)

            disparity_pool_results = pool.map(process_disparity, disparity_pool_inputs)

            for images_path in image_left_pool_results:
                train_images_left_paths.append(images_path)

            for images_path in image_right_pool_results:
                train_images_right_paths.append(images_path)

            for disparity_path, depth_path in disparity_pool_results:
                train_disparity_paths.append(disparity_path)
                train_depth_paths.append(depth_path)

        print('Processed {} samples for {}'.format(
            n_sample, dirpath_basename))

    '''
    Write paths to file
    '''
    print('Storing {} left training stereo images file paths into: {}'.format(
        len(train_images_left_paths),
        TRAIN_IMAGES_LEFT_FILEPATH))
    data_utils.write_paths(
        TRAIN_IMAGES_LEFT_FILEPATH,
        train_images_left_paths)

    print('Storing {} right training stereo images file paths into: {}'.format(
        len(train_images_right_paths),
        TRAIN_IMAGES_RIGHT_FILEPATH))
    data_utils.write_paths(
        TRAIN_IMAGES_RIGHT_FILEPATH,
        train_images_right_paths)

    print('Storing {} training intrinsics file paths into: {}'.format(
        len(train_intrinsics_paths),
        TRAIN_INTRINSICS_FILEPATH))
    data_utils.write_paths(
        TRAIN_INTRINSICS_FILEPATH,
        train_intrinsics_paths)

    print('Storing {} training focal length and baseline file paths into: {}'.format(
        len(train_focal_length_baseline_paths),
        TRAIN_FOCAL_LENGTH_BASELINE_FILEPATH))
    data_utils.write_paths(
        TRAIN_FOCAL_LENGTH_BASELINE_FILEPATH,
        train_focal_length_baseline_paths)

    print('Storing {} training disparity file paths into: {}'.format(
        len(train_disparity_paths),
        TRAIN_DISPARITY_FILEPATH))
    data_utils.write_paths(
        TRAIN_DISPARITY_FILEPATH,
        train_disparity_paths)

    print('Storing {} training depth file paths into: {}'.format(
        len(train_depth_paths),
        TRAIN_DEPTH_FILEPATH))
    data_utils.write_paths(
        TRAIN_DEPTH_FILEPATH,
        train_depth_paths)

    train_images_all_paths = train_images_left_paths + train_images_right_paths
    train_intrinsics_all_paths = train_intrinsics_paths + train_intrinsics_paths

    print('Storing {} all training images file paths into: {}'.format(
        len(train_images_all_paths),
        TRAIN_IMAGES_ALL_FILEPATH))
    data_utils.write_paths(
        TRAIN_IMAGES_ALL_FILEPATH,
        train_images_all_paths)

    print('Storing {} all training intrinsics file paths into: {}'.format(
        len(train_intrinsics_all_paths),
        TRAIN_INTRINSICS_ALL_FILEPATH))
    data_utils.write_paths(
        TRAIN_INTRINSICS_ALL_FILEPATH,
        train_intrinsics_all_paths)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--paths_only', action='store_true', help='If set, then generate paths only')
    parser.add_argument('--n_thread',  type=int, default=8)

    args = parser.parse_args()

    # Create directories for output files
    for dirpath in [TRAIN_REF_DIRPATH, VAL_REF_DIRPATH, TEST_REF_DIRPATH]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    # Set up dataset
    setup_dataset_cityscapes_training(args.paths_only, args.n_thread)
