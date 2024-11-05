import os, sys, glob, argparse, cv2
import numpy as np
import multiprocessing as mp
sys.path.insert(0, 'src')
import data_utils


'''
Paths for KITTI dataset
'''
KITTI_RAW_DATA_DIRPATH = os.path.join('data', 'kitti_raw_data')

KITTI_CALIBRATION_FILENAME = 'calib_cam_to_cam.txt'

KITTI_STATIC_FRAMES_FILEPATH = os.path.join('setup', 'kitti_static_frames.txt')
KITTI_STATIC_FRAMES_PATHS = data_utils.read_paths(KITTI_STATIC_FRAMES_FILEPATH)

KITTI_SCENE_FLOW_DIRPATH = os.path.join('data', 'kitti_scene_flow', 'training')

# Expected focal length and baselines for image sizes for KITTI 2012, 2015
F = {
    1226 : 707.0912,
    1242 : 721.5377,
    1241 : 718.856,
    1238 : 718.3351,
    1224 : 707.0493
}
B = {
    1226 : 0.5379045,
    1242 : 0.53272545,
    1241 : 0.5323319,
    1238 : 0.53014046,
    1224 : 0.5372559
}

'''
Output paths
'''
KITTI_RAW_DATA_DERIVED_DIRPATH = os.path.join(
    'data', 'kitti_raw_data_unsupervised_segmentation')

KITTI_SCENE_FLOW_DERIVED_DIRPATH = os.path.join(
    'data', 'kitti_scene_flow_unsupervised_segmentation')

TRAIN_REF_DIRPATH = os.path.join('training', 'kitti')
VAL_REF_DIRPATH = os.path.join('validation', 'kitti')
TEST_REF_DIRPATH = os.path.join('testing', 'kitti')

# Training data file path
TRAIN_IMAGES_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_images_left.txt')
TRAIN_IMAGES_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_images_right.txt')
TRAIN_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_focal_length_baseline_left.txt')
TRAIN_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_focal_length_baseline_right.txt')
TRAIN_INTRINSICS_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_intrinsics_left.txt')
TRAIN_INTRINSICS_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_intrinsics_right.txt')

# Nonstatic data meaning camera was non static
TRAIN_NONSTATIC_IMAGES_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_images_left.txt')
TRAIN_NONSTATIC_IMAGES_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_images_right.txt')
TRAIN_NONSTATIC_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_focal_length_baseline_left.txt')
TRAIN_NONSTATIC_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_focal_length_baseline_right.txt')
TRAIN_NONSTATIC_INTRINSICS_LEFT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_intrinsics_left.txt')
TRAIN_NONSTATIC_INTRINSICS_RIGHT_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'kitti_train_nonstatic_intrinsics_right.txt')

# Testing data file path
TEST_IMAGE_LEFT0_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_image_left0.txt')
TEST_IMAGE_LEFT1_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_image_left1.txt')
TEST_IMAGE_RIGHT0_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_image_right0.txt')
TEST_IMAGE_RIGHT1_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_image_right1.txt')
TEST_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_image.txt')
TEST_DISPARITY_LEFT0_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_disparity_left0.txt')
TEST_FLOW0TO1_LEFT0_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_flow0to1_left0.txt')
TEST_DEPTH_LEFT0_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_depth_left0.txt')
TEST_DEPTH_RIGHT0_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_depth_right0.txt')
TEST_DEPTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_depth.txt')
TEST_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_focal_length_baseline_left.txt')
TEST_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_focal_length_baseline_right.txt')
TEST_FOCAL_LENGTH_BASELINE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_focal_length_baseline.txt')
TEST_INTRINSICS_LEFT_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_intrinsics_left.txt')
TEST_INTRINSICS_RIGHT_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_intrinsics_right.txt')
TEST_INTRINSICS_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'kitti_test_intrinsics.txt')

def map_intrinsics_and_focal_length_baseline(paths_only=False):
    '''
    Map camera intrinsics and focal length + baseline to directories

    Arg(s):
        paths_only : bool
            boolean flag if set then create paths only
    Returns:
        dict[str, str] : sequence dates and camera id to focal length and baseline paths
        dict[str, str] : sequence dates and camera id to camera intrinsics paths
    '''

    # Build a mapping between the camera intrinsics to the directories
    intrinsics_files = sorted(glob.glob(
        os.path.join(KITTI_RAW_DATA_DIRPATH, '*', KITTI_CALIBRATION_FILENAME)))

    intrinsics_dkeys = {}
    focal_length_baseline_dkeys = {}
    for intrinsics_file in intrinsics_files:
        # Example: data/kitti_raw_data_unsupervised_segmentation/2011_09_26/intrinsics_left.npy
        intrinsics_left_path = intrinsics_file \
            .replace(KITTI_RAW_DATA_DIRPATH, KITTI_RAW_DATA_DERIVED_DIRPATH) \
            .replace(KITTI_CALIBRATION_FILENAME, 'intrinsics_left.npy')

        intrinsics_right_path = intrinsics_file \
            .replace(KITTI_RAW_DATA_DIRPATH, KITTI_RAW_DATA_DERIVED_DIRPATH) \
            .replace(KITTI_CALIBRATION_FILENAME, 'intrinsics_right.npy')

        # Example: data/kitti_raw_data_unsupervised_segmentation/2011_09_26/focal_length_baseline2.npy
        focal_length_baseline_left_path = intrinsics_file \
            .replace(KITTI_RAW_DATA_DIRPATH, KITTI_RAW_DATA_DERIVED_DIRPATH) \
            .replace(KITTI_CALIBRATION_FILENAME, 'focal_length_baseline_left.npy')

        focal_length_baseline_right_path = intrinsics_file \
            .replace(KITTI_RAW_DATA_DIRPATH, KITTI_RAW_DATA_DERIVED_DIRPATH) \
            .replace(KITTI_CALIBRATION_FILENAME, 'focal_length_baseline_right.npy')

        sequence_dirpath = os.path.split(intrinsics_left_path)[0]

        if not os.path.exists(sequence_dirpath):
            os.makedirs(sequence_dirpath)

        calib = data_utils.load_calibration(intrinsics_file)
        camera_left = np.reshape(calib['P_rect_02'], [3, 4]).astype(np.float32)
        camera_right = np.reshape(calib['P_rect_03'], [3, 4]).astype(np.float32)

        # Focal length of the cameras
        focal_length_left = camera_left[0, 0]
        focal_length_right = camera_right[0, 0]

        # camera2 is left of camera0 (-6cm) camera3 is right of camera2 (+53.27cm)
        translation_left = camera_left[0, 3] / focal_length_left
        translation_right = camera_right[0, 3] / focal_length_right
        baseline = translation_left - translation_right

        position_left = camera_left[0:3, 3] / focal_length_left
        position_right = camera_right[0:3, 3] / focal_length_right

        # Baseline should be just translation along x
        error_baseline = np.abs(baseline - np.linalg.norm(position_left - position_right))
        assert error_baseline < 0.01, \
            'baseline={}'.format(baseline)

        # Concatenate together as fB
        focal_length_baseline_left = np.concatenate([
            np.expand_dims(focal_length_left, axis=-1),
            np.expand_dims(baseline, axis=-1)],
            axis=-1)

        focal_length_baseline_right = np.concatenate([
            np.expand_dims(focal_length_right, axis=-1),
            np.expand_dims(baseline, axis=-1)],
            axis=-1)

        # Extract camera parameters
        intrinsics_left = camera_left[:3, :3]
        intrinsics_right = camera_right[:3, :3]

        # Store as numpy
        if not paths_only:
            np.save(focal_length_baseline_left_path, focal_length_baseline_left)
            np.save(focal_length_baseline_right_path, focal_length_baseline_right)

            np.save(intrinsics_left_path, intrinsics_left)
            np.save(intrinsics_right_path, intrinsics_right)

        # Add as keys to instrinsics and focal length baseline dictionaries
        sequence_date = intrinsics_file.split(os.sep)[2]

        focal_length_baseline_dkeys[(sequence_date, 'image_02')] = focal_length_baseline_left_path
        focal_length_baseline_dkeys[(sequence_date, 'image_03')] = focal_length_baseline_right_path

        intrinsics_dkeys[(sequence_date, 'image_02')] = intrinsics_left_path
        intrinsics_dkeys[(sequence_date, 'image_03')] = intrinsics_right_path

    return focal_length_baseline_dkeys, intrinsics_dkeys

def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple[str]
            path to image path at current time step,
            path to image path at previous time step,
            path to image path at next time step,
            path to focal length and baseline between stereo pair,
            path to camera intrinsics,
            boolean flag if set then create paths only
    Returns:
        str : output concatenated image path
        str : focal length and baseline path
        str : camera intrinsics path
    '''

    image_curr_path, \
        image_prev_path, \
        image_next_path, \
        focal_length_baseline_path, \
        intrinsics_path, \
        paths_only = inputs

    if not paths_only:
        # Read images and concatenate together
        image_curr = cv2.imread(image_curr_path)
        image_prev = cv2.imread(image_prev_path)
        image_next = cv2.imread(image_next_path)
        images = np.concatenate([image_curr, image_prev, image_next], axis=1)

    images_path = image_curr_path \
        .replace(KITTI_RAW_DATA_DIRPATH, KITTI_RAW_DATA_DERIVED_DIRPATH)

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

    return (images_path,
            focal_length_baseline_path,
            intrinsics_path)

def filter_static_frames(inputs):
    '''
    Return tuple of list of paths containing the nonstatic scenes

    Arg(s):
        inputs : tuple(list[str])
            image paths
            focal length/ baseline paths
            intrinsics paths
    Returns:
        list[str] : nonstatic image paths
        list[str] : nonstatic focal length and baseline paths
        list[str] : nonstatic intrinsics paths
        int : number of frames removed
    '''

    # Process static frames file
    kitti_static_frames_parts = []
    for path in KITTI_STATIC_FRAMES_PATHS:
        parts = path.split(' ')
        kitti_static_frames_parts.append((parts[1], parts[2]))

    image_paths, \
        focal_length_baseline_paths, \
        intrinsics_paths = inputs

    n_removed = 0
    n_sample = len(image_paths)

    assert n_sample == len(focal_length_baseline_paths)
    assert n_sample == len(intrinsics_paths)

    # Allocate lists to keep nonstatic paths
    nonstatic_image_paths = []
    nonstatic_focal_length_baseline_paths = []
    nonstatic_intrinsics_paths = []

    for idx in range(n_sample):
        image_path = image_paths[idx]
        focal_length_baseline_path = focal_length_baseline_paths[idx]
        intrinsics_path = intrinsics_paths[idx]

        is_static = False

        # Flag if file is in static frames file
        for parts in kitti_static_frames_parts:
            # If the sequence and image id are in image path
            if parts[0] in image_path and parts[1] in image_path:
                is_static = True
                break

        if is_static:
            n_removed = n_removed + 1
        else:
            nonstatic_image_paths.append(image_path)
            nonstatic_focal_length_baseline_paths.append(focal_length_baseline_path)
            nonstatic_intrinsics_paths.append(intrinsics_path)

        sys.stdout.write(
            'Processed {}/{} examples \r'.format(idx + 1, n_sample))
        sys.stdout.flush()

    print('Removed {} static frames from {} examples'.format(n_removed, n_sample))

    return (nonstatic_image_paths,
            nonstatic_focal_length_baseline_paths,
            nonstatic_intrinsics_paths)

def setup_dataset_kitti_training(paths_only=False, n_thread=8):
    '''
    Fetch image, intrinsics, focal length and baseline paths for training

    Arg(s):
        paths_only : bool
            if set, then only produces paths
        n_thread : int
            number of threads to use in multiprocessing
    '''

    # Get intrinsics, focal length and baseline from calibration
    focal_length_baseline_dkeys, intrinsics_dkeys = \
        map_intrinsics_and_focal_length_baseline(paths_only=paths_only)

    # Allocate list to hold output paths
    train_images_left_paths = []
    train_images_right_paths = []
    train_focal_length_baseline_left_paths = []
    train_focal_length_baseline_right_paths = []
    train_intrinsics_left_paths = []
    train_intrinsics_right_paths = []

    '''
    Fetch paths for training
    '''
    # Example: data/kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync
    sequence_date_dirpaths = sorted(glob.glob(
        os.path.join(os.path.join(KITTI_RAW_DATA_DIRPATH, '*', '*/'))))

    # Iterate through KITTI raw data directories
    for sequence_date_dirpath in sequence_date_dirpaths:
        # Extract date of the sequence date
        sequence_date = sequence_date_dirpath.split(os.sep)[-3]

        image_left_paths = sorted(glob.glob(
            os.path.join(sequence_date_dirpath, 'image_02', 'data', '*.png')))

        image_right_paths = sorted(glob.glob(
            os.path.join(sequence_date_dirpath, 'image_03', 'data', '*.png')))

        n_sample = len(image_left_paths)

        # Check that data streams are aligned
        assert n_sample == len(image_right_paths)

        for image_left_path, image_right_path in zip(image_left_paths, image_right_paths):
            # And that file names match
            assert os.path.basename(image_left_path) == os.path.basename(image_right_path)

        # Same intrinsics, focal length and baseline for sequence
        intrinsics_left_paths = \
            [intrinsics_dkeys[(sequence_date, 'image_02')]] * n_sample
        intrinsics_right_paths = \
            [intrinsics_dkeys[(sequence_date, 'image_03')]] * n_sample

        focal_length_baseline_left_paths = \
            [focal_length_baseline_dkeys[(sequence_date, 'image_02')]] * n_sample
        focal_length_baseline_right_paths = \
            [focal_length_baseline_dkeys[(sequence_date, 'image_03')]] * n_sample

        # Add paths to running lists
        data_left_paths = \
            (image_left_paths, focal_length_baseline_left_paths, intrinsics_left_paths)
        data_right_paths = \
            (image_right_paths, focal_length_baseline_right_paths, intrinsics_right_paths)

        # Save triplets of images as frames
        for camera_id, data_paths in zip([0, 1], [data_left_paths, data_right_paths]):

            # Unpack data
            image_paths, focal_length_baseline_paths, intrinsics_paths = data_paths
            pool_inputs = []

            for idx in range(1, n_sample-1):
                # Fetch data tuple to be processed
                image_curr_path = image_paths[idx]
                image_prev_path = image_paths[idx-1]
                image_next_path = image_paths[idx+1]

                focal_length_baseline_path = focal_length_baseline_paths[idx]
                intrinsics_path = intrinsics_paths[idx]

                pool_inputs.append((
                    image_curr_path,
                    image_prev_path,
                    image_next_path,
                    focal_length_baseline_path,
                    intrinsics_path,
                    paths_only))

            with mp.Pool(n_thread) as pool:
                pool_results = pool.map(process_frame, pool_inputs)

            for result in pool_results:
                images_path, \
                    focal_length_baseline_path, \
                    intrinsics_path = result

                # Save paths for left images
                if camera_id == 0:
                    train_images_left_paths.append(images_path)
                    train_intrinsics_left_paths.append(intrinsics_path)
                    train_focal_length_baseline_left_paths.append(focal_length_baseline_path)
                # Save paths for right images
                else:
                    train_images_right_paths.append(images_path)
                    train_focal_length_baseline_right_paths.append(focal_length_baseline_path)
                    train_intrinsics_right_paths.append(intrinsics_path)

            print('Processed {} samples using KITTI sequence={}'.format(
                n_sample, sequence_date_dirpath))

    # Sort training paths alphabetically
    train_images_left_paths = sorted(train_images_left_paths)
    train_images_right_paths = sorted(train_images_right_paths)
    train_focal_length_baseline_left_paths = sorted(train_focal_length_baseline_left_paths)
    train_focal_length_baseline_right_paths = sorted(train_focal_length_baseline_right_paths)
    train_intrinsics_left_paths = sorted(train_intrinsics_left_paths)
    train_intrinsics_right_paths = sorted(train_intrinsics_right_paths)

    '''
    Filter out static frames from training set
    '''
    print('Filtering static frames from the left camera training set')
    filtered_data_left = filter_static_frames((
        train_images_left_paths,
        train_focal_length_baseline_left_paths,
        train_intrinsics_left_paths))

    train_nonstatic_images_left_paths, \
        train_nonstatic_focal_length_baseline_left_paths, \
        train_nonstatic_intrinsics_left_paths = filtered_data_left

    print('Filtering static frames from the right camera training set')
    filtered_data_right = filter_static_frames((
        train_images_right_paths,
        train_focal_length_baseline_right_paths,
        train_intrinsics_right_paths))

    train_nonstatic_images_right_paths, \
        train_nonstatic_focal_length_baseline_right_paths, \
        train_nonstatic_intrinsics_right_paths = filtered_data_right

    # Check that both streams still have the same number of examples
    assert len(train_nonstatic_images_left_paths) == len(train_nonstatic_images_right_paths)
    assert len(train_nonstatic_focal_length_baseline_left_paths) == len(train_nonstatic_focal_length_baseline_right_paths)
    assert len(train_nonstatic_intrinsics_left_paths) == len(train_nonstatic_intrinsics_right_paths)

    '''
    Assert that the filtered paths are aligned
    '''
    nonstatic_image_paths = zip(
        train_nonstatic_images_left_paths,
        train_nonstatic_images_right_paths)

    for left_path, right_path in nonstatic_image_paths:
        # Check filename and sequence date
        assert os.path.basename(left_path) == os.path.basename(right_path)
        assert left_path.split(os.sep)[-4] == right_path.split(os.sep)[-4]

    nonstatic_focal_length_baseline_paths = zip(
        train_nonstatic_focal_length_baseline_left_paths,
        train_nonstatic_focal_length_baseline_right_paths)

    for left_path, right_path in nonstatic_focal_length_baseline_paths:
        # Check sequence date
        assert left_path.split(os.sep)[-2] == right_path.split(os.sep)[-2]

    nonstatic_intrinsics_paths = zip(
        train_nonstatic_intrinsics_left_paths,
        train_nonstatic_intrinsics_right_paths)

    for left_path, right_path in nonstatic_intrinsics_paths:
        # Check sequence date
        assert left_path.split(os.sep)[-2] == right_path.split(os.sep)[-2]

    '''
    Write paths to file
    '''
    print('Storing {} left training stereo image file paths into: {}'.format(
        len(train_images_left_paths),
        TRAIN_IMAGES_LEFT_FILEPATH))
    data_utils.write_paths(
        TRAIN_IMAGES_LEFT_FILEPATH,
        train_images_left_paths)

    print('Storing {} right training stereo image file paths into: {}'.format(
        len(train_images_right_paths),
        TRAIN_IMAGES_RIGHT_FILEPATH))
    data_utils.write_paths(
        TRAIN_IMAGES_RIGHT_FILEPATH,
        train_images_right_paths)

    print('Storing {} left training focal length baseline file paths into: {}'.format(
        len(train_focal_length_baseline_left_paths),
        TRAIN_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH))
    data_utils.write_paths(
        TRAIN_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH,
        train_focal_length_baseline_left_paths)

    print('Storing {} right training focal length baseline file paths into: {}'.format(
        len(train_focal_length_baseline_right_paths),
        TRAIN_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH))
    data_utils.write_paths(
        TRAIN_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH,
        train_focal_length_baseline_right_paths)

    print('Storing {} left training intrinsics file paths into: {}'.format(
        len(train_intrinsics_left_paths),
        TRAIN_INTRINSICS_LEFT_FILEPATH))
    data_utils.write_paths(
        TRAIN_INTRINSICS_LEFT_FILEPATH,
        train_intrinsics_left_paths)

    print('Storing {} right training intrinsics file paths into: {}'.format(
        len(train_intrinsics_right_paths),
        TRAIN_INTRINSICS_RIGHT_FILEPATH))
    data_utils.write_paths(
        TRAIN_INTRINSICS_RIGHT_FILEPATH,
        train_intrinsics_right_paths)

    print('Storing {} nonstatic left training stereo image file paths into: {}'.format(
        len(train_nonstatic_images_left_paths),
        TRAIN_NONSTATIC_IMAGES_LEFT_FILEPATH))
    data_utils.write_paths(
        TRAIN_NONSTATIC_IMAGES_LEFT_FILEPATH,
        train_nonstatic_images_left_paths)

    print('Storing {} nonstatic right training stereo image file paths into: {}'.format(
        len(train_nonstatic_images_right_paths),
        TRAIN_NONSTATIC_IMAGES_RIGHT_FILEPATH))
    data_utils.write_paths(
        TRAIN_NONSTATIC_IMAGES_RIGHT_FILEPATH,
        train_nonstatic_images_right_paths)

    print('Storing {} nonstatic left training focal length baseline file paths into: {}'.format(
        len(train_nonstatic_focal_length_baseline_left_paths),
        TRAIN_NONSTATIC_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH))
    data_utils.write_paths(
        TRAIN_NONSTATIC_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH,
        train_nonstatic_focal_length_baseline_left_paths)

    print('Storing {} nonstatic right training focal length baseline file paths into: {}'.format(
        len(train_nonstatic_focal_length_baseline_right_paths),
        TRAIN_NONSTATIC_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH))
    data_utils.write_paths(
        TRAIN_NONSTATIC_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH,
        train_nonstatic_focal_length_baseline_right_paths)

    print('Storing {} nonstatic left training intrinsics file paths into: {}'.format(
        len(train_nonstatic_intrinsics_left_paths),
        TRAIN_NONSTATIC_INTRINSICS_LEFT_FILEPATH))
    data_utils.write_paths(
        TRAIN_NONSTATIC_INTRINSICS_LEFT_FILEPATH,
        train_nonstatic_intrinsics_left_paths)

    print('Storing {} nonstatic right training intrinsics file paths into: {}'.format(
        len(train_nonstatic_intrinsics_right_paths),
        TRAIN_NONSTATIC_INTRINSICS_RIGHT_FILEPATH))
    data_utils.write_paths(
        TRAIN_NONSTATIC_INTRINSICS_RIGHT_FILEPATH,
        train_nonstatic_intrinsics_right_paths)

def setup_dataset_kitti_testing(paths_only=False, n_thread=8):
    '''
    Fetch image, ground truth (convert from disparity to depth) paths for testing

    Arg(s):
        paths_only : bool
            if set, then only produces paths
    '''

    # Fetch KITTI scene flow (2015) dataset paths
    test_image_left0_paths = sorted(
        glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'image_2', '*_10.png')))
    test_image_left1_paths = sorted(
        glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'image_2', '*_11.png')))
    test_image_right0_paths = sorted(
        glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'image_3', '*_10.png')))
    test_image_right1_paths = sorted(
        glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'image_3', '*_11.png')))
    test_disparity_left0_paths = sorted(
        glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'disp_occ_0', '*_10.png')))
    test_disparity_right0_paths = sorted(
        glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'disp_occ_1', '*_10.png')))
    test_flow0to1_left0_paths = sorted(
        glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'flow_occ', '*_10.png')))
    test_calibration_paths = sorted(
        glob.glob(os.path.join(KITTI_SCENE_FLOW_DIRPATH, 'calib_cam_to_cam', '*.txt')))

    n_sample = len(test_image_left0_paths)

    data_paths = [
        test_image_left0_paths,
        test_image_left1_paths,
        test_image_right0_paths,
        test_image_right1_paths,
        test_disparity_left0_paths,
        test_disparity_right0_paths,
        test_flow0to1_left0_paths,
        test_calibration_paths
    ]

    for paths in data_paths:
        assert len(paths) == n_sample

    test_data_paths = zip(
        test_image_left0_paths,
        test_image_left1_paths,
        test_image_right0_paths,
        test_image_right1_paths,
        test_disparity_left0_paths,
        test_disparity_right0_paths,
        test_flow0to1_left0_paths,
        test_calibration_paths)

    '''
    Derive depth, intrinsics, focal length and baseline from data
    '''
    test_depth_left0_paths = []
    test_depth_right0_paths = []
    test_intrinsics_left_paths = []
    test_intrinsics_right_paths = []
    test_focal_length_baseline_left_paths = []
    test_focal_length_baseline_right_paths = []

    for data_paths in test_data_paths:

        # Unpack data paths and check filenames
        _, _, _, _, \
            disparity_left0_path, \
            disparity_right0_path, \
            _, \
            calibration_path = data_paths

        filename = os.path.splitext(os.path.basename(calibration_path))[0]

        for path in data_paths:
            assert filename in path

        '''
        Extract intrinsics and convert disparity to depth
        '''
        # Load calibration
        calibration = data_utils.load_calibration(calibration_path)

        # Obtain calibration for camera 0 and camera 1
        camera_left = np.reshape(np.asarray(calibration['P_rect_02'], np.float32), [3, 4])
        camera_right = np.reshape(np.asarray(calibration['P_rect_03'], np.float32), [3, 4])

        # Extract camera parameters
        intrinsics_left = camera_left[:3, :3]
        intrinsics_right = camera_right[:3, :3]

        # Load disparity
        disparity_left0 = data_utils.load_disparity(disparity_left0_path)
        disparity_right0 = data_utils.load_disparity(disparity_right0_path)

        assert disparity_left0.shape == disparity_right0.shape
        _, n_width = disparity_left0.shape

        # Assert that focal length and baseline match expected for image size
        focal_length_left = camera_left[0, 0]
        focal_length_right = camera_right[0, 0]

        assert np.abs(focal_length_left - F[n_width]) < 0.01, \
            'focal_length_left={}, F={}'.format(focal_length_left, F[n_width])

        assert np.abs(focal_length_right - F[n_width]) < 0.01, \
            'focal_length_right={}, F={}'.format(focal_length_right, F[n_width])

        position_left = camera_left[0:3, 3] / focal_length_left
        position_right = camera_right[0:3, 3] / focal_length_right

        baseline = np.linalg.norm(position_left - position_right)

        assert np.abs(baseline - B[n_width]) < 0.01, \
            'baseline={}, B={}'.format(baseline, B[n_width])

        # Concatenate together as fB
        focal_length_baseline_left = np.concatenate([
            np.expand_dims(focal_length_left, axis=-1),
            np.expand_dims(baseline, axis=-1)],
            axis=-1)

        focal_length_baseline_right = np.concatenate([
            np.expand_dims(focal_length_right, axis=-1),
            np.expand_dims(baseline, axis=-1)],
            axis=-1)

        # Convert disparity to depth
        if not paths_only:
            depth_left0 = np.where(
                disparity_left0 > 0,
                (focal_length_left * baseline) / disparity_left0,
                0.0)
            depth_right0 = np.where(
                disparity_right0 > 0,
                (focal_length_right * baseline) / disparity_right0,
                0.0)

        '''
        Save intrinsics
        '''
        intrinsics_left_path = calibration_path \
            .replace(KITTI_SCENE_FLOW_DIRPATH, KITTI_SCENE_FLOW_DERIVED_DIRPATH) \
            .replace('calib_cam_to_cam', 'camera_left')[:-3] + 'npy'

        intrinsics_right_path = calibration_path \
            .replace(KITTI_SCENE_FLOW_DIRPATH, KITTI_SCENE_FLOW_DERIVED_DIRPATH) \
            .replace('calib_cam_to_cam', 'camera_right')[:-3] + 'npy'

        focal_length_baseline_left_path = calibration_path \
            .replace(KITTI_SCENE_FLOW_DIRPATH, KITTI_SCENE_FLOW_DERIVED_DIRPATH) \
            .replace('calib_cam_to_cam', 'focal_length_baseline_left')[:-3] + 'npy'

        focal_length_baseline_right_path = calibration_path \
            .replace(KITTI_SCENE_FLOW_DIRPATH, KITTI_SCENE_FLOW_DERIVED_DIRPATH) \
            .replace('calib_cam_to_cam', 'focal_length_baseline_right')[:-3] + 'npy'

        if not paths_only:
            intrinsics_left_dirpath = os.path.dirname(intrinsics_left_path)
            intrinsics_right_dirpath = os.path.dirname(intrinsics_right_path)

            focal_length_baseline_left_dirpath = \
                os.path.dirname(focal_length_baseline_left_path)
            focal_length_baseline_right_dirpath = \
                os.path.dirname(focal_length_baseline_right_path)

            dirpaths = [
                intrinsics_left_dirpath,
                intrinsics_right_dirpath,
                focal_length_baseline_left_dirpath,
                focal_length_baseline_right_dirpath
            ]

            for dirpath in dirpaths:
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)

            np.save(intrinsics_left_path, intrinsics_left)
            np.save(intrinsics_right_path, intrinsics_right)

            np.save(focal_length_baseline_left_path, focal_length_baseline_left)
            np.save(focal_length_baseline_right_path, focal_length_baseline_right)

        test_intrinsics_left_paths.append(intrinsics_left_path)
        test_intrinsics_right_paths.append(intrinsics_right_path)

        test_focal_length_baseline_left_paths.append(focal_length_baseline_left_path)
        test_focal_length_baseline_right_paths.append(focal_length_baseline_right_path)

        '''
        Save depth
        '''
        depth_left0_path = disparity_left0_path \
            .replace(KITTI_SCENE_FLOW_DIRPATH, KITTI_SCENE_FLOW_DERIVED_DIRPATH) \
            .replace('disp_occ_0', 'depth_left')[:-3] + 'png'

        depth_right0_path = disparity_right0_path \
            .replace(KITTI_SCENE_FLOW_DIRPATH, KITTI_SCENE_FLOW_DERIVED_DIRPATH) \
            .replace('disp_occ_1', 'depth_right')[:-3] + 'png'

        if not paths_only:
            depth_left0_dirpath = os.path.dirname(depth_left0_path)
            depth_right0_dirpath = os.path.dirname(depth_right0_path)

            if not os.path.exists(depth_left0_dirpath):
                os.makedirs(depth_left0_dirpath)

            if not os.path.exists(depth_right0_dirpath):
                os.makedirs(depth_right0_dirpath)

            data_utils.save_depth(depth_left0, depth_left0_path, multiplier=256.0)
            data_utils.save_depth(depth_right0, depth_right0_path, multiplier=256.0)

        test_depth_left0_paths.append(depth_left0_path)
        test_depth_right0_paths.append(depth_right0_path)

    # Concatenate paths together for single image case
    test_image_paths = \
        test_image_left0_paths + \
        test_image_right0_paths
    test_depth_paths = \
        test_depth_left0_paths + \
        test_depth_right0_paths
    test_focal_length_baseline_paths = \
        test_focal_length_baseline_left_paths + \
        test_focal_length_baseline_right_paths
    test_intrinsics_paths = \
        test_intrinsics_left_paths + \
        test_intrinsics_right_paths

    '''
    Write paths to file
    '''
    print('Storing {} left testing stereo image at time 0 file paths into: {}'.format(
        len(test_image_left0_paths),
        TEST_IMAGE_LEFT0_FILEPATH))
    data_utils.write_paths(
        TEST_IMAGE_LEFT0_FILEPATH,
        test_image_left0_paths)

    print('Storing {} left testing stereo image at time 1 file paths into: {}'.format(
        len(test_image_left1_paths),
        TEST_IMAGE_LEFT1_FILEPATH))
    data_utils.write_paths(
        TEST_IMAGE_LEFT1_FILEPATH,
        test_image_left1_paths)

    print('Storing {} right testing stereo image at time 0 file paths into: {}'.format(
        len(test_image_right0_paths),
        TEST_IMAGE_RIGHT0_FILEPATH))
    data_utils.write_paths(
        TEST_IMAGE_RIGHT0_FILEPATH,
        test_image_right0_paths)

    print('Storing {} right testing stereo image at time 1 file paths into: {}'.format(
        len(test_image_right1_paths),
        TEST_IMAGE_RIGHT1_FILEPATH))
    data_utils.write_paths(
        TEST_IMAGE_RIGHT1_FILEPATH,
        test_image_right1_paths)

    print('Storing {} testing image file paths into: {}'.format(
        len(test_image_paths),
        TEST_IMAGE_FILEPATH))
    data_utils.write_paths(
        TEST_IMAGE_FILEPATH,
        test_image_paths)

    print('Storing {} left testing disparity at time 0 file paths into: {}'.format(
        len(test_disparity_left0_paths),
        TEST_DISPARITY_LEFT0_FILEPATH))
    data_utils.write_paths(
        TEST_DISPARITY_LEFT0_FILEPATH,
        test_disparity_left0_paths)

    print('Storing {} left testing flow from time 0 to 1 file paths into: {}'.format(
        len(test_flow0to1_left0_paths),
        TEST_FLOW0TO1_LEFT0_FILEPATH))
    data_utils.write_paths(
        TEST_FLOW0TO1_LEFT0_FILEPATH,
        test_flow0to1_left0_paths)

    print('Storing {} left testing depth at time 0 file paths into: {}'.format(
        len(test_depth_left0_paths),
        TEST_DEPTH_LEFT0_FILEPATH))
    data_utils.write_paths(
        TEST_DEPTH_LEFT0_FILEPATH,
        test_depth_left0_paths)

    print('Storing {} right testing depth at time 0 file paths into: {}'.format(
        len(test_depth_right0_paths),
        TEST_DEPTH_RIGHT0_FILEPATH))
    data_utils.write_paths(
        TEST_DEPTH_RIGHT0_FILEPATH,
        test_depth_right0_paths)

    print('Storing {} testing depth file paths into: {}'.format(
        len(test_depth_paths),
        TEST_DEPTH_FILEPATH))
    data_utils.write_paths(
        TEST_DEPTH_FILEPATH,
        test_depth_paths)

    print('Storing {} left testing focal length baseline file paths into: {}'.format(
        len(test_focal_length_baseline_left_paths),
        TEST_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH))
    data_utils.write_paths(
        TEST_FOCAL_LENGTH_BASELINE_LEFT_FILEPATH,
        test_focal_length_baseline_left_paths)

    print('Storing {} right testing focal length baseline file paths into: {}'.format(
        len(test_focal_length_baseline_right_paths),
        TEST_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH))
    data_utils.write_paths(
        TEST_FOCAL_LENGTH_BASELINE_RIGHT_FILEPATH,
        test_focal_length_baseline_right_paths)

    print('Storing {} testing focal length baseline file paths into: {}'.format(
        len(test_focal_length_baseline_paths),
        TEST_FOCAL_LENGTH_BASELINE_FILEPATH))
    data_utils.write_paths(
        TEST_FOCAL_LENGTH_BASELINE_FILEPATH,
        test_focal_length_baseline_paths)

    print('Storing {} left testing intrinsics file paths into: {}'.format(
        len(test_intrinsics_left_paths),
        TEST_INTRINSICS_LEFT_FILEPATH))
    data_utils.write_paths(
        TEST_INTRINSICS_LEFT_FILEPATH,
        test_intrinsics_left_paths)

    print('Storing {} right testing intrinsics file paths into: {}'.format(
        len(test_intrinsics_right_paths),
        TEST_INTRINSICS_RIGHT_FILEPATH))
    data_utils.write_paths(
        TEST_INTRINSICS_RIGHT_FILEPATH,
        test_intrinsics_right_paths)

    print('Storing {} testing intrinsics file paths into: {}'.format(
        len(test_intrinsics_paths),
        TEST_INTRINSICS_FILEPATH))
    data_utils.write_paths(
        TEST_INTRINSICS_FILEPATH,
        test_intrinsics_paths)


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
    setup_dataset_kitti_training(args.paths_only, args.n_thread)

    setup_dataset_kitti_testing(args.paths_only)
