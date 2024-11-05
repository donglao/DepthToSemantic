import sys, os, cv2
sys.path.insert(0, 'src')
import numpy as np
import data_utils


DATA_SPLIT_DIRPATH = 'data_split'
KITTI_ROOT_DIRPATH = os.path.join('data', 'kitti_raw_data')

EIGEN_TRAIN_PATHS_FILE = os.path.join(DATA_SPLIT_DIRPATH, 'kitti_eigen_train.txt')
EIGEN_VAL_PATHS_FILE = os.path.join(DATA_SPLIT_DIRPATH, 'kitti_eigen_validation.txt')
EIGEN_TEST_PATHS_FILE = os.path.join(DATA_SPLIT_DIRPATH, 'kitti_eigen_test.txt')

TRAIN_REFS_DIRPATH = 'training'
VAL_REFS_DIRPATH = 'validation'
TEST_REFS_DIRPATH = 'testing'

TRAIN_IMAGE0_FILEPATH = os.path.join(TRAIN_REFS_DIRPATH, 'kitti_eigen_train_image0.txt')
TRAIN_IMAGE1_FILEPATH = os.path.join(TRAIN_REFS_DIRPATH, 'kitti_eigen_train_image1.txt')
TRAIN_IMAGES_FILEPATH = os.path.join(TRAIN_REFS_DIRPATH, 'kitti_eigen_train_images.txt')
TRAIN_CAMERA_FILEPATH = os.path.join(TRAIN_REFS_DIRPATH, 'kitti_eigen_train_camera.txt')
TRAIN_GROUND_TRUTH_FILEPATH = os.path.join(TRAIN_REFS_DIRPATH, 'kitti_eigen_train_ground_truth.txt')
VAL_IMAGE0_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'kitti_eigen_val_image0.txt')
VAL_IMAGE1_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'kitti_eigen_val_image1.txt')
VAL_CAMERA_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'kitti_eigen_val_camera.txt')
VAL_GROUND_TRUTH_FILEPATH = os.path.join(VAL_REFS_DIRPATH, 'kitti_eigen_val_ground_truth.txt')
TEST_IMAGE0_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_eigen_test_image0.txt')
TEST_IMAGE1_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_eigen_test_image1.txt')
TEST_CAMERA_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_eigen_test_camera.txt')
TEST_GROUND_TRUTH_FILEPATH = os.path.join(TEST_REFS_DIRPATH, 'kitti_eigen_test_ground_truth.txt')

OUTPUT_DIRPATH = os.path.join('data', 'kitti_raw_data_single_image_depth_stereo')

eigen_train_paths = data_utils.read_paths(EIGEN_TRAIN_PATHS_FILE)
eigen_val_paths = data_utils.read_paths(EIGEN_VAL_PATHS_FILE)
eigen_test_paths = data_utils.read_paths(EIGEN_TEST_PATHS_FILE)

for dirpath in [TRAIN_REFS_DIRPATH, VAL_REFS_DIRPATH, TEST_REFS_DIRPATH]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

eigen_train_image0_paths = []
eigen_train_image1_paths = []
eigen_val_image0_paths = []
eigen_val_image1_paths = []
eigen_test_image0_paths = []
eigen_test_image1_paths = []

data_paths = [
    ['training', TRAIN_IMAGE0_FILEPATH, TRAIN_IMAGE1_FILEPATH, \
        eigen_train_paths, eigen_train_image0_paths, eigen_train_image1_paths],
    ['validation', VAL_IMAGE0_FILEPATH, VAL_IMAGE1_FILEPATH, \
        eigen_val_paths, eigen_val_image0_paths, eigen_val_image1_paths],
    ['testing', TEST_IMAGE0_FILEPATH, TEST_IMAGE1_FILEPATH, \
        eigen_test_paths, eigen_test_image0_paths, eigen_test_image1_paths]]

for paths in data_paths:
    data_split, image0_output_filepath, image1_output_filepath, \
        image_paths, image0_output_paths, image1_output_paths = paths

    for idx in range(len(image_paths)):
        sys.stdout.write(
            'Reading {}/{} {} image file paths...\r'.format(idx+1, len(image_paths), data_split))
        sys.stdout.flush()
        image0_path, image1_path = image_paths[idx].split()
        image0_output_paths.append(os.path.join(KITTI_ROOT_DIRPATH, image0_path)[:-3]+'png')
        image1_output_paths.append(os.path.join(KITTI_ROOT_DIRPATH, image1_path)[:-3]+'png')

    print('Completed reading {}/{} {} image file paths'.format(idx+1, len(image_paths), data_split))

    print('Storing {} image 0 file paths into: {}'.format(data_split, image0_output_filepath))
    with open(image0_output_filepath, "w") as o:
        for idx in range(len(image0_output_paths)):
            o.write(image0_output_paths[idx]+'\n')

    print('Storing {} image 1 file paths into: {}'.format(data_split, image1_output_filepath))
    with open(image1_output_filepath, "w") as o:
        for idx in range(len(image1_output_paths)):
            o.write(image1_output_paths[idx]+'\n')


eigen_train_images_paths = eigen_train_image0_paths+eigen_train_image1_paths
print('Storing all training image file paths into: {}'.format(TRAIN_IMAGES_FILEPATH))
with open(TRAIN_IMAGES_FILEPATH, "w") as o:
    for idx in range(len(eigen_train_images_paths)):
        o.write(eigen_train_images_paths[idx]+'\n')

# Generate camera parameters and depth maps
data_paths = [
    ['training', eigen_train_image0_paths,
        TRAIN_CAMERA_FILEPATH, TRAIN_GROUND_TRUTH_FILEPATH, ],
    ['validation', eigen_val_image0_paths,
        VAL_CAMERA_FILEPATH, VAL_GROUND_TRUTH_FILEPATH, ],
    ['testing', eigen_test_image0_paths,
        TEST_CAMERA_FILEPATH, TEST_GROUND_TRUTH_FILEPATH, ]]

for paths in data_paths:
    data_split, image0_output_paths, \
        camera_output_filepath, ground_truth_output_filepath = paths
    camera_output_paths = []
    ground_truth_output_paths = []
    for idx, image_path in enumerate(image0_output_paths):
        # Extract useful components from paths
        _, _, date, sequence, camera, _, filename = image_path.split(os.sep)
        camera_id = np.int32(camera[-1])
        file_id, ext = os.path.splitext(filename)
        velodyne_path = os.path.join(
            KITTI_ROOT_DIRPATH, date, sequence, 'velodyne_points', 'data', file_id+'.bin')
        calibration_dirpath = os.path.join(KITTI_ROOT_DIRPATH, date)

        # Get focal length and baseline
        camera_dirpath = calibration_dirpath.replace(KITTI_ROOT_DIRPATH, OUTPUT_DIRPATH)
        if not os.path.exists(camera_dirpath):
            os.makedirs(camera_dirpath)

        camera_path = os.path.join(camera_dirpath, 'fB.npy')

        if not os.path.exists(camera_path):
            f, B = data_utils.load_focal_length_baseline(calibration_dirpath, camera_id=2)
            camera = np.concatenate([np.expand_dims(f, axis=-1), np.expand_dims(B, axis=-1)], axis=-1)
            np.save(camera_path, camera)

        camera_output_paths.append(camera_path)

        # Get groundtruth depth from velodyne
        shape = cv2.imread(image_path).shape[0:2]
        ground_truth = data_utils.velodyne2depth(calibration_dirpath, velodyne_path, shape, camera_id=2)

        # Construct ground truth output path
        ground_truth_path = image_path \
                .replace('image', 'ground_truth') \
                .replace(KITTI_ROOT_DIRPATH, OUTPUT_DIRPATH)
        ground_truth_path = ground_truth_path[:-3]+'npy'

        if not os.path.exists(os.path.dirname(ground_truth_path)):
            os.makedirs(os.path.dirname(ground_truth_path))

        np.save(ground_truth_path, ground_truth)
        ground_truth_output_paths.append(ground_truth_path)

        sys.stdout.write(
            'Processing {}/{} {} groundtruth file paths...\r'.format(
                idx+1, len(image0_output_paths), data_split))
        sys.stdout.flush()

    print('Completed processing {}/{} {} groundtruth file paths'.format(
        idx+1, len(image0_output_paths), data_split))

    print('Storing {} groundtruth file paths into: {}'.format(data_split, ground_truth_output_filepath))
    with open(ground_truth_output_filepath, "w") as o:
        for idx in range(len(ground_truth_output_paths)):
            o.write(ground_truth_output_paths[idx]+'\n')

    print('Storing {} focal length baseline filepaths into: {}'.format(data_split, camera_output_filepath))
    with open(camera_output_filepath, "w") as o:
        for idx in range(len(camera_output_paths)):
            o.write(camera_output_paths[idx]+'\n')
