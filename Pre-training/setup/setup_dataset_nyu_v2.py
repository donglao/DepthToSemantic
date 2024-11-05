import warnings
warnings.filterwarnings("ignore")

import os, sys, glob, cv2, argparse
import multiprocessing as mp
import numpy as np
sys.path.insert(0, 'src')
import data_utils


N_CLUSTER = 1500
O_HEIGHT = 480
O_WIDTH = 640
N_HEIGHT = 416
N_WIDTH = 576
MIN_POINTS = 1100
TEMPORAL_WINDOW = 21
RANDOM_SEED = 1

parser = argparse.ArgumentParser()

parser.add_argument('--temporal_window',          type=int, default=TEMPORAL_WINDOW)
parser.add_argument('--n_height',                 type=int, default=N_HEIGHT)
parser.add_argument('--n_width',                  type=int, default=N_WIDTH)

args = parser.parse_args()


NYU_ROOT_DIRPATH = \
    os.path.join('data', 'nyu_v2')
NYU_OUTPUT_DIRPATH = \
    os.path.join('data', 'nyu_v2_unsupervised_segmentation')

NYU_TEST_IMAGE_SPLIT_FILEPATH = \
    os.path.join('data_split', 'nyu_v2_test_image.txt')
NYU_TEST_DEPTH_SPLIT_FILEPATH = \
    os.path.join('data_split', 'nyu_v2_test_depth.txt')

TRAIN_REF_DIRPATH = os.path.join('training', 'nyu_v2')
VAL_REF_DIRPATH = os.path.join('validation', 'nyu_v2')
TEST_REF_DIRPATH = os.path.join('testing', 'nyu_v2')

TRAIN_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_REF_DIRPATH, 'nyu_v2_train_image.txt')
TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_REF_DIRPATH, 'nyu_v2_train_ground_truth.txt')
TRAIN_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_REF_DIRPATH, 'nyu_v2_train_intrinsics.txt')

VAL_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_image.txt')
VAL_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_ground_truth.txt')
VAL_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_intrinsics.txt')

TEST_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_image.txt')
TEST_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_ground_truth.txt')
TEST_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_intrinsics.txt')


def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple
            image path at time t=0,
            image path at time t=1,
            image path at time t=-1,
            ground truth path at time t=0
    Returns:
        str : output concatenated image path at time t=0
        str : output ground truth path at time t=0
    '''

    image0_path, image1_path, image2_path, ground_truth_path = inputs

    # Read images
    image0 = cv2.imread(image0_path)
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    d_height = O_HEIGHT - args.n_height
    d_width = O_WIDTH - args.n_width

    y_start = d_height // 2
    x_start = d_width // 2
    y_end = y_start + args.n_height
    x_end = x_start + args.n_width

    # Load dense depth
    ground_truth = data_utils.load_depth(ground_truth_path)

    assert image0.shape[0] == ground_truth.shape[0] and image0.shape[1] == ground_truth.shape[1]
    assert image0.shape[0] == O_HEIGHT and image0.shape[1] == O_WIDTH

    if args.n_height != O_HEIGHT or args.n_width != O_WIDTH:
        image0 = image0[y_start:y_end, x_start:x_end, :]
        image1 = image1[y_start:y_end, x_start:x_end, :]
        image2 = image2[y_start:y_end, x_start:x_end, :]
        ground_truth = ground_truth[y_start:y_end, x_start:x_end]

    imagec = np.concatenate([image1, image0, image2], axis=1)

    # Example: nyu/training/depths/raw_data/bedroom_0001/r-1294886360.208451-2996770081.png
    image_output_path = image0_path \
        .replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH)
    ground_truth_output_path = ground_truth_path \
        .replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH) \
        .replace('depth', 'ground_truth')

    image_output_dirpath = os.path.dirname(image_output_path)
    ground_truth_output_dirpath = os.path.dirname(ground_truth_output_path)

    # Create output directories
    output_dirpaths = [
        image_output_dirpath,
        ground_truth_output_dirpath,
    ]

    for dirpath in output_dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

    # Write to file
    cv2.imwrite(image_output_path, imagec)
    data_utils.save_depth(ground_truth, ground_truth_output_path)

    return (image_output_path,
            ground_truth_output_path)


# Create output directories first
dirpaths = [
    NYU_OUTPUT_DIRPATH,
    TRAIN_REF_DIRPATH,
    VAL_REF_DIRPATH,
    TEST_REF_DIRPATH
]

for dirpath in dirpaths:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


'''
Setup intrinsics (values are copied from camera_params.m)
'''
fx_rgb = 518.85790117450188
fy_rgb = 519.46961112127485
cx_rgb = 325.58244941119034
cy_rgb = 253.73616633400465
intrinsic_matrix = np.array([
    [fx_rgb,   0.,     cx_rgb],
    [0.,       fy_rgb, cy_rgb],
    [0.,       0.,     1.    ]], dtype=np.float32)


if args.n_height != O_HEIGHT or args.n_width != O_WIDTH:

    d_height = O_HEIGHT - args.n_height
    d_width = O_WIDTH - args.n_width

    y_start = d_height // 2
    x_start = d_width // 2

    intrinsic_matrix = intrinsic_matrix + [[0.0, 0.0, -x_start],
                                           [0.0, 0.0, -y_start],
                                           [0.0, 0.0, 0.0     ]]

intrinsics_output_path = os.path.join(NYU_OUTPUT_DIRPATH, 'intrinsics.npy')
np.save(intrinsics_output_path, intrinsic_matrix)


'''
Process training paths
'''
train_image_output_paths = []
train_ground_truth_output_paths = []
train_intrinsics_output_paths = [intrinsics_output_path]

train_image_sequences = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'training', 'images', 'raw_data', '*/')))
train_depth_sequences = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'training', 'depths', 'raw_data', '*/')))

w = int(args.temporal_window // 2)

for image_sequence, depth_sequence in zip(train_image_sequences, train_depth_sequences):

    # Fetch image and dense depth from sequence directory
    image_paths = \
        sorted(glob.glob(os.path.join(image_sequence, '*.png')))
    ground_truth_paths = \
        sorted(glob.glob(os.path.join(depth_sequence, '*.png')))

    n_sample = len(image_paths)

    for image_path, ground_truth_path in zip(image_paths, ground_truth_paths):
        assert os.path.join(*(image_path.split(os.sep)[-3:])) == os.path.join(*(image_path.split(os.sep)[-3:]))

    pool_input = [
        (image_paths[idx], image_paths[idx-w], image_paths[idx+w], ground_truth_paths[idx])
        for idx in range(w, n_sample - w)
    ]

    print('Processing {} samples in: {}'.format(n_sample - 2 * w + 1, image_sequence))

    with mp.Pool() as pool:
        pool_results = pool.map(process_frame, pool_input)

        for result in pool_results:
            image_output_path, \
                ground_truth_output_path = result

            # Collect filepaths
            train_image_output_paths.append(image_output_path)
            train_ground_truth_output_paths.append(ground_truth_output_path)

train_intrinsics_output_paths = train_intrinsics_output_paths * len(train_image_output_paths)

print('Storing {} training image file paths into: {}'.format(
    len(train_image_output_paths), TRAIN_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_IMAGE_OUTPUT_FILEPATH, train_image_output_paths)

print('Storing {} training ground truth file paths into: {}'.format(
    len(train_ground_truth_output_paths), TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_GROUND_TRUTH_OUTPUT_FILEPATH, train_ground_truth_output_paths)

print('Storing {} training intrinsics file paths into: {}'.format(
    len(train_intrinsics_output_paths), TRAIN_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_INTRINSICS_OUTPUT_FILEPATH, train_intrinsics_output_paths)


'''
Process validation and testing paths
'''
test_image_split_paths = data_utils.read_paths(NYU_TEST_IMAGE_SPLIT_FILEPATH)

val_image_output_paths = []
val_ground_truth_output_paths = []
val_intrinsics_output_paths = [intrinsics_output_path]

test_image_output_paths = []
test_ground_truth_output_paths = []
test_intrinsics_output_paths = [intrinsics_output_path]

test_image_paths = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'testing', 'images', '*.png')))
test_ground_truth_paths = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'testing', 'depths', '*.png')))

n_sample = len(test_image_paths)

for image_path, ground_truth_path in zip(test_image_paths, test_ground_truth_paths):
    assert os.path.join(*(image_path.split(os.sep)[-3:])) == os.path.join(*(image_path.split(os.sep)[-3:]))

print('Processing {} samples for validation and testing'.format(n_sample))

for image_output_path, ground_truth_output_path in zip(test_image_paths, test_ground_truth_paths):

    test_split = False
    for test_image_path in test_image_split_paths:
        if test_image_path in image_output_path:
            test_split = True

    if test_split:
        # Collect test filepaths
        test_image_output_paths.append(image_output_path)
        test_ground_truth_output_paths.append(ground_truth_output_path)
    else:
        # Collect validation filepaths
        val_image_output_paths.append(image_output_path)
        val_ground_truth_output_paths.append(ground_truth_output_path)

val_intrinsics_output_paths = val_intrinsics_output_paths * len(val_image_output_paths)
test_intrinsics_output_paths = test_intrinsics_output_paths * len(test_image_output_paths)

'''
Write validation output paths
'''
print('Storing {} validation image file paths into: {}'.format(
    len(val_image_output_paths), VAL_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_IMAGE_OUTPUT_FILEPATH, val_image_output_paths)

print('Storing {} validation dense depth file paths into: {}'.format(
    len(val_ground_truth_output_paths), VAL_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_GROUND_TRUTH_OUTPUT_FILEPATH, val_ground_truth_output_paths)

print('Storing {} validation intrinsics file paths into: {}'.format(
    len(val_intrinsics_output_paths), VAL_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_INTRINSICS_OUTPUT_FILEPATH, val_intrinsics_output_paths)


'''
Write testing output paths
'''
print('Storing {} testing image file paths into: {}'.format(
    len(test_image_output_paths), TEST_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_IMAGE_OUTPUT_FILEPATH, test_image_output_paths)

print('Storing {} testing dense depth file paths into: {}'.format(
    len(test_ground_truth_output_paths), TEST_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_GROUND_TRUTH_OUTPUT_FILEPATH, test_ground_truth_output_paths)

print('Storing {} testing intrinsics file paths into: {}'.format(
    len(test_intrinsics_output_paths), TEST_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_INTRINSICS_OUTPUT_FILEPATH, test_intrinsics_output_paths)
