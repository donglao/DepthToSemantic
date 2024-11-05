import os, glob, argparse


'''
Paths for Cityscapes dataset
'''
CITYSCAPES_DATA_DIRPATH = os.path.join('data', 'cityscapes')


def sanitize_dataset_cityscapes_training(dry_run=True):
    '''
    Fetch image, intrinsics, focal length and baseline paths and see if they match
    '''

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

    print('Found {} directories'.format(n_dirpath))
    unmatched_paths = []

    '''
    Process each directory
    '''
    for dirpath_index in range(n_dirpath):

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

        disparity_filenames = [
                '_'.join((os.path.splitext(os.path.split(path)[-1])[0]).split('_')[:-1])
            for path in disparity_paths
        ]

        dirpath_basename = \
            os.path.basename(train_image_left_dirpaths[dirpath_index][:-1])

        for paths in [image_left_paths, image_right_paths, intrinsics_paths]:

            filenames = [
                '_'.join((os.path.splitext(os.path.split(path)[-1])[0]).split('_')[:-1])
                for path in paths
            ]

            for filename, path in zip(filenames, paths):

                if filename not in disparity_filenames:
                    print('Not found in disparity list: {}'.format(path))
                    unmatched_paths.append(path)

    # Remove paths
    if dry_run:
        print('Paths to be removed:')
    else:
        print('Removing:')

    for path in unmatched_paths:
        print(path)

        if not dry_run:
            os.remove(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dry_run', action='store_true', help='If set, then show paths to be removed')

    args = parser.parse_args()


    # Sanitize up dataset
    sanitize_dataset_cityscapes_training(args.dry_run)
