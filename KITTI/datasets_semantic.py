import torch
from PIL import Image
import numpy as np
import data_utils


def load_triplet_image(path, normalize=True, data_format='CHW'):
    '''
    Load in triplet frames from path

    Arg(s):
        path : str
            path to image triplet
        normalize : bool
            if set, normalize to [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : image at t
        numpy[float32] : image at t - 1
        numpy[float32] : image at t + 1
    '''

    images = data_utils.load_image(
        path,
        normalize=normalize,
        data_format=data_format)

    image0, image1, image2 = np.split(
        images,
        indices_or_sections=3,
        axis=-1 if data_format == 'CHW' else 1)

    return image0, image1, image2

def horizontal_flip(images_arr):
    '''
    Perform horizontal flip on each sample

    Arg(s):
        images_arr : list[np.array[float32]]
            list of N x C x H x W tensors
    Returns:
        list[np.array[float32]] : list of transformed N x C x H x W image tensors
    '''

    for i, image in enumerate(images_arr):
        if len(image.shape) != 3:
            raise ValueError('Can only flip C x H x W images in dataloader.')

        flipped_image = np.flip(image, axis=-1).copy()

        # Sanity check
        assert (image == flipped_image[..., ::-1]).all()

        images_arr[i] = flipped_image

    return images_arr

def random_crop(inputs, shape, intrinsics=None, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : list[numpy[float32]]
            list of 3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        list[numpy[float32]] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width+1)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height+1)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width
    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    # Adjust intrinsics
    if intrinsics is not None:
        offset_principal_point = [[0.0, 0.0, -x_start],
                                  [0.0, 0.0, -y_start],
                                  [0.0, 0.0, 0.0     ]]

        intrinsics = [
            in_ + offset_principal_point for in_ in intrinsics
        ]

        return outputs, intrinsics
    else:
        return outputs

class SingleImageDepthInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching images

    Arg(s):
        image_paths : list[str]
            paths to images
    '''

    def __init__(self, image_paths):

        self.n_sample = len(image_paths)
        self.image_paths = image_paths

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        image = data_utils.load_image(
            path=self.image_paths[index],
            normalize=False,
            data_format=self.data_format)

        return image.astype(int)

    def __len__(self):
        return self.n_sample


class SingleImageDepthStereoTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) left camera image
        (2) left camera image at t - 1
        (3) left camera image at t + 1
        (4) right camera image
        (5) left intrinsic camera calibration matrix
        (6) right intrinsic camera calibration matrix
        (7) left focal length and baseline
        (8) right focal length and baseline

    Arg(s):
        image_left_paths : list[str]
            paths to left camera images
        image_right_paths : list[str]
            paths to right camera images
        intrinsics_left_paths : list[str]
            paths to intrinsic left camera calibration matrix
        intrinsics_right_paths : list[str]
            paths to intrinsic right camera calibration matrix
        focal_length_baseline_left_paths : list[str]
            paths to focal length and baseline for left camera
        focal_length_baseline_right_paths : list[str]
            paths to focal length and baseline for right camera
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
        random_swap_left_right : bool
            Whether to perform random left right image swapping as data augmentation
    '''

    def __init__(self,
                 image_left_paths,
                 image_right_paths,
                 intrinsics_left_paths,
                 intrinsics_right_paths,
                 focal_length_baseline_left_paths,
                 focal_length_baseline_right_paths,
                 random_crop_shape=None,
                 random_crop_type=None,
                 random_swap_left_right=False):

        self.n_sample = len(image_left_paths)

        input_paths = [
            image_left_paths,
            intrinsics_left_paths,
            intrinsics_right_paths,
            focal_length_baseline_left_paths,
            focal_length_baseline_right_paths
        ]

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.image_left_paths = image_left_paths
        self.image_right_paths = image_right_paths

        self.intrinsics_left_paths = intrinsics_left_paths
        self.intrinsics_right_paths = intrinsics_right_paths

        self.focal_length_baseline_left_paths = focal_length_baseline_left_paths
        self.focal_length_baseline_right_paths = focal_length_baseline_right_paths

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.do_random_swap_left_right = random_swap_left_right

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Swap and flip a stereo video stream
        do_swap = self.do_random_swap_left_right and np.random.uniform() < 0.5

        if do_swap:
            # Swap paths for left and right camera
            image0_path = self.image_right_paths[index]
            intrinsics0_path = self.intrinsics_right_paths[index]
            focal_length_baseline0_path = self.focal_length_baseline_right_paths[index]

            image3_path = self.image_left_paths[index]
        else:
            # Keep paths consistent
            image0_path = self.image_left_paths[index]
            intrinsics0_path = self.intrinsics_left_paths[index]
            focal_length_baseline0_path = self.focal_length_baseline_left_paths[index]

            image3_path = self.image_right_paths[index]

        # Load images: t, t-1, t+1
        image0, image1, image2 = load_triplet_image(
            path=image0_path,
            normalize=False,
            data_format=self.data_format)
        # 3: stereo pair for 0
        image3, _, _ = load_triplet_image(
            path=image3_path,
            normalize=False,
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics0 = np.load(intrinsics0_path)

        # Load camera intrinsics
        focal_length_baseline0 = np.load(focal_length_baseline0_path)

        inputs = [
            image0,
            image1,
            image2,
            image3
        ]

        # If we swapped left and right, also need to horizontally flip images
        if do_swap:
            inputs = horizontal_flip(inputs)

        # Crop input images and depth maps and adjust intrinsics
        if self.do_random_crop:
            inputs, [intrinsics0] = random_crop(
                inputs=inputs,
                shape=self.random_crop_shape,
                intrinsics=[intrinsics0],
                crop_type=self.random_crop_type)

        # Convert intrinsics, focal length and baseline to float32
        intrinsics0 = intrinsics0.astype(np.float32)
        focal_length_baseline0 = focal_length_baseline0.astype(np.float32)

        # Convert images to int
        inputs = [
            T.astype(int)
            for T in inputs
        ]

        inputs = inputs + [intrinsics0, focal_length_baseline0]

        return inputs

    def __len__(self):
        return self.n_sample

def load_image(path, shape, normalize=True, load_triplet=False, use_resize=True, do_augment=False):

    n_height, n_width = shape
    # Load image and resize
    image = Image.open(path).convert('RGB')

    if load_triplet:
        image, _, _ = np.split(np.array(image), indices_or_sections=3, axis=1)
        image = Image.fromarray(image.astype(np.uint8))

    o_width, o_height = image.size

    if use_resize:
        image = image.resize((n_width, n_height), Image.LANCZOS)

    image = np.asarray(image, np.float32)
    image = np.transpose(image, (2, 0, 1))

    if not use_resize and do_augment:
        [image] = random_crop(
           [image],
           shape,
           intrinsics=None,
           crop_type=['bottom', 'anchored', 'horizontal'])

    # Normalize
    image = image/255.0 if normalize else image

    return image, (o_height, o_width)

def augment_image_color(images,
                        color_range=None,
                        intensity_range=None,
                        gamma_range=None,
                        normalize=True):
    if color_range is not None:
        color = np.random.uniform(color_range[0], color_range[1], 3)
        images = [np.reshape(color, [3, 1, 1])*image for image in images]

    if intensity_range is not None:
        intensity = np.random.uniform(intensity_range[0], intensity_range[1], 1)
        images = [intensity*image for image in images]

    if gamma_range is not None:
        gamma = np.random.uniform(gamma_range[0], gamma_range[1], 1)
        images = [np.power(image, gamma) for image in images]

    if normalize:
        images = [np.clip(image, 0.0, 1.0).astype(np.float32) for image in images]
    else:
        images = [np.clip(image, 0.0, 255.0).astype(np.float32) for image in images]

    return images

def resize(inputs, shape, intrinsics=None, focal_length_baseline=None):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : list[numpy[float32]]
            list of 3 x 3 camera intrinsics matrix
        focal_length_baseline : list[numpy[float32]]
            list of 1 x 2 camera focal length and baseline
    Return:
        list[numpy[float32]] : list of cropped inputs
        list[numpy[float32]] : if given, 3 x 3 adjusted camera intrinsics matrix
        list[numpy[float32]] : if given, 1 x 2 adjusted camera focal length and baseline
    '''

    n_height, n_width = shape
    o_height, o_width, _ = inputs[0].shape

    scale_height = float(n_height) / float(o_height)
    scale_width = float(n_width) / float(o_width)

    # Resize inputs
    for idx, in_ in enumerate(inputs):
        # Convert to Pillow and resize
        in_ = Image.fromarray(in_.astype(np.uint8))
        in_ = in_.resize((n_width, n_height), Image.LANCZOS)

        # Convert back to numpy
        inputs[idx] = np.asarray(in_, np.uint8)

    # Adjust camera parameters
    if intrinsics is not None:
        scale_factor = np.array(
            [[scale_width, 1.0,          scale_width],
             [1.0,         scale_height, scale_height],
             [1.0,         1.0,          1.0     ]])

        intrinsics = [
            in_ * scale_factor for in_ in intrinsics
        ]

    if focal_length_baseline is not None:
        focal_length_baseline = [
            in_ * np.array([scale_width, 1.0]) for in_ in focal_length_baseline
        ]

    # Select returns
    if intrinsics is not None and focal_length_baseline is not None:
        return inputs, intrinsics, focal_length_baseline
    elif intrinsics is not None:
        return inputs, intrinsics
    elif focal_length_baseline is not None:
        return inputs, focal_length_baseline
    else:
        return inputs


class SingleImageDepthResizedTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) left camera image
        (2) left camera image at t - 1
        (3) left camera image at t + 1
        (4) right camera image
        (5) left intrinsic camera calibration matrix
        (6) right intrinsic camera calibration matrix
        (7) left focal length and baseline
        (8) right focal length and baseline

    Arg(s):
        images_left_paths : list[str]
            paths to left camera images
        images_right_paths : list[str]
            paths to right camera images
        intrinsics_left_paths : list[str]
            paths to intrinsic left camera calibration matrix
        intrinsics_right_paths : list[str]
            paths to intrinsic right camera calibration matrix
        focal_length_baseline_left_paths : list[str]
            paths to focal length and baseline for left camera
        focal_length_baseline_right_paths : list[str]
            paths to focal length and baseline for right camera
        resize_shape : tuple[int]
            shape (height, width) to resize inputs
        random_swap_left_right : bool
            Whether to perform random left right image swapping as data augmentation
    '''

    def __init__(self,
                 images_left_paths,
                 images_right_paths,
                 intrinsics_left_paths,
                 intrinsics_right_paths,
                 focal_length_baseline_left_paths,
                 focal_length_baseline_right_paths,
                 resize_shape=None,
                 random_swap_left_right=False):

        self.n_sample = len(images_left_paths)

        input_paths = [
            images_left_paths,
            intrinsics_left_paths,
            intrinsics_right_paths,
            focal_length_baseline_left_paths,
            focal_length_baseline_right_paths
        ]

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.images_left_paths = images_left_paths
        self.images_right_paths = images_right_paths

        self.intrinsics_left_paths = intrinsics_left_paths
        self.intrinsics_right_paths = intrinsics_right_paths

        self.focal_length_baseline_left_paths = focal_length_baseline_left_paths
        self.focal_length_baseline_right_paths = focal_length_baseline_right_paths

        self.resize_shape = resize_shape

        self.do_random_swap_left_right = random_swap_left_right

        self.data_format = 'HWC'

    def __getitem__(self, index):

        # Swap and flip a stereo video stream
        do_swap = self.do_random_swap_left_right and np.random.uniform() < 0.5

        if do_swap:
            # Swap paths for left and right camera
            image0_path = self.images_right_paths[index]
            intrinsics0_path = self.intrinsics_right_paths[index]
            focal_length_baseline0_path = self.focal_length_baseline_right_paths[index]

            image3_path = self.images_left_paths[index]
        else:
            # Keep paths consistent
            image0_path = self.images_left_paths[index]
            intrinsics0_path = self.intrinsics_left_paths[index]
            focal_length_baseline0_path = self.focal_length_baseline_left_paths[index]

            image3_path = self.images_right_paths[index]

        # Load images: t, t-1, t+1
        image0, image1, image2 = load_triplet_image(
            path=image0_path,
            normalize=False,
            data_format=self.data_format)
        # 3: stereo pair for 0
        image3, _, _ = load_triplet_image(
            path=image3_path,
            normalize=False,
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics0 = np.load(intrinsics0_path)

        # Load focal length baseline
        focal_length_baseline0 = np.load(focal_length_baseline0_path)

        images = [
            image0,
            image1,
            image2,
            image3
        ]

        # Resize inputs
        images, [intrinsics0], [focal_length_baseline0] = resize(
            inputs=images,
            shape=self.resize_shape,
            intrinsics=[intrinsics0],
            focal_length_baseline=[focal_length_baseline0])

        # Switch from H x W x C to C x H x W
        images = [
            np.transpose(image, (2, 0, 1)).astype(np.uint8) for image in images
        ]

        # If we swapped left and right, also need to horizontally flip images
        if do_swap:
            images = horizontal_flip(images)

        # (f * b) / ((n / o) / n) = (f * b) / o
        focal_length_baseline0 = \
            np.prod(focal_length_baseline0) / np.asarray(self.resize_shape[-1], np.float32)

        focal_length_baseline0 = np.reshape(focal_length_baseline0, [1, 1, 1])

        outputs = images + [
            intrinsics0.astype(np.float32), focal_length_baseline0.astype(np.float32)
        ]

        return outputs

    def __len__(self):
        return self.n_sample

class SingleImageDepthResizedInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching images

    Arg(s):
        image_paths : list[str]
            paths to images
        focal_length_baseline_paths : list[str]
            paths to focal length and baseline for camera
        resize_shape : tuple[int]
            shape (height, width) to resize inputs
    '''

    def __init__(self, image_paths, focal_length_baseline_paths, resize_shape=None):

        self.n_sample = len(image_paths)
        self.image_paths = image_paths
        self.focal_length_baseline_paths = focal_length_baseline_paths

        self.resize_shape = resize_shape

        self.data_format = 'HWC'

    def __getitem__(self, index):

        # Load image
        image = data_utils.load_image(
            path=self.image_paths[index],
            normalize=False,
            data_format=self.data_format)

        # Load focal length baseline
        focal_length_baseline = \
            np.load(self.focal_length_baseline_paths[index])

        # Resize inputs
        [image], [focal_length_baseline] = resize(
            inputs=[image],
            shape=self.resize_shape,
            focal_length_baseline=[focal_length_baseline])

        image = np.transpose(image, (2, 0, 1))

        # (f * b) / ((n / o) / n) = (f * b) / o
        focal_length_baseline = \
            np.prod(focal_length_baseline) / np.asarray(self.resize_shape[-1], np.float32)

        focal_length_baseline = np.reshape(focal_length_baseline, [1, 1, 1])

        return image.astype(np.uint8), focal_length_baseline.astype(np.float32)

    def __len__(self):
        return self.n_sample


class Monodepth2InferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching images

    Arg(s):
        image_paths : list[str]
            paths to images
        resize_shape : list[int]
            height and width tuple
        load_image_triplets : bool
            if set then load image triplets instead of image
    '''

    def __init__(self,
                 image_paths,
                 intrinsics_paths,
                 resize_shape=None,
                 load_image_triplets=False,
                 return_image_info=False):

        self.n_sample = len(image_paths)
        self.image_paths = image_paths
        self.intrinsics_paths = intrinsics_paths

        self.resize_shape = resize_shape

        self.load_image_triplets = load_image_triplets

        self.return_image_info = return_image_info

        self.data_format = 'HWC'

    def __getitem__(self, index):

        # Load camera intrinsics
        if self.intrinsics_paths[index] is not None:
            intrinsics = np.load(self.intrinsics_paths[index])
        else:
            intrinsics = np.eye(3)

        if self.load_image_triplets:
            # Load images: t, t-1, t+1
            image0, image1, image2 = load_triplet_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

            image_info = {
                'height' : image0.shape[0],
                'width' : image0.shape[1]
            }

            # Resize inputs
            [image0, image1, image2], [intrinsics] = resize(
                inputs=[image0, image1, image2],
                shape=self.resize_shape,
                focal_length_baseline=None,
                intrinsics=[intrinsics])

            images = [
                np.transpose(image, (2, 0, 1))
                for image in [image0, image1, image2]
            ]
        else:
            # Load a single image
            image = data_utils.load_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

            image_info = {
                'height' : image.shape[0],
                'width' : image.shape[1]
            }

            # Resize inputs
            [image], [intrinsics] = resize(
                inputs=[image],
                shape=self.resize_shape,
                focal_length_baseline=None,
                intrinsics=[intrinsics])

            images = [np.transpose(image, (2, 0, 1))]

        if self.return_image_info:
            return images + [intrinsics.astype(np.float32)] + [image_info]
        else:
            return images + [intrinsics.astype(np.float32)]

    def __len__(self):
        return self.n_sample


class ImagePairCameraDataset(torch.utils.data.Dataset):

    def __init__(self,
                 image0_paths,
                 image1_paths,
                 camera_paths,
                 shape,
                 normalize=True,
                 augment=False,
                 use_resize=True,
                 training=True,
                 load_triplet=False,
                 color_range=[0.5, 2.0],
                 intensity_range=[0.8, 1.2],
                 gamma_range=[0.8, 1.2]):
        self.image0_paths = image0_paths
        self.image1_paths = image1_paths
        self.camera_paths = camera_paths
        self.n_height = shape[0]
        self.n_width = shape[1]
        self.normalize = normalize
        self.use_resize = use_resize
        self.augment = augment
        self.training = training
        self.load_triplet = load_triplet
        self.color_range = color_range
        self.intensity_range = intensity_range
        self.gamma_range = gamma_range

    def __getitem__(self, index):

        # Load image
        image0, (height, width) = load_image(
            self.image0_paths[index],
            shape=(self.n_height, self.n_width),
            normalize=self.normalize,
            load_triplet=self.load_triplet,
            use_resize=self.use_resize,
            do_augment=self.training)
        image1, _  = load_image(
            self.image1_paths[index],
            shape=(self.n_height, self.n_width),
            normalize=self.normalize,
            load_triplet=self.load_triplet,
            use_resize=self.use_resize,
            do_augment=self.training)

        # Load camera (focal length and baseline)
        camera = np.load(self.camera_paths[index]).astype(np.float32)

        # (f * b) / ((n / o) / n) = (f * b) / o
        camera = np.prod(camera) / np.asarray(width, np.float32)
        camera = np.reshape(camera, list(camera.shape)+[1, 1, 1])

        # Color augmentation
        if self.augment and np.random.uniform(0.0, 1.0, 1) > 0.50:
            image0, image1 = augment_image_color([image0, image1],
                color_range=self.color_range,
                intensity_range=self.intensity_range,
                gamma_range=self.gamma_range,
                normalize=self.normalize)

        return image0, image1, camera

    def __len__(self):
        return len(self.image0_paths)


class SemanticSegmentationDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching images

    Arg(s):
        image_paths : list[str]
            paths to images
        focal_length_baseline_paths : list[str]
            paths to focal length and baseline for camera
        resize_shape : tuple[int]
            shape (height, width) to resize inputs
    '''

    def __init__(self, image_paths, label_paths):

        self.n_sample = len(image_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths

        self.data_format = 'HWC'

    def __getitem__(self, index):

        # Load image
        image = data_utils.load_image(
            path=self.image_paths[index],
            normalize=False,
            data_format=self.data_format)

        # Load labels
        label = data_utils.load_semantic_label(
            path=self.image_paths[index],
            data_format=self.data_format)

        image = np.transpose(image, (2, 0, 1))

        return image.astype(np.uint8), label.astype(np.uint8)

    def __len__(self):
        return self.n_sample