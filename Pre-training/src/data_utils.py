import os, cv2
import numpy as np
from PIL import Image
from collections import Counter


def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path)

    return path_list

def write_paths(filepath, paths):
    '''
    Stores line delimited paths into file

    Arg(s):
        filepath : str
            path to file to save paths
        paths : list
            paths to write into file
    '''

    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def load_image(path, normalize=True, data_format='HWC'):
    '''
    Loads an RGB image

    Arg(s):
        path : str
            path to RGB image
        normalize : bool
            if set, then normalize image between [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : H x W x C or C x H x W image
    '''

    # Load image
    image = Image.open(path).convert('RGB')

    # Convert to numpy
    image = np.asarray(image, np.float32)

    if data_format == 'CHW':
        image = np.transpose(image, (2, 0, 1))

    # Normalize
    image = image / 255.0 if normalize else image

    return image

def load_depth(path, multiplier=256.0, data_format='HW'):
    '''
    Loads a depth map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0

    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z

def load_depth_with_validity_map(path, data_format='HW'):
    '''
    Loads a depth map and validity map from a 16-bit PNG file
    Args:
        path : str
            path to 16-bit PNG file
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
        numpy[float32] : binary validity map for available depth measurement locations
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / 256.0
    z[z <= 0] = 0.0
    v = z.astype(np.float32)
    v[z > 0] = 1.0

    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
        v = np.expand_dims(v, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
        v = np.expand_dims(v, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z, v

def save_depth(z, path, multiplier=256.0):
    '''
    Saves a depth map to a 16-bit PNG file

    Arg(s):
        z : numpy[float32]
            depth map
        path : str
            path to store depth map
    '''

    z = np.uint32(z * multiplier)
    z = Image.fromarray(z, mode='I')
    z.save(path)

def load_disparity(path, multiplier=256.0, data_format='HW'):
    '''
    Loads a disparity image

    Arg(s):
        path : str
            path to disparity image
        multiplier : float
            depth factor multiplier for saving and loading in 16/32 bit png
        data_format : str
            sets output format to HW, HWC or CHW
    Returns:
        numpy : H x W disparity image
    '''

    # Load image and resize
    disparity = Image.open(path).convert('I')

    # Convert unsigned int 16/32 to disparity values
    disparity = np.asarray(disparity, np.uint16)
    disparity = np.asarray(disparity / multiplier, np.float32)

    # Rearrange dimensions based on output format
    if data_format == 'CHW':
        disparity = np.expand_dims(disparity, axis=0)
    elif data_format == 'HWC':
        disparity = np.expand_dims(disparity, axis=-1)
    elif data_format == 'HW':
        pass
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return disparity

def load_flow(path, data_format='HWC'):
    '''
    Loads optical flow

    Arg(s):
        path: str
            path to optical flow
        data_format : str
            sets output format to HWC or CHW
    Returns:
        numpy[float32] : H x W x 2 or 2 x H x W (u, v) flow
        numpy[float32] : H x W x 1 or 1 x H x W validity map
    '''

    # Load flow as BGR image
    flow = np.load(path)

    # Convert u, v to true flow value
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[:, :, 2] = np.where(flow[:, :, 2] == 0, 0, 1)

    # Mask out invalid flow
    invalid_idx = (flow[:, :, 2] == 0)
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0

    # Rearrange dimensions based on output format
    if data_format == 'CHW':
        flow = np.transpose(flow, (2, 0, 1))
    elif data_format == 'HWC':
        pass
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return flow

def save_flow(flow, path, data_format='HWC'):
    '''
    Saves optical flow of image as 16 bit .png file in path

    Arg(s):
        flow : numpy[float32]
            H x W x C or C x H x W flow array
        path : str
            path to save .png file (must end with .png)
        output_format : str
            whether the flow array is formated to HWC or CHW
    '''

    # Swap channels based on input format
    if data_format == 'CHW':
        flow = np.transpose(flow, (1, 2, 0))
    elif data_format == 'HWC':
        pass
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    height, width, channel = flow.shape

    if channel == 2:
        flow = np.concatenate([flow, np.ones([height, width, 1])], axis=-1)
    elif channel == 3:
        flow[..., 2] = np.where(flow[..., 2] > 0, 1, 0)
    else:
        raise ValueError('Invalid number of channels: {}'.format(channel))

    # Convert u, v to preserve precision
    flow[..., 0:2] = flow[..., 0:2] * 64.0 + 2 ** 15

    # Save image
    np.save(path, flow)

def load_calibration(path):
    '''
    Loads the calibration matrices for each camera (KITTI) and stores it as map

    Arg(s):
        path : str
            path to file to be read
    Returns:
        dict : map containing camera intrinsics keyed by camera id
    '''

    float_chars = set("0123456789.e+- ")
    data = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.asarray(
                        [float(x) for x in value.split(' ')])
                except ValueError:
                    pass
    return data

def load_velodyne(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # Homogeneous
    return points

def load_focal_length_baseline(calibration_dir, camera_id):
    cam2cam = load_calibration(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))
    P2_rect = cam2cam['P_rect_02'].reshape(3, 4)
    P3_rect = cam2cam['P_rect_03'].reshape(3, 4)
    # camera2 is left of camera0 (-6cm) camera3 is right of camera2 (+53.27cm)
    b2 = P2_rect[0, 3]/-P2_rect[0, 0]
    b3 = P3_rect[0, 3]/-P3_rect[0, 0]
    baseline = b3-b2
    if camera_id == 2:
        focal_length = P2_rect[0, 0]
    elif camera_id == 3:
        focal_length = P3_rect[0, 0]

    return focal_length, baseline

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def velodyne2depth(calibration_dir, velodyne_path, shape, camera_id=2):
    # Load calibration files
    cam2cam = load_calibration(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))
    velo2cam = load_calibration(os.path.join(calibration_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    # Compute projection matrix from velodyne to image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(camera_id)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)
    # Load velodyne points and remove all that are behind image plane (approximation)
    # Each row of the velodyne data refers to forward, left, up, reflectance
    velo = load_velodyne(velodyne_path)
    velo = velo[velo[:, 0] >= 0, :]
    # Project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]
    velo_pts_im[:, 2] = velo[:, 0]
    # Check if in bounds (use minus 1 to get the exact same value as KITTI matlab code)
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0])-1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1])-1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < shape[1]) & (velo_pts_im[:, 1] < shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]
    # Project to image
    depth = np.zeros(shape)
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]
    # Find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    # Clip all depth values less than 0 to 0
    depth[depth < 0] = 0
    return depth.astype(np.float32)

def resize(T, shape, interp_type='lanczos', data_format='HWC'):
    '''
    Resizes a tensor

    Args:
        T : numpy
            tensor to resize
        shape : tuple[int]
            (height, width) to resize tensor
        interp_type : str
            interpolation for resize
        data_format : str
            'CHW', or 'HWC', 'CDHW', 'DHWC'
    Returns:
        numpy : image resized to height and width
    '''

    dtype = T.dtype

    if interp_type == 'nearest':
        interp_type = cv2.INTER_NEAREST
    elif interp_type == 'area':
        interp_type = cv2.INTER_AREA
    elif interp_type == 'bilinear':
        interp_type = cv2.INTER_LINEAR
    elif interp_type == 'lanczos':
        interp_type = cv2.INTER_LANCZOS4
    else:
        raise ValueError('Unsupport interpolation type: {}'.format(interp_type))

    if shape is None or any([x is None or x <= 0 for x in shape]):
        return T

    n_height, n_width = shape

    # Resize tensor
    if data_format == 'CHW':
        # Tranpose from CHW to HWC
        R = np.transpose(T, (1, 2, 0))

        # Resize and transpose back to CHW
        R = cv2.resize(R, dsize=(n_width, n_height), interpolation=interp_type)
        R = np.reshape(R, (n_height, n_width, T.shape[0]))
        R = np.transpose(R, (2, 0, 1))

    elif data_format == 'HWC':
        R = cv2.resize(T, dsize=(n_width, n_height), interpolation=interp_type)
        R = np.reshape(R, (n_height, n_width, T.shape[2]))

    elif data_format == 'CDHW':
        # Transpose CDHW to DHWC
        D = np.transpose(T, (1, 2, 3, 0))

        # Resize and transpose back to CDHW
        R = np.zeros((D.shape[0], n_height, n_width, D.shape[3]))

        for d in range(R.shape[0]):
            r = cv2.resize(D[d, ...], dsize=(n_width, n_height), interpolation=interp_type)
            R[d, ...] = np.reshape(r, (n_height, n_width, D.shape[3]))

        R = np.transpose(R, (3, 0, 1, 2))

    elif data_format == 'DHWC':
        R = np.zeros((T.shape[0], n_height, n_width, T.shape[3]))
        for d in range(R.shape[0]):
            r = cv2.resize(T[d, ...], dsize=(n_width, n_height), interpolation=interp_type)
            R[d, ...] = np.reshape(r, (n_height, n_width, T.shape[3]))

    else:
        raise ValueError('Unsupport data format: {}'.format(data_format))

    return R.astype(dtype)

def flip_horizontal(T, data_format='HWC'):
    '''
    Perform horizontal flip on a tensor

    Args:
        T : numpy
            tensor to horizontally flip
        data_format : str
            'CHW', or 'HWC', 'CDHW', 'DHWC'
    Returns:
        numpy : horizontally flipped tensor
    '''

    n_dim = len(T.shape)

    # If shape 1D or greater than 4D
    if n_dim < 2 or n_dim > 4:
        raise ValueError('Unsupport data shape: {}'.format(T.shape))

    if data_format == 'CHW':
        return T[:, :, ::-1].copy()
    elif data_format == 'HWC':
        return T[:, ::-1, :].copy()
    elif data_format == 'CDHW':
        return T[:, :, :, ::-1].copy()
    elif data_format == 'DHWC':
        return T[:, :, ::-1, :].copy()
    else:
        raise ValueError('Unsupport data format: {}'.format(data_format))

def flip_vertical(T, data_format='HWC'):
    '''
    Perform vertical flip on a tensor

    Args:
        T : numpy
            tensor to vertically flip
        data_format : str
            CHW', 'HWC', 'CDHW', 'DHWC'
    Returns:
        numpy : vertically flipped tensor
    '''

    n_dim = len(T.shape)

    # If shape 1D or greater than 4D
    if n_dim < 3 or n_dim > 4:
        raise ValueError('Unsupport data shape: {}'.format(T.shape))

    if data_format == 'CHW':
        return T[:, ::-1, :].copy()
    elif data_format == 'HWC':
        return T[::-1, :, :].copy()
    elif data_format == 'CDHW':
        return T[:, :, ::-1, :].copy()
    elif data_format == 'DHWC':
        return T[:, ::-1, :, :].copy()
    else:
        raise ValueError('Unsupport data format: {}'.format(data_format))

def translate_vertical(T, translate_by=0.0, data_format='HWC'):
    '''
    Translate a tensor up or down

    Args:
        T : numpy
            tensor to vertically flip
        translate_by: int
            number of pixels to translate by
        data_format : str
            CHW', 'HWC', 'CDHW', 'DHWC'
    Returns:
        translated tensor
    '''

    n_dim = len(T.shape)

    # If shape 1D or greater than 4D
    if n_dim < 3 or n_dim > 4:
        raise ValueError('Unsupport data shape: {}'.format(T.shape))

    if translate_by == 0:
        return T

    is_up = True if translate_by > 0 else False
    translate_by = np.abs(translate_by)

    if data_format == 'CHW':
        if is_up:
            T = np.pad(T, ((0, 0), (0, translate_by), (0, 0)), mode='edge')
            T = T[:, translate_by:, :]
        else:
            T = np.pad(T, ((0, 0), (translate_by, 0), (0, 0)), mode='edge')
            T = T[:, :-translate_by, :]
    elif data_format == 'HWC':
        if is_up:
            T = np.pad(T, ((0, translate_by), (0, 0), (0, 0)), mode='edge')
            T = T[translate_by:, :, :]
        else:
            T = np.pad(T, ((translate_by, 0), (0, 0), (0, 0)), mode='edge')
            T = T[:-translate_by, :, :]
    elif data_format == 'CDHW':
        if is_up:
            T = np.pad(T, ((0, 0), (0, 0), (0, translate_by), (0, 0)), mode='edge')
            T = T[:, :, translate_by:, :]
        else:
            T = np.pad(T, ((0, 0), (0, 0), (translate_by, 0), (0, 0)), mode='edge')
            T = T[:, :, :-translate_by, :]
    elif data_format == 'DHWC':
        if is_up:
            T = np.pad(T, ((0, 0), (0, translate_by), (0, 0), (0, 0)), mode='edge')
            T = T[:, translate_by:, :, :]
        else:
            T = np.pad(T, ((0, 0), (translate_by, 0), (0, 0), (0, 0)), mode='edge')
            T = T[:, :-translate_by, :, :]
    else:
        raise ValueError('Unsupport data format: {}'.format(data_format))

    return T

def translate_horizontal(T, translate_by=0.0, data_format='HWC'):
    '''
    Translate a tensor by left and right

    Args:
        T : numpy
            tensor to vertically flip
        translate_by: int
            number of pixels to translate by
        data_format : str
            CHW', 'HWC', 'CDHW', 'DHWC'
    Returns:
        tensor of translated tensors
    '''

    n_dim = len(T.shape)

    # If shape 1D or greater than 4D
    if n_dim < 3 or n_dim > 4:
        raise ValueError('Unsupport data shape: {}'.format(T.shape))

    if translate_by == 0:
        return T

    is_right = True if translate_by > 0 else False
    translate_by = np.abs(translate_by)

    if data_format == 'CHW':
        if is_right:
            T = np.pad(T, ((0, 0), (0, 0), (translate_by, 0)), mode='edge')
            T = T[:, :, :-translate_by]
        else:
            T = np.pad(T, ((0, 0), (0, 0), (0, translate_by)), mode='edge')
            T = T[:, :, translate_by:]
    elif data_format == 'HWC':
        if is_right:
            T = np.pad(T, ((0, 0), (translate_by, 0), (0, 0)), mode='edge')
            T = T[:, :-translate_by, :]
        else:
            T = np.pad(T, ((0, 0), (0, translate_by), (0, 0)), mode='edge')
            T = T[:, translate_by:, :]
    elif data_format == 'CDHW':
        if is_right:
            T = np.pad(T, ((0, 0), (0, 0), (0, 0), (translate_by, 0)), mode='edge')
            T = T[:, :, :, :-translate_by]
        else:
            T = np.pad(T, ((0, 0), (0, 0), (0, 0), (0, translate_by)), mode='edge')
            T = T[:, :, :, translate_by:]
    elif data_format == 'DHWC':
        if is_right:
            T = np.pad(T, ((0, 0), (0, 0), (translate_by, 0), (0, 0)), mode='edge')
            T = T[:, :, :-translate_by, :]
        else:
            T = np.pad(T, ((0, 0), (0, 0), (0, translate_by), (0, 0)), mode='edge')
            T = T[:, :, translate_by:, :]
    else:
        raise ValueError('Unsupport data format: {}'.format(data_format))

    return T

def rotate(T, angle, interp_method='nearest', data_format='HWC'):
    '''
    Rotate tensor based on angle

    Args:
        T : numpy
            input tensor
        angle : float
            angle for rotation in degrees
        interp_method : str
            method to interpolate affine warp
        data_format : str
            CHW', 'HWC', 'CDHW', 'DHWC'
    Returns:
        numpy : rotated tensor
    '''

    dtype = T.dtype
    shape = T.shape
    n_dim = len(shape)

    if n_dim < 3 or n_dim > 4:
        raise ValueError('Unsupport data shape: {}'.format(T.shape))

    if interp_method == 'bilinear':
        interp_method = cv2.INTER_LINEAR
    elif interp_method == 'nearest':
        interp_method = cv2.INTER_NEAREST
    else:
        raise ValueError('Unsupported interpolation: {}'.format(interp_method))

    if data_format == 'CHW':
        rotation_matrix = cv2.getRotationMatrix2D(
            center=(shape[2] / 2, shape[1] / 2),
            angle=angle,
            scale=1.0)

        R = cv2.warpAffine(
            src=np.transpose(T, (1, 2, 0)),
            M=rotation_matrix,
            dsize=(shape[2], shape[1]),
            flags=interp_method)

        if len(R.shape) == 3:
            R = np.transpose(R, (2, 0, 1))

    elif data_format == 'HWC':
        rotation_matrix = cv2.getRotationMatrix2D(
            center=(shape[1] / 2, shape[0] / 2),
            angle=angle,
            scale=1.0)

        R = cv2.warpAffine(
            src=T,
            M=rotation_matrix,
            dsize=(shape[1], shape[0]),
            flags=interp_method)

    elif data_format == 'CDHW':
        rotation_matrix = cv2.getRotationMatrix2D(
            center=(shape[3] / 2, shape[2] / 2),
            angle=angle,
            scale=1.0)

        T = np.transpose(T, (1, 2, 3, 0))
        R = np.zeros_like(T)

        for d in range(shape[1]):
            r = cv2.warpAffine(
                src=T[d, :, :, :],
                M=rotation_matrix,
                dsize=(shape[3], shape[2]),
                flags=interp_method)
            R[d, :, :, :] = np.reshape(r, R[d, :, :, :].shape)

        T = np.transpose(T, (3, 0, 1, 2))

        if len(R.shape) == 4:
            R = np.transpose(R, (3, 0, 1, 2))

    elif data_format == 'DHWC':
        rotation_matrix = cv2.getRotationMatrix2D(
            center=(shape[2] / 2, shape[1] / 2),
            angle=angle,
            scale=1.0)

        R = np.zeros_like(T)

        for d in range(shape[0]):
            r = cv2.warpAffine(
                src=T[d, :, :, :],
                M=rotation_matrix,
                dsize=(shape[2], shape[1]),
                flags=interp_method)
            R[d, :, :, :] = np.reshape(r, R[d, :, :, :].shape)

    else:
        raise ValueError('Unsupport data format: {}'.format(data_format))

    # In case there are dimensions of 1s in shape
    R = np.reshape(R, T.shape).astype(dtype)

    return R
