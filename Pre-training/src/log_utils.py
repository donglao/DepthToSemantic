import os
import torch
import numpy as np
from matplotlib import pyplot as plt


def log(s, filepath=None, to_console=True):
    '''
    Logs a string to either file or console

    Arg(s):
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''

    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, 'w+') as o:
               o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')

def colorize(T, colormap='magma'):
    '''
    Colorizes a 1-channel tensor with matplotlib colormaps

    Arg(s):
        T : torch.Tensor[float32]
            1-channel tensor
        colormap : str
            matplotlib colormap
    '''

    cm = plt.cm.get_cmap(colormap)
    shape = T.shape

    # Convert to numpy array and transpose
    if shape[0] > 1:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)))
    else:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)), axis=-1)

    # Colorize using colormap and transpose back
    color = np.concatenate([
        np.expand_dims(cm(T[n, ...])[..., 0:3], 0) for n in range(T.shape[0])],
        axis=0)
    color = np.transpose(color, (0, 3, 1, 2))

    # Convert back to tensor
    return torch.from_numpy(color.astype(np.float32))


def cartesian2polar(x, y):
    '''
    Converts cartesian coordinates to polar coordinates
    Args:
        x : torch.Tensor[float32]
            x-component
        y : torch.Tensor[float32]
            y-component
    Returns:
        torch.Tensor : rho, phi
    '''

    rho = torch.sqrt(x ** 2 + y ** 2)
    phi = torch.atan2(y, x)
    return rho, phi

def hsv2rgb(hsv):
    '''
    Converts HSV to RGB
    Args:
        hsv : torch.Tensor[float32]
            HSV image
    Returns:
        tensor : rho, phi
    '''

    h = hsv[..., 0, :, :]
    s = hsv[..., 1, :, :]
    v = hsv[..., 2, :, :]

    hi = torch.floor(h * 6.0)
    f = h * 6.0 - hi
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    # Quantize based on hue
    rgb = torch.stack([hi, hi, hi], dim=1) % 6
    rgb[rgb == 0] = torch.stack((v, t, p), dim=1)[rgb == 0]
    rgb[rgb == 1] = torch.stack((q, v, p), dim=1)[rgb == 1]
    rgb[rgb == 2] = torch.stack((p, v, t), dim=1)[rgb == 2]
    rgb[rgb == 3] = torch.stack((p, q, v), dim=1)[rgb == 3]
    rgb[rgb == 4] = torch.stack((t, p, v), dim=1)[rgb == 4]
    rgb[rgb == 5] = torch.stack((v, p, q), dim=1)[rgb == 5]

    return rgb

def make_colorwheel(use_torch=True):
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Args:
        use_torch : bool
            if set, then use torch functions else use numpy
    Returns:
        torch.Tensor or numpy : Color wheel
    '''

    if use_torch:
        # Set to torch functions
        zeros = torch.zeros
        dtype_float = torch.float
        floor = torch.floor
        arange = torch.arange
    else:
        # Set numpy functions
        zeros = np.zeros
        dtype_float = np.float32
        floor = torch.floor
        arange = torch.arange

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = zeros((ncols, 3), dtype=dtype_float)
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = floor(255 * arange(0.0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - floor(255 * arange(0.0, YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = floor(255 * arange(0.0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - floor(255 * arange(0.0, CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = floor(255 * arange(0.0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - floor(255 * arange(0.0, MR) / MR)
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False, use_torch=True):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u : torch.Tensor[float32]
            input horizontal flow of shape [H,W]
        v : torch.Tensor[float32]
            input vertical flow of shape [H,W]
        convert_to_bgr : bool
            convert output image to BGR. Defaults to False
        use_torch : bool
            if set, then use torch functions else use numpy
    Returns:
        torch.Tensor[float32] or numpy[float32] : flow visualization image of shape [H, W, 3]
    '''

    if use_torch:
        # Set to torch functions
        zeros = torch.zeros
        dtype_uint8 = torch.uint8
        floor = torch.floor
        power = torch.pow
        sqrt = torch.sqrt
        atan2 = torch.atan2
        shape = (u.shape[0], u.shape[1], u.shape[2], 3)
    else:
        # Set numpy functions
        zeros = np.zeros
        dtype_uint8 = np.uint8
        floor = np.floor
        power = np.power
        sqrt = np.sqrt
        atan2 = np.arctan2
        shape = (u.shape[0], u.shape[1], 3)

    flow_image = zeros(shape, dtype=dtype_uint8)

    # Shape = [55 x 3]
    colorwheel = make_colorwheel(use_torch=use_torch)
    ncols = colorwheel.shape[0]

    rad = sqrt(power(u, 2) + power(v, 2))
    a = atan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)

    if use_torch:
        k0 = floor(fk).int()
    else:
        k0 = floor(fk).astype(np.int32)

    k1 = k0 + 1
    k1[k1 == ncols] = 0

    if use_torch:
        f = fk - k0.float()
        k0 = k0.long()
        k1 = k1.long()
    else:
        f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[..., ch_idx] = floor(255 * col)

    return flow_image

def flow2color(flow_uv, clip_flow=None, convert_to_bgr=False, use_torch=True):
    '''
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv : torch.Tensor[float32]
            flow u, v image of shape [H, W, 2]
        clip_flow : float
            clip maximum of flow values. Defaults to None.
        convert_to_bgr : bool
            convert output image to BGR. Defaults to False.
        use_torch : bool
            if set, then use torch functions else use numpy
    Returns:
        torch.Tensor[float32] or numpy[float32] : flow visualization image of shape [H, W, 3]
    '''

    if use_torch:
        clamp = torch.clamp
        sqrt = torch.sqrt
        power = torch.pow
        maximum = torch.max
        stack = torch.stack
        array = torch.Tensor
    else:
        clamp = np.clip
        sqrt = np.sqrt
        power = np.power
        maximum = np.max
        stack = np.stack
        array = np.array

    if use_torch:
        assert flow_uv.ndim == 4, 'input flow must have four dimensions'
        assert flow_uv.shape[3] == 2, 'input flow must have shape [B, H, W, 2]'
    else:
        assert flow_uv.ndim == 3, 'input flow must have three dimensions'
        assert flow_uv.shape[2] == 2, 'input flow must have shape [H, W, 2]'

    if clip_flow is not None:
        flow_uv = clamp(flow_uv, 0, clip_flow)

    u = flow_uv[..., 0]
    v = flow_uv[..., 1]

    rad = sqrt(power(u, 2) + power(v, 2))
    rad_max = [maximum(n) for n in rad]
    rad_max = array(rad_max)

    epsilon = 1e-5

    u = stack([u[i] / (rad_max[i] + epsilon) for i in range(len(rad_max))])
    v = stack([v[i] / (rad_max[i] + epsilon) for i in range(len(rad_max))])

    return flow_uv_to_colors(u, v, convert_to_bgr, use_torch=use_torch)
