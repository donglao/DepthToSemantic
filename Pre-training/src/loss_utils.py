import torch


'''
Utility functions for photometric reprojection
'''
def warp1d_horizontal(image, disparity, padding_mode='border'):
    '''
    Performs horizontal 1d warping

    Arg(s):
        image : torch.Tensor[float32]
            N x C x H x W image
        disparity : torch.Tensor[float32]
            N x 1 x H x W disparity
    Returns:
        torch.Tensor[float32] : N x C x H x W image shifted by disparity
    '''

    n_batch, _, n_height, n_width = image.shape

    # Original coordinates of pixels
    x = torch.linspace(0, 1, n_width, dtype=torch.float32, device=image.device) \
        .repeat(n_batch, n_height, 1)
    y = torch.linspace(0, 1, n_height, dtype=torch.float32, device=image.device) \
        .repeat(n_batch, n_width, 1) \
        .transpose(1, 2)

    # Apply shift in X direction, disparity is passed in NCHW format with 1 channel
    dx = disparity[:, 0, :, :] / n_width
    flow_field = torch.stack((x + dx, y), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    return torch.nn.functional.grid_sample(
        image,
        grid=(2 * flow_field - 1),
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True)

def meshgrid(n_batch, n_height, n_width, device, homogeneous=True):
    '''
    Creates N x 2 x H x W meshgrid in x, y directions

    Arg(s):
        n_batch : int
            batch size
        n_height : int
            height of tensor
        n_width : int
            width of tensor
        device : torch.device
            device on which to create meshgrid
        homoegenous : bool
            if set, then add homogeneous coordinates (N x H x W x 3)
    Return:
        torch.Tensor[float32]: N x 2 x H x W meshgrid of x, y and 1 (if homogeneous)
    '''

    x = torch.linspace(start=0.0, end=n_width-1, steps=n_width, device=device)
    y = torch.linspace(start=0.0, end=n_height-1, steps=n_height, device=device)

    # Create H x W grids
    grid_y, grid_x = torch.meshgrid(y, x)

    if homogeneous:
        # Create 3 x H x W grid (x, y, 1)
        grid_xy = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
    else:
        # Create 2 x H x W grid (x, y)
        grid_xy = torch.stack([grid_x, grid_y], dim=0)

    grid_xy = torch.unsqueeze(grid_xy, dim=0) \
        .repeat(n_batch, 1, 1, 1)

    return grid_xy

def backproject_to_camera(depth, intrinsics, shape):
    '''
    Backprojects pixel coordinates to 3D camera coordinates

    Arg(s):
        depth : torch.Tensor[float32]
            N x 1 x H x W depth map
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 4 x (H x W)
    '''

    n_batch, _, n_height, n_width = shape

    # Create homogeneous coordinates [x, y, 1]
    xy_h = meshgrid(n_batch, n_height, n_width, device=depth.device, homogeneous=True)

    # Reshape pixel coordinates to N x 3 x (H x W)
    xy_h = xy_h.view(n_batch, 3, -1)

    # Reshape depth as N x 1 x (H x W)
    depth = depth.view(n_batch, 1, -1)

    # K^-1 [x, y, 1] z
    points = torch.matmul(torch.inverse(intrinsics), xy_h) * depth

    # Make homogeneous
    return torch.cat([points, torch.ones_like(depth)], dim=1)

def project_to_pixel(points, pose, intrinsics, shape):
    '''
    Projects points in camera coordinates to 2D pixel coordinates

    Arg(s):
        points : torch.Tensor[float32]
            N x 4 x (H x W) depth map
        pose : torch.Tensor[float32]
            N x 4 x 4 transformation matrix
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 2 x H x W
    '''

    n_batch, _, n_height, n_width = shape

    # Convert camera intrinsics to homogeneous coordinates
    column = torch.zeros([n_batch, 3, 1], device=points.device)
    row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=points.device) \
        .view(1, 1, 4) \
        .repeat(n_batch, 1, 1)
    intrinsics = torch.cat([intrinsics, column], dim=2)
    intrinsics = torch.cat([intrinsics, row], dim=1)

    # Apply the transformation and project: \pi K g p
    T = torch.matmul(intrinsics, pose)
    T = T[:, 0:3, :]
    points = torch.matmul(T, points)
    points = points / (torch.unsqueeze(points[:, 2, :], dim=1) + 1e-7)
    points = points[:, 0:2, :]

    # Reshape to N x 2 x H x W
    return points.view(n_batch, 2, n_height, n_width)

def grid_sample(image, target_xy, shape, padding_mode='border'):
    '''
    Samples the image at x, y locations to target x, y locations

    Arg(s):
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        target_xy : torch.Tensor[float32]
            N x 2 x H x W target x, y locations in image space
        shape : list[int]
            shape of tensor in (N, C, H, W)
        padding_mode : str
            padding to use when sampled out of bounds
    Return:
        torch.Tensor[float32] : N x 3 x H x W RGB image
    '''

    _, _, n_height, n_width = shape

    # Swap dimensions to N x H x W x 2 for grid sample
    target_xy = target_xy.permute(0, 2, 3, 1)

    # Normalize coordinates between -1 and 1
    target_xy[..., 0] /= (n_width - 1.0)
    target_xy[..., 1] /= (n_height - 1.0)
    target_xy = 2.0 * (target_xy - 0.5)

    # Sample the image at normalized target x, y locations
    return torch.nn.functional.grid_sample(
        image,
        grid=target_xy,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True)


def color_consistency_loss_func(src, tgt):
    return torch.mean(torch.abs(tgt-src))

def ssim_loss_func(src, tgt):
    return torch.mean(ssim(src, tgt))

def smoothness_loss_func(predict, image):
    predict_dy, predict_dx = gradient_yx(predict)
    image_dy, image_dx = gradient_yx(image)

    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    smoothness_x = torch.mean(weights_x*torch.abs(predict_dx))
    smoothness_y = torch.mean(weights_y*torch.abs(predict_dy))

    return smoothness_x+smoothness_y

def left_right_consistency_loss_func(src, tgt):
    return torch.mean(torch.abs(src-tgt))


'''
Helper functions for constructing loss functions
'''
def gradient_yx(T):
    '''
    Computes gradients in the y and x directions

    Arg(s):
        T : torch.Tensor[float32]
            N x C x H x W tensor
    Returns:
        torch.Tensor[float32] : gradients in y direction
        torch.Tensor[float32] : gradients in x direction
    '''

    dx = T[:, :, :, :-1] - T[:, :, :, 1:]
    dy = T[:, :, :-1, :] - T[:, :, 1:, :]
    return dy, dx

def ssim(x, y):
    '''
    Computes Structural Similarity Index distance between two images

    Arg(s):
        x : torch.Tensor[float32]
            N x 3 x H x W RGB image
        y : torch.Tensor[float32]
            N x 3 x H x W RGB image
    Returns:
        torch.Tensor[float32] : SSIM distance between two images
    '''

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.nn.AvgPool2d(3, 1)(x)
    mu_y = torch.nn.AvgPool2d(3, 1)(y)
    mu_xy = mu_x * mu_y
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2

    sigma_x = torch.nn.AvgPool2d(3, 1)(x ** 2) - mu_xx
    sigma_y = torch.nn.AvgPool2d(3, 1)(y ** 2) - mu_yy
    sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_xy

    numer = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denom = (mu_xx + mu_yy + C1) * (sigma_x + sigma_y + C2)
    score = numer / denom

    return torch.clamp((1.0 - score) / 2.0, 0.0, 1.0)
