import torch


def activation_func(activation_fn):
    '''
    Select activation function

    Arg(s):
        activation_fn : str
            name of activation function
    Returns:
        func : activation function
    '''

    if 'linear' in activation_fn:
        return None
    elif 'leaky_relu' in activation_fn:
        return torch.nn.LeakyReLU(negative_slope=0.20, inplace=True)
    elif 'relu' in activation_fn:
        return torch.nn.ReLU()
    elif 'elu' in activation_fn:
        return torch.nn.ELU()
    elif 'sigmoid' in activation_fn:
        return torch.nn.Sigmoid()
    else:
        raise ValueError('Unsupported activation function: {}'.format(activation_fn))


'''
Network layers
'''
class Conv2d(torch.nn.Module):
    '''
    2D convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(Conv2d, self).__init__()

        self.use_batch_norm = use_batch_norm
        padding = kernel_size // 2

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)

        self.activation_func = activation_func

        if self.use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through the convolutional layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        conv = self.conv(x)
        conv = self.batch_norm(conv) if self.use_batch_norm else conv

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class TransposeConv2d(torch.nn.Module):
    '''
    Transpose convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(TransposeConv2d, self).__init__()

        self.use_batch_norm = use_batch_norm
        padding = kernel_size // 2

        self.deconv = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)

        self.activation_func = activation_func

        if self.use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through the transposed convolutional layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x 2H x 2W output tensor
        '''

        deconv = self.deconv(x)
        deconv = self.batch_norm(deconv) if self.use_batch_norm else deconv

        if self.activation_func is not None:
            return self.activation_func(deconv)
        else:
            return deconv


class UpConv2d(torch.nn.Module):
    '''
    Up-convolution (upsample + convolution) block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        shape : list[int]
            two element tuple of ints (height, width)
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(UpConv2d, self).__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

    def forward(self, x, shape):
        '''
        Forward input x through the up-convolutional layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        upsample = torch.nn.functional.interpolate(x, size=shape)
        conv = self.conv(upsample)
        return conv


'''
Network encoder blocks
'''
class ResNetBlock(torch.nn.Module):
    '''
    Basic ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(ResNetBlock, self).__init__()

        self.activation_func = activation_func

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.projection = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False)

    def forward(self, x):
        '''
        Forward input x through a basic ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x H/s x W/s output tensor
        '''

        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv2 + X)


class ResNetBottleneckBlock(torch.nn.Module):
    '''
    ResNet bottleneck block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(ResNetBottleneckBlock, self).__init__()

        self.activation_func = activation_func

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv3 = Conv2d(
            out_channels,
            4 * out_channels,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.projection = Conv2d(
            in_channels,
            4 * out_channels,
            kernel_size=1,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False)

    def forward(self, x):
        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x

        # f(x) + x
        return self.activation_func(conv3 + X)


class VGGNetBlock(torch.nn.Module):
    '''
    VGGNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_convolution : int
            number of convolution layers
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False):
        super(VGGNetBlock, self).__init__()

        layers = []
        for _ in range(n_convolution - 1):
            conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
            layers.append(conv)
            in_channels = out_channels

        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


'''
Network decoder blocks
'''
class DecoderBlock(torch.nn.Module):
    '''
    Decoder block with skip connection

    Arg(s):
        in_channels : int
            number of input channels
        skip_channels : int
            number of skip connection channels
        out_channels : int
            number of output channels
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        deconv_type : str
            deconvolution types: transpose, up
    '''

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 deconv_type='transpose'):
        super(DecoderBlock, self).__init__()

        self.skip_channels = skip_channels
        self.deconv_type = deconv_type

        if deconv_type == 'transpose':
            self.deconv = TransposeConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif deconv_type == 'up':
            self.deconv = UpConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

        concat_channels = skip_channels + out_channels

        self.conv = Conv2d(
            concat_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

    def forward(self, x, skip=None, shape=None):
        '''
        Forward input x through a decoder block and fuse with skip connection

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            skip : torch.Tensor[float32]
                N x F x h x w skip connection
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        if self.deconv_type == 'transpose':
            deconv = self.deconv(x)
        elif self.deconv_type == 'up':

            if skip is not None:
                shape = skip.shape[2:4]
            elif shape is not None:
                pass
            else:
                n_height, n_width = x.shape[2:4]
                shape = (int(2 * n_height), int(2 * n_width))

            deconv = self.deconv(x, shape=shape)

        if self.skip_channels > 0:
            concat = torch.cat([deconv, skip], dim=1)
        else:
            concat = deconv

        return self.conv(concat)


'''
Pose regression layer
'''
def pose_matrix(v, rotation_parameterization='axis'):
    '''
    Convert 6 DoF parameters to transformation matrix

    Arg(s):
        v : torch.Tensor[float32]
            N x 6 vector in the order of tx, ty, tz, rx, ry, rz
        rotation_parameterization : str
            axis
    Returns:
        torch.Tensor[float32] : N x 4 x 4 homogeneous transformation matrix
    '''

    # Select N x 3 element rotation vector
    r = v[..., :3]
    # Select N x 3 element translation vector
    t = v[..., 3:]

    if rotation_parameterization == 'axis':
        Rt = axis_angle(torch.unsqueeze(r, dim=1), t)
    else:
        raise ValueError('Unsupported rotation parameterization: {}'.format(rotation_parameterization))

    return Rt

def axis_angle(w, t):
    '''
    Convert axis angle rotation paramters and translation vector a 4 x 4 matrix

    Arg(s):
        w : torch.Tensor[float32]
            N x 1 x 3 axis angle rotation parameters
        t : torch.Tensor[float32]
            N x 1 x 3 translation vector
    Returns:
        torch.Tensor[float32] : 4 x 4 pose matrix
    '''

    R = to_rotation_matrix(w)
    T = to_translation_matrix(t)

    return torch.matmul(T, R)

def to_rotation_matrix(w):
    '''
    Convert axis angle rotation parameters into a 4 x 4 transformation matrix

    Arg(s):
        w : torch.Tensor[float32]
            N x 1 x 3 rotation parameters
    Returns:
        torch.Tensor[float32] : N x 4 x 4 rotation matrix
    '''

    angle = torch.norm(w, 2, 2, True)
    axis = w / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((w.shape[0], 4, 4)).to(device=w.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def to_translation_matrix(t):
    '''
    Convert a translation vector into a 4 x 4 transformation matrix

    Arg(s):
        t : torch.Tensor[float32]
            N x 1 x 3 translation vector
    Returns:
        torch.Tensor[float32] : N x 4 x 4 translation matrix
    '''

    T = torch.zeros(t.shape[0], 4, 4).to(device=t.device)

    t = t.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def warp1d_horizontal(image, disparity):
    n_batch, _, n_height, n_width = image.shape

    # Original coordinates of pixels
    x = torch.linspace(0, 1, n_width, dtype=torch.float32, device=image.device) \
        .repeat(n_batch, n_height, 1)
    y = torch.linspace(0, 1, n_height, dtype=torch.float32, device=image.device) \
        .repeat(n_batch, n_width, 1) \
        .transpose(1, 2)

    # Apply shift in X direction
    dx = disparity[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x+dx, y), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    return torch.nn.functional.grid_sample(
        image,
        grid=2*flow_field-1,
        mode='bilinear',
        padding_mode='zeros')
