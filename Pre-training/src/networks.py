import torch
import net_utils


'''
Encoder architectures
'''
class ResNetEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections

    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(ResNetEncoder, self).__init__()

        use_bottleneck = False

        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 50:
            n_blocks = [3, 4, 6, 3]
            use_bottleneck = True
            resnet_block = net_utils.ResNetBottleneckBlock
        else:
            raise ValueError('Only supports 18, 34, 50 layer architecture')

        for n in range(len(n_filters) - len(n_blocks) - 1):
            n_blocks = n_blocks + [n_blocks[-1]]

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert network_depth == len(n_blocks) + 1

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        activation_func = net_utils.activation_func(activation_func)

        in_channels, out_channels = [input_channels, n_filters[filter_idx]]

        # Resolution 1/1 -> 1/2
        self.conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

        filter_idx = filter_idx + 1

        blocks2 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):

            if n == 0:
                pass
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels

            stride = 1

            block = resnet_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks2.append(block)

        self.blocks2 = torch.nn.Sequential(*blocks2)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks3 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):

            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                stride = 2
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                stride = 1

            block = resnet_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks3.append(block)

        self.blocks3 = torch.nn.Sequential(*blocks3)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks4 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):

            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                stride = 2
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                stride = 1

            block = resnet_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks4.append(block)

        self.blocks4 = torch.nn.Sequential(*blocks4)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks5 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):

            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                stride = 2
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                stride = 1

            block = resnet_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks5.append(block)

        self.blocks5 = torch.nn.Sequential(*blocks5)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            blocks6 = []
            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
            for n in range(n_blocks[block_idx]):

                if n == 0:
                    in_channels = 4 * in_channels if use_bottleneck else in_channels
                    stride = 2
                else:
                    in_channels = 4 * out_channels if use_bottleneck else out_channels
                    stride = 1

                block = resnet_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)

                blocks6.append(block)

            self.blocks6 = torch.nn.Sequential(*blocks6)
        else:
            self.blocks6 = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            blocks7 = []
            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
            for n in range(n_blocks[block_idx]):

                if n == 0:
                    in_channels = 4 * in_channels if use_bottleneck else in_channels
                    stride = 2
                else:
                    in_channels = 4 * out_channels if use_bottleneck else out_channels
                    stride = 1

                block = resnet_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)

                blocks7.append(block)

            self.blocks7 = torch.nn.Sequential(*blocks7)
        else:
            self.blocks7 = None

    def forward(self, x):
        '''
        Forward input x through the ResNet model

        Arg(s):
            x : torch.Tensor[float32]
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))

        # Resolution 1/4 -> 1/8
        layers.append(self.blocks3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.blocks4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.blocks5(layers[-1]))

        # Resolution 1/32 -> 1/64
        if self.blocks6 is not None:
            layers.append(self.blocks6(layers[-1]))

        # Resolution 1/64 -> 1/128
        if self.blocks7 is not None:
            layers.append(self.blocks7(layers[-1]))

        return layers[-1], layers[1:-1]


class VGGNetEncoder(torch.nn.Module):
    '''
    VGGNet encoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input data
        n_layer : int
            architecture type based on layers: 8, 11, 13
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(VGGNetEncoder, self).__init__()

        if n_layer == 8:
            n_convolutions = [1, 1, 1, 1, 1]
        elif n_layer == 11:
            n_convolutions = [1, 1, 2, 2, 2]
        elif n_layer == 13:
            n_convolutions = [2, 2, 2, 2, 2]
        else:
            raise ValueError('Only supports 8, 11, 13 layer architecture')

        assert len(n_filters) == len(n_convolutions)

        activation_func = net_utils.activation_func(activation_func)

        # Resolution 1/1 -> 1/2
        stride = 1 if n_convolutions[0] - 1 > 0 else 2
        in_channels, out_channels = [input_channels, n_filters[0]]

        conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        if n_convolutions[0] - 1 > 0:
            self.conv1 = torch.nn.Sequential(
                conv1,
                net_utils.VGGNetBlock(
                    out_channels,
                    out_channels,
                    n_convolution=n_convolutions[0] - 1,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm))
        else:
            self.conv1 = conv1

        # Resolution 1/2 -> 1/4
        in_channels, out_channels = [n_filters[0], n_filters[1]]
        self.conv2 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[1],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/4 -> 1/8
        in_channels, out_channels = [n_filters[1], n_filters[2]]
        self.conv3 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[2],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/8 -> 1/16
        in_channels, out_channels = [n_filters[2], n_filters[3]]
        self.conv4 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[3],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/16 -> 1/32
        in_channels, out_channels = [n_filters[3], n_filters[4]]
        self.conv5 = net_utils.VGGNetBlock(
            in_channels,
            out_channels,
            n_convolution=n_convolutions[4],
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

    def forward(self, x):
        '''
        Forward input x through the VGGNet model

        Arg(s):
            x : torch.Tensor[float32]
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''
        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        layers.append(self.conv2(layers[-1]))

        # Resolution 1/4 -> 1/8
        layers.append(self.conv3(layers[-1]))

        # Resolution 1/8 -> 1/32
        layers.append(self.conv4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.conv5(layers[-1]))

        return layers[-1], layers[1:-1]


'''
Decoder architectures
'''
class MultiScaleDecoder(torch.nn.Module):
    '''
    Multi-scale decoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
        n_resolution : int
            number of output resolutions (scales) for multi-scale prediction
        n_filters : int list
            number of filters to use at each decoder block
        n_skips : int list
            number of filters from skip connections
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        output_func : func
            activation function for output
        use_batch_norm : bool
            if set, then applied batch normalization
        deconv_type : str
            deconvolution types available: transpose, up
    '''

    def __init__(self,
                 input_channels=256,
                 output_channels=1,
                 n_resolution=4,
                 n_filters=[256, 128, 64, 32, 16],
                 n_skips=[256, 128, 64, 32, 0],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 output_func='linear',
                 use_batch_norm=False,
                 deconv_type='transpose'):
        super(MultiScaleDecoder, self).__init__()

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert n_resolution > 0 and n_resolution < network_depth

        self.n_resolution = n_resolution
        self.output_func = output_func

        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)

        # Upsampling from lower to full resolution requires multi-scale
        if 'upsample' in self.output_func and self.n_resolution < 2:
            self.n_resolution = 2

        filter_idx = 0

        in_channels, skip_channels, out_channels = [
            input_channels, n_skips[filter_idx], n_filters[filter_idx]
        ]

        # Resolution 1/128 -> 1/64
        if network_depth > 6:
            self.deconv6 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv6 = None

        # Resolution 1/64 -> 1/32
        if network_depth > 5:
            self.deconv5 = net_utils.DecoderBlock(
                in_channels,
                skip_channels,
                out_channels,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                deconv_type=deconv_type)

            filter_idx = filter_idx + 1

            in_channels, skip_channels, out_channels = [
                n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
            ]
        else:
            self.deconv5 = None

        # Resolution 1/32 -> 1/16
        self.deconv4 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        # Resolution 1/16 -> 1/8
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        self.deconv3 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        self.output3 = net_utils.Conv2d(out_channels, output_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=False)

        # Resolution 1/8 -> 1/4
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 3:
            skip_channels = skip_channels + output_channels

        self.deconv2 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        self.output2 = net_utils.Conv2d(
            out_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=output_func,
            use_batch_norm=False)

        # Resolution 1/4 -> 1/2
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 2:
            skip_channels = skip_channels + output_channels

        self.deconv1 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        self.output1 = net_utils.Conv2d(
            out_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=output_func,
            use_batch_norm=False)

        # Resolution 1/2 -> 1/1
        filter_idx = filter_idx + 1

        in_channels, skip_channels, out_channels = [
            n_filters[filter_idx-1], n_skips[filter_idx], n_filters[filter_idx]
        ]

        if self.n_resolution > 1:
            skip_channels = skip_channels + output_channels

        self.deconv0 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            deconv_type=deconv_type)

        self.output0 = net_utils.Conv2d(
            out_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=output_func,
            use_batch_norm=False)

    def forward(self, x, skips, shape=None):
        '''
        Forward latent vector x through decoder network

        Arg(s):
            x : torch.Tensor[float32]
                latent vector
            skips : list[torch.Tensor[float32]]
                list of skip connection tensors (earlier are larger resolution)
        Returns:
            list[torch.Tensor[float32]] : list of outputs at multiple scales
        '''

        layers = [x]
        outputs = []

        # Start at the end and walk backwards through skip connections
        n = len(skips) - 1

        # Resolution 1/128 -> 1/64
        if self.deconv6 is not None:
            layers.append(self.deconv6(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/64 -> 1/32
        if self.deconv5 is not None:
            layers.append(self.deconv5(layers[-1], skips[n]))
            n = n - 1

        # Resolution 1/32 -> 1/16
        layers.append(self.deconv4(layers[-1], skips[n]))

        # Resolution 1/16 -> 1/8
        n = n - 1

        layers.append(self.deconv3(layers[-1], skips[n]))

        if self.n_resolution > 3:
            output3 = self.output3(layers[-1])
            outputs.append(output3)

            upsample_output3 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/8 -> 1/4
        n = n - 1

        skip = torch.cat([skips[n], upsample_output3], dim=1) if self.n_resolution > 3 else skips[n]
        layers.append(self.deconv2(layers[-1], skip))

        if self.n_resolution > 2:
            output2 = self.output2(layers[-1])
            outputs.append(output2)

            upsample_output2 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/4 -> 1/2
        n = n - 1

        skip = torch.cat([skips[n], upsample_output2], dim=1) if self.n_resolution > 2 else skips[n]
        layers.append(self.deconv1(layers[-1], skip))

        if self.n_resolution > 1:
            output1 = self.output1(layers[-1])
            outputs.append(output1)

            upsample_output1 = torch.nn.functional.interpolate(
                input=outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/2 -> 1/1
        n = n - 1

        if 'upsample' in self.output_func:
            output0 = upsample_output1
        else:
            if self.n_resolution > 1:
                # If there is skip connection at layer 0
                skip = torch.cat([skips[n], upsample_output1], dim=1) if n == 0 else upsample_output1
                layers.append(self.deconv0(layers[-1], skip))
            else:

                if n == 0:
                    layers.append(self.deconv0(layers[-1], skips[n]))
                else:
                    layers.append(self.deconv0(layers[-1], shape=shape[-2:]))

            output0 = self.output0(layers[-1])

        outputs.append(output0)

        return outputs


class DisparityDecoder(torch.nn.Module):

    def __init__(self,
                 input_channels=256,
                 output_channels=2,
                 n_pyramid=4,
                 n_filters=[256, 128, 64, 32, 16],
                 n_skips=[256, 128, 64, 32, 0],
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 deconv_type='transpose'):
        super(DisparityDecoder, self).__init__()

        network_depth = 5
        assert(n_pyramid > 0 and n_pyramid < network_depth)
        assert(len(n_filters) == network_depth)
        assert(len(n_skips) == network_depth)

        activation_func = net_utils.activation_func(activation_func)

        self.n_pyramid = n_pyramid

        # Resolution 1/32 -> 1/16
        in_channels, skip_channels, out_channels = [input_channels, n_skips[0], n_filters[0]]

        self.deconv4 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            activation_func=activation_func,
            deconv_type=deconv_type)

        # Resolution 1/16 -> 1/8
        in_channels, skip_channels, out_channels = [n_filters[0], n_skips[1], n_filters[1]]

        self.deconv3 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            activation_func=activation_func,
            deconv_type=deconv_type)

        self.output3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            torch.nn.Sigmoid())

        # Resolution 1/8 -> 1/4
        in_channels, skip_channels, out_channels = [n_filters[1], n_skips[2], n_filters[2]]

        if self.n_pyramid > 3:
            skip_channels = skip_channels+output_channels

        self.deconv2 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            activation_func=activation_func,
            deconv_type=deconv_type)

        self.output2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            torch.nn.Sigmoid())

        # Resolution 1/4 -> 1/2
        in_channels, skip_channels, out_channels = [n_filters[2], n_skips[3], n_filters[3]]

        if self.n_pyramid > 2:
            skip_channels = skip_channels+output_channels

        self.deconv1 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            activation_func=activation_func,
            deconv_type=deconv_type)

        self.output1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            torch.nn.Sigmoid())

        # Resolution 1/2 -> 1/1
        in_channels, skip_channels, out_channels = [n_filters[3], n_skips[4], n_filters[4]]

        if self.n_pyramid > 1:
            skip_channels = skip_channels+output_channels

        self.deconv0 = net_utils.DecoderBlock(
            in_channels,
            skip_channels,
            out_channels,
            activation_func=activation_func,
            deconv_type=deconv_type)

        self.output0 = torch.nn.Sequential(
            torch.nn.Conv2d(
                out_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            torch.nn.Sigmoid())

    def forward(self, x, skips, shape):
        layers = [x]
        outputs = []

        # Resolution 1/32 -> 1/16
        n = len(skips)-1
        layers.append(self.deconv4(layers[-1], skips[n]))

        # Resolution 1/16 -> 1/8
        n = n-1
        layers.append(self.deconv3(layers[-1], skips[n]))

        if self.n_pyramid > 3:
            outputs.append(self.output3(layers[-1]))
            upsample_output3 = torch.nn.functional.interpolate(
                outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/8 -> 1/4
        n = n-1
        skip = torch.cat([skips[n], upsample_output3], dim=1) if self.n_pyramid > 3 else skips[n]
        layers.append(self.deconv2(layers[-1], skip))

        if self.n_pyramid > 2:
            outputs.append(self.output2(layers[-1]))
            upsample_output2 = torch.nn.functional.interpolate(
                outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/4 -> 1/2
        n = n-1
        skip = torch.cat([skips[n], upsample_output2], dim=1) if self.n_pyramid > 2 else skips[n]
        layers.append(self.deconv1(layers[-1], skip))

        if self.n_pyramid > 1:
            outputs.append(self.output1(layers[-1]))
            upsample_output1 = torch.nn.functional.interpolate(
                outputs[-1],
                scale_factor=2,
                mode='bilinear',
                align_corners=True)

        # Resolution 1/2 -> 1/1
        if self.n_pyramid > 1:
            layers.append(self.deconv0(layers[-1], upsample_output1))
        else:
            layers.append(self.deconv0(layers[-1], shape=shape))

        outputs.append(self.output0(layers[-1]))

        return outputs


class PoseDecoder(torch.nn.Module):
    '''
    Pose decoder that outputs 4 x 4 pose matrix

    Arg(s):
        rotation_parameterization : str
            axis
        input_channels : int
            number of channels in input latent vector
        n_filters : int list
            number of filters to use at each decoder block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 rotation_parameterization,
                 input_channels=256,
                 n_filters=[],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(PoseDecoder, self).__init__()

        self.rotation_parameterization = rotation_parameterization

        activation_func = net_utils.activation_func(activation_func)

        if len(n_filters) > 0:
            layers = []
            in_channels = input_channels

            for out_channels in n_filters:
                conv = net_utils.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                layers.append(conv)
                in_channels = out_channels

            conv = net_utils.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=None,
                use_batch_norm=False)
            layers.append(conv)

            self.conv = torch.nn.Sequential(*layers)
        else:
            self.conv = net_utils.Conv2d(
                in_channels=input_channels,
                out_channels=6,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=None,
                use_batch_norm=False)

    def forward(self, x):
        '''
        Forward through pose decoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x H latent vector
        Returns:
            torch.Tensor[float32] : N x 4 x 4 pose matrix
        '''

        conv_output = self.conv(x)
        pose_mean = torch.mean(conv_output, [2, 3])
        dof = 0.01 * pose_mean
        pose_matrix = net_utils.pose_matrix(
            dof,
            rotation_parameterization=self.rotation_parameterization)

        return pose_matrix
