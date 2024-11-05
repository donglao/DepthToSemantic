import torch, torchvision
import log_utils, losses, loss_utils, networks


class DepthNetModel(object):
    '''
    Single image depth estimation network model

    Arg(s):
        encoder_type : str
            encoder type
        input_channels : int
            number of channels in the image
        n_filters_encoder : list[int]
            list of filters for each layer in encoder
        decoder_type : str
            decoder type
        n_resolution_decoder_output : int
            number of resolutions of multiscale outputs
        n_filters_decoder : list[int]
            list of filters for each layer in decoder
        activation_func : str
            activation function for network
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        use_batch_norm : bool
            if set, then applied batch normalization
        min_predict_depth : float
            minimum predicted depth
        max_predict_depth : float
            maximum predicted depth
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 encoder_type='resnet18',
                 input_channels=3,
                 n_filters_encoder=[32, 64, 96, 128, 256],
                 decoder_type='multiscale',
                 n_resolution_decoder_output=1,
                 n_filters_decoder=[256, 128, 96, 64, 32],
                 activation_func='leaky_relu',
                 weight_initializer='xavier_normal',
                 use_batch_norm=False,
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.device = device

        # Build encoder
        if 'resnet18' in encoder_type:
            self.encoder = networks.ResNetEncoder(
                n_layer=18,
                input_channels=input_channels,
                n_filters=n_filters_encoder,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif 'resnet34' in encoder_type:
            self.encoder = networks.ResNetEncoder(
                n_layer=34,
                input_channels=input_channels,
                n_filters=n_filters_encoder,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif 'vggnet08' in encoder_type:
            self.encoder = networks.VGGNetEncoder(
                n_layer=8,
                input_channels=input_channels,
                n_filters=n_filters_encoder,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        else:
            raise ValueError('Encoder type {} not supported.'.format(encoder_type))

        # Calculate number of channels for latent and skip connections
        latent_channels = n_filters_encoder[-1]
        n_skips = n_filters_encoder[:-1]
        n_skips = n_skips[::-1] + [0]

        # Build decoder
        if 'multiscale' in decoder_type:
            self.decoder = networks.MultiScaleDecoder(
                input_channels=latent_channels,
                output_channels=1,
                n_resolution=n_resolution_decoder_output,
                n_filters=n_filters_decoder,
                n_skips=n_skips,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='sigmoid',
                use_batch_norm=use_batch_norm,
                deconv_type='up')
        else:
            raise ValueError('Decoder type {} not supported.'.format(decoder_type))

        # Move to device
        self.data_parallel()
        self.to(self.device)

    def forward(self, image):
        '''
        Forwards the inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
        Returns:
            torch.Tensor[float32] : N x 1 x H x W output dense depth
        '''

        latent, skips = self.encoder(image)

        outputs = self.decoder(latent, skips, shape=image.shape[-2:])

        output_depth = outputs[-1]

        output_depth = \
            self.min_predict_depth / (output_depth + self.min_predict_depth / self.max_predict_depth)

        return output_depth

    def compute_loss(self,
                     output_depth0,
                     supervision_type=['monocular', 'stereo'],
                     image0=None,
                     image1=None,
                     image2=None,
                     image3=None,
                     pose0to1=None,
                     pose0to2=None,
                     intrinsics0=None,
                     focal_length_baseline0=None,
                     w_color=0.15,
                     w_structure=0.95,
                     w_smoothness=0.05):
        '''
        Computes loss function

        Arg(s):
            output_depth0 : torch.Tensor[float32]
                N x 1 x H x W output depth for left image
            supervision_type : list[str]
                monocular, stereo
            image0 : torch.Tensor[float32]
                N x 3 x H x W left image
            image1 : torch.Tensor[float32]
                N x 3 x H x W t-1 image
            image2 : torch.Tensor[float32]
                N x 3 x H x W t+1 image
            image3 : torch.Tensor[float32]
                N x 3 x H x W right image
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from image 0 to image 1
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from image 0 to image 1
            intrinsics0 : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix for left camera
            focal_length_baseline0 : torch.Tensor[float32]
                N x 2 focal length and baseline for left camera
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
            w_smoothness : float
                weight of smoothness term
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        shape = image0.shape

        '''
        Photometric reprojection losses
        '''
        if 'monocular' in supervision_type:
            points0 = loss_utils.backproject_to_camera(output_depth0, intrinsics0, shape)

            # Perform rigid warping from image 1 to 0 and compute loss
            xy0to1 = loss_utils.project_to_pixel(points0, pose0to1, intrinsics0, shape)
            image1to0 = loss_utils.grid_sample(image1, xy0to1, shape)

            loss_color1to0 = losses.color_consistency_loss(image0, image1to0)
            loss_structure1to0 = losses.structural_consistency_loss_func(image0, image1to0)

            # Perform rigid warping from image 2 to 0 and compute loss
            xy0to2 = loss_utils.project_to_pixel(points0, pose0to2, intrinsics0, shape)
            image2to0 = loss_utils.grid_sample(image2, xy0to2, shape)

            loss_color2to0 = losses.color_consistency_loss(image0, image2to0)
            loss_structure2to0 = losses.structural_consistency_loss_func(image0, image2to0)
        else:
            image1to0 = image1
            loss_color1to0 = 0.0
            loss_structure1to0 = 0.0

            image2to0 = image2
            loss_color2to0 = 0.0
            loss_structure2to0 = 0.0

        if 'stereo' in supervision_type:
            # Get focal length baseline for 1d horizontal shift
            fb = focal_length_baseline0[:, 0] * focal_length_baseline0[:, 1]
            fb = fb[:, None, None, None]

            # Perform disparity shift from stereo image and compute loss
            image3to0 = loss_utils.warp1d_horizontal(image3, -fb / output_depth0)
            loss_color3to0 = losses.color_consistency_loss(image0, image3to0)
            loss_structure3to0 = losses.structural_consistency_loss_func(image0, image3to0)
        else:
            image3to0 = image3
            loss_color3to0 = 0.0
            loss_structure3to0 = 0.0

        loss_color = loss_color1to0 + loss_color2to0 + loss_color3to0
        loss_structure = loss_structure1to0 + loss_structure2to0 + loss_structure3to0

        # Local smoothness loss
        loss_smoothness = losses.smoothness_loss_func(output_depth0, image0)

        # Compute total loss
        loss = w_color * loss_color + \
            w_structure * loss_structure + \
            w_smoothness * w_smoothness

        # Save loss info to be logged
        loss_info = {
            'loss_color': loss_color,
            'loss_structure': loss_structure,
            'loss_smoothness': loss_smoothness,
            'loss' : loss,
            'image1to0': image1to0,
            'image2to0': image2to0,
            'image3to0': image3to0,
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = \
            list(self.encoder.parameters()) + \
            list(self.decoder.parameters())

        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder.train()
        self.decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder.eval()
        self.decoder.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        # Move to device
        self.encoder.to(device)
        self.decoder.to(device)

    def save_model(self, checkpoint_path, step, optimizer, epoch):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
            epoch : int
                current training epoch
        '''

        checkpoint = {}
        checkpoint['train_step'] = step
        checkpoint['epoch'] = epoch
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Load weights for sparse_to_dense_depth, encoder, and decoder
        checkpoint['encoder_state_dict'] = self.encoder.state_dict()
        checkpoint['decoder_state_dict'] = self.decoder.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def restore_model(self, checkpoint_path, optimizer=None):
        '''
        Restore weights of the model

        Arg(s):
            checkpoint_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore weights for encoder, and decoder
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        try:
            epoch = checkpoint['epoch']
        except Exception:
            epoch = None

        return checkpoint['train_step'], optimizer, epoch

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image0=None,
                    image1to0=None,
                    image2to0=None,
                    image3to0=None,
                    output_depth0=None,
                    ground_truth0=None,
                    pose0to1=None,
                    pose0to2=None,
                    scalars={},
                    n_display=4):
        '''
        Logs summary to Tensorboard
        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image0 : torch.Tensor[float32]
                image from left camera
            image1to0 : torch.Tensor[float32]
                image at time step t-1 warped to time step t
            image2to0 : torch.Tensor[float32]
                image at time step t+1 warped to time step t
            image3to0 : torch.Tensor[float32]
                image from right camera reprojected to left camera
            output_depth0 : torch.Tensor[float32]
                output depth for left camera
            ground_truth0 : torch.Tensor[float32]
                ground truth depth for left camera
            scalars : dict[str, float]
                dictionary of scalars to log
            n_display : int
                number of images to display
        '''

        with torch.no_grad():

            display_summary_image = []
            display_summary_depth = []

            display_summary_image_text = tag
            display_summary_depth_text = tag

            if image0 is not None:
                image0_summary = image0[0:n_display, ...]

                display_summary_image_text += '_image0'
                display_summary_depth_text += '_image0'

                # Add to list of images to log
                display_summary_image.append(
                    torch.cat([
                        image0_summary.cpu(),
                        torch.zeros_like(image0_summary, device=torch.device('cpu'))],
                        dim=-1))

                display_summary_depth.append(display_summary_image[-1])

            image_list = []
            image_text_list = []

            if image1to0 is not None:
                image_list.append(image1to0)
                image_text_list.append('_image1to0_output-error')

            if image2to0 is not None:
                image_list.append(image2to0)
                image_text_list.append('_image2to0_output-error')

            if image3to0 is not None:
                image_list.append(image3to0)
                image_text_list.append('_image3to0_output-error')

            for image, image_text in zip(image_list, image_text_list):

                if image0 is not None and image is not None:
                    image_summary = image[0:n_display, ...]

                    display_summary_image_text += image_text

                    # Compute reconstruction error w.r.t. image 0
                    image_error_summary = torch.mean(
                        torch.abs(image0_summary - image_summary),
                        dim=1,
                        keepdim=True)

                    # Add to list of images to log
                    image_error_summary = log_utils.colorize(
                        (image_error_summary / 0.10).cpu(),
                        colormap='inferno')

                    display_summary_image.append(
                        torch.cat([
                            image_summary.cpu(),
                            image_error_summary],
                            dim=3))

            if output_depth0 is not None:
                output_depth0_summary = output_depth0[0:n_display, ...]

                display_summary_depth_text += '_output0'

                # Add to list of images to log
                n_batch, _, n_height, n_width = output_depth0_summary.shape

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            (output_depth0_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth0_distro', output_depth0, global_step=step)

            if output_depth0 is not None and ground_truth0 is not None:
                validity_map0 = torch.unsqueeze(ground_truth0[:, 1, :, :], dim=1)
                ground_truth0 = torch.unsqueeze(ground_truth0[:, 0, :, :], dim=1)

                validity_map0_summary = validity_map0[0:n_display]
                ground_truth0_summary = ground_truth0[0:n_display]

                display_summary_depth_text += '_groundtruth0-error'

                # Compute output error w.r.t. ground truth
                ground_truth0_error_summary = \
                    torch.abs(output_depth0_summary - ground_truth0_summary)

                ground_truth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (ground_truth0_error_summary + 1e-8) / (ground_truth0_summary + 1e-8),
                    validity_map0_summary)

                # Add to list of images to log
                ground_truth0_summary = log_utils.colorize(
                    (ground_truth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                ground_truth0_error_summary = log_utils.colorize(
                    (ground_truth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        ground_truth0_summary,
                        ground_truth0_error_summary],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth0_distro', ground_truth0, global_step=step)

            if pose0to1 is not None:
                # Log distribution of pose 1 to 0translation vector
                summary_writer.add_histogram(tag + '_tx0to1_distro', pose0to1[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty0to1_distro', pose0to1[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz0to1_distro', pose0to1[:, 2, 3], global_step=step)

            if pose0to2 is not None:
                # Log distribution of pose 2 to 0 translation vector
                summary_writer.add_histogram(tag + '_tx0to2_distro', pose0to2[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty0to2_distro', pose0to2[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz0to2_distro', pose0to2[:, 2, 3], global_step=step)

            # Log scalars to tensorboard
            for (name, value) in scalars.items():
                summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

            # Log image summaries to tensorboard
            if len(display_summary_image) > 1:
                display_summary_image = torch.cat(display_summary_image, dim=2)

                summary_writer.add_image(
                    display_summary_image_text,
                    torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                    global_step=step)

            if len(display_summary_depth) > 1:
                display_summary_depth = torch.cat(display_summary_depth, dim=2)

                summary_writer.add_image(
                    display_summary_depth_text,
                    torchvision.utils.make_grid(display_summary_depth, nrow=n_display),
                    global_step=step)
