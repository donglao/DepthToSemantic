import torch, torchvision
import networks
import losses, loss_utils, log_utils


class PoseNetModel(object):
    '''
    Pose network for computing relative pose between a pair of images

    Arg(s):
        encoder_type : str
            posenet, resnet18, resnet34
        rotation_parameterization : str
            currently only supports axis
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        device : torch.device
            device for running model
    '''

    def __init__(self,
                 encoder_type='posenet',
                 rotation_parameterization='axis',
                 weight_initializer='xavier_normal',
                 activation_func='leaky_relu',
                 device=torch.device('cuda')):

        self.device = device

        # Create pose encoder
        if 'resnet18' in encoder_type:
            self.encoder = networks.ResNetEncoder(
                n_layer=18,
                input_channels=6,
                n_filters=[16, 32, 64, 128, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        elif 'resnet34' in encoder_type:
            self.encoder = networks.ResNetEncoder(
                n_layer=34,
                input_channels=6,
                n_filters=[16, 32, 64, 128, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        elif 'vggnet08' in encoder_type:
            self.encoder = networks.VGGNetEncoder(
                n_layer=8,
                input_channels=6,
                n_filters=[16, 32, 64, 128, 256],
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True)
        else:
            raise ValueError('Unsupported PoseNet encoder type: {}'.format(encoder_type))

        # Create pose decoder
        self.decoder = networks.PoseDecoder(
            rotation_parameterization=rotation_parameterization,
            input_channels=256,
            n_filters=[256, 256],
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=True)

        # Move to device
        self.data_parallel()
        self.to(self.device)

    def forward(self, image0, image1):
        '''
        Forwards the inputs through the network

        Arg(s):
            image0 : torch.Tensor[float32]
                image at time step 0
            image1 : torch.Tensor[float32]
                image at time step 1
        Returns:
            torch.Tensor[float32] : pose from time step 1 to 0
        '''

        # Forward through the network
        latent, _ = self.encoder(torch.cat([image0, image1], dim=1))
        output = self.decoder(latent)

        return output

    def compute_loss(self,
                     depth0,
                     image0,
                     image1,
                     image2,
                     pose0to1,
                     pose0to2,
                     intrinsics,
                     w_color=0.15,
                     w_structure=0.85):
        '''
        Computes loss function

        Arg(s):
            depth0 : list[torch.Tensor[float32]]
                N x 1 x H x W depth for image
            image0 : torch.Tensor[float32]
                N x 3 x H x W t image
            image1 : torch.Tensor[float32]
                N x 3 x H x W t-1 image
            image2 : torch.Tensor[float32]
                N x 3 x H x W t+1 image
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from image 0 to image 1
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from image 0 to image 1
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix for camera
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        shape = image0.shape

        # Define losses
        loss_color = 0.0
        loss_structure = 0.0

        points0 = loss_utils.backproject_to_camera(depth0, intrinsics, shape)

        # Perform rigid warping from image 1 to 0 and compute loss
        xy0to1 = loss_utils.project_to_pixel(points0, pose0to1, intrinsics, shape)
        image1to0 = loss_utils.grid_sample(image1, xy0to1, shape)

        loss_color1to0 = losses.color_consistency_loss(image0, image1to0)
        loss_structure1to0 = losses.structural_consistency_loss_func(image0, image1to0)

        # Perform rigid warping from image 2 to 0 and compute loss
        xy0to2 = loss_utils.project_to_pixel(points0, pose0to2, intrinsics, shape)
        image2to0 = loss_utils.grid_sample(image2, xy0to2, shape)

        loss_color2to0 = losses.color_consistency_loss(image0, image2to0)
        loss_structure2to0 = losses.structural_consistency_loss_func(image0, image2to0)

        loss_color = loss_color + \
            loss_color1to0 + \
            loss_color2to0

        loss_structure = loss_structure + \
            loss_structure1to0 + \
            loss_structure2to0

        loss = \
            w_color * loss_color + \
            w_structure * loss_structure

        # Save loss info to be logged
        loss_info = {
            'loss_color': loss_color,
            'loss_structure': loss_structure,
            'loss' : loss,
            'image1to0': image1to0,
            'image2to0': image2to0,
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the set of parameters
        '''

        return list(self.encoder.parameters()) + list(self.decoder.parameters())

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
        # Save training state
        checkpoint['train_step'] = step
        checkpoint['epoch'] = epoch
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save encoder and decoder weights
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
            int : epoch in optimization
        '''

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore encoder and decoder weights
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
                    max_display_depth=100.0,
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
            output_depth0 : torch.Tensor[float32]
                output depth for left camera
            scalars : dict[str, float]
                dictionary of scalars to log
            max_display_depth : float
                max value of depth to display in Tensorboard
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
                            (output_depth0_summary / max_display_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth0_distro', output_depth0, global_step=step)

            # Log distribution of pose 1 to 0 translation vector
            summary_writer.add_histogram(tag + '_tx0to1_distro', pose0to1[:, 0, 3], global_step=step)
            summary_writer.add_histogram(tag + '_ty0to1_distro', pose0to1[:, 1, 3], global_step=step)
            summary_writer.add_histogram(tag + '_tz0to1_distro', pose0to1[:, 2, 3], global_step=step)

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
