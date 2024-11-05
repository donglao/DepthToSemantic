import os, sys, torch, torchvision
import losses, loss_utils, log_utils
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'monodepth2'))
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'monodepth2', 'networks'))
from resnet_encoder import ResnetEncoder
from depth_decoder import DepthDecoder


class OpticalFlowModel(object):
    '''
    Optical flow model
    '''

    def __init__(self, encoder_type=['resnet18'], device=torch.device('cuda')):

        self.device = device

        # Create encoder and decoder network
        if 'resnet18' in encoder_type:
            n_layer = 18
        elif 'resnet50' in encoder_type:
            n_layer = 50
        else:
            raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

        self.encoder = ResnetEncoder(n_layer, pretrained='pretrained' in encoder_type)

        num_ch_enc = [
            2 * n for n in self.encoder.num_ch_enc
        ]
        self.decoder_flow = DepthDecoder(
            num_ch_enc=num_ch_enc,
            scales=range(4),
            num_output_channels=2)

        # Move to device
        self.to(self.device)

    def forward(self, image0, image1, return_all_output_resolutions=False):
        '''
        Forwards the inputs through the network

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W image
            image1 : torch.Tensor[float32]
                N x 3 x H x W image
            return_all_output_resolutions : bool
                if set then return a list of all outputs at each resolution
        Returns:
            torch.Tensor[float32] : N x 1 x H x W output flow from image 0 to 1
        '''

        # Forward through network
        latent0 = self.encoder(image0)
        latent1 = self.encoder(image1)

        latent = [
            torch.cat([feature0, feature1], dim=1)
            for feature0, feature1 in zip(latent0, latent1)
        ]

        outputs = self.decoder_flow(latent)

        if return_all_output_resolutions:
            return [0.20 * 2.0 * (outputs[('disp', i)] - 0.50) for i in self.decoder_flow.scales]
        else:
            return 0.20 * 2.0 * (outputs[("disp", 0)] - 0.50)

    def compute_loss(self,
                     flows0to1,
                     image0=None,
                     image1=None,
                     w_color=0.15,
                     w_structure=0.85,
                     w_smoothness=0.10):
        '''
        Computes loss function

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W left image
            image1 : torch.Tensor[float32]
                N x 3 x H x W t-1 image
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

        # Define losses
        loss_color = 0.0
        loss_structure = 0.0
        loss_smoothness = 0.0

        # Interpolate all resolutions to shape
        flows0to1 = [
            torch.nn.functional.interpolate(
                flow0to1, size=shape[-2:], mode='bilinear', align_corners=True)
            for flow0to1 in flows0to1
        ]

        n_batch, _, n_height, n_width = shape
        scale = torch.ones_like(flows0to1[0])
        scale[:, 0, :, :] *= float(n_width)
        scale[:, 1, :, :] *= float(n_height)

        xy0 = loss_utils.meshgrid(
            n_batch,
            n_height,
            n_width,
            homogeneous=False,
            device=self.device)

        for resolution, flow0to1 in enumerate(flows0to1):

            resolution = float(resolution)
            w_resolution = 1.0 / (2 ** resolution)

            # Perform warping from image 1 to 0 and compute loss
            xy0to1 = flow0to1 * scale + xy0

            image1to0 = loss_utils.grid_sample(image1, xy0to1, shape)

            loss_color1to0 = losses.color_consistency_loss(image0, image1to0)
            loss_structure1to0 = losses.structural_consistency_loss_func(image0, image1to0)

            loss_color = loss_color + w_resolution * loss_color1to0
            loss_structure = loss_structure + w_resolution * loss_structure1to0

            # Compute smoothness loss function
            loss_smoothness = loss_smoothness + \
                w_resolution * losses.smoothness_loss_func(flow0to1[:, 0:1, :, :], image0) + \
                w_resolution * losses.smoothness_loss_func(flow0to1[:, 1:2, :, :], image0)

        loss = \
            w_color * loss_color + \
            w_structure * loss_structure + \
            w_smoothness * loss_smoothness

        # Save loss info to be logged
        loss_info = {
            'loss_color': loss_color,
            'loss_structure': loss_structure,
            'loss_smoothness': loss_smoothness,
            'loss' : loss,
            'image1to0': image1to0,
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
            list(self.decoder_flow.parameters())

        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder.train()
        self.decoder_flow.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder.eval()
        self.decoder_flow.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        # Move to device
        self.encoder.to(device)
        self.decoder_flow.to(device)

    def save_model(self,
                   encoder_checkpoint_path,
                   decoder_flow_checkpoint_path,
                   step,
                   optimizer,
                   epoch):
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

        checkpoint_encoder = {}
        checkpoint_encoder['train_step'] = step
        checkpoint_encoder['epoch'] = epoch
        checkpoint_encoder['optimizer_state_dict'] = optimizer.state_dict()

        checkpoint_encoder['model_state_dict'] = self.encoder.state_dict()

        torch.save(checkpoint_encoder, encoder_checkpoint_path)

        checkpoint_decoder_flow = {}
        checkpoint_decoder_flow['model_state_dict'] = self.decoder_flow.state_dict()

        torch.save(checkpoint_decoder_flow, decoder_flow_checkpoint_path)

    def restore_model(self,
                      encoder_restore_path=None,
                      decoder_flow_restore_path=None):
        '''
        Restores model from checkpoint path

        Args:
            encoder_restore_path : str
                checkpoint for encoder
            decoder_flow_restore_path : str
                checkpoint for decoder for flow
        '''

        if encoder_restore_path is not None:
            # Restore weights for encoder
            loaded_dict_encoder = torch.load(encoder_restore_path, map_location=self.device)

            if 'model_state_dict' in loaded_dict_encoder.keys():
                loaded_dict_encoder = loaded_dict_encoder['model_state_dict']

            filtered_dict_encoder = {
                k : v for k, v in loaded_dict_encoder.items() if k in self.encoder.state_dict()
            }

            self.encoder.load_state_dict(filtered_dict_encoder)

        if decoder_flow_restore_path is not None:
            # Restore weights for decoder for flow
            loaded_dict_decoder = torch.load(decoder_flow_restore_path, map_location=self.device)

            if 'model_state_dict' in loaded_dict_decoder.keys():
                loaded_dict_decoder = loaded_dict_decoder['model_state_dict']

            self.decoder_flow.load_state_dict(loaded_dict_decoder)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder_flow = torch.nn.DataParallel(self.decoder_flow)

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image0=None,
                    image1=None,
                    image1to0=None,
                    flow0to1=None,
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
            flow0to1 : torch.Tensor[float32]
                flow from time step t to t-1
            scalars : dict[str, float]
                dictionary of scalars to log
            n_display : int
                number of images to display
        '''

        with torch.no_grad():

            display_summary_image = []
            display_summary_flow0to1 = []

            display_summary_image_text = tag
            display_summary_flow0to1_text = tag

            if image0 is not None and image1 is not None:
                image0_summary = image0[0:n_display, ...]
                image1_summary = image1[0:n_display, ...]

                display_summary_image_text += '_image01'
                display_summary_flow0to1_text += '_image01'

                # Add to list of images to log
                display_summary_image.append(
                    torch.cat([
                        image0_summary.cpu(),
                        image1_summary.cpu()],
                        dim=-1))

                display_summary_flow0to1.append(display_summary_image[-1])

            image_list = []
            image_text_list = []

            if image1to0 is not None:
                image_list.append(image1to0)
                image_text_list.append('_image1to0_output-error')

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

            if flow0to1 is not None:
                flow0to1_summary = flow0to1[0:n_display, ...]

                display_summary_flow0to1_text += '_output0'

                # Add to list of images to log
                n_batch, _, n_height, n_width = flow0to1_summary.shape

                scale = torch.ones_like(flow0to1_summary)
                scale[:, 0, :, :] *= float(n_width)
                scale[:, 1, :, :] *= float(n_height)

                flow0to1_summary = flow0to1_summary.permute(0, 2, 3, 1).float()

                display_summary_flow0to1.append(
                    torch.cat([
                        log_utils.flow2color(flow0to1_summary.cpu()).permute(0, 3, 1, 2).float(),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu')).float()],
                        dim=3))

                # Log distribution of flow 0to1
                summary_writer.add_histogram(tag + '_flow0to1x_distro', flow0to1[:, 0, :, :], global_step=step)
                summary_writer.add_histogram(tag + '_flow0to1y_distro', flow0to1[:, 1, :, :], global_step=step)

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

            if len(display_summary_flow0to1) > 1:
                display_summary_flow0to1 = torch.cat(display_summary_flow0to1, dim=2)

                summary_writer.add_image(
                    display_summary_flow0to1_text,
                    torchvision.utils.make_grid(display_summary_flow0to1, nrow=n_display),
                    global_step=step)
