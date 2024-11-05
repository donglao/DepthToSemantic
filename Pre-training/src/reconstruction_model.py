import os, sys, torch, torchvision
import log_utils
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'monodepth2'))
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'monodepth2', 'networks'))
from resnet_encoder import ResnetEncoder
from depth_decoder import DepthDecoder


class ReconstructionModel(object):
    '''
    Reconstruction model
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

        self.decoder_reconstruction = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            scales=range(4),
            num_output_channels=3)

        # Move to device
        self.to(self.device)

    def forward(self, image, return_all_output_resolutions=False):
        '''
        Forwards an image through the network to output depth

        Args:
            image : tensor
                N x C x H x W tensor
            return_all_restolutions : bool
                if set then return outputs from all resolutions
        Returns:
            tensor : N x C x H x W depth
        '''

        # Forward through network
        latent = self.encoder(image)
        outputs = self.decoder_reconstruction(latent)

        if return_all_output_resolutions:
            outputs = [
                outputs[("disp", i)] for i in range(4)
            ]

            return outputs

        else:
            output = outputs[("disp", 0)]

            return output

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = \
            list(self.encoder.parameters()) + \
            list(self.decoder_reconstruction.parameters())

        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder.train()
        self.decoder_reconstruction.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder.eval()
        self.decoder_reconstruction.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Arg(s):
            device : torch.device
                device for running model
        '''

        # Move to device
        self.encoder.to(device)
        self.decoder_reconstruction.to(device)

    def save_model(self,
                   encoder_checkpoint_path,
                   decoder_reconstruction_checkpoint_path,
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

        checkpoint_decoder_reconstruction = {}
        checkpoint_decoder_reconstruction['model_state_dict'] = self.decoder_reconstruction.state_dict()

        torch.save(checkpoint_decoder_reconstruction, decoder_reconstruction_checkpoint_path)

    def restore_model(self,
                      encoder_restore_path=None,
                      decoder_reconstruction_restore_path=None):
        '''
        Restores model from checkpoint path

        Args:
            encoder_restore_path : str
                checkpoint for encoder
            decoder_reconstruction_restore_path : str
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

        if decoder_reconstruction_restore_path is not None:
            # Restore weights for decoder for flow
            loaded_dict_decoder = torch.load(decoder_reconstruction_restore_path, map_location=self.device)

            if 'model_state_dict' in loaded_dict_decoder.keys():
                loaded_dict_decoder = loaded_dict_decoder['model_state_dict']

            self.decoder_reconstruction.load_state_dict(loaded_dict_decoder)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder_reconstruction = torch.nn.DataParallel(self.decoder_reconstruction)

    def compute_loss(self,
                     output_images,
                     target_images,
                     w_reconstruction=1.00):
        '''
        Computes the loss function

        Arg(s):
            output_images : torch.Tensor[float32]
                output images
            target_images : torch.Tensor[float32]
                target images
            w_reconstruction : float
                weight of reconstruction loss
        Returns:
            torch.Tensor[float32] : loss (scalar)
        '''

        target_height, target_width = target_images.shape[-2:]

        loss_reconstruction = 0.0

        for s in range(4):
            # Compute loss at each resolution
            w_resolution = 1.0 / (4 ** (4 - s - 1))

            # Resize to ground truth height and width
            output_images[s] = torch.nn.functional.interpolate(
                output_images[s],
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=True)

            if w_reconstruction > 0.0:
                loss_reconstruction = loss_reconstruction + w_resolution * \
                    torch.nn.functional.l1_loss(
                        input=output_images[s],
                        target=target_images,
                        reduction='mean')

        loss = w_reconstruction * loss_reconstruction

        loss_info = {
            'loss' : loss
        }

        return loss, loss_info

    def log_summary(self,
                    summary_writer,
                    images=None,
                    output_images=None,
                    ground_truths=None,
                    scalars=None,
                    tag='train',
                    step=0,
                    n_display=4):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            images : torch.Tensor[float32]
                N x C x H x W images
            output_images : torch.Tensor[float32]
                N x C x H x W images
            scalars : dict[str, float]
                dictionary of scalars as name value pairs
            step : int
                step in training for model
            n_display : int
                number of images to log
        '''

        with torch.no_grad():

            display_summary = []
            display_summary_text = tag

            if images is not None:
                # Setup image summary
                images = images[0:n_display, ...]
                images = images.cpu()

                if ground_truths is not None:
                    # If resize to ground truth shape
                    height, width = ground_truths.shape[-2:]

                    images = torch.nn.functional.interpolate(
                        images,
                        size=(height, width),
                        mode='bilinear',
                        align_corners=True)

                if images.shape[1] == 1:
                    images_summary = log_utils.colorize(
                        images,
                        colormap='viridis')
                else:
                    images_summary = images

                display_summary.append(images_summary)
                display_summary_text = display_summary_text + '_images'

            if output_images is not None:
                # Setup output image summary
                output_images = output_images[0:n_display, ...]

                if ground_truths is not None:
                    # If resize to ground truth shape
                    height, width = ground_truths.shape[-2:]

                    output_images = torch.nn.functional.interpolate(
                        output_images,
                        size=(height, width),
                        mode='bilinear',
                        align_corners=True)

                output_images_summary = output_images.cpu()

                display_summary.append(output_images_summary)
                display_summary_text = display_summary_text + '_outputs'

            if ground_truths is not None:
                # Setup ground truth summary
                ground_truths = ground_truths[0:n_display, ...]

                ground_truths_summary = ground_truths.cpu()

                display_summary.append(ground_truths_summary)
                display_summary_text = display_summary_text + '_groundtruths'

                if output_images is not None:

                    output_error = \
                        torch.abs(ground_truths - output_images) / ground_truths

                    error_summary = log_utils.colorize(
                        torch.mean(output_error, dim=1, keepdim=True) / 0.10,
                        colormap='hot')

                    display_summary.append(error_summary)
                    display_summary_text = display_summary_text + '_error'

            if len(display_summary) > 0:
                display_summary = torch.cat(display_summary, dim=3)

            summary_writer.add_image(
                display_summary_text,
                torchvision.utils.make_grid(display_summary, nrow=1),
                global_step=step)

            # Log scalars
            if scalars is not None:
                for scalar_name, scalar_value in scalars.items():
                    summary_writer.add_scalar(
                        tag + '_' + scalar_name,
                        scalar_value,
                        global_step=step)
