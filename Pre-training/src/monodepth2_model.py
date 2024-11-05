'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Safa Cicek <safacicek@ucla.edu>
If this code is useful to you, please consider citing the following paper:
A. Wong, S. Cicek, and S. Soatto. Targeted Adversarial Perturbations for Monocular Depth Prediction.
https://arxiv.org/pdf/2006.08602.pdf
@inproceedings{wong2020targeted,
    title={Targeted Adversarial Perturbations for Monocular Depth Prediction},
    author={Wong, Alex and Safa Cicek and Soatto, Stefano},
    booktitle={Advances in neural information processing systems},
    year={2020}
}
'''
import os, sys
import torch, torchvision
import log_utils, net_utils
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'monodepth2'))
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'monodepth2', 'networks'))
from resnet_encoder import ResnetEncoder
from depth_decoder import DepthDecoder
from pose_decoder import PoseDecoder
from layers import BackprojectDepth, Project3D, SSIM, get_smooth_loss


STEREO_SCALE_FACTOR             = 5.4
MIN_PREDICT_DEPTH               = 0.1
MAX_PREDICT_DEPTH               = 100.0


class Monodepth2Model(object):
    '''
    Wrapper class for Monodepth2

    Arg(s):
        device : torch.device
            device to run on
    '''
    def __init__(self,
                 encoder_type=['resnet18'],
                 network_modules=['depth', 'pose'],
                 scale_factor_depth=STEREO_SCALE_FACTOR,
                 min_predict_depth=MIN_PREDICT_DEPTH,
                 max_predict_depth=MAX_PREDICT_DEPTH,
                 device=torch.device('cuda')):

        self.n_resolution = 4

        # Depth range settings
        self.scale_factor_depth = scale_factor_depth
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth

        self.device = device

        if 'resnet18' in encoder_type:
            n_layers = 18
        elif 'resnet50' in encoder_type:
            n_layers = 50
        else:
            raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

        self.encoder = ResnetEncoder(
            num_layers=n_layers,
            pretrained='pretrained' in encoder_type)

        self.decoder_depth = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            scales=range(self.n_resolution))

        if 'pose' in network_modules:
            self.decoder_pose = PoseDecoder(
                num_ch_enc=self.encoder.num_ch_enc,
                num_input_features=2,
                num_frames_to_predict_for=2)
        else:
            self.decoder_pose = None

        if 'mask' in network_modules:
            self.decoder_mask = DepthDecoder(
                num_ch_enc=self.encoder.num_ch_enc,
                scales=range(self.n_resolution),
                num_output_channels=1)
        else:
            self.decoder_mask = None

        if 'mask' in network_modules or 'auto_mask' in network_modules:
            self.do_masking = True
        else:
            self.do_masking = False

        # Move to device
        self.to(self.device)

    def forward_depth(self, image, return_all_output_resolutions=False):
        '''
        Forwards an image through the network to output depth

        Args(s):
            image : torch.Tensor[float32]
                N x C x H x W tensor
            return_all_restolutions : bool
                if set then return outputs from all resolutions
        Returns:
            torch.Tensor[float32] : N x C x H x W depth
        '''

        # Forward through network
        latent = self.encoder(image)
        outputs = self.decoder_depth(latent)

        # Constants for converting disparity to depth
        min_disparity = 1.0 / self.max_predict_depth
        max_disparity = 1.0 / self.min_predict_depth

        if return_all_output_resolutions:
            output_disparities = [
                outputs[("disp", i)] for i in range(self.n_resolution)
            ]

            output_disparities = [
                min_disparity + (max_disparity - min_disparity) * disparity
                for disparity in output_disparities
            ]

            output_depths = [
                self.scale_factor_depth / disparity
                for disparity in output_disparities
            ]

            return output_depths

        else:
            output_disparity = outputs[("disp", 0)]

            output_disparity = \
                min_disparity + (max_disparity - min_disparity) * output_disparity

            output_depth = self.scale_factor_depth / output_disparity

            return output_depth

    def forward_pose(self, image0, image1):
        '''
        Forwards a pair of images through the network to output pose from time 0 to 1

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W tensor
            image1 : torch.Tensor[float32]
                N x C x H x W tensor
        Returns:
            torch.Tensor[float32] : N x C x H x W depth
        '''

        assert self.decoder_pose is not None

        inputs = [self.encoder(image0), self.encoder(image1)]
        rotation, translation = self.decoder_pose(inputs)

        rotation = torch.squeeze(rotation[:, 0, ...])
        translation = torch.squeeze(translation[:, 0, ...])

        pose_matrix = net_utils.pose_matrix(
            torch.cat([rotation, translation], dim=-1),
            rotation_parameterization='axis')

        if len(pose_matrix.shape) == 4:
            pose_matrix = pose_matrix[:, 0:1, :, :]

        return pose_matrix

    def forward_mask(self, image0, image1):
        '''
        Forwards a pair of images through the network to output mask
        from image 0 to image 1 and image1 to image0

        Arg(s):
            image0 : tensor
                N x C x H x W tensor
            image1 : tensor
                N x C x H x W tensor
        Returns:
            list[torch.Tensor] : list of N x 2 x H x W mask
        '''

        assert self.decoder_mask is not None

        inputs = [self.encoder(image0), self.encoder(image1)]
        outputs = self.decoder_mask(inputs)

        output_masks = [
            outputs[("disp", i)] for i in range(self.n_resolution)
        ]

        return output_masks

    def compute_unsupervised_loss(self,
                                  output_depths0,
                                  image0=None,
                                  image1=None,
                                  image2=None,
                                  masks0to1=None,
                                  masks0to2=None,
                                  pose0to1=None,
                                  pose0to2=None,
                                  intrinsics=None,
                                  w_photometric=1.00,
                                  w_smoothness=0.10):
        '''
        Computes loss

        Arg(s):
            output_depths0 : list[torch.Tensor[float32]]
                list of N x 1 x H x W depth
            image0 : torch.Tensor[float32]
                N x 3 x H x W image at t
            image1 : torch.Tensor[float32]
                N x 3 x H x W image at t-1
            image2 : torch.Tensor[float32]
                N x 3 x H x W image at t+1
            masks0to1 : list[torch.Tensor[float32]]
                list of N x 1 x H x W mask
            masks0to2 : list[torch.Tensor[float32]]
                list of N x 1 x H x W mask
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from image 0 to image 1
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from image 0 to image 1
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
            w_photometric : float
                weight of photometric consistency term
            w_smoothness : float
                weight of smoothness term
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        Returns:
            float : loss
        '''

        loss = 0.0

        n_batch, _, n_height, n_width = image0.shape
        shape = (n_height, n_width)

        if masks0to1 is None:
            masks0to1 = [None] * len(output_depths0)

        if masks0to2 is None:
            masks0to2 = [None] * len(output_depths0)

        backproject_depth_func = BackprojectDepth(n_batch, n_height, n_width)
        project_3d_func = Project3D(n_batch, n_height, n_width)
        ssim_func = SSIM()

        backproject_depth_func.to(self.device)
        project_3d_func.to(self.device)
        ssim_func.to(self.device)

        column = torch.zeros([n_batch, 3, 1], device=self.device)
        row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device) \
            .view(1, 1, 4) \
            .repeat(n_batch, 1, 1)
        intrinsics = torch.cat([intrinsics, column], dim=2)
        intrinsics = torch.cat([intrinsics, row], dim=1)

        for resolution, (output_depth, mask0to1, mask0to2) in enumerate(zip(output_depths0, masks0to1, masks0to2)):

            # Interpolate all resolutions to shape
            output_depth = torch.nn.functional.interpolate(
                output_depth, size=shape[-2:], mode='bilinear', align_corners=True)

            # Weight loss by resolution
            resolution = float(resolution)
            w_resolution = 1.0 / (2 ** resolution)

            points = backproject_depth_func(output_depth, torch.inverse(intrinsics))

            target_xy0to1 = project_3d_func(points, intrinsics, pose0to1)
            target_xy0to2 = project_3d_func(points, intrinsics, pose0to2)

            image1to0 = torch.nn.functional.grid_sample(
                image1,
                target_xy0to1,
                padding_mode='border',
                align_corners=True)
            image2to0 = torch.nn.functional.grid_sample(
                image2,
                target_xy0to2,
                padding_mode='border',
                align_corners=True)

            if resolution == 0:
                image1to0_full_resolution = image1to0
                image2to0_full_resolution = image2to0

            # Photometric reconstruction losses
            loss_photometric_color1to0 = torch.mean(torch.abs(image1to0 - image0), dim=1, keepdim=True)
            loss_photometric_color2to0 = torch.mean(torch.abs(image2to0 - image0), dim=1, keepdim=True)

            loss_photometric_ssim1to0 = torch.mean(ssim_func(image1to0, image0), dim=1, keepdim=True)
            loss_photometric_ssim2to0 = torch.mean(ssim_func(image2to0, image0), dim=1, keepdim=True)

            loss_photometric1to0 = 0.85 * loss_photometric_ssim1to0 + 0.15 * loss_photometric_color1to0
            loss_photometric2to0 = 0.85 * loss_photometric_ssim2to0 + 0.15 * loss_photometric_color2to0

            losses_photometric = torch.cat([loss_photometric1to0, loss_photometric2to0], dim=1)

            if self.do_masking:

                if mask0to1 is None and mask0to2 is None:
                    # Identity automasking
                    loss_identity_color1to0 = torch.mean(torch.abs(image1 - image0), dim=1, keepdim=True)
                    loss_identity_color2to0 = torch.mean(torch.abs(image2 - image0), dim=1, keepdim=True)

                    loss_identity_ssim1to0 = torch.mean(ssim_func(image1, image0), dim=1, keepdim=True)
                    loss_identity_ssim2to0 = torch.mean(ssim_func(image2, image0), dim=1, keepdim=True)

                    loss_identity1to0 = 0.85 * loss_identity_ssim1to0 + 0.15 * loss_identity_color1to0
                    loss_identity2to0 = 0.85 * loss_identity_ssim2to0 + 0.15 * loss_identity_color2to0

                    losses_identity = torch.cat([loss_identity1to0, loss_identity2to0], dim=1)
                    losses_identity += torch.randn_like(losses_identity) * 0.00001

                    loss_photometric = torch.cat((losses_identity, losses_photometric), dim=1)
                    loss_photometric, _ = torch.min(loss_photometric, dim=1)
                else:
                    # Add a loss pushing mask to 1 (using nn.BCELoss for stability)
                    loss_mask = \
                        torch.nn.BCELoss()(mask0to1, torch.ones_like(mask0to1)) + \
                        torch.nn.BCELoss()(mask0to1, torch.ones_like(mask0to1))
                    loss = loss + 0.2 * loss_mask

                    # Interpolate all resolutions to shape
                    mask0to1 = torch.nn.functional.interpolate(
                        mask0to1, size=shape[-2:], mode='bilinear', align_corners=True)

                    mask0to2 = torch.nn.functional.interpolate(
                        mask0to2, size=shape[-2:], mode='bilinear', align_corners=True)

                    loss_photometric = losses_photometric * torch.cat([mask0to1, mask0to1], dim=1)
            else:
                loss_photometric = losses_photometric

            loss_photometric = w_photometric * torch.mean(loss_photometric)

            # Smoothness loss
            mean_output_depth = torch.mean(output_depth, dim=[2, 3], keepdim=True)
            normalized_output_depth = output_depth / (mean_output_depth + 1e-7)

            loss_smoothness = \
                w_resolution * w_smoothness * get_smooth_loss(normalized_output_depth, image0)

            loss = loss + loss_photometric + loss_smoothness

        loss = loss / float(self.n_resolution)

        # Save loss info to be logged
        loss_info = {
            'loss' : loss,
            'image1to0' : image1to0_full_resolution,
            'image2to0' : image2to0_full_resolution
        }

        return loss, loss_info

    def compute_supervised_loss(self, output_depths, ground_truth):
        '''
        Computes loss

        Arg(s):
            output_depths : list[tensor]
                list of N x 1 x H x W tensor
            ground_truth : tensor
                N x 1 x H x W tensor
        Returns:
            float : loss
        '''

        shape = ground_truth.shape[-2:]

        ground_truth = torch.where(
            ground_truth > self.max_predict_depth,
            torch.full_like(ground_truth, self.max_predict_depth),
            ground_truth)

        validity_map = torch.where(
            ground_truth > 0,
            torch.ones_like(ground_truth),
            torch.zeros_like(ground_truth))

        loss = 0.0

        for resolution, output_depth in enumerate(output_depths):

            # Interpolate all resolutions to shape
            output_depth = torch.nn.functional.interpolate(
                output_depth, size=shape[-2:], mode='bilinear', align_corners=True)

            # Weight loss by resolution
            resolution = float(resolution)
            w_resolution = 1.0 / (2 ** resolution)

            loss_l1 = torch.mean(validity_map * torch.abs(output_depth - ground_truth))

            loss = loss + w_resolution * loss_l1

        # Save loss info to be logged
        loss_info = {
            'loss' : loss,
        }

        return loss, loss_info

    def to(self, device):
        '''
        Moves model to device

        Args:
            device : torch.device
        '''

        self.encoder.to(device)
        self.decoder_depth.to(device)

        if self.decoder_pose is not None:
            self.decoder_pose.to(device)

        if self.decoder_mask is not None:
            self.decoder_mask.to(device)

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder.train()
        self.decoder_depth.train()

        if self.decoder_pose is not None:
            self.decoder_pose.train()

        if self.decoder_mask is not None:
            self.decoder_mask.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder.eval()
        self.decoder_depth.eval()

        if self.decoder_pose is not None:
            self.decoder_pose.eval()

        if self.decoder_mask is not None:
            self.decoder_mask.eval()

    def parameters(self):
        '''
        Fetches model parameters

        Returns:
            list : list of model parameters
        '''

        parameters = \
            list(self.encoder.parameters()) + \
            list(self.decoder_depth.parameters())

        if self.decoder_pose is not None:
            parameters = parameters + list(self.decoder_pose.parameters())

        if self.decoder_mask is not None:
            parameters = parameters + list(self.decoder_mask.parameters())

        return parameters

    def restore_model(self,
                      encoder_restore_path=None,
                      decoder_depth_restore_path=None,
                      decoder_pose_restore_path=None,
                      decoder_mask_restore_path=None):
        '''
        Restores model from checkpoint path

        Args:
            encoder_restore_path : str
                checkpoint for encoder
            decoder_depth_restore_path : str
                checkpoint for decoder for depth
            decoder_pose_restore_path : str
                checkpoint for decoder for pose
            decoder_mask_restore_path : str
                checkpoint for decoder for mask
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

        if decoder_depth_restore_path is not None:
            # Restore weights for decoder for depth
            loaded_dict_decoder = torch.load(decoder_depth_restore_path, map_location=self.device)

            if 'model_state_dict' in loaded_dict_decoder.keys():
                loaded_dict_decoder = loaded_dict_decoder['model_state_dict']

            self.decoder_depth.load_state_dict(loaded_dict_decoder)

        if decoder_pose_restore_path is not None and self.decoder_pose is not None:
            # Restore weights for decoder for pose
            loaded_dict_decoder = torch.load(decoder_pose_restore_path, map_location=self.device)

            if 'model_state_dict' in loaded_dict_decoder.keys():
                loaded_dict_decoder = loaded_dict_decoder['model_state_dict']

            self.decoder_pose.load_state_dict(loaded_dict_decoder)

        if decoder_mask_restore_path is not None and self.decoder_mask is not None:
            # Restore weights for decoder for mask
            loaded_dict_decoder = torch.load(decoder_mask_restore_path, map_location=self.device)

            if 'model_state_dict' in loaded_dict_decoder.keys():
                loaded_dict_decoder = loaded_dict_decoder['model_state_dict']

            self.decoder_mask.load_state_dict(loaded_dict_decoder)

    def save_model(self,
                   encoder_checkpoint_path=None,
                   decoder_depth_checkpoint_path=None,
                   decoder_pose_checkpoint_path=None,
                   decoder_mask_checkpoint_path=None,
                   step=-1,
                   optimizer=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            encoder_checkpoint_path : str
                path to save checkpoint
            decoder_depth_checkpoint_path : str
                path to save checkpoint
            decoder_pose_checkpoint_path : str
                path to save checkpoint
            decoder_mask_checkpoint_path : str
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
        checkpoint_encoder['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint_encoder['model_state_dict'] = self.encoder.state_dict()

        torch.save(checkpoint_encoder, encoder_checkpoint_path)

        checkpoint_decoder_depth = {}
        checkpoint_decoder_depth['train_step'] = step
        checkpoint_decoder_depth['model_state_dict'] = self.decoder_depth.state_dict()

        torch.save(checkpoint_decoder_depth, decoder_depth_checkpoint_path)

        if self.decoder_pose is not None and decoder_pose_checkpoint_path is not None:
            checkpoint_decoder_pose = {}
            checkpoint_decoder_pose['train_step'] = step
            checkpoint_decoder_pose['model_state_dict'] = self.decoder_pose.state_dict()

            torch.save(checkpoint_decoder_pose, decoder_pose_checkpoint_path)

        if self.decoder_mask is not None and decoder_mask_checkpoint_path is not None:
            checkpoint_decoder_mask = {}
            checkpoint_decoder_mask['train_step'] = step
            checkpoint_decoder_mask['model_state_dict'] = self.decoder_mask.state_dict()

            torch.save(checkpoint_decoder_mask, decoder_mask_checkpoint_path)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder_depth = torch.nn.DataParallel(self.decoder_depth)

        if self.decoder_pose is not None:
            self.decoder_pose = torch.nn.DataParallel(self.decoder_pose)

        if self.decoder_mask is not None:
            self.decoder_mask = torch.nn.DataParallel(self.decoder_mask)

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image0=None,
                    image1to0=None,
                    image2to0=None,
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
            ground_truth0 : torch.Tensor[float32]
                ground truth depth for left camera
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

                min_predict_depth, _ = torch.min(output_depth0_summary.view(n_batch, -1, 1, 1), dim=1, keepdim=True)
                max_predict_depth, _ = torch.max(output_depth0_summary.view(n_batch, -1, 1, 1), dim=1, keepdim=True)

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            ((output_depth0_summary - min_predict_depth) / (max_predict_depth - min_predict_depth)).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth0_distro', output_depth0, global_step=step)

            if output_depth0 is not None and ground_truth0 is not None:
                ground_truth0 = torch.unsqueeze(ground_truth0[:, 0, :, :], dim=1)

                ground_truth0_summary = ground_truth0[0:n_display]

                display_summary_depth_text += '_groundtruth0-error'

                ground_truth0_summary = torch.nn.functional.interpolate(
                    ground_truth0_summary, size=(n_height, n_width), mode='nearest')

                validity_map0_summary = torch.where(
                    ground_truth0_summary > 0,
                    torch.ones_like(ground_truth0_summary),
                    ground_truth0_summary)

                # Compute output error w.r.t. ground truth
                ground_truth0_error_summary = \
                    torch.abs(output_depth0_summary - ground_truth0_summary)

                ground_truth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (ground_truth0_error_summary + 1e-8) / (ground_truth0_summary + 1e-8),
                    validity_map0_summary)

                # Add to list of images to log
                ground_truth0_summary = log_utils.colorize(
                    (ground_truth0_summary / max_display_depth).cpu(),
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
