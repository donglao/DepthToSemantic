import os, sys
import torch, torchvision
import log_utils, loss_utils, losses
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'deeplab'))
import deeplab_networks as deeplab

MIN_PREDICT_DEPTH               = 0.1
MAX_PREDICT_DEPTH               = 100.0


class DeepLabModel(object):
    '''
    Wrapper class for DeepLab

    Arg(s):
        device : torch.device
            device to run on
    '''
    def __init__(self,
                 encoder_type=['resnet18'],
                 min_predict_depth=MIN_PREDICT_DEPTH,
                 max_predict_depth=MAX_PREDICT_DEPTH,
                 device=torch.device('cuda')):

        # Depth range settings
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth

        self.device = device

        if 'resnet18' in encoder_type:
            resnet = deeplab.resnet18
        elif 'resnet50' in encoder_type:
            resnet = deeplab.resnet50
        elif 'resnet101' in encoder_type:
            resnet = deeplab.resnet101
        elif 'resnet152' in encoder_type:
            resnet = deeplab.resnet152
        else:
            raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

        self.network = resnet(
            pretrained='pretrained' in encoder_type,
            num_classes=1,
            num_groups=None,
            weight_std=False,
            beta=False)

        # Move to device
        self.to(self.device)

    def forward(self, image, return_all_output_resolutions=False):
        '''
        Forwards an image through the network to output depth

        Arg(s):
            image : tensor
                N x C x H x W tensor
            return_all_restolutions : bool
                if set then return outputs from all resolutions
        Returns:
            tensor : N x C x H x W depth
        '''

        # Forward through network
        output = self.network(image)

        # Converting to depth
        output_depth = torch.sigmoid(output)

        output_depth = \
            self.min_predict_depth / (output_depth + self.min_predict_depth / self.max_predict_depth)

        if return_all_output_resolutions:
            return [output_depth]
        else:
            return output_depth

    def compute_unsupervised_loss(self,
                                  output_depth0,
                                  image0,
                                  image1,
                                  image2,
                                  pose0to1,
                                  pose0to2,
                                  intrinsics,
                                  w_color=0.15,
                                  w_structure=0.95,
                                  w_smoothness=0.05):
        '''
        Computes loss function

        Arg(s):
            output_depth0 : torch.Tensor[float32]
                N x 1 x H x W output depth at time t
            image0 : torch.Tensor[float32]
                N x 3 x H x W image at time t
            image1 : torch.Tensor[float32]
                N x 3 x H x W image at time t-1
            image2 : torch.Tensor[float32]
                N x 3 x H x W image at time t+1
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from image at t to t-1
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 transformation matrix from image at t to t+1
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
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
        points0 = loss_utils.backproject_to_camera(output_depth0, intrinsics, shape)

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

        # Save loss info to be logged
        loss_color = loss_color1to0 + loss_color2to0
        loss_structure = loss_structure1to0 + loss_structure2to0

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
            'image2to0': image2to0
        }

        return loss, loss_info

    def compute_supervised_loss(self, output_depth, ground_truth):
        '''
        Computes loss function

        Arg(s):
            output_depths : torch.Tensor[float32]
                N x 1 x H x W output depth at time t
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground truth depth at time t
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        shape = ground_truth.shape[-2:]

        # Clip the ground truth
        ground_truth = torch.where(
            ground_truth > 2 * self.max_predict_depth,
            torch.zeros_like(ground_truth),
            ground_truth)

        ground_truth = torch.where(
            ground_truth > self.max_predict_depth,
            self.max_predict_depth * torch.ones_like(ground_truth),
            ground_truth)

        validity_map = torch.where(
            ground_truth > 0,
            torch.ones_like(ground_truth),
            torch.zeros_like(ground_truth))

        loss = 0.0

        # Interpolate all resolutions to shape
        output_depth = torch.nn.functional.interpolate(
            output_depth, size=shape[-2:], mode='bilinear', align_corners=True)

        loss_l1 = torch.sum(
            validity_map * torch.abs(output_depth - ground_truth),
            dim=[1, 2, 3])
        loss_l1 = torch.mean(loss_l1 / torch.sum(validity_map, dim=[1, 2, 3]))

        loss = loss_l1

        # Save loss info to be logged
        loss_info = {
            'loss' : loss
        }

        return loss, loss_info

    def to(self, device):
        '''
        Moves model to device

        Args:
            device : torch.device
        '''

        self.network.to(device)

    def train(self):
        '''
        Sets model to training mode
        '''

        self.network.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.network.eval()

    def parameters(self):
        '''
        Fetches model parameters

        Returns:
            list : list of model parameters
        '''

        parameters = list(self.network.parameters())

        return parameters

    def restore_model(self, restore_path=None):
        '''
        Restores model from checkpoint path

        Arg(s):
            restore_path : str
                checkpoint for network
        '''

        if restore_path is not None:
            # Restore weights for network
            loaded_dict_model = torch.load(restore_path, map_location=self.device)

            if 'model_state_dict' in loaded_dict_model.keys():
                loaded_dict_model = loaded_dict_model['model_state_dict']

            filtered_dict_model = {
                k : v for k, v in loaded_dict_model.items() if k in self.network.state_dict()
            }

            self.network.load_state_dict(filtered_dict_model)

    def save_model(self, checkpoint_path=None, step=-1, optimizer=None):
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
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if isinstance(self.network, torch.nn.DataParallel):
            checkpoint['model_state_dict'] = self.network.module.state_dict()
        else:
            checkpoint['model_state_dict'] = self.network.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.network = torch.nn.DataParallel(self.network)

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
                image from time step t
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
                            (output_depth0_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth0_distro', output_depth0, global_step=step)

            if output_depth0 is not None and ground_truth0 is not None:

                ground_truth0_summary = ground_truth0[0:n_display]
                validity_map0_summary = torch.where(
                    ground_truth0_summary > 0,
                    torch.ones_like(ground_truth0_summary),
                    ground_truth0_summary)

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
