import os, time
# import  cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils
# import eval_utils
from log_utils import log
from deeplab_model import DeepLabModel
from posenet_model import PoseNetModel
from transforms import Transforms


def train(train_images_left_path,
          train_images_right_path,
          train_intrinsics_left_path,
          train_intrinsics_right_path,
          train_focal_length_baseline_left_path,
          train_focal_length_baseline_right_path,
          train_ground_truth_path,
          val_image_path,
          val_focal_length_baseline_path,
          val_ground_truth_path,
          # Input settings
          n_batch,
          n_height,
          n_width,
          normalized_image_range,
          # Network settings
          encoder_type,
          min_predict_depth,
          max_predict_depth,
          # Training settings
          learning_rates,
          learning_schedule,
          # Augmentation settings
          augmentation_probabilities,
          augmentation_schedule,
          # Photometric data augmentations
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_gamma,
          augmentation_random_hue,
          augmentation_random_saturation,
          # Geometric data augmentations
          augmentation_random_swap_left_right,
          augmentation_random_crop_to_shape,
          augmentation_random_flip_type,
          augmentation_random_rotate_max,
          augmentation_random_crop_and_pad,
          augmentation_random_resize_and_crop,
          augmentation_random_resize_and_pad,
          # Loss settings
          supervision_type,
          w_color,
          w_structure,
          w_smoothness,
          w_weight_decay_depth,
          w_weight_decay_pose,
          # Evaluation settings
          min_evaluate_depth,
          max_evaluate_depth,
          # Checkpoint settings
          checkpoint_path,
          n_checkpoint,
          n_summary,
          n_summary_display,
          validation_start_step,
          depth_model_restore_path,
          pose_model_restore_path,
          # Hardware settings
          device='cuda',
          n_thread=8):

    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and event paths
    depth_model_checkpoint_path = os.path.join(checkpoint_path, 'depth_model-{}.pth')
    pose_model_checkpoint_path = os.path.join(checkpoint_path, 'pose_model-{}.pth')

    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    '''
    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty
    }
    '''

    '''
    Set up paths for training
    '''
    train_images_left_paths = data_utils.read_paths(train_images_left_path)
    train_images_right_paths = data_utils.read_paths(train_images_right_path)

    train_intrinsics_left_paths = data_utils.read_paths(train_intrinsics_left_path)
    train_intrinsics_right_paths = data_utils.read_paths(train_intrinsics_right_path)

    n_train_sample = len(train_images_left_paths)

    if train_focal_length_baseline_left_path is None:
        train_focal_length_baseline_left_paths = [None] * n_train_sample
    else:
        train_focal_length_baseline_left_paths = \
            data_utils.read_paths(train_focal_length_baseline_left_path)

    if train_focal_length_baseline_right_path is None:
        train_focal_length_baseline_right_paths = [None] * n_train_sample
    else:
        train_focal_length_baseline_right_paths = \
            data_utils.read_paths(train_focal_length_baseline_right_path)

    if train_ground_truth_path is None:
        train_ground_truth_paths = [None] * n_train_sample
    else:
        train_ground_truth_paths = \
            data_utils.read_paths(train_ground_truth_path)

    # Make sure number of paths match number of training sample
    input_paths = [
        train_images_left_paths,
        train_images_right_paths,
        train_intrinsics_left_paths,
        train_intrinsics_right_paths,
        train_focal_length_baseline_left_paths,
        train_focal_length_baseline_right_paths,
        train_ground_truth_paths
    ]

    for paths in input_paths:
        assert len(paths) == n_train_sample

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.floor(n_train_sample / n_batch).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.SingleImageDepthResizedTrainingDataset(
            images_left_paths=train_images_left_paths,
            images_right_paths=train_images_right_paths,
            intrinsics_left_paths=train_intrinsics_left_paths,
            intrinsics_right_paths=train_intrinsics_right_paths,
            focal_length_baseline_left_paths=train_focal_length_baseline_left_paths,
            focal_length_baseline_right_paths=train_focal_length_baseline_right_paths,
            ground_truth_paths=train_ground_truth_paths,
            resize_shape=(n_height, n_width),
            random_swap_left_right=augmentation_random_swap_left_right),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=True)

    train_transforms_photometric = Transforms(
        normalized_image_range=normalized_image_range,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_gamma=augmentation_random_gamma,
        random_hue=augmentation_random_hue,
        random_saturation=augmentation_random_saturation)

    train_transforms_geometric = Transforms(
        random_crop_to_shape=augmentation_random_crop_to_shape,
        random_flip_type=augmentation_random_flip_type,
        random_rotate_max=augmentation_random_rotate_max,
        random_crop_and_pad=augmentation_random_crop_and_pad,
        random_resize_and_crop=augmentation_random_resize_and_crop,
        random_resize_and_pad=augmentation_random_resize_and_pad)

    # Map interpolation mode names to enums
    interpolation_modes = \
        train_transforms_geometric.map_interpolation_mode_names_to_enums(['nearest'])

    '''
    Set up paths for validation
    '''
    '''
    val_image_paths = data_utils.read_paths(val_image_path)
    val_focal_length_baseline_paths = data_utils.read_paths(val_focal_length_baseline_path)
    val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

    assert len(val_image_paths) == len(val_focal_length_baseline_paths)
    assert len(val_image_paths) == len(val_ground_truth_paths)

    # Load ground truth depths
    val_ground_truths = []
    for path in val_ground_truth_paths:
        ext = os.path.splitext(path)[-1]
        if ext == '.png':
            ground_truth, validity_map = data_utils.load_depth_with_validity_map(
                path,
                data_format='CHW')
        elif ext == '.npy':
            ground_truth = np.expand_dims(np.load(path), axis=0)
            validity_map = np.where(ground_truth > 0, 1, 0)

        ground_truth = np.concatenate([ground_truth, validity_map], axis=0)
        val_ground_truths.append(np.expand_dims(ground_truth, axis=0))

    val_dataloader = torch.utils.data.DataLoader(
        datasets.SingleImageDepthResizedInferenceDataset(
            image_paths=val_image_paths,
            focal_length_baseline_paths=val_focal_length_baseline_paths,
            resize_shape=(n_height, n_width)),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    val_transforms = Transforms(
        normalized_image_range=normalized_image_range)
    '''

    '''
    Build network
    '''
    # Build network to output depth
    depth_model = DeepLabModel(
        encoder_type=encoder_type,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    if 'freeze_batchnorm' in encoder_type:
        depth_model.eval()
    else:
        depth_model.train()

    parameters_depth_model = depth_model.parameters()

    if torch.cuda.device_count() > 1:
        depth_model.data_parallel()

    # Bulid PoseNet (only needed for training) network
    if 'monocular' in supervision_type:
        pose_model = PoseNetModel(
            encoder_type='vggnet08',
            rotation_parameterization='axis',
            weight_initializer='kaiming_uniform',
            activation_func='relu',
            device=device)

        if 'freeze_batchnorm' in encoder_type:
            pose_model.eval()
        else:
            pose_model.train()

        parameters_pose_model = pose_model.parameters()
    else:
        pose_model = None
        parameters_pose_model = None

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')

    log('Training input paths:', log_path)
    train_input_paths = [
        train_images_left_path,
        train_images_right_path,
        train_intrinsics_left_path,
        train_intrinsics_right_path,
        train_focal_length_baseline_left_path,
        train_focal_length_baseline_right_path,
        train_ground_truth_path
    ]
    for path in train_input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_image_path,
        val_focal_length_baseline_path,
        val_ground_truth_path
    ]
    for path in val_input_paths:
        if path is not None:
            log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        # Batch settings
        n_batch=n_batch,
        n_height=n_height,
        n_width=n_width,
        # Input settings
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Depth network settings
        encoder_type=encoder_type,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        parameters_depth_model=parameters_depth_model,
        parameters_pose_model=parameters_pose_model)

    log_training_settings(
        log_path,
        # Training settings
        n_batch=n_batch,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        # Augmentation settings
        augmentation_probabilities=augmentation_probabilities,
        augmentation_schedule=augmentation_schedule,
        augmentation_random_swap_left_right=augmentation_random_swap_left_right,
        augmentation_random_brightness=augmentation_random_brightness,
        augmentation_random_contrast=augmentation_random_contrast,
        augmentation_random_gamma=augmentation_random_gamma,
        augmentation_random_hue=augmentation_random_hue,
        augmentation_random_saturation=augmentation_random_saturation,
        augmentation_random_crop_to_shape=augmentation_random_crop_to_shape,
        augmentation_random_flip_type=augmentation_random_flip_type,
        augmentation_random_rotate_max=augmentation_random_rotate_max,
        augmentation_random_crop_and_pad=augmentation_random_crop_and_pad,
        augmentation_random_resize_and_crop=augmentation_random_resize_and_crop,
        augmentation_random_resize_and_pad=augmentation_random_resize_and_pad)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        supervision_type=supervision_type,
        w_color=w_color,
        w_structure=w_structure,
        w_smoothness=w_smoothness,
        w_weight_decay_depth=w_weight_decay_depth,
        w_weight_decay_pose=w_weight_decay_pose)

    log_evaluation_settings(
        log_path,
        min_evaluate_depth=min_evaluate_depth,
        max_evaluate_depth=max_evaluate_depth)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        n_checkpoint=n_checkpoint,
        summary_event_path=event_path,
        n_summary=n_summary,
        n_summary_display=n_summary_display,
        validation_start_step=validation_start_step,
        depth_model_restore_path=depth_model_restore_path,
        pose_model_restore_path=pose_model_restore_path,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    # Set up weights and optimizer for training
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    optimizer_parameters = [
        {
            'params' : parameters_depth_model,
            'weight_decay' : w_weight_decay_depth
        }
    ]

    if 'monocular' in supervision_type:
        optimizer_parameters.append(
            {
                'params' : parameters_pose_model,
                'weight_decay' : w_weight_decay_pose
            })

    optimizer = torch.optim.Adam(
        optimizer_parameters,
        lr=learning_rate)

    # Start training
    train_step = 0

    epoch = None
    if depth_model_restore_path is not None and depth_model_restore_path != '':
        train_step, optimizer, epoch = depth_model.restore_model(depth_model_restore_path, optimizer)

    if pose_model_restore_path is not None and pose_model_restore_path != '':
        pose_model.restore_model(pose_model_restore_path)

    time_start = time.time()

    log('Begin training...', log_path)
    if epoch is None:
        start_epoch = 1
    else:
        start_epoch = epoch

    for epoch in range(start_epoch, learning_schedule[-1] + 1):

        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

        # Set augmentation schedule
        if -1 not in augmentation_schedule and epoch > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1
            augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        for inputs in train_dataloader:

            train_step = train_step + 1

            # Fetch data
            inputs = [
                in_.to(device) for in_ in inputs
            ]

            image0, \
                image1, \
                image2, \
                image3, \
                intrinsics0, \
                focal_length_baseline0, \
                ground_truth0 = inputs

            # Do data augmentation
            [image0, image1, image2] = train_transforms_photometric.transform(
                images_arr=[image0, image1, image2],
                random_transform_probability=augmentation_probability)

            [image0, image1, image2, ground_truth0] = train_transforms_geometric.transform(
                images_arr=[image0, image1, image2, ground_truth0],
                interpolation_modes=interpolation_modes,
                random_transform_probability=augmentation_probability)

            # Forward through the network
            output_depth0 = depth_model.forward(
                image0,
                return_all_output_resolutions=False)

            if 'monocular' in supervision_type:
                pose0to1 = pose_model.forward(image0, image1)
                pose0to2 = pose_model.forward(image0, image2)

                # Compute loss function
                loss, loss_info = depth_model.compute_unsupervised_loss(
                    output_depth0,
                    image0,
                    image1,
                    image2,
                    pose0to1,
                    pose0to2,
                    intrinsics0,
                    w_color=w_color,
                    w_structure=w_structure,
                    w_smoothness=w_smoothness)
            elif 'ground_truth' in supervision_type:
                # Compute loss function
                loss, loss_info = depth_model.compute_supervised_loss(
                    output_depth0,
                    ground_truth0)

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_summary) == 0:
                # Get images and depth map to log
                if 'monocular' in supervision_type:
                    image1to0 = loss_info.pop('image1to0')
                    image2to0 = loss_info.pop('image2to0')
                elif 'ground_truth' in supervision_type:
                    image1to0 = image1
                    image2to0 = image2

                # Log summary
                depth_model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train',
                    step=train_step,
                    image0=image0,
                    image1to0=image1to0.detach().clone(),
                    image2to0=image2to0.detach().clone(),
                    output_depth0=output_depth0.detach().clone(),
                    ground_truth0=ground_truth0,
                    scalars=loss_info,
                    n_display=min(n_batch, n_summary_display))

            # Log results and save checkpoints
            if (train_step % n_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                # Save checkpoints
                depth_model.save_model(
                    depth_model_checkpoint_path.format(train_step), train_step, optimizer)

                if 'monocular' in supervision_type:
                    pose_model.save_model(
                        pose_model_checkpoint_path.format(train_step), train_step, optimizer, epoch)

    # Save checkpoints and close summary
    train_summary_writer.close()

    depth_model.save_model(
        depth_model_checkpoint_path.format(train_step), train_step, optimizer)

    if 'monocular' in supervision_type:
        pose_model.save_model(
            pose_model_checkpoint_path.format(train_step), train_step, optimizer, epoch)


'''
Helper functions for logging
'''
def log_input_settings(log_path,
                       n_batch=None,
                       n_height=None,
                       n_width=None,
                       input_channels=3,
                       normalized_image_range=[0, 255]):

    batch_settings_text = ''
    batch_settings_vars = []

    if n_batch is not None:
        batch_settings_text = batch_settings_text + 'n_batch={}'
        batch_settings_vars.append(n_batch)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_height is not None:
        batch_settings_text = batch_settings_text + 'n_height={}'
        batch_settings_vars.append(n_height)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_width is not None:
        batch_settings_text = batch_settings_text + 'n_width={}'
        batch_settings_vars.append(n_width)

    log('Input settings:', log_path)

    if len(batch_settings_vars) > 0:
        log(batch_settings_text.format(*batch_settings_vars),
            log_path)

    log('input_channels={}'.format(
        input_channels),
        log_path)
    log('normalized_image_range={}'.format(normalized_image_range),
        log_path)
    log('', log_path)

def log_network_settings(log_path,
                         # Depth network settings
                         encoder_type,
                         min_predict_depth,
                         max_predict_depth,
                         parameters_depth_model=[],
                         parameters_pose_model=[]):

    # Computer number of parameters
    if parameters_depth_model is not None:
        n_parameter_depth = sum(p.numel() for p in parameters_depth_model)
    else:
        n_parameter_depth = 0

    if parameters_pose_model is not None:
        n_parameter_pose = sum(p.numel() for p in parameters_pose_model)
    else:
        n_parameter_pose = 0

    n_parameter = n_parameter_depth + n_parameter_pose

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    if n_parameter_depth > 0 :
        n_parameter_text = n_parameter_text + 'n_parameter_depth={}'
        n_parameter_vars.append(n_parameter_depth)

    n_parameter_text = \
        n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

    if n_parameter_pose > 0 :
        n_parameter_text = n_parameter_text + 'n_parameter_pose={}'
        n_parameter_vars.append(n_parameter_pose)

    n_parameter_text = \
        n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

    log('Depth network settings:', log_path)
    log('encoder_type={}'.format(encoder_type),
        log_path)
    log('min_predict_depth={:.2f}'.format(min_predict_depth),
        log_path)
    log('max_predict_depth={:.2f}'.format(max_predict_depth),
        log_path)
    log('', log_path)

    log('Weight settings:', log_path)
    log('n_parameter={}  n_parameter_depth={}  n_parameter_pose={}'.format(
        n_parameter, n_parameter_depth, n_parameter_pose),
        log_path)
    log('', log_path)

def log_training_settings(log_path,
                          # Training settings
                          n_batch,
                          n_train_sample,
                          n_train_step,
                          learning_rates,
                          learning_schedule,
                          # Augmentation settings
                          augmentation_probabilities,
                          augmentation_schedule,
                          # Photometric data augmentations
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_gamma,
                          augmentation_random_hue,
                          augmentation_random_saturation,
                          # Geometric data augmentations
                          augmentation_random_swap_left_right,
                          augmentation_random_crop_to_shape,
                          augmentation_random_flip_type,
                          augmentation_random_rotate_max,
                          augmentation_random_crop_and_pad,
                          augmentation_random_resize_and_crop,
                          augmentation_random_resize_and_pad):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)

    log('Photometric data augmentations:', log_path)
    log('augmentation_random_brightness={}'.format(augmentation_random_brightness),
        log_path)
    log('augmentation_random_contrast={}'.format(augmentation_random_contrast),
        log_path)
    log('augmentation_random_gamma={}'.format(augmentation_random_gamma),
        log_path)
    log('augmentation_random_hue={}'.format(augmentation_random_hue),
        log_path)
    log('augmentation_random_saturation={}'.format(augmentation_random_saturation),
        log_path)

    log('Geometric data augmentations:', log_path)
    log('augmentation_random_swap_left_right={}'.format(augmentation_random_swap_left_right),
        log_path)
    log('augmentation_random_crop_to_shape={}'.format(augmentation_random_crop_to_shape),
        log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
        log_path)
    log('augmentation_random_rotate_max={}'.format(augmentation_random_rotate_max),
        log_path)
    log('augmentation_random_crop_and_pad={}'.format(augmentation_random_crop_and_pad),
        log_path)
    log('augmentation_random_crop_and_pad={}'.format(augmentation_random_crop_and_pad),
        log_path)
    log('augmentation_random_resize_and_crop={}'.format(augmentation_random_resize_and_crop),
        log_path)
    log('augmentation_random_resize_and_pad={}'.format(augmentation_random_resize_and_pad),
        log_path)

    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           supervision_type,
                           w_color,
                           w_structure,
                           w_smoothness,
                           w_weight_decay_depth,
                           w_weight_decay_pose):

    log('Loss function settings:', log_path)
    log('supervision_type={}'.format(
        supervision_type),
        log_path)
    log('w_color={:.1e}  w_structure={:.1e}'.format(
        w_color, w_structure),
        log_path)
    log('w_smoothness={:.1e}'.format(w_smoothness),
        log_path)
    log('w_weight_decay_depth={:.1e}  w_weight_decay_pose={:.1e}'.format(
        w_weight_decay_depth, w_weight_decay_pose),
        log_path)
    log('', log_path)

def log_evaluation_settings(log_path,
                            min_evaluate_depth,
                            max_evaluate_depth):

    log('Evaluation settings:', log_path)
    log('min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        min_evaluate_depth, max_evaluate_depth),
        log_path)
    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_path,
                        n_checkpoint=None,
                        summary_event_path=None,
                        n_summary=None,
                        n_summary_display=None,
                        validation_start_step=None,
                        depth_model_restore_path=None,
                        pose_model_restore_path=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)

        if n_checkpoint is not None:
            log('checkpoint_save_frequency={}'.format(n_checkpoint), log_path)

        if validation_start_step is not None:
            log('validation_start_step={}'.format(validation_start_step), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_summary is not None:
        summary_settings_text = summary_settings_text + 'log_summary_frequency={}'
        summary_settings_vars.append(n_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if n_summary_display is not None:
        summary_settings_text = summary_settings_text + 'n_summary_display={}'
        summary_settings_vars.append(n_summary_display)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if depth_model_restore_path is not None and depth_model_restore_path != '':
        log('depth_model_restore_path={}'.format(depth_model_restore_path),
            log_path)

    if pose_model_restore_path is not None and pose_model_restore_path != '':
        log('pose_model_restore_path={}'.format(pose_model_restore_path),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
