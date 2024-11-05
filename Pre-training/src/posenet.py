import os, time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils
from log_utils import log
from posenet_model import PoseNetModel
from monodepth2_model import Monodepth2Model
from transforms import Transforms


def train(train_images_left_path,
          train_images_right_path,
          train_intrinsics_left_path,
          train_intrinsics_right_path,
          # Input settings
          n_batch,
          n_height,
          n_width,
          normalized_image_range,
          # Network settings
          encoder_type,
          # Weight settings
          weight_initializer,
          activation_func,
          use_batch_norm,
          # Training settings
          learning_rates,
          learning_schedule,
          augmentation_probabilities,
          augmentation_schedule,
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_saturation,
          # Loss settings
          w_color,
          w_structure,
          w_weight_decay_pose,
          # Checkpoint settings
          checkpoint_path,
          n_checkpoint,
          n_summary,
          n_summary_display,
          monodepth2_encoder_restore_path,
          monodepth2_decoder_restore_path,
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
    pose_model_checkpoint_path = os.path.join(checkpoint_path, 'pose_model-{}.pth')

    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    '''
    Set up paths for training
    '''
    train_images_left_paths = data_utils.read_paths(train_images_left_path)
    train_images_right_paths = data_utils.read_paths(train_images_right_path)

    train_intrinsics_left_paths = data_utils.read_paths(train_intrinsics_left_path)
    train_intrinsics_right_paths = data_utils.read_paths(train_intrinsics_right_path)

    n_train_sample = len(train_images_left_paths)

    # Make sure number of paths match number of training sample
    input_paths = [
        train_images_left_paths,
        train_images_right_paths,
        train_intrinsics_left_paths,
        train_intrinsics_right_paths,
    ]

    for paths in input_paths:
        assert len(paths) == n_train_sample

    train_images_paths = train_images_left_paths + train_images_right_paths
    train_intrinsics_paths = train_intrinsics_left_paths + train_intrinsics_right_paths

    n_train_sample = len(train_images_paths)

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.Monodepth2InferenceDataset(
            image_paths=train_images_paths,
            intrinsics_paths=train_intrinsics_paths,
            resize_shape=(n_height, n_width),
            load_image_triplets=True),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=False)

    train_transforms = Transforms(
        normalized_image_range=normalized_image_range,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_saturation=augmentation_random_saturation)

    '''
    Build networks
    '''
    # Build network to output depth
    monodepth2_model = Monodepth2Model(device=device)
    monodepth2_model.eval()

    monodepth2_model.restore_model(encoder_restore_path=monodepth2_encoder_restore_path)
    monodepth2_model.restore_model(decoder_depth_restore_path=monodepth2_decoder_restore_path)

    monodepth2_model.eval()

    # Build network to output pose
    pose_model = PoseNetModel(
        encoder_type=encoder_type,
        rotation_parameterization='axis',
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        device=device)

    pose_model.train()

    parameters_pose_model = pose_model.parameters()

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')

    log('Training input paths:', log_path)
    train_input_paths = [
        train_images_left_path,
        train_images_right_path,
        train_intrinsics_left_path,
        train_intrinsics_right_path,
    ]
    for path in train_input_paths:
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
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        use_batch_norm=use_batch_norm,
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
        augmentation_random_brightness=augmentation_random_brightness,
        augmentation_random_contrast=augmentation_random_contrast,
        augmentation_random_saturation=augmentation_random_saturation)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        w_color=w_color,
        w_structure=w_structure,
        w_weight_decay_pose=w_weight_decay_pose)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        n_checkpoint=n_checkpoint,
        summary_event_path=event_path,
        n_summary=n_summary,
        n_summary_display=n_summary_display,
        monodepth2_encoder_restore_path=monodepth2_encoder_restore_path,
        monodepth2_decoder_restore_path=monodepth2_decoder_restore_path,
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
            'params' : parameters_pose_model,
            'weight_decay' : w_weight_decay_pose
        }
    ]

    optimizer = torch.optim.Adam(
        optimizer_parameters,
        lr=learning_rate)

    # Start training
    train_step = 0

    epoch = None
    if pose_model_restore_path is not None and pose_model_restore_path != '':
        train_step, optimizer, epoch = pose_model.restore_model(
            pose_model_restore_path,
            optimizer)

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
                intrinsics = inputs

            # Forward through the network
            with torch.no_grad():
                # Separate image for depth so it does not change with augmentation
                [image] = train_transforms.transform(
                    images_arr=[image0],
                    random_transform_probability=0.0)

                depth0 = monodepth2_model.forward_depth(image)

                # Do data augmentation
                [image0, image1, image2] = train_transforms.transform(
                    images_arr=[image0, image1, image2],
                    random_transform_probability=augmentation_probability)

            pose0to1 = pose_model.forward(image0, image1)
            pose0to2 = pose_model.forward(image0, image2)

            # Compute loss function
            loss, loss_info = pose_model.compute_loss(
                depth0=depth0,
                image0=image0,
                image1=image1,
                image2=image2,
                pose0to1=pose0to1,
                pose0to2=pose0to2,
                intrinsics=intrinsics,
                w_color=w_color,
                w_structure=w_structure)

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_summary) == 0:
                # Get images and depth map to log
                image1to0 = loss_info.pop('image1to0')
                image2to0 = loss_info.pop('image2to0')

                # Log summary
                pose_model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train',
                    step=train_step,
                    image0=image0,
                    image1to0=image1to0.detach().clone(),
                    image2to0=image2to0.detach().clone(),
                    output_depth0=depth0,
                    pose0to1=pose0to1,
                    pose0to2=pose0to2,
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
                pose_model.save_model(
                    pose_model_checkpoint_path.format(train_step), train_step, optimizer, epoch)

    # Save checkpoints and close summary
    train_summary_writer.close()

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
                         # Weight settings
                         weight_initializer,
                         activation_func,
                         use_batch_norm,
                         parameters_pose_model=[]):

    # Computer number of parameters
    if parameters_pose_model is not None:
        n_parameter_pose = sum(p.numel() for p in parameters_pose_model)
    else:
        n_parameter_pose = 0

    log('Depth network settings:', log_path)
    log('encoder_type={}'.format(encoder_type),
        log_path)
    log('', log_path)

    log('Weight settings:', log_path)
    log('n_parameter_pose={}'.format(
        n_parameter_pose),
        log_path)
    log('weight_initializer={}  activation_func={}  use_batch_norm={}'.format(
        weight_initializer, activation_func, use_batch_norm),
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
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_saturation):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{}({}):{}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), le, v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{}({}):{}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), le, v)
            for ls, le, v in zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)
    log('augmentation_random_brightness=%s' %
        (augmentation_random_brightness), log_path)
    log('augmentation_random_contrast=%s' %
        (augmentation_random_contrast), log_path)
    log('augmentation_random_saturation=%s' %
        (augmentation_random_saturation), log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           w_color,
                           w_structure,
                           w_weight_decay_pose):

    log('Loss function settings:', log_path)
    log('w_color={:.1e}  w_structure={:.1e}'.format(
        w_color, w_structure),
        log_path)
    log('w_weight_decay_pose={:.1e}'.format(
        w_weight_decay_pose),
        log_path)
    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_path,
                        n_checkpoint=None,
                        summary_event_path=None,
                        n_summary=None,
                        n_summary_display=None,
                        monodepth2_encoder_restore_path=None,
                        monodepth2_decoder_restore_path=None,
                        pose_model_restore_path=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)

        if n_checkpoint is not None:
            log('checkpoint_save_frequency={}'.format(n_checkpoint), log_path)

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

    if monodepth2_encoder_restore_path is not None and monodepth2_encoder_restore_path != '':
        log('monodepth2_encoder_restore_path={}'.format(monodepth2_encoder_restore_path),
            log_path)

    if monodepth2_decoder_restore_path is not None and monodepth2_decoder_restore_path != '':
        log('monodepth2_decoder_restore_path={}'.format(monodepth2_decoder_restore_path),
            log_path)

    if pose_model_restore_path is not None and pose_model_restore_path != '':
        log('pose_model_restore_path={}'.format(pose_model_restore_path),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
