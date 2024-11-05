import os, time, tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils
from log_utils import log
from transforms import Transforms
from monodepth2_model import Monodepth2Model


def train(train_images_path,
          train_intrinsics_path,
          train_ground_truth_path,
          # Input settings
          n_batch,
          n_height,
          n_width,
          normalized_image_range,
          # Network settings
          encoder_type,
          network_modules,
          scale_factor_depth,
          min_predict_depth,
          max_predict_depth,
          # Training settings
          learning_rates,
          learning_schedule,
          augmentation_probabilities,
          augmentation_schedule,
          augmentation_random_crop_type,
          augmentation_random_crop_shape,
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_saturation,
          augmentation_random_crop_to_shape,
          augmentation_random_flip_type,
          # Loss settings
          w_photometric,
          w_smoothness,
          # Checkpoint settings
          checkpoint_path,
          n_checkpoint,
          n_summary,
          n_summary_display,
          validation_start_step,
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
    encoder_checkpoint_path = os.path.join(checkpoint_path, 'encoder-{}.pth')
    decoder_depth_checkpoint_path = os.path.join(checkpoint_path, 'decoder_depth-{}.pth')
    decoder_pose_checkpoint_path = os.path.join(checkpoint_path, 'decoder_pose-{}.pth')
    decoder_mask_checkpoint_path = os.path.join(checkpoint_path, 'decoder_mask-{}.pth')

    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    '''
    Set up paths for training
    '''
    train_images_paths = data_utils.read_paths(train_images_path)
    train_intrinsics_paths = data_utils.read_paths(train_intrinsics_path)

    n_train_sample = len(train_images_paths)

    if train_ground_truth_path is not None:
        train_ground_truth_paths = data_utils.read_paths(train_ground_truth_path)
    else:
        train_ground_truth_paths = [None] * n_train_sample

    # Make sure number of paths match number of training sample
    input_paths = [
        train_images_paths,
        train_intrinsics_paths,
        train_ground_truth_paths,
    ]

    for paths in input_paths:
        assert len(paths) == n_train_sample

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.Monodepth2TrainingDataset(
            image_paths=train_images_paths,
            intrinsics_paths=train_intrinsics_paths,
            ground_truth_paths=train_ground_truth_paths,
            resize_shape=(n_height, n_width),
            random_crop_shape=augmentation_random_crop_shape,
            random_crop_type=augmentation_random_crop_type,
            load_image_triplets=True,
            return_image_info=False),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=False)

    train_transforms = Transforms(
        normalized_image_range=normalized_image_range,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_saturation=augmentation_random_saturation,
        random_crop_to_shape=augmentation_random_crop_to_shape,
        random_flip_type=augmentation_random_flip_type)

    '''
    Build network
    '''
    # Build network
    monodepth2_model = Monodepth2Model(
        encoder_type=encoder_type,
        network_modules=network_modules,
        scale_factor_depth=scale_factor_depth,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    monodepth2_model.train()

    parameters_monodepth2_model = monodepth2_model.parameters()

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')

    log('Training input paths:', log_path)
    train_input_paths = [
        train_images_path,
        train_intrinsics_path,
        train_ground_truth_path
    ]
    for path in train_input_paths:
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
        augmentation_random_saturation=augmentation_random_saturation,
        augmentation_random_crop_to_shape=augmentation_random_crop_to_shape,
        augmentation_random_flip_type=augmentation_random_flip_type)

    log_network_settings(
        log_path,
        # Network settings
        encoder_type=encoder_type,
        network_modules=network_modules,
        scale_factor_depth=scale_factor_depth,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        parameters_model=parameters_monodepth2_model)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        w_photometric=w_photometric,
        w_smoothness=w_smoothness)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        n_checkpoint=n_checkpoint,
        summary_event_path=event_path,
        n_summary=n_summary,
        n_summary_display=n_summary_display,
        validation_start_step=validation_start_step,
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
            'params' : parameters_monodepth2_model,
            'weight_decay' : 0.0
        }
    ]

    optimizer = torch.optim.Adam(
        optimizer_parameters,
        lr=learning_rate)

    # Start training
    train_step = 0

    time_start = time.time()

    log('Begin training...', log_path)
    start_epoch = 1

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

        for inputs in tqdm.tqdm(train_dataloader, desc='Epoch: {}/{}  Batch'.format(epoch, learning_schedule[-1] + 1)):

            train_step = train_step + 1

            # Fetch data
            inputs = [
                in_.to(device) for in_ in inputs
            ]

            image0, image1, image2, intrinsics, ground_truth = inputs

            # Do data augmentation
            [image0, image1, image2], [ground_truth] = train_transforms.transform(
                images_arr=[image0, image1, image2],
                range_maps_arr=[ground_truth],
                random_transform_probability=augmentation_probability)

            if False and 'cityscapes' in train_images_path:
                n_height_image, n_width_image = image0.shape[-2:]

                crop_start_height_image = int(0.10 * n_height_image)
                crop_end_height_image = int(0.90 * n_height_image)

                crop_start_width_image = int(0.10 * n_width_image)
                crop_end_width_image = int(0.90 * n_width_image)

                image0 = image0[:, :, crop_start_height_image:crop_end_height_image, crop_start_width_image:crop_end_width_image]
                image1 = image2[:, :, crop_start_height_image:crop_end_height_image, crop_start_width_image:crop_end_width_image]
                image2 = image2[:, :, crop_start_height_image:crop_end_height_image, crop_start_width_image:crop_end_width_image]

                n_height_ground_truth, n_width_ground_truth = ground_truth.shape[-2:]

                crop_start_height_ground_truth = int(0.10 * n_height_ground_truth)
                crop_end_height_ground_truth = int(0.90 * n_height_ground_truth)

                crop_start_width_ground_truth = int(0.10 * n_width_ground_truth)
                crop_end_width_ground_truth = int(0.90 * n_width_ground_truth)

                ground_truth = ground_truth[:, :, crop_start_height_ground_truth:crop_end_height_ground_truth, crop_start_width_ground_truth:crop_end_width_ground_truth]

                intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * 0.80
                intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * 0.80

            # Forward through the network
            output_depths = monodepth2_model.forward_depth(
                image0,
                return_all_output_resolutions=True)

            if 'mask' in network_modules:
                masks0to1 = monodepth2_model.forward_mask(image0, image1)
                masks0to2 = monodepth2_model.forward_mask(image0, image2)
            else:
                masks0to1 = None
                masks0to2 = None

            if 'pose' in network_modules:
                pose0to1 = monodepth2_model.forward_pose(image0, image1)
                pose0to2 = monodepth2_model.forward_pose(image0, image2)

                loss, loss_info = monodepth2_model.compute_unsupervised_loss(
                    output_depths,
                    image0=image0,
                    image1=image1,
                    image2=image2,
                    masks0to1=masks0to1,
                    masks0to2=masks0to2,
                    pose0to1=pose0to1,
                    pose0to2=pose0to2,
                    intrinsics=intrinsics,
                    w_photometric=w_photometric,
                    w_smoothness=w_smoothness)
            else:
                pose0to1 = None
                pose0to2 = None

                # Compute loss function
                loss, loss_info = monodepth2_model.compute_supervised_loss(
                    output_depths=output_depths,
                    ground_truth=ground_truth)

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_summary) == 0:
                # Get images and depth map to log
                output_depth0 = output_depths[0].clone().detach()

                validity_map = torch.where(ground_truth > 0, torch.ones_like(ground_truth), ground_truth)

                image1to0 = loss_info.pop('image1to0').clone().detach() if 'image1to0' in loss_info.keys() else None
                image2to0 = loss_info.pop('image2to0').clone().detach() if 'image2to0' in loss_info.keys() else None

                # Log summary
                monodepth2_model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train',
                    step=train_step,
                    image0=image0,
                    image1to0=image1to0,
                    image2to0=image2to0,
                    output_depth0=output_depth0,
                    ground_truth0=torch.cat([ground_truth, validity_map], dim=1),
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
                monodepth2_model.save_model(
                    encoder_checkpoint_path.format(train_step),
                    decoder_depth_checkpoint_path.format(train_step),
                    decoder_pose_checkpoint_path.format(train_step),
                    decoder_mask_checkpoint_path.format(train_step),
                    train_step,
                    optimizer)

    log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
        train_step, n_train_step, loss.item(), time_elapse, time_remain),
        log_path)

    # Save checkpoints
    monodepth2_model.save_model(
        encoder_checkpoint_path.format(train_step),
        decoder_depth_checkpoint_path.format(train_step),
        decoder_pose_checkpoint_path.format(train_step),
        decoder_mask_checkpoint_path.format(train_step),
        train_step,
        optimizer)

    # Save checkpoints and close summary
    train_summary_writer.close()


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
                          augmentation_random_saturation,
                          augmentation_random_flip_type,
                          augmentation_random_crop_to_shape):

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
    log('augmentation_random_flip_type=%s' %
        (augmentation_random_flip_type), log_path)
    log('augmentation_random_crop_to_shape=%s' %
        (augmentation_random_crop_to_shape), log_path)
    log('', log_path)


def log_network_settings(log_path,
                         # Depth network settings
                         encoder_type,
                         network_modules,
                         scale_factor_depth,
                         min_predict_depth,
                         max_predict_depth,
                         parameters_model=[]):

    # Computer number of parameters
    if parameters_model is not None:
        n_parameter = sum(p.numel() for p in parameters_model)
    else:
        n_parameter = 0

    log('n_parameter={}'.format(n_parameter), log_path)

    log('Network settings:', log_path)
    log('encoder_type={}'.format(encoder_type),
        log_path)
    log('network_modules={}'.format(network_modules),
        log_path)
    log('scale_factor_depth={:.2f}  min_predict_depth={:.2f}  max_predict_depth={:.2f}'.format(
        scale_factor_depth, min_predict_depth, max_predict_depth),
        log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           w_photometric,
                           w_smoothness):

    log('Loss function settings:', log_path)
    log('w_photometric={:.1e}  w_smoothness={:.1e}'.format(
        w_photometric, w_smoothness),
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
