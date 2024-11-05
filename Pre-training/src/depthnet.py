import os, time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils, eval_utils
from log_utils import log
from transforms import Transforms
from depthnet_model import DepthNetModel
from posenet_model import PoseNetModel


def train(train_image_left_path,
          train_image_right_path,
          train_intrinsics_left_path,
          train_intrinsics_right_path,
          train_focal_length_baseline_left_path,
          train_focal_length_baseline_right_path,
          val_image_path,
          val_ground_truth_path,
          # Batch settings
          n_batch,
          n_height,
          n_width,
          # Input settings
          input_channels,
          normalized_image_range,
          # Depth network settings
          encoder_type,
          n_filters_encoder,
          decoder_type,
          n_resolution_decoder_output,
          n_filters_decoder,
          min_predict_depth,
          max_predict_depth,
          # Weight settings
          weight_initializer,
          activation_func,
          use_batch_norm,
          # Training settings
          learning_rates,
          learning_schedule,
          augmentation_probabilities,
          augmentation_schedule,
          augmentation_random_swap_left_right,
          augmentation_random_crop_type,
          augmentation_random_flip_type,
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_saturation,
          # Loss function settings
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

    if device == 'gpu' or device == 'cuda':
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

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty
    }

    '''
    Load input paths and set up dataloaders
    '''
    # Read paths for training
    train_image_left_paths = data_utils.read_paths(train_image_left_path)
    train_image_right_paths = data_utils.read_paths(train_image_right_path)

    train_intrinsics_left_paths = data_utils.read_paths(train_intrinsics_left_path)
    train_intrinsics_right_paths = data_utils.read_paths(train_intrinsics_right_path)

    train_focal_length_baseline_left_paths = \
        data_utils.read_paths(train_focal_length_baseline_left_path)
    train_focal_length_baseline_right_paths = \
        data_utils.read_paths(train_focal_length_baseline_right_path)

    n_train_sample = len(train_image_left_paths)

    # Make sure number of paths match number of training sample
    input_paths = [
        train_image_left_paths,
        train_image_right_paths,
        train_intrinsics_left_paths,
        train_intrinsics_right_paths,
        train_focal_length_baseline_left_paths,
        train_focal_length_baseline_right_paths
    ]

    for paths in input_paths:
        assert len(paths) == n_train_sample

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.SingleImageDepthStereoTrainingDataset(
            image_left_paths=train_image_left_paths,
            image_right_paths=train_image_right_paths,
            intrinsics_left_paths=train_intrinsics_left_paths,
            intrinsics_right_paths=train_intrinsics_right_paths,
            focal_length_baseline_left_paths=train_focal_length_baseline_left_paths,
            focal_length_baseline_right_paths=train_focal_length_baseline_right_paths,
            random_crop_shape=(n_height, n_width),
            random_crop_type=augmentation_random_crop_type,
            random_swap_left_right=augmentation_random_swap_left_right),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=False)

    train_transforms = Transforms(
        normalized_image_range=normalized_image_range,
        random_flip_type=augmentation_random_flip_type,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_saturation=augmentation_random_saturation)

    # Load validation data if it is available
    validation_available = val_image_path is not None and \
        val_ground_truth_path is not None

    if validation_available:
        val_image_paths = data_utils.read_paths(val_image_path)
        val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

        n_val_sample = len(val_image_paths)

        assert len(val_ground_truth_paths) == n_val_sample

        ground_truths = []
        for path in val_ground_truth_paths:
            ground_truth, validity_map = data_utils.load_depth_with_validity_map(
                path,
                data_format='CHW')
            ground_truth = np.concatenate([ground_truth, validity_map], axis=0)
            ground_truths.append(np.expand_dims(ground_truth, axis=0))

        val_dataloader = torch.utils.data.DataLoader(
            datasets.SingleImageDepthInferenceDataset(
                image_paths=val_image_paths),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        val_transforms = Transforms(
            normalized_image_range=normalized_image_range)

    '''
    Set up the model
    '''
    # Build depth prediction network
    depth_model = DepthNetModel(
        encoder_type=encoder_type,
        input_channels=input_channels,
        n_filters_encoder=n_filters_encoder,
        decoder_type=decoder_type,
        n_resolution_decoder_output=n_resolution_decoder_output,
        n_filters_decoder=n_filters_decoder,
        activation_func=activation_func,
        weight_initializer=weight_initializer,
        use_batch_norm=use_batch_norm,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    parameters_depth_model = depth_model.parameters()

    depth_model.train()

    # Bulid PoseNet (only needed for training) network
    pose_model = PoseNetModel(
        encoder_type='resnet18',
        rotation_parameterization='axis',
        weight_initializer=weight_initializer,
        activation_func='relu',
        device=device)

    parameters_pose_model = pose_model.parameters()

    pose_model.train()

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    '''
    Log input paths
    '''
    log('Training input paths:', log_path)
    train_input_paths = [
        train_image_left_path,
        train_image_right_path,
        train_intrinsics_left_path,
        train_intrinsics_right_path,
        train_focal_length_baseline_left_path,
        train_focal_length_baseline_right_path,
    ]
    for path in train_input_paths:
        log(path, log_path)
    log('', log_path)

    log('Validation input paths:', log_path)
    val_input_paths = [
        val_image_path,
        val_ground_truth_path
    ]
    for path in val_input_paths:
        log(path, log_path)
    log('', log_path)

    log_input_settings(
        log_path,
        # Batch settings
        n_batch=n_batch,
        n_height=n_height,
        n_width=n_width,
        # Input settings
        input_channels=input_channels,
        normalized_image_range=normalized_image_range)

    log_network_settings(
        log_path,
        # Depth network settings
        encoder_type=encoder_type,
        n_filters_encoder=n_filters_encoder,
        decoder_type=decoder_type,
        n_resolution_decoder_output=n_resolution_decoder_output,
        n_filters_decoder=n_filters_decoder,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        weight_initializer=weight_initializer,
        activation_func=activation_func,
        use_batch_norm=use_batch_norm,
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
        augmentation_random_crop_type=augmentation_random_crop_type,
        augmentation_random_flip_type=augmentation_random_flip_type,
        augmentation_random_brightness=augmentation_random_brightness,
        augmentation_random_contrast=augmentation_random_contrast,
        augmentation_random_saturation=augmentation_random_saturation)

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

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    optimizer = torch.optim.Adam([
        {
            'params' : parameters_depth_model,
            'weight_decay' : w_weight_decay_depth
        },
        {
            'params' : parameters_pose_model,
            'weight_decay' : w_weight_decay_pose
        }],
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
                focal_length_baseline0 = inputs

            # Do data augmentation
            [image0, image1, image2, image3] = train_transforms.transform(
                images_arr=[image0, image1, image2, image3],
                random_transform_probability=augmentation_probability)

            # Forward through the network
            output_depth0 = depth_model.forward(image0)

            pose0to1 = pose_model.forward(image0, image1)
            pose0to2 = pose_model.forward(image0, image2)

            # Compute loss function
            loss, loss_info = depth_model.compute_loss(
                output_depth0=output_depth0,
                supervision_type=supervision_type,
                image0=image0,
                image1=image1,
                image2=image2,
                image3=image3,
                pose0to1=pose0to1,
                pose0to2=pose0to2,
                intrinsics0=intrinsics0,
                focal_length_baseline0=focal_length_baseline0,
                w_color=w_color,
                w_structure=w_structure,
                w_smoothness=w_smoothness)

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_summary) == 0:

                image1to0 = loss_info.pop('image1to0')
                image2to0 = loss_info.pop('image2to0')
                image3to0 = loss_info.pop('image3to0')

                # Log summary
                depth_model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train',
                    step=train_step,
                    image0=image0,
                    image1to0=image1to0.detach().clone(),
                    image2to0=image2to0.detach().clone(),
                    image3to0=image3to0.detach().clone(),
                    output_depth0=output_depth0.detach().clone(),
                    pose0to1=pose0to1.detach().clone(),
                    pose0to2=pose0to2.detach().clone(),
                    scalars=loss_info,
                    n_display=min(n_batch, n_summary_display))

            # Log results and save checkpoints
            if (train_step % n_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                if train_step >= validation_start_step and validation_available:
                    # Switch to validation mode
                    depth_model.eval()

                    with torch.no_grad():
                        # Perform validation
                        best_results = validate(
                            depth_model=depth_model,
                            dataloader=val_dataloader,
                            transforms=val_transforms,
                            ground_truths=ground_truths,
                            step=train_step,
                            best_results=best_results,
                            median_scale_depth='stereo' not in supervision_type,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            device=device,
                            summary_writer=val_summary_writer,
                            n_summary_display=n_summary_display,
                            log_path=log_path)

                    # Switch back to training
                    depth_model.train()

                # Save checkpoints
                depth_model.save_model(
                    depth_model_checkpoint_path.format(train_step), train_step, optimizer, epoch)

                pose_model.save_model(
                    pose_model_checkpoint_path.format(train_step), train_step, optimizer, epoch)

    # Perform validation for final step
    depth_model.eval()

    with torch.no_grad():
        best_results = validate(
            depth_model=depth_model,
            dataloader=val_dataloader,
            transforms=val_transforms,
            ground_truths=ground_truths,
            step=train_step,
            best_results=best_results,
            median_scale_depth='stereo' not in supervision_type,
            min_evaluate_depth=min_evaluate_depth,
            max_evaluate_depth=max_evaluate_depth,
            device=device,
            summary_writer=val_summary_writer,
            n_summary_display=n_summary_display,
            log_path=log_path)

    # Save checkpoints
    depth_model.save_model(
        depth_model_checkpoint_path.format(train_step), train_step, optimizer, epoch)

    pose_model.save_model(
        pose_model_checkpoint_path.format(train_step), train_step, optimizer, epoch)

def validate(depth_model,
             dataloader,
             transforms,
             ground_truths,
             step,
             best_results,
             median_scale_depth,
             min_evaluate_depth,
             max_evaluate_depth,
             device,
             summary_writer,
             n_summary_display=4,
             n_summary_display_interval=250,
             log_path=None):

    n_sample = len(dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    image_summary = []
    output_depth_summary = []
    ground_truth_summary = []

    for idx, (image, ground_truth) in enumerate(zip(dataloader, ground_truths)):

        # Move inputs to device
        image = image.to(device)

        ground_truth = torch.from_numpy(ground_truth).to(device)

        [image] = transforms.transform(
            images_arr=[image],
            random_transform_probability=0.0)

        # Forward through network
        output_depth = depth_model.forward(image)

        if (idx % n_summary_display_interval) == 0 and summary_writer is not None:
            image_summary.append(image)
            output_depth_summary.append(output_depth)
            ground_truth_summary.append(ground_truth)

        # Convert to numpy to validate
        output_depth = np.squeeze(output_depth.cpu().numpy())
        ground_truth = np.squeeze(ground_truth.cpu().numpy())

        validity_map = ground_truth[1, :, :]
        ground_truth = ground_truth[0, :, :]

        # Select valid regions to evaluate
        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        output_depth = output_depth[mask]
        ground_truth = ground_truth[mask]

        # Scale output depth by median
        if median_scale_depth:
            output_depth = output_depth * np.median(ground_truth) / np.median(output_depth)

        # Compute validation metrics
        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    # Compute mean metrics
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)

    # Log to tensorboard
    if summary_writer is not None:
        depth_model.log_summary(
            summary_writer=summary_writer,
            tag='eval',
            step=step,
            image0=torch.cat(image_summary, dim=0),
            output_depth0=torch.cat(output_depth_summary, dim=0),
            ground_truth0=torch.cat(ground_truth_summary, dim=0),
            scalars={'mae' : mae, 'rmse' : rmse, 'imae' : imae, 'irmse': irmse},
            n_display=n_summary_display)

    # Print validation results to console
    log('Validation results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step, mae, rmse, imae, irmse),
        log_path)

    n_improve = 0
    if np.round(mae, 2) <= np.round(best_results['mae'], 2):
        n_improve = n_improve + 1
    if np.round(rmse, 2) <= np.round(best_results['rmse'], 2):
        n_improve = n_improve + 1
    if np.round(imae, 2) <= np.round(best_results['imae'], 2):
        n_improve = n_improve + 1
    if np.round(irmse, 2) <= np.round(best_results['irmse'], 2):
        n_improve = n_improve + 1

    if n_improve > 2:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse

    log('Best results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        best_results['step'],
        best_results['mae'],
        best_results['rmse'],
        best_results['imae'],
        best_results['irmse']), log_path)

    return best_results


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
                         n_filters_encoder,
                         decoder_type,
                         n_resolution_decoder_output,
                         n_filters_decoder,
                         min_predict_depth,
                         max_predict_depth,
                         # Weight settings
                         weight_initializer,
                         activation_func,
                         use_batch_norm,
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
    log('n_filters_encoder={}'.format(n_filters_encoder),
        log_path)
    log('n_filters_decoder={}  n_resolution_decoder={}'.format(
        n_filters_decoder, n_resolution_decoder_output),
        log_path)
    log('decoder_type={}'.format(decoder_type),
        log_path)
    log('min_predict_depth={:.2f}  max_predict_depth={:.2f}'.format(
        min_predict_depth, max_predict_depth),
        log_path)
    log('', log_path)

    log('Weight settings:', log_path)
    log('n_parameter={}  n_parameter_depth={}  n_parameter_pose={}'.format(
        n_parameter, n_parameter_depth, n_parameter_pose),
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
                          augmentation_random_swap_left_right,
                          augmentation_random_crop_type,
                          augmentation_random_flip_type,
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
    log('augmentation_random_swap_left_right={}'.format(augmentation_random_swap_left_right),
        log_path)
    log('augmentation_random_crop_type={}'.format(augmentation_random_crop_type),
        log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
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
