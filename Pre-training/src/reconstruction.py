import os, time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils
from log_utils import log
from reconstruction_model import ReconstructionModel
from transforms import Transforms


def train(train_images_left_path,
          train_images_right_path,
          # Input settings
          n_batch,
          n_height,
          n_width,
          normalized_image_range,
          # Network settings
          encoder_type,
          # Training settings
          learning_rates,
          learning_schedule,
          augmentation_probabilities,
          augmentation_schedule,
          augmentation_random_swap_left_right,
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_saturation,
          # Loss settings
          w_reconstruction,
          remove_percent_range,
          remove_patch_size,
          # Checkpoint settings
          checkpoint_path,
          n_checkpoint,
          n_summary,
          n_summary_display,
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
    decoder_reconstruction_checkpoint_path = os.path.join(checkpoint_path, 'decoder_reconstruction-{}.pth')

    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    '''
    Set up paths for training
    '''
    train_images_left_paths = data_utils.read_paths(train_images_left_path)
    train_images_right_paths = data_utils.read_paths(train_images_right_path)

    n_train_sample = len(train_images_left_paths)

    # Make sure number of paths match number of training sample
    input_paths = [
        train_images_left_paths,
        train_images_right_paths,
    ]

    for paths in input_paths:
        assert len(paths) == n_train_sample

    # Set up training dataloader
    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.ReconstructionTrainingDataset(
            images_left_paths=train_images_left_paths,
            images_right_paths=train_images_right_paths,
            resize_shape=(n_height, n_width),
            random_swap_left_right=augmentation_random_swap_left_right),
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
    Build network
    '''
    # Build network
    reconstruction_model = ReconstructionModel(
        encoder_type=encoder_type,
        device=device)

    reconstruction_model.train()

    parameters_reconstruction_model = reconstruction_model.parameters()

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')

    log('Training input paths:', log_path)
    train_input_paths = [
        train_images_left_path,
        train_images_right_path,
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
        augmentation_random_saturation=augmentation_random_saturation)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        remove_percent_range=remove_percent_range,
        remove_patch_size=remove_patch_size,
        w_reconstruction=w_reconstruction)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        n_checkpoint=n_checkpoint,
        summary_event_path=event_path,
        n_summary=n_summary,
        n_summary_display=n_summary_display,
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
            'params' : parameters_reconstruction_model,
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

        for train_images in train_dataloader:

            train_step = train_step + 1

            # Fetch data
            train_images = train_images.to(device)

            # Do data augmentation
            [train_images] = train_transforms.transform(
                images_arr=[train_images],
                random_transform_probability=augmentation_probability)

            # Remove patches
            batch, _, height, width = train_images.shape
            masks = torch.ones([batch, 1, height, width], device=device)

            masks = remove_random_patches(
                masks,
                remove_percent_range=remove_percent_range,
                remove_patch_size=remove_patch_size)

            input_images = masks * train_images

            # Forward through the network
            output_images = reconstruction_model.forward(
                input_images,
                return_all_output_resolutions=True)

            # Compute loss function
            loss, loss_info = reconstruction_model.compute_loss(
                output_images=output_images,
                target_images=train_images,
                w_reconstruction=w_reconstruction)

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_summary) == 0:
                reconstruction_model.log_summary(
                    summary_writer=train_summary_writer,
                    images=input_images,
                    output_images=output_images[-1].detach().clone(),
                    ground_truths=train_images,
                    scalars=loss_info,
                    tag='train',
                    step=train_step)

            # Log results and save checkpoints
            if (train_step % n_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                # Save checkpoints
                reconstruction_model.save_model(
                    encoder_checkpoint_path.format(train_step),
                    decoder_reconstruction_checkpoint_path.format(train_step),
                    train_step,
                    optimizer,
                     epoch)

    # Save checkpoints and close summary
    train_summary_writer.close()


'''
Help functions for loss
'''
def remove_random_patches(images, remove_percent_range, remove_patch_size):
    '''
    Remove random patches for each sample

    Arg(s):
        images_arr : list[torch.Tensor[float32]]
            list of N x C x H x W tensors
        remove_percent_range : list[float]
            min and max remove percentage
        remove_patch_size : list[int]
            size of patch to remove
    Returns:
        torch.Tensor[float32] : list of transformed N x C x H x W image tensors
    '''

    n_batch = images.shape[0]
    device = images.device

    values = torch.rand(n_batch, device=device)

    remove_percent_min, remove_percent_max = remove_percent_range

    densities = \
        (remove_percent_max - remove_percent_min) * values + remove_percent_min

    for b, image in enumerate(images):

        nonzero_indices = random_nonzero(image, density=densities[b])
        image[nonzero_indices] = 0.0

        image = -torch.nn.functional.max_pool2d(
            input=-image,
            kernel_size=remove_patch_size,
            stride=1,
            padding=[int(k // 2) for k in remove_patch_size])

        images[b, ...] = image

    return images

def random_nonzero(T, density=0.10):
    '''
    Randomly selects nonzero elements

    Arg(s):
        T : torch.Tensor[float32]
            N x C x H x W tensor
        density : float
            percentage of nonzero elements to select
    Returns:
        list[tuple[torch.Tensor[float32]]] : list of tuples of indices
    '''

    # Find all nonzero indices
    nonzero_indices = (T > 0).nonzero(as_tuple=True)

    # Randomly choose a subset of the indices
    random_subset = torch.randperm(nonzero_indices[0].shape[0], device=T.device)
    random_subset = random_subset[0:int(density * random_subset.shape[0])]

    random_nonzero_indices = [
        indices[random_subset] for indices in nonzero_indices
    ]

    return random_nonzero_indices


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
                          augmentation_random_swap_left_right,
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
    log('augmentation_random_brightness=%s' %
        (augmentation_random_brightness), log_path)
    log('augmentation_random_contrast=%s' %
        (augmentation_random_contrast), log_path)
    log('augmentation_random_saturation=%s' %
        (augmentation_random_saturation), log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           remove_percent_range,
                           remove_patch_size,
                           w_reconstruction):

    log('Loss function settings:', log_path)
    log('remove_percent_range={}  remove_patch_size={}'.format(
        remove_percent_range, remove_patch_size),
        log_path)
    log('w_reconstruction={:.1e}'.format(
        w_reconstruction),
        log_path)
    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_path,
                        n_checkpoint=None,
                        summary_event_path=None,
                        n_summary=None,
                        n_summary_display=None,
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

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
