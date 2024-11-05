import os, time, cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils, eval_utils
from log_utils import log
from monodepth_model import MonodepthModel


def train(train_image0_path,
          train_image1_path,
          train_camera_path,
          val_image0_path,
          val_camera_path,
          val_ground_truth_path,
          load_triplet,
          use_resize,
          # Batch settings
          n_batch,
          n_height,
          n_width,
          encoder_type,
          decoder_type,
          activation_func,
          n_pyramid,
          # Training settings
          learning_rates,
          learning_schedule,
          use_augment,
          w_color,
          w_ssim,
          w_smoothness,
          w_left_right,
          # Depth range settings
          scale_factor,
          min_evaluate_depth,
          max_evaluate_depth,
          # Checkpoint settings
          n_summary,
          n_checkpoint,
          checkpoint_path,
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
    decoder_checkpoint_path = os.path.join(checkpoint_path, 'decoder-{}.pth')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    # Read paths for training
    train_image0_paths = data_utils.read_paths(train_image0_path)
    train_image1_paths = data_utils.read_paths(train_image1_path)
    train_camera_paths = data_utils.read_paths(train_camera_path)
    assert(len(train_image0_paths) == len(train_image1_paths))
    assert(len(train_image0_paths) == len(train_camera_paths))

    n_train_sample = len(train_image0_paths)
    n_train_step = learning_schedule[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)

    # Read paths for validation
    val_image0_paths = data_utils.read_paths(val_image0_path)
    val_camera_paths = data_utils.read_paths(val_camera_path)
    val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)
    assert(len(val_image0_paths) == len(val_camera_paths))
    assert(len(val_image0_paths) == len(val_ground_truth_paths))

    # Load ground truth depths
    val_ground_truths = []
    for path in val_ground_truth_paths:
        val_ground_truths.append(np.load(path))

    train_dataloader = torch.utils.data.DataLoader(
        datasets.ImagePairCameraDataset(
            train_image0_paths,
            train_image1_paths,
            train_camera_paths,
            shape=(n_height, n_width),
            augment=use_augment,
            load_triplet=load_triplet,
            use_resize=use_resize,
            training=True),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=False)

    val_dataloader = torch.utils.data.DataLoader(
        datasets.ImagePairCameraDataset(
            val_image0_paths,
            val_image0_paths,
            val_camera_paths,
            shape=(n_height, n_width),
            augment=False,
            use_resize=use_resize,
            training=False),
        batch_size=1,
        shuffle=False,
        num_workers=n_thread,
        drop_last=False)

    # Build network
    model = MonodepthModel(
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        activation_func=activation_func,
        n_pyramid=n_pyramid,
        scale_factor=scale_factor,
        device=device)
    train_summary = SummaryWriter(event_path)
    parameters = model.parameters()
    n_param = sum(p.numel() for p in parameters)

    # Start training
    model.train()

    log('Network settings:', log_path)
    log('n_batch=%d  n_height=%d  n_width=%d  n_param=%d' %
        (n_batch, n_height, n_width, n_param), log_path)
    log('encoder_type=%s  decoder_type=%s  activation_func=%s  n_pyramid=%d' %
        (encoder_type, decoder_type, activation_func, n_pyramid), log_path)
    log('', log_path)

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}  use_augment={}  use_resize={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step, use_augment, use_resize),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{}({}):{}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), le, v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    log('w_color=%.2f  w_ssim=%.2f  w_smoothness=%.2f  w_left_right=%.2f' %
        (w_color, w_ssim, w_smoothness, w_left_right), log_path)
    log('', log_path)

    log('Depth range settings:', log_path)
    log('scale_factor=%.2f' %
        (scale_factor), log_path)
    log('min_evaluate_depth=%.2f  max_evaluate_depth=%.2f' %
        (min_evaluate_depth, max_evaluate_depth), log_path)
    log('Checkpoint settings:', log_path)
    log('depth_model_checkpoint_path=%s' % checkpoint_path, log_path)

    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    optimizer = torch.optim.Adam([
        {
            'params' : parameters,
            'weight_decay' : 0.0
        }],
        lr=learning_rate)

    # Start training
    train_step = 0
    time_start = time.time()

    log('Begin training...', log_path)

    for epoch in range(1, learning_schedule[-1] + 1):

        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

        for train_image0, train_image1, train_camera in train_dataloader:

            train_step = train_step+1
            # Fetch data
            if device.type == 'cuda':
                train_image0 = train_image0.cuda()
                train_image1 = train_image1.cuda()
                train_camera = train_camera.cuda()

            # Forward through the network
            model.forward(train_image0, train_camera)

            # Compute loss function
            loss = model.compute_loss(train_image0, train_image1,
                    w_color=w_color,
                    w_ssim=w_ssim,
                    w_smoothness=w_smoothness,
                    w_left_right=w_left_right)

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_summary) == 0:
                model.log_summary(
                    summary_writer=train_summary,
                    step=train_step)

            # Log results and save checkpoints
            if (train_step % n_checkpoint) == 0:
                time_elapse = (time.time()-time_start)/3600
                time_remain = (n_train_step-train_step)*time_elapse/train_step
                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain), log_path)

                # Save checkpoints
                torch.save({
                    'train_step': train_step,
                    'model_state_dict': model.encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, encoder_checkpoint_path.format(train_step))
                torch.save({
                    'train_step': train_step,
                    'model_state_dict': model.decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, decoder_checkpoint_path.format(train_step))

                # Switch to eval mode and perform validation
                model.eval()
                with torch.no_grad():
                    validate(
                        model,
                        dataloader=val_dataloader,
                        ground_truths=val_ground_truths,
                        min_evaluate_depth=min_evaluate_depth,
                        max_evaluate_depth=max_evaluate_depth,
                        device=device,
                        log_path=log_path)
                # Switch back to training
                model.train()

    # Save checkpoints and close summary
    train_summary.close()
    torch.save({
        'train_step': train_step,
        'model_state_dict': model.encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, encoder_checkpoint_path.format(train_step))
    torch.save({
        'train_step': train_step,
        'model_state_dict': model.decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, decoder_checkpoint_path.format(train_step))

def validate(model,
             dataloader,
             ground_truths,
             # Depth evaluation range
             min_evaluate_depth,
             max_evaluate_depth,
             device,
             log_path):

    n_sample = len(dataloader)
    eigen_absrel = np.zeros(n_sample)
    eigen_sqrel = np.zeros(n_sample)
    eigen_rmse = np.zeros(n_sample)
    eigen_logrmse = np.zeros(n_sample)
    eigen_a1 = np.zeros(n_sample)
    eigen_a2 = np.zeros(n_sample)
    eigen_a3 = np.zeros(n_sample)
    for idx, ((image0, image1, camera), ground_truth) in enumerate(zip(dataloader, ground_truths)):
        if device.type == 'cuda':
            image0 = image0.cuda()
            image1 = image1.cuda()
            camera = camera.cuda()
        o_height, o_width = ground_truth.shape[0:2]

        # Forward through network
        _, _, depth0_output, _ = model.forward(image0, camera)

        # Convert to numpy to validate
        if device.type == 'cuda':
            depth0_output = np.squeeze(depth0_output.cpu().numpy())
        else:
            depth0_output = np.squeeze(depth0_output.numpy())

        # Resize to original to compute Eigen metrics
        depth0_output = cv2.resize(depth0_output, (o_width, o_height))

        mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)

        crop = np.array([
            0.40810811*o_height, 0.99189189*o_height,
            0.03594771*o_width,  0.96405229*o_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1.0
        mask = np.logical_and(mask, crop_mask)

        depth0_output = depth0_output[mask]
        ground_truth = ground_truth[mask]

        # Cap depth between min and max evaluation depth
        depth0_output[depth0_output < min_evaluate_depth] = min_evaluate_depth
        depth0_output[depth0_output > max_evaluate_depth] = max_evaluate_depth

        # Compute Eigen benchmark metrics
        eigen_absrel[idx] = eval_utils.abs_rel_err(depth0_output, ground_truth)
        eigen_sqrel[idx] = eval_utils.sq_rel_err(depth0_output, ground_truth)
        eigen_rmse[idx] = eval_utils.root_mean_sq_err(depth0_output, ground_truth)
        eigen_logrmse[idx] = eval_utils.log_root_mean_sq_err(depth0_output, ground_truth)
        eigen_a1[idx] = eval_utils.thresh_ratio_err(depth0_output, ground_truth, 1.25)
        eigen_a2[idx] = eval_utils.thresh_ratio_err(depth0_output, ground_truth, 1.25**2)
        eigen_a3[idx] = eval_utils.thresh_ratio_err(depth0_output, ground_truth, 1.25**3)

    eigen_absrel = np.mean(eigen_absrel)
    eigen_sqrel = np.mean(eigen_sqrel)
    eigen_rmse = np.mean(eigen_rmse)
    eigen_logrmse = np.mean(eigen_logrmse)
    eigen_a1 = np.mean(eigen_a1)
    eigen_a2 = np.mean(eigen_a2)
    eigen_a3 = np.mean(eigen_a3)
    log('Validation results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'AbsRel', 'SqRel', 'RMSE', 'logRMSE', 'a1', 'a2', 'a3'),
        log_path)
    log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        eigen_absrel, eigen_sqrel, eigen_rmse, eigen_logrmse, eigen_a1, eigen_a2, eigen_a3),
        log_path)
