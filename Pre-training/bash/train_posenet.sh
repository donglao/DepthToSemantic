export CUDA_VISIBLE_DEVICES=0

python src/train_posenet.py \
--train_images_left_path \
    training/kitti/kitti_train_nonstatic_images_left.txt \
--train_images_right_path \
    training/kitti/kitti_train_nonstatic_images_right.txt \
--train_intrinsics_left_path \
    training/kitti/kitti_train_nonstatic_intrinsics_left.txt \
--train_intrinsics_right_path \
    training/kitti/kitti_train_nonstatic_intrinsics_right.txt \
--n_batch 8 \
--n_height 192 \
--n_width 640 \
--normalized_image_range 0 1 \
--encoder_type resnet18 \
--weight_initializer kaiming_uniform \
--activation_func relu \
--use_batch_norm \
--learning_rates 1.0e-4 5e-5 \
--learning_schedule 25 50 \
--augmentation_probabilities 0.00 \
--augmentation_schedule -1 \
--augmentation_random_brightness -1 -1 \
--augmentation_random_contrast -1 -1 \
--augmentation_random_saturation -1 -1 \
--w_color 0.15 \
--w_structure 0.85 \
--w_weight_decay_pose 0.00 \
--checkpoint_path trained_posenet/res18_bn_relu_kuni_8x192x640_co15_st85_lr0-1e4_25-5e-5_50_aug0-000_50_swapno_bri000-000_con000-000_sat000-000 \
--n_checkpoint 5000 \
--n_summary 5000 \
--n_summary_display 4 \
--monodepth2_encoder_restore_path \
    pretrained_models/monodepth2/mono_stereo_640x192/encoder.pth \
--monodepth2_decoder_restore_path \
    pretrained_models/monodepth2/mono_stereo_640x192/depth.pth \
--device gpu \
--n_thread 8
