export CUDA_VISIBLE_DEVICES=0

python src/train_depthnet.py \
--train_image_left_path \
    training/kitti/kitti_train_nonstatic_images_left.txt \
--train_image_right_path \
    training/kitti/kitti_train_nonstatic_images_right.txt \
--train_intrinsics_left_path \
    training/kitti/kitti_train_nonstatic_intrinsics_left.txt \
--train_intrinsics_right_path \
    training/kitti/kitti_train_nonstatic_intrinsics_right.txt \
--train_focal_length_baseline_left_path \
    training/kitti/kitti_train_nonstatic_focal_length_baseline_left.txt \
--train_focal_length_baseline_right_path \
    training/kitti/kitti_train_nonstatic_focal_length_baseline_right.txt \
--val_image_path \
    testing/kitti/kitti_test_image.txt \
--val_ground_truth_path \
    testing/kitti/kitti_test_depth.txt \
--n_batch 8 \
--n_height 320 \
--n_width 768 \
--input_channels 3 \
--normalized_image_range 0 1 \
--encoder_type resnet18 \
--n_filters_encoder 64 128 256 512 512 \
--decoder_type multiscale \
--n_resolution_decoder_output 1 \
--n_filters_decoder 256 128 128 64 32 \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--weight_initializer xavier_normal \
--activation_func leaky_relu \
--use_batch_norm \
--learning_rates 5e-5 2e-5 \
--learning_schedule 50 60 \
--augmentation_probabilities 0.00 \
--augmentation_schedule -1 \
--augmentation_random_crop_type horizontal bottom anchored \
--augmentation_random_flip_type none \
--augmentation_random_brightness -1 -1 \
--augmentation_random_contrast -1 -1 \
--augmentation_random_saturation -1 -1 \
--supervision_type monocular \
--w_color 0.20 \
--w_structure 0.80 \
--w_smoothness 0.10 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 1e-8 \
--max_evaluate_depth 100.0 \
--checkpoint_path trained_depthnet/res18ms1_8x320x768_co020_st080_sm010_lr0-1e4_25-5e5_50_aug0-000_50_hbacrop \
--n_checkpoint 5000 \
--n_summary 10000 \
--n_summary_display 4 \
--validation_start_step 5000 \
--device gpu \
--n_thread 8
