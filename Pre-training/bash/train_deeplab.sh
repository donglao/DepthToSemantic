export CUDA_VISIBLE_DEVICES=0

python src/train_deeplab.py \
--train_images_left_path \
    training/kitti/kitti_train_nonstatic_images_left.txt \
--train_images_right_path \
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
--val_focal_length_baseline_path \
    testing/kitti/kitti_test_focal_length_baseline.txt \
--val_ground_truth_path \
    testing/kitti/kitti_test_depth.txt \
--n_batch 8 \
--n_height 192 \
--n_width 640 \
--normalized_image_range 0 1 \
--encoder_type resnet50 \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--learning_rates 1.0e-4 5e-5 \
--learning_schedule 25 50 \
--augmentation_probabilities 0.00 \
--augmentation_schedule -1 \
--augmentation_random_brightness -1 -1 \
--augmentation_random_contrast -1 -1 \
--augmentation_random_saturation -1 -1 \
--supervision_type monocular stereo \
--w_color 0.15 \
--w_structure 0.85 \
--w_smoothness 1.00 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 1e-3 \
--max_evaluate_depth 80.0 \
--checkpoint_path trained_deeplab/res50ms1_8x192x640_co15_st85_sm100_lr0-1e4_25-5e-5_50_aug0-000_50_swapno_bri000-000_con000-000_sat000-000 \
--n_checkpoint 5000 \
--n_summary 10000 \
--n_summary_display 4 \
--validation_start_step 5000 \
--device gpu \
--n_thread 4
