export CUDA_VISIBLE_DEVICES=0,1

python src/train_deeplab.py \
--train_images_left_path \
    training/cityscapes/cityscapes_train_images_left.txt \
--train_images_right_path \
    training/cityscapes/cityscapes_train_images_right.txt \
--train_intrinsics_left_path \
    training/cityscapes/cityscapes_train_intrinsics.txt \
--train_intrinsics_right_path \
    training/cityscapes/cityscapes_train_intrinsics.txt \
--train_ground_truth_path \
    training/cityscapes/cityscapes_train_depth.txt \
--val_image_path \
    testing/kitti/kitti_test_image.txt \
--val_focal_length_baseline_path \
    testing/kitti/kitti_test_focal_length_baseline.txt \
--val_ground_truth_path \
    testing/kitti/kitti_test_depth.txt \
--n_batch 20 \
--n_height 576 \
--n_width 640 \
--normalized_image_range 0.485 0.456 0.406 0.229 0.224 0.225 \
--encoder_type resnet50 pretrained \
--min_predict_depth 1.0 \
--max_predict_depth 100.0 \
--learning_rates 1e-4 5e-4 2e-4 1e-4 \
--learning_schedule 5 40 70 80 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_brightness 0.5 1.5 \
--augmentation_random_contrast 0.5 1.5 \
--augmentation_random_saturation 0.5 1.5 \
--augmentation_random_crop_to_shape 512 512 \
--augmentation_random_flip_type horizontal \
--augmentation_random_brightness 0.50 1.50 \
--augmentation_random_contrast 0.50 1.50 \
--augmentation_random_gamma -1 -1 \
--augmentation_random_hue -1 -1 \
--augmentation_random_saturation 0.50 1.50 \
--augmentation_random_crop_to_shape 512 512 \
--augmentation_random_flip_type horizontal vertical \
--augmentation_random_rotate_max 10 \
--augmentation_random_resize_and_crop 1.00 1.50 \
--augmentation_random_crop_and_pad 0.90 1.00 \
--supervision_type ground_truth \
--w_color 0.15 \
--w_structure 0.85 \
--w_smoothness 1.00 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 1e-3 \
--max_evaluate_depth 80.0 \
--checkpoint_path trained_deeplab/cityscapes/res50pt_16x576x640-512x512_lr0-1e4-5-5e4_40-2e4_70-1e4_80_bri050-150_con050-150_sat050-150_hvflip_rot10_resize100-150_crop_pad90-100 \
--n_checkpoint 5000 \
--n_summary 5000 \
--n_summary_display 4 \
--validation_start_step 5000 \
--device gpu \
--n_thread 8
