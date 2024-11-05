export CUDA_VISIBLE_DEVICES=0

python src/train_reconstruction.py \
--train_images_left_path \
    training/kitti/kitti_train_nonstatic_images_left.txt \
--train_images_right_path \
    training/kitti/kitti_train_nonstatic_images_right.txt \
--n_batch 8 \
--n_height 192 \
--n_width 640 \
--normalized_image_range 0 1 \
--encoder_type resnet50 \
--learning_rates 1e-4 5e-5 \
--learning_schedule 25 50 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_swap \
--augmentation_random_brightness -1 -1 \
--augmentation_random_contrast -1 -1 \
--augmentation_random_saturation -1 -1 \
--remove_percent_range 0.001 0.01 \
--remove_patch_size 7 7 \
--w_reconstruction 1.00 \
--checkpoint_path \
    trained_reconstruction/res50_8x192x640_rec100_rmpct1e3-1e2_rmsize7x7_lr0-1e4_25-5e5_50_aug0-100_50_swap_bri000-000_con000-000_sat000-000 \
--n_checkpoint 5000 \
--n_summary 5000 \
--n_summary_display 4 \
--device gpu \
--n_thread 8
