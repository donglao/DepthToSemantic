export CUDA_VISIBLE_DEVICES=1

python src/train_optical_flow.py \
--train_images_left_path \
    training/kitti/kitti_train_nonstatic_images_left.txt \
--train_images_right_path \
    training/kitti/kitti_train_nonstatic_images_right.txt \
--n_batch 8 \
--n_height 192 \
--n_width 640 \
--normalized_image_range 0 1 \
--encoder_type resnet18 pretrained \
--learning_rates 5e-5 \
--learning_schedule 50 \
--augmentation_probabilities 0.00 \
--augmentation_schedule -1 \
--augmentation_random_swap \
--augmentation_random_brightness -1 -1 \
--augmentation_random_contrast -1 -1 \
--augmentation_random_saturation -1 -1 \
--w_color 0.15 \
--w_structure 0.85 \
--w_smoothness 2.00 \
--checkpoint_path trained_flow/res18pt_8x192x640_co15_st85_sm200_lr0-5e5_25-5e5_50_aug0-000_50_swap_bri000-000_con000-000_sat000-000 \
--n_checkpoint 1000 \
--n_summary 1000 \
--n_summary_display 4 \
--validation_start_step 5000000 \
--device gpu \
--n_thread 8
