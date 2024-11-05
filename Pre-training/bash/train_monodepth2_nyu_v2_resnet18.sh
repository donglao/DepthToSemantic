export CUDA_VISIBLE_DEVICES=0

python src/train_monodepth2.py \
--train_images_path \
    training/nyu_v2/nyu_v2_train_image.txt \
--train_intrinsics_path \
    training/nyu_v2/nyu_v2_train_intrinsics.txt \
--train_ground_truth_path \
    training/nyu_v2/nyu_v2_train_ground_truth.txt \
--n_batch 8 \
--n_height 384 \
--n_width 448 \
--normalized_image_range 0 1 \
--encoder_type resnet18 pretrained \
--network_modules depth \
--scale_factor_depth 5.4 \
--min_predict_depth 0.1 \
--max_predict_depth 10.0 \
--learning_rates 1.0e-4 5e-5 \
--learning_schedule 5 10 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_brightness 0.80 1.20 \
--augmentation_random_contrast 0.80 1.20 \
--augmentation_random_saturation 0.80 1.20 \
--w_color 0.15 \
--w_structure 0.85 \
--w_smoothness 1.00 \
--checkpoint_path trained_monodepth2/nyu_v2/res18pt_8x384x448_lr0-1e4_5-5e-5_10_aug0-100_10_bri080-120_con080-120_sat080-120 \
--n_checkpoint 5000 \
--n_summary 5000 \
--n_summary_display 4 \
--validation_start_step 5000 \
--device gpu \
--n_thread 8
