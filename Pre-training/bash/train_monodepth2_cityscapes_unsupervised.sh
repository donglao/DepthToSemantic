export CUDA_VISIBLE_DEVICES=0

python src/train_monodepth2.py \
--train_images_path \
    training/cityscapes/cityscapes_train_images_all.txt \
--train_intrinsics_path \
    training/cityscapes/cityscapes_train_intrinsics_all.txt \
--n_batch 8 \
--n_height 320 \
--n_width 800 \
--normalized_image_range 0 1 \
--encoder_type resnet18 pretrained \
--network_modules pose \
--scale_factor_depth 5.4 \
--min_predict_depth 0.1 \
--max_predict_depth 100.0 \
--learning_rates 5e-5 \
--learning_schedule 50 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_brightness 0.80 1.20 \
--augmentation_random_contrast 0.80 1.20 \
--augmentation_random_saturation 0.80 1.20 \
--w_photometric 1.00 \
--w_smoothness 6.00 \
--checkpoint_path \
    trained_monodepth2/cityscapes/unsupervised/res18pt_8x320x800_pose_nomask_ph100_sm600_lr0-5e5_50_bri080-120_con080-120_sat080-120 \
--n_checkpoint 1000 \
--n_summary 1000 \
--n_summary_display 4 \
--validation_start_step 0 \
--device gpu \
--n_thread 8
