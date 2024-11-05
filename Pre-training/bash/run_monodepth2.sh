export CUDA_VISIBLE_DEVICES=0

python src/run_monodepth2.py \
--image_path \
    testing/kitti_eigen_test_image0.txt \
--ground_truth_path \
    testing/kitti_eigen_test_ground_truth.txt \
--n_height 192 \
--n_width 640 \
--normalized_image_range 0 1 \
--median_scale_depth \
--min_evaluate_depth 1e-3 \
--max_evaluate_depth 80.0 \
--monodepth2_encoder_restore_path \
    pretrained_models/monodepth2/mono_stereo_640x192/encoder.pth \
--monodepth2_decoder_restore_path \
    pretrained_models/monodepth2/mono_stereo_640x192/depth.pth \
--posenet_model_restore_path \
    pretrained_models/posenet/pose_model-420000.pth \
--output_dirpath \
    pretrained_models/monodepth2/mono_stereo_640x192/evaluation_results \
--tasks depth \
--device gpu

python src/run_monodepth2.py \
--image_path \
    training/kitti/kitti_train_nonstatic_images_left.txt \
--intrinsics_path \
    training/kitti/kitti_train_nonstatic_intrinsics_left.txt \
--n_height 192 \
--n_width 640 \
--load_image_triplets \
--normalized_image_range 0 1 \
--max_evaluate_sample 5000 \
--median_scale_depth \
--min_evaluate_depth 1e-3 \
--max_evaluate_depth 80.0 \
--monodepth2_encoder_restore_path \
    pretrained_models/monodepth2/mono_stereo_640x192/encoder.pth \
--monodepth2_decoder_restore_path \
    pretrained_models/monodepth2/mono_stereo_640x192/depth.pth \
--posenet_model_restore_path \
    pretrained_models/posenet/pose_model-420000.pth \
--output_dirpath \
    pretrained_models/monodepth2/mono_stereo_640x192/evaluation_results \
--tasks rigid_flow \
--device gpu
