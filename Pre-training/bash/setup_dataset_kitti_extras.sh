export CUDA_VISIBLE_DEVICES=0

python setup/setup_dataset_kitti_extras.py \
--n_height 192 \
--n_width 640 \
--normalized_image_range 0 1 \
--monodepth2_encoder_restore_path \
    pretrained_models/monodepth2/mono_stereo_640x192/encoder.pth \
--monodepth2_decoder_restore_path \
    pretrained_models/monodepth2/mono_stereo_640x192/depth.pth \
--posenet_model_restore_path \
    pretrained_models/posenet/pose_model-420000.pth \
