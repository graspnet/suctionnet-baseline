CUDA_VISIBLE_DEVICES=1 python train.py \
--model deeplabv3plus_resnet101_depth \
--camera realsense \
--log_dir /DATA2/Benchmark/suction/models/log_kinect_depth_test \
--data_root /DATA2/Benchmark/graspnet \
--label_root /DATA2/Benchmark/suction/graspnet_label \
--batch_size 8
