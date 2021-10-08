CUDA_VISIBLE_DEVICES=0 python train.py \
--model deeplabv3plus_resnet101 \
--camera realsense \
--log_dir /DATA2/Benchmark/suction/models/log_kinectV6_test \
--data_root /DATA2/Benchmark/graspnet \
--label_root /ssd1/hanwen/grasping/graspnet_label \
--batch_size 8
