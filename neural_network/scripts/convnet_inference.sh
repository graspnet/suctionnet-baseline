CUDA_VISIBLE_DEVICES=5 python inference.py --model convnet_resnet101 \
--checkpoint_path ~/codes/graspnet/network/log_convnet_kinect/checkpoint_40 \
--split test_seen \
--camera realsense \
--dataset_root /DATA2/Benchmark/graspnet \
--save_dir /DATA2/Benchmark/suction/inference_results/convnet \
--save_visu 
