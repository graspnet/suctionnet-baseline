CUDA_VISIBLE_DEVICES=0 python inference.py --model deeplabv3plus_resnet101 \
--checkpoint_path /DATA2/Benchmark/suction/models/log_kinectV6/checkpoint_90 \
--split test_seen \
--camera realsense \
--dataset_root /DATA2/Benchmark/graspnet \
--save_dir /DATA2/Benchmark/suction/inference_results/deeplabV3plus_v6_epoch90 \
--save_visu

