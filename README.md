# Baseline Methods for SuctionNet-1Billion

Baseline methods in RA-L paper "SuctionNet-1Billion:  A  Large-Scale  Benchmark  for  Suction  Grasping" 

![Image Title](https://github.com/graspnet/suctionnet-baseline/blob/master/framework3.jpg)

## Dataset

Download data and labels from our [SuctionNet webpage](https://graspnet.net/suction).

## Environment

The code has been tested with `CUDA 10.1` and `pytorch 1.4.0` on ubuntu `16.04`

## Training Prerequisites

To train the networks, you need additional labels including 2D mappings of seal labels, bounding boxes of objects and object centers.

change the directory to `neural_network`

```
cd neural_network
```

To generate the 2D mappings of seal label, run the following command:

```
python score_mapping.py \
--dataset_root /path/to/SuctionNet/dataset \
--saveroot /path/to/save/additional/labels \
--camera realsense \ # kinect or realsense
--sigma 4 \	# sigmal of the 2D gaussian kernel to map the score
--pool_size 10 \ # number of cpu threads to use
--save_visu # whether to save visualizations
```

or modify [scripts/score_mapping.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/score_mapping.sh) and run `sh scripts/score_mapping.sh`.

To get bounding boxes and centers of the objects, run the following command:

```
python cal_center_bbox.py \
--dataset_root /path/to/SuctionNet/dataset \
--saveroot /path/to/save/additional/labels \
--camera realsense \ # kinect or realsense
--pool_size 10 \ # number of cpu threads to use
--save_visu # whether to save visualizations
```

or modify the [scripts/cal_center_bbox.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/cal_center_bbox.sh) and run `sh scripts/cal_center_bbox.sh`.

Please make sure the `--saveroot` args are the same for the above two commands.

Note that the 2D mappings of seal label can take up to `177 G` disk space. We save them in advance to make the training process more efficient. You may also modify the mapping to an online version but this will be much slower for training.

## Usage

### Neural Networks

Change the directory to `neural_network`:

```
cd neural_network
```

For training, use the following command:

```
python train.py \
--model model_name \ 
--camera realsense \ # realsense or kinect
--log_dir /path/to/save/the/model/weights \
--data_root /path/to/SuctionNet/dataset \
--label_root /path/to/the/additional/labels \
--batch_size 8
```

or modify [scripts/deeplabv3plus_train.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_train.sh), [scripts/deeplabv3plus_train_depth.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_inference_depth.sh), [scripts/convnet_train.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/convnet_train.sh) for training our RGB-D model, depth model and fully convolutional network (FCN) model.

For inference, use the following command: 

```
python inference.py \
--model model_name \
--checkpoint_path /path/to/the/saved/model/weights \
--split test_seen \ # can be test, test_seen, test_similar, test_novel
--camera realsense \ # realsense or kinect
--dataset_root /path/to/SuctionNet/dataset \
--save_dir /path/to/save/the/inference/results \
--save_visu # whether to save the visualizations
```

or modify [scripts/deeplabv3plus_inference.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_inference.sh), [scripts/deeplabv3plus_inference_depth.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/deeplabv3plus_inference_depth.sh), [scripts/convnet_inference.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/neural_network/scripts/convnet_inference.sh) to inference with our RGB-D model, depth model and fully convolutional network (FCN) model.

### Normal STD

Change the directory to `normal_std` by:

```
cd normal_std
```

This method does not need to train, you can directly inference with the following command:

```
python inference.py 
--split test_seen \ # can be test, test_seen, test_similar, test_novel
--camera realsense \ # realsense or kinect
--save_root /path/to/save/the/inference/results \
--dataset_root /path/to/SuctionNet/dataset \
--save_visu
```

or modify [inference.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/normal_std/inference.sh) and run `sh inference.sh`

## Pre-trained Models

### RGB-D Models

We provide models including [our model for realsense](https://drive.google.com/file/d/18TbctdhpNXEKLYDWFzI9cT1Wnhe-tn9h/view?usp=sharing), [our model for kinect](https://drive.google.com/file/d/1gOz_KmIugBGUtpcyHAgYO01T0h5ZqOl9/view?usp=sharing), [Fully Conv Net for realsense](https://drive.google.com/file/d/1hgYYIvw5Xy-r5C8IitKizswtuMV_EqPP/view?usp=sharing) ,[Fully Conv Net for kinect](https://drive.google.com/file/d/1A6K5EmItBuDaxrWyz5g8zSHY5Kw1_NnX/view?usp=sharing).

### Depth Models

Our models only taking in depth images are also provided [for realsense](https://drive.google.com/file/d/1q2W2AV663PNT4_TYo5zZtYxjenZJ7GAb/view?usp=sharing) and [for kinect](https://drive.google.com/file/d/1mAzFC9dlEDBuoHQp7JGTcTkKGSwFnVth/view?usp=sharing).

## Citation

if you find our work useful, please cite

```
@ARTICLE{suctionnet,
  author={Cao, Hanwen and Fang, Hao-Shu and Liu, Wenhai and Lu, Cewu},
  journal={IEEE Robotics and Automation Letters}, 
  title={SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping}, 
  year={2021},
  volume={6},
  number={4},
  pages={8718-8725},
  doi={10.1109/LRA.2021.3115406}}
```

