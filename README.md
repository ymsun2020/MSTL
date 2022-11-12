# Multi-source Templates Learning for Real-time Aerial Object Tracking

This is the official code for the paper "Multi-source Templates Learning for Real-time Aerial Object Tracking".In this work, we present an efficient Aerial Object Tracking method via Multi-source Templates named MSTL. 

## Highlights

### Real-Time Speed.

Our tracker can run **~200fps on GPU, ~100fps on CPU, and ~20** on Nvidia jetson-xavier-nx platform. After using ONNX to accelerate, the speed can reach **509** fps on GPU.



### Demo

![demo_gif](demo_gif.gif)

## Quick Start

### Environment Preparing

```
python 3.7.3
pytorch 1.11.0
opencv-python 4.5.5.64
```

### Training

First, you need to set paths for training datasets in lib/train/admin/local.py.

Then, run the following commands for training.

```bash
python lib/train/run_training.py
```

### Evaluation

First, you need to set paths for this project in lib/test/evaluation/local.py.

Then put the checkpoint file to ./tracking/networks

Finally, run the following commands for evaluation on four datasets.

- UAV123

```bash
python tracking/test.py MSTL MSTL --dataset uav
```

- UAV20L

```bash
python tracking/test.py MSTL MSTL --dataset uavl
```

- UAV@10fps

```
python tracking/test.py MSTL MSTL --dataset uav10
```

- UAV -x

```
python tracking/test.py MSTL MSTL --dataset uavd
```



### Trained model and Row results

The trained models and the raw tracking results are provided


## Acknowledgement

This is a modified version of the python framework [PyTracking](https://github.com/visionml/pytracking)  based on **Pytorch**. We would like to thank their authors for providing great frameworks and toolkits.

