# CS6140 Project - Lane Detection Using K-Lane LiDAR Point Cloud Dataset. 
---
This project involves lane detection using the KAIST-LANE (K-Lane) dataset and builds upon the LLDN-GFC architecture. We have made several modifications to create a **lightweight version** of the **LLDN-GFC** architecture for lane detection using the **KAIST-LANE (K-Lane) dataset**. The goal of this project is to improve the **efficiency** of the model without compromising accuracy. We have made several modifications to the original architecture to reduce the computational burden while not compromising too much performance in lane detection tasks.

## Modifications

We made the following modifications to the original **LLDN-GFC** architecture:

1. **Transformer Replacement**:
   - Replaced the ResNet34 encoder used for the **BEV encoder** with a new version in `net/pcencoder/projector_dense.py` based on DenseNet121.
   
2. **Backbone Replacement**:
   - Replaced the **Attention** and **FeedForward** modules of the transformer with our **Global Feature Correlator** backbone located in `baseline/models/backbone/ml_transformer.py`.

These changes were designed to enhance the model's lane detection capabilities while reducing overall parameter count and model complexity.


## Dataset
We used the **KAIST-LANE (K-Lane)** dataset for training and testing the model. This dataset provides LiDAR point cloud data, which is essential for detecting lanes on road surfaces.


## Running Instructions

### Prerequisites

We tested the K-Lane detection frameworks on the following environment:
* Python 3.7 / 3.8
* Ubuntu 18.04
* Torch 1.7.1
* CUDA 11.2


### Requirements

1. Clone the repository
```
git clone ...
```

2. Install the dependencies using requirements.txt
* Install the commented-out packages manually
```
pip3 install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```
* Install the rest of the package in batch
```
pip3 install -r requirements.txt
```

### Setup the Dataset

* You can get a dataset using the Google Drive Urls below:

1. <a href="https://drive.google.com/drive/folders/1NE9DT8wZSbkL95Z9bQWm22EjGRb9SsYM?usp=sharing" title="K-Lane Dataset">link for download seq_01 to 04</a> 
2. <a href="https://drive.google.com/drive/folders/1YBz5iaDLAcTH5IOjpaMrLt2iFu2m_Ui_?usp=sharing" title="K-Lane Dataset">link for download seq_05 to 12</a>
3. <a href="https://drive.google.com/drive/folders/1dUIYuOhnKwM1Uf5Q-nC0X0piCZFL8zCQ?usp=sharing" title="K-Lane Dataset">link for download seq_13 to 14</a>
4. <a href="https://drive.google.com/drive/folders/12aLITzR_tE8eVi-Q4OWoomX9VR3Olea7?usp=sharing" title="K-Lane Dataset">link for download seq_15, test, and     description</a>

* After all files are downloaded, please arrange the workspace directory with the following structure (datasets must be unzipped):
```
KLaneFrameworks
├── annot_tool
├── baseline 
├── configs
      ├── config_vis.py
      ├── Proj28_GFC-T3_RowRef_82_73.py
      ├── Proj28_GFC-T3_RowRef_82_73.pth
├── data
      ├── KLane
            ├── test
            ├── train
                  ├── seq_1
                  :
                  ├── seq_15
            ├── description_frames_test.txt
            ├── description_test_lightcurve.txt
├── logs
```
2. **Download Pre-trained Models**:

   * We have provided two pre-trained models:
   - `ml_base.pth`: The training result of the baseline model (without our modifications).
   - `ml_curr_best.pth`: The training result of our model (with the modifications mentioned above).

   * You can download these pretrained models from our Google Drive [Google Drive link](https://drive.google.com/drive/u/0/folders/1N3_2hsOI_295krR_cEMITu_RAbtyVbCJ). Then put them in `\configs` folder.

### Training & Testing
* Configure the config files in `\configs` folder. In Proj28_GFC-T3_RowRef_82_73.py, make sure `epochs = 20` and `lr = 0.0001` before running train_gpu_0.py, OR `epochs = 1` and `lr = 0.0003` before running train_gpu_full and validate_gpu_0.py

* To train the model, prepare the total dataset, and in train_gpu_0.py, make sure `path_config = './configs/Proj28_GFC-T3_RowRef_82_73.py'` is uncommented for our model, OR make sure `path_config = './configs/baseline_config.py'` is uncommented for base model, and run
```
python3 train_gpu_0.py ...
```
* Then, in train_gpu_full.py, make sure `path_config = './configs/Proj28_GFC-T3_RowRef_82_73.py'` and `ckpt_path = './configs/ml_curr_best.pth'` are uncommented for our model, OR make sure `ckpt_path = './configs/ml_base.pth'` and `path_config = './configs/baseline_config.py'` are uncommented for base model, and run
```
python3 train_gpu_full ...
```
* To test from a pretrained model, and in validate_gpu_0.py, make sure `path_config = './configs/Proj28_GFC-T3_RowRef_82_73.py'` and `path_ckpt = './configs/ml_curr_best.pth'` are uncommented for our model, OR make sure `path_config = './configs/baseline_config.py'` and `path_ckpt = './configs/ml_curr_best_proj28.pth'` are uncommented for base model, and run
```
python3 validate_gpu_0.py ...
```

## License
`K-Lane` is released under the Apache-2.0 license.

## Acknowledgement
The K-Lane benchmark is contributed by [Dong-Hee Paek](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=5), [Kevin Tirta Wijaya](https://www.ktirta.xyz/), [Dong-In Kim](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=13), [Min-Hyeok Sun](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=14), advised by [Seung-Hyun Kong](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_1).

We thank the maintainers of the following projects that enable us to develop `K-Lane`:
[`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) by MMLAB, [`TuRoad`](https://github.com/Turoad/lanedet) bu TuZheng.

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. 2021R1A2C3008370).
