# cs6140 Project - Lane Detection using K-Lane LiDAR dataset. 
---

# K-Lane Detection Frameworks
This is the documentation for how to use our detection frameworks with K-Lane dataset.
We tested the K-Lane detection frameworks on the following environment:
* Python 3.7 / 3.8
* Ubuntu 18.04
* Torch 1.7.1
* CUDA 11.2

## Preparing the Dataset
You can get a dataset using the Google Drive Urls below:

1. <a href="https://drive.google.com/drive/folders/1NE9DT8wZSbkL95Z9bQWm22EjGRb9SsYM?usp=sharing" title="K-Lane Dataset">link for download seq_01 to 04</a> 
2. <a href="https://drive.google.com/drive/folders/1YBz5iaDLAcTH5IOjpaMrLt2iFu2m_Ui_?usp=sharing" title="K-Lane Dataset">link for download seq_05 to 12</a>
3. <a href="https://drive.google.com/drive/folders/1dUIYuOhnKwM1Uf5Q-nC0X0piCZFL8zCQ?usp=sharing" title="K-Lane Dataset">link for download seq_13 to 14</a>
4. <a href="https://drive.google.com/drive/folders/12aLITzR_tE8eVi-Q4OWoomX9VR3Olea7?usp=sharing" title="K-Lane Dataset">link for download seq_15, test, and description</a>

After all files are downloaded, please arrange the workspace directory with the following structure:
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


## Requirements

1. Clone the repository
```
git clone ...
```

2. Install the dependencies using requirements.txt
   1. Install the commented-out packages manually
      ```
      pip3 install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
      ```
   2. Install the rest of the package in batch
      ```
      pip3 install -r requirements.txt
      ```

## Training & Testing
* To train the model, prepare the total dataset and run
```
python3 train_gpu_0.py ...
```
https://drive.google.com/drive/u/0/folders/1N3_2hsOI_295krR_cEMITu_RAbtyVbCJ
* To test from a pretrained model (e.g., ml_curr_best.pth or ml_base.pth), download the pretrained model from our Google Drive <a href="https://drive.google.com/drive/u/0/folders/1N3_2hsOI_295krR_cEMITu_RAbtyVbCJ" title="pth">Model</a> and run
```
python3 validate_gpu_0.py ...
```


# Lane Detection with LiDAR Point Cloud

This project involves lane detection using the **KAIST-LANE (K-Lane) dataset** and builds upon the **LLDN-GFC** architecture. We have made several modifications to improve the model's performance, specifically by replacing parts of the architecture for better feature extraction and lane detection accuracy.

## Modifications

We made the following modifications to the original **LLDN-GFC** architecture:

1. **Transformer Replacement**:
   - Replaced the transformer used for the **BEV encoder** with a new version from `net/pcencoder/projector_dense.py`.
   
2. **Backbone Replacement**:
   - Replaced the **Attention** and **FeedForward** modules of the transformer with our **Global Feature Correlator** backbone located in `baseline/models/backbone/ml_transformer.py`.

These changes were designed to enhance the model's lane detection capabilities and improve overall performance.

## Dataset

We used the **KAIST-LANE (K-Lane)** dataset for training and testing the model. This dataset provides LiDAR point cloud data, which is essential for detecting lanes on road surfaces.

## Running Instructions

### Prerequisites

Before running the model, make sure the following dependencies are installed:

- Python 3.8 or higher
- PyTorch
- NumPy
- OpenCV
- Other dependencies listed in the `requirements.txt` file

### Setup

1. **Download Pre-trained Models**:

   We have provided two pre-trained models:
   - `ml_base.pth`: The training result of the baseline model (without our modifications).
   - `ml_curr_best.pth`: The training result of our model (with the modifications mentioned above).

   You can download these models from the following [Google Drive link](https://drive.google.com/drive/u/0/folders/1N3_2hsOI_295krR_cEMITu_RAbtyVbCJ).

2. **Place the Model in the Config Folder**:
   
   Once you have downloaded the `.pth` files, select one of them:
   - If you selected `ml_base.pth`, rename it to **`ml_curr_best.pth`**.
   - Put the selected file inside the `configs/` folder of the project.

### Running the Model

To run the lane detection model, use the following command:

```bash
python main.py --config <path_to_config_file> --checkpoint <path_to_ml_curr_best.pth>



## License
`K-Lane` is released under the Apache-2.0 license.

## Acknowledgement
The K-Lane benchmark is contributed by [Dong-Hee Paek](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=5), [Kevin Tirta Wijaya](https://www.ktirta.xyz/), [Dong-In Kim](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=13), [Min-Hyeok Sun](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_2&wr_id=14), advised by [Seung-Hyun Kong](http://ave.kaist.ac.kr/bbs/board.php?bo_table=sub1_1).

We thank the maintainers of the following projects that enable us to develop `K-Lane`:
[`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) by MMLAB, [`TuRoad`](https://github.com/Turoad/lanedet) bu TuZheng.

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. 2021R1A2C3008370).

## Citation

If you find this work is useful for your research, please consider citing:
```
@InProceedings{paek2022klane,
  title     = {K-Lane: Lidar Lane Dataset and Benchmark for Urban Roads and Highways},
  author    = {Paek, Dong-Hee and Kong, Seung-Hyun and Wijaya, Kevin Tirta},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshop on Autonomous Driving (WAD)},
  month     = {June},
  year      = {2022}
}
```
