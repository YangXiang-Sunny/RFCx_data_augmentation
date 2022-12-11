# RFCx_data_augmentation
Audio data augmentation for Puerto Rico rainforest species audio data

## About 

* Group members: Yang Xiang, Ziye Tao, Li Sun, Yanqi Luo, Meichen Dong 

## Requirements

The pipeline requires some Python packages to run, including `tensorflow`, `librosa` and `nlpaug`. The dependencies are stored in `environment.yml`, and it could be activated with the following command. 
```shell
conda env create -f environment.yml
```

## Usage

* Design doc 
	* https://docs.google.com/document/d/1Fxcv6K84TplhNJIzEOL_pvuI2S2_y8KVa4ERp4DwUfo/edit?usp=sharing

* Execute the following command to run the pipeline 

  ```shell
  python ./code/main.py --input audio --train_val_split 0.2 --aug noise_injection --loss masked_loss
  ```

