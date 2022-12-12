# RFCx_data_augmentation
The repo includes a pipeline for running data augmentation experiments with audio data of Puerto Rico rainforest species, provided by [Rainforest Connection](https://rfcx.org/). 

## About 

Tropical ecosystems are particularly characterized by the high number of rare and inconspicuous species. However, it is hard to collect sufficient data samples corresponding to these rare species in the tropical forests, while a good model will require sufficient training data. Therefore, we aim to help RFCx improve the data augmentation process to increase the training data size for rare species, better understand the sounds of rare species, and promote the model performance of rare species. Our project group members are Yang Xiang, Ziye Tao, Li Sun, Yanqi Luo and Meichen Dong, supervised by Jack Lebien, Marconi Campos and Nelson Buainain from Rainforest Connection Team, and Orhan Eren Akgun and Weiwei Pan from Harvard. 

This repo allows you to reproduce some of our experiment results, for both audio augmentation and spectrogram augmentation methods. 

## Data 

The Puerto Rico audio data can be obtained with gcloud CLI. 

1. Install the gcloud CLI with the link [here](https://cloud.google.com/sdk/docs/install). 

2. Authorize the account with the following command

   ````shell
   gcloud auth activate-service-account rfcx-harvard-ds --key-file=./rfcx-models-dev-fb2d3411b649.json
   ````

3. Use `gsutil` to access data in the `rfcx-harvard-ds` bucket.

   * List bucket contents 

     ````shell
     gsutil ls gs://rfcx-harvard-ds/
     ````

   * Download all data

     ````shell
     gsutil -m cp -r gs://rfcx-harvard-ds/* ./
     ````

## Requirements

The pipeline requires some Python packages to run, including `tensorflow`, `librosa` and `nlpaug`. The dependencies are stored in `environment.yml`, and it could be activated with the following command. 
```shell
conda env create -f environment.yml
```

## Usage

The pipeline helps you to split the data into training and validation set, generate spectrogram data with or without audio or spectrogram augmentation, train the model and evaluate the model on training, validation and test set. It also allows you to skip some of the procedures to accelerate the process. For example, you could skip generating the spectrogram data if the spectrogram data is already there, by setting the spectrogram data as the input. The model training process could also be skipped if you already have a model and want to evaluate it. 

Here are some sample usages. 

* Audio data input without augmentation 

```shell
python ./code/main.py --input audio --input_path INPUT_AUDIO_DATA_PATH --output_spec_path OUTPUT_SPECTROGRAM_DATA_PATH --train_val_split 0.8 --loss binary_crossentropy --model_path OUTPUT_MODEL_PATH --test_path TEST_DATA_PATH --output_test_path OUTPUT_TEST_DATA_PATH --test_label_path TEST_LABEL_CSV_PATH 
```

* Audio data input with noise injection augmentation 

```shell
python ./code/main.py --input audio --input_path INPUT_AUDIO_DATA_PATH --output_spec_path OUTPUT_NOISE_INJECTION_SPECTROGRAM_DATA_PATH --train_val_split 0.8 --aug noise_injection --loss binary_crossentropy --model_path OUTPUT_MODEL_PATH --test_path TEST_DATA_PATH --output_test_path OUTPUT_TEST_DATA_PATH --test_label_path TEST_LABEL_CSV_PATH 
```

* spectrogram input with frequency mask augmentation, skip training process 

```shell
python ./code/main.py --input spec --input_path FREQUENCY_MASK_SPECTROGRAM_DATA_PATH --model_path FREQUENCY_MASK_MODEL_PATH --skip_train --test_path TEST_DATA_PATH --output_test_path OUTPUT_TEST_DATA_PATH --test_label_path TEST_LABEL_CSV_PATH 
```

The Design document link is [here](https://docs.google.com/document/d/1Fxcv6K84TplhNJIzEOL_pvuI2S2_y8KVa4ERp4DwUfo/edit?usp=sharing).

