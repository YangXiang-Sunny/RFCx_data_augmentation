import os, sys
import time 
import pandas as pd
import numpy as np
import librosa

from specinput import wave_to_mel_spec, load_audio, params
from datagen import get_files_and_labels


def remove_file(data_path, file_to_remove='.DS_Store'):
    """
    Remove unnecessary .DS_Store
    """
    queue = os.listdir(data_path)

    while queue:
        for i in range(len(queue)):
            file = queue.pop(0)
            
            if file.split('/')[-1] == file_to_remove:
                # Remove target file 
                os.remove(data_path + file)
                print("File removed:", data_path + file)
                continue
                
            if os.path.isdir(data_path + file):
                # Update contents of folder to queue 
                queue += ['/' + file + '/' + f for f in os.listdir(data_path + file)]
    

def generate_class_freqs(path):
    """
    Generate class-freqs.npy based on class-meta.csv
    """
#     # path to class-meta.csv file
#     path = './class-meta.csv' 

    # amount to expand each class' frequency band by on either side
    buffer_hertz = 1000

    # get a vector of frequencies corresponding to the spectrogram rows based on the specinput.py params
    melfreqs = librosa.mel_frequencies(n_mels=params.mel_bands, fmin=params.mel_min_hz, fmax=params.mel_max_hz)

    # compute min/max frequency indices
    df = pd.read_csv(path, index_col=0)
    spf = []
    for i in sorted(list(set(df['Class']))):
        spf.append([i.replace(' ',''),
                   np.argmin(np.abs(melfreqs - np.min(df[df['Class']==i]['Min Frequency (Hz)'] - buffer_hertz))), 
                   np.argmin(np.abs(melfreqs - np.max(df[df['Class']==i]['Max Frequency (Hz)'] + buffer_hertz)))])
    spf = np.stack(spf)

    np.save('class-freqs.npy', spf)

    
def transfer_data(files_input, path_output, mode="npy_to_npy"):
    """
    files_input: a list of all input data files 
    path_output: directory for saving output data files 
    mode: npy_to_npy / wav_to_npy

    Example - transfer_data(files_train_p, path_train_p)
    """
    files_output = []

    start = time.time()

    for i, file in enumerate(files_input):

        if i % 1000 == 0:
            print(i, " / ", len(files_input))
            end = time.time()
            print("time elapsed: ", round(end - start, 2), "s")
#             time.sleep(1)
            start = end

        class_name = file.split('/')[-2]
        p_or_n = file.split('/')[-3]
        
        if not os.path.exists(path_output + '/' + p_or_n + '/' + class_name):
            folder_path = path_output + '/' + p_or_n + '/' + class_name
            os.mkdir(folder_path)
            print("Created folder: ", folder_path)
        
        if mode == "npy_to_npy":
            # Copy the data file to new path 
            sample_data = np.load(file, allow_pickle=False)
            file_output_path = path_output + "/" + p_or_n + "/" + class_name + "/" + file.split('/')[-1]
            np.save(file_output_path, sample_data)
            files_output.append(file_output_path)
#             print("Saved file: ", file_output_path)
        elif mode == "wav_to_npy":
            if file[-4::] != ".wav":
                print("Skip file: ", file)
                continue
            data, sample_rate = load_audio(file)
            file_output_path = path_output + "/" + p_or_n + "/" + class_name + "/" + file.split('/')[-1][0:-4] + ".npy"
            spec_data = wave_to_mel_spec(data)
            np.save(file_output_path, spec_data)
            files_output.append(file_output_path)
#             print("Saved file: ", file_output_path) 
        else:
            assert False, "unhandled mode"
        
    return files_output 


def get_spec_data(data_path):
    """
    Grab train-val split spectrogram data 
    """
    class_list = os.listdir(data_path + '/train/p/')

    # generate positive train file paths
    files_train_p, _, labels = get_files_and_labels(data_path + '/train/p/', 
                                                    train_split = 1,
                                                    classes = class_list)

    # generate negative train file paths
    files_train_n, _, labels_train_n = get_files_and_labels(data_path + '/train/n/',
                                                            train_split = 1,
                                                            classes = class_list) 
    
    # generate positive validation file paths
    files_val_p, _, labels_val_p = get_files_and_labels(data_path + '/val/p/',
                                                        train_split = 1,
                                                        classes = class_list)

    # generate negative train file paths
    files_val_n, _, labels_val_n = get_files_and_labels(data_path + '/val/n/',
                                                        train_split = 1,
                                                        classes = class_list) 
    
    files_train_n = [i for i in files_train_n if i.split('/')[-2] in list(labels.keys())]
    files_val_n = [i for i in files_val_n if i.split('/')[-2] in list(labels.keys())]
    
    files_train = files_train_p + files_train_n 
    files_val = files_val_p + files_val_n 
    
    return files_train, files_val, labels 