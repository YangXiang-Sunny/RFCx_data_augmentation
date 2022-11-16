import os, sys, getopt
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback

from datagen import get_files_and_labels, scalespec, preprocess, DataGenerator
from learningrate import warmup_cosine_decay, WarmUpCosineDecayScheduler
from specinput import wave_to_mel_spec, load_audio, params



def usage():
    """
    Print usage of the file
    """
    print("Usage: ....")
    return 
    
    

def parse():
    """
    Parse command line arguments 
    """
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:s:a:l:", ["help", "input=", "train_val_split=", "aug=", "loss="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    
    input_file_type = None 
    train_val_split = None 
    aug_method = None 
    loss_name = None 
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-i", "--input"):
            input_file_type = arg
        elif opt in ("-s", "--train_val_split"):
            train_val_split = float(arg)
        elif opt in ("-a", "--aug"):
            aug_method = arg 
        elif opt in ("-l", "--loss"):
            loss_name = arg
        else:
            assert False, "unhandled option"
    
    return input_file_type, train_val_split, aug_method, loss_name
            
            

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
    

def train_val_split(data_path, train_split=0.8):
    """    
    Split data into train and vaildation sets by a given percentage
    
    Parameters
    ----------
    data_path: directory of data, should contain two folders p/ and n/ 
    train_split: given percentage of training data size 
    
    Return
    ------
    files_train: list of data files for training set
    files_val: list of data files for validation set
     
    """
    
    class_list = os.listdir(data_path + 'p')

    # generate positive train file paths
    files_train_p, files_val_p, labels = get_files_and_labels(data_path + '/p/',
                                                              train_split=train_split,
                                                              random_state=42,
                                                              classes=class_list)

    # generate negative train file paths
    files_train_n, files_val_n, labels_n = get_files_and_labels(data_path + '/n/',
                                                                train_split=train_split,
                                                                random_state=42,
                                                                classes=class_list) 
    
    labels_rev = dict((v,k) for (k,v) in labels.items())
    files_train_n = [i for i in files_train_n if i.split('/')[-2] in list(labels.keys())]
    
    files_train = files_train_p + files_train_n
    files_val = files_val_p + files_val_n
    
    return files_train, files_val



def generate_aug_data():
    return 



def generate_spec_data(path, files_train, files_val):
    
    # ----- Creat directory -----
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path + "/train/"):
        os.mkdir(path + "/train/")
    if not os.path.exists(path + "/val/"):
        os.mkdir(path + "/val/")

    path_train_p = path + "/train/p/"
    path_train_n = path + "/train/n/"
    path_val_p = path + "/val/p/"
    path_val_n = path + "/val/n/"

    path_all = [path_train_p, path_train_n, path_val_p, path_val_n]

    for p in path_all:
        if not os.path.exists(p):
            os.mkdir(p)  
    # --------------------------
    
    # ----- Save data -----
    start = time.time()

    for i, file in enumerate(files_input):

        if i % 1000 == 0:
            print(i, " / ", len(files_input))
            end = time.time()
            print("time elapsed: ", round(end - start, 2), "s")
            time.sleep(1)
            start = end

        class_name = file.split('/')[-2]
        if not os.path.exists(path_output + class_name):
            os.mkdir(path_output + class_name)

        # Copy the data file to new path 
        sample_data = np.load(file, allow_pickle=False)
        file_output_path = path_output + class_name + "/" + file.split('/')[-1]
        np.save(file_output_path, sample_data)
    
    
    return 

def train():
    return 

def evaluate():
    return 


# -------------------------------------- Functions for data augmentation --------------------------------------

def noise_injection(data, sample_rate):
    # noize_factor random between 0.001 to 0.005
    noise_factor = np.random.uniform(0.001, 0.005)
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shift_time(data, sampling_rate):
    #shift time between 0.1 to 1 second
    shift_max= np.random.uniform(0.1, 1)
    shift = int(np.round(sampling_rate * shift_max)) # np.random.randint(
    direction = np.random.randint(0, 2)
    if direction == 1:
        shift = -shift
    augmented_data = np.roll(data, shift)
    return augmented_data

def change_pitch(data, sample_rate):
    n_step = np.random.uniform(-2,2)
    augmented_data = librosa.effects.pitch_shift(y=data, sr=sample_rate, n_steps=n_step)
    return augmented_data

def change_speed(data, sample_rate):
    speed_factor= np.random.uniform(0.75, 1.25) 
    augmented_data = librosa.effects.time_stretch(data, speed_factor)
    return augmented_data

# --------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    # Parse command line arguments 
    input_file_type, train_val_split, aug_method, loss_name = parse() 
    
    print(input_file_type, train_val_split, aug_method, loss_name)
    
    sys.exit()
    
    # Remove .DS_Store files in data directory 
    print("------------- Start file removal  -------------")
    file_to_remove='.DS_Store'
    remove_file(audio_data_path)
    print("------------- File removal completed -------------")

    # Train validation split 
    audio_data_path = "/n/home11/yxiang/data/rfcx-harvard-ds/puerto-rico/train/audio/"
    train_split = 0.8
    files_train, files_val = train_val_split(audio_data_path, train_split)

    # Generate augmented data 


    # Generate spectrogram data 
    output_spec_path = "/n/home11/yxiang/data/spec_data/"


